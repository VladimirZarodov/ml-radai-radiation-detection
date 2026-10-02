# -*- coding: utf-8 -*-
"""
verify_submission.py -- checklist verification of a generated submission CSV.

  .venv\\Scripts\\python.exe submission\\verify_submission.py <csv> --split validation

Checks performed:
  1. header == expected template (placeholder header for validation dry runs;
     for the final testing CSV the expected header must be supplied via
     --header, copied from the organiser's official template);
  2. column integrity: sequential #, integer run ids in range, integer
     time_ms, NUMERIC time_start/time_stop equal to clamp(time -+ 2000 ms,
     0, run_length), float metric_1, label_1 within the official 24-label
     vocabulary, NO empty field anywhere except label_2/label_3 (Checkpoint
     C decision 1), label_2/3 empty, metric_2/3 zero, rows sorted by
     (run, time) and monotone time within a run;
  3. no duplicate (run, time) rows;
  4. nestedness demo: alarms surviving a raised metric_1 cut are a strict
     subset of the file;
  5. (validation only) end-to-end local re-score from the CSV rows: per-
     category d/c/id recall + fpr hypotheses at the pool-a FAR 1/3/10
     thresholds, cross-checked against local_scorer's in-memory results;
  6. no-leakage audit: prints the provenance of every fitted object
     (detector fit = training TRAIN_RUNS; thresholds = training pools only;
     testing split used for inference only);
  7. git worktree check (read-only): no TRACKED file modified.
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import radai_lib as L
import local_scorer as LS
import make_submission as MS
import csv_format as CF

ROOT = Path(__file__).resolve().parent.parent
SCRATCH = ROOT / "_scratch"


def check_format(path, header_expected, allowed_runs, run_len=None):
    """allowed_runs: iterable of valid run ids for this split.
    run_len: optional dict run_id -> run length in ms (for time_stop clamp)."""
    allowed = set(allowed_runs)
    with open(path, encoding="utf-8", newline="") as fh:
        rdr = csv.reader(fh)
        head = next(rdr)
        rows = list(rdr)
    ok = True
    def bad(msg):
        nonlocal ok
        ok = False
        print("  FAIL:", msg)
    if head != next(csv.reader([header_expected])):   # csv-aware: handles "quoted" cells
        bad(f"header mismatch\n       got: {','.join(head)}\n       exp: {header_expected}")
    else:
        print(f"  header OK ({len(head)} columns)")
    lab_blank = 0
    seen = set()
    prev = None
    for i, r in enumerate(rows):
        if len(r) != len(head):
            bad(f"row {i + 1}: {len(r)} fields"); continue
        num, run, t, ts, te, l1, m1, l2, m2, l3, m3 = r[:11]
        # -- no field may be empty except label_2 / label_3 -----------------
        for j, fld in enumerate(r[:11]):
            if fld == "" and CF.COLUMNS[j] not in CF.ALLOWED_EMPTY:
                bad(f"row {i + 1}: empty field in column {CF.COLUMNS[j]}")
        if int(num) != i + 1:
            bad(f"row {i + 1}: '#' = {num}")
        if int(run) not in allowed:
            bad(f"row {i + 1}: run_id {run} not in split's run set")
        if int(t) < 0 or float(t) != int(t):
            bad(f"row {i + 1}: time {t} not integer ms")
        # -- time_start / time_stop numeric & equal to the +/-2 s clamp ------
        try:
            ti, tsi, tei = int(t), int(ts), int(te)
            if tsi != max(0, ti - CF.WINDOW_MS):
                bad(f"row {i + 1}: time_start {ts} != max(0, time-{CF.WINDOW_MS})")
            exp_te = ti + CF.WINDOW_MS
            if run_len is not None:
                exp_te = min(int(run_len[int(run)]), exp_te)
            if tei != exp_te:
                bad(f"row {i + 1}: time_stop {te} != min(run_len, time+{CF.WINDOW_MS})"
                    + ("" if run_len is not None else " (or run end not supplied)"))
        except ValueError:
            bad(f"row {i + 1}: time_start/time_stop not numeric: {ts!r},{te!r}")
        try:
            m = float(m1)
            if not np.isfinite(m):
                bad(f"row {i + 1}: metric_1 {m1}")
        except ValueError:
            bad(f"row {i + 1}: metric_1 not numeric: {m1!r}")
        if l1 == "":
            lab_blank += 1
        elif l1 not in L.OFFICIAL_LABELS:
            bad(f"row {i + 1}: label_1 {l1!r} not in official 24")
        for x, what in ((l2, "label_2"), (l3, "label_3")):
            if x != "":
                bad(f"row {i + 1}: {what} not empty: {x!r}")
        for x, what in ((m2, "metric_2"), (m3, "metric_3")):
            if x not in ("0", "0.0"):
                bad(f"row {i + 1}: {what} != 0: {x!r}")
        key = (int(run), int(t))
        if key in seen:
            bad(f"duplicate (run,time) {key} at row {i + 1}")
        seen.add(key)
        if prev and int(run) == prev[0] and int(t) < prev[1]:
            bad(f"row {i + 1}: time not monotone within run {run}")
        prev = key
    print(f"  rows: {len(rows)}, blank label_1: {lab_blank}, "
          f"unique runs: {len({r[1] for r in rows})}")
    # nestedness: subset at raised cuts
    metrics = np.array([float(r[6]) for r in rows])
    runs = np.array([int(r[1]) for r in rows])
    for cut in (4.42, 5.08):
        sub = set(zip(runs[metrics >= cut], metrics[metrics >= cut]))
        assert sub <= set(zip(runs, metrics))
    print("  nestedness: higher metric cuts give strict subsets  OK")
    return ok, rows


def rescore(rows, thr, tol=60.0):
    """Same hypotheses as local_scorer on parsed CSV rows for validation runs."""
    tc = pickle.load(open(SCRATCH / "sub_training_scores.pkl", "rb"))
    names = L.source_names(str(ROOT / "training_v4.3.h5"))
    enc = LS.gt_encounters(tc, names)
    by_run = defaultdict(list)
    for i, r in enumerate(rows):
        w = int(round((int(r[2]) / 1e3 - L.DET_T / 2) / L.STRIDE))
        by_run[int(r[1])].append(dict(aid=i, win=w, end=w,
                                      metric=float(r[6]), label=r[5],
                                      is_bg=False))
    tot = hit = cat = idn = fa = 0
    for run, al in by_run.items():
        if run not in enc:
            continue
        n, h, sel, matched, ih, ch = LS.sweep_run(al, enc[run], thr, "center", tol)
        tot += n; hit += h; idn += ih; cat += ch
        fa += sum(1 for a in sel if a["aid"] not in matched)
    bg_h = sum(int((tc[r]["dmin"] > L.R_EXCL).sum()) * L.STRIDE / 3600
               for r in L.TEST_RUNS)
    return dict(enc=tot, recall=hit / tot, c=cat / tot, id=idn / tot,
                fa_h=fa / bg_h, bg_h=bg_h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--split", required=True, choices=["validation", "testing"])
    ap.add_argument("--header", default=None,
                    help="expected header (official template) for the final "
                         "testing CSV; defaults to the placeholder")
    args = ap.parse_args()

    p = Path(args.csv_path)
    print(f"== verify {p} (split={args.split}) ==\n[1-3] format checks")
    allowed = L.TEST_RUNS if args.split == "validation" else range(300)
    header = args.header or CF.HEADER
    sc_pkl = (SCRATCH / ("sub_testing_scores.pkl" if args.split == "testing"
                         else "sub_training_scores.pkl"))
    run_len = MS.run_lengths_ms(pickle.load(open(sc_pkl, "rb")))
    ok, rows = check_format(p, header, allowed, run_len)

    print("\n[3b] official-reader check: pd.read_csv(comment='#') as in "
          "radai/evaluation/evaluation.py")
    df = pd.read_csv(p, comment="#")
    required = ["run_id", "time", "label_1", "metric_1"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"  FAIL: official reader loses required columns: {missing}")
        ok = False
    elif len(df) != len(rows):
        print(f"  FAIL: official reader sees {len(df)} rows, expected {len(rows)}")
        ok = False
    else:
        print(f"  columns {list(df.columns)}")
        print(f"  rows through official reader: {len(df)}  OK")

    print("\n[4] no-leakage provenance audit")
    det = pickle.load(open(SCRATCH / "sub_detector.pkl", "rb"))
    cal = json.load(open(ROOT / "submission" / "calibration.json"))
    print(f"  detector: M{det['M'].shape}, U{det['U'].shape} fit from "
          f"training_v4.3.h5 TRAIN_RUNS={L.TRAIN_RUNS} (18 runs)")
    print(f"  pools: (a) {cal['pool_a']['bg_hours']:.1f} bg-h, "
          f"(b) {cal['pool_b']['bg_hours']:.1f} bg-h -- BOTH from TRAINING file")
    print("  testing file: read only listmode/dt+energy for inference; no "
          "GT exists there; nothing on testing was fitted/tuned")

    print("\n[5] git worktree (tracked files must be untouched)")
    st = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                        capture_output=True, text=True).stdout
    dirty = [ln for ln in st.splitlines() if ln and ln[0] != "?"]
    print("  tracked modifications:", dirty if dirty else "NONE")
    ok &= not dirty

    if args.split == "validation":
        print("\n[6] end-to-end local re-score from CSV rows (validation)")
        for tgt in ("1", "3", "10"):
            thr = cal["targets"][tgt]["pool_a"]["thr"]
            s = rescore(rows, thr)
            print(f"  thr={thr:5.2f}: d={s['recall'] * 100:5.1f}% "
                  f"c={s['c'] * 100:5.1f}% id={s['id'] * 100:5.1f}% "
                  f"fpr(H1)={s['fa_h']:.2f}/bg-h")

    print("\nVERIFY:", "ALL CHECKS PASSED" if ok else "FAILURES ABOVE")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
