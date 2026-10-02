# -*- coding: utf-8 -*-
"""
local_scorer.py -- validation-only local scoring & sweep-consistency proof.

Modes (TRAINING-file GT only; testing has no GT here, so nothing in this file
ever touches testing outputs):

  proof  : show that sweeping metric_1 over EMITTED mx31-peak alarms
           reproduces the notebook detection behaviour on the internal
           hold-out (runs 25-124):
             * "plateau" rule -- an alarm counts for encounter CA if its
               mx31 plateau overlaps [CA-60, CA+60].  Mathematically the
               same criterion as the notebook's "max mx31 in CA +- 60 >=
               thr", modulo peaks removed by min-separation thinning.
             * "center" rule  -- the alarm's peak window itself within
               CA +- 60 s (simpler, portal-plausible).
           Prints recall diffs vs the notebook criterion, FA/alarm rates,
           tolerance and separation sensitivity, a full sweep curve CSV, and
           verifies nestedness (higher threshold -> subset of alarms).

  table  : per-category d/c/id recall + fpr hypotheses at one threshold.

Hypotheses (unconfirmed by organiser; documented, tested for sensitivity):
  * detection tolerance: |alarm - CA| <= 60 s (sensitivity: 30/60/120);
  * FPR DEFINITION (Checkpoint C decision): false alarms per BACKGROUND hour,
    measured here on the validation runs' own background time (dmin > 150 s,
    notebook FARM mask).  For extrapolating to the official leaderboard we
    ASSUME the portal normalises by ~286.5 background hours in testing --
    evidence: leaderboard fpr values are integer counts / 286.5 (~40 checked);
    the true portal denominator is UNCONFIRMED, so all per-hour numbers here
    are "per background hour" under this assumption;
  * per-category fpr attributes false alarms to the category of the
    ALARM'S label_1 (user hypothesis);
  * TP label for c/id recall: highest-metric matched alarm (ties: earlier).
"""
from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

import radai_lib as L

ROOT = Path(__file__).resolve().parent.parent
SCRATCH = ROOT / "_scratch"
OUTDIR = Path(__file__).resolve().parent

TRAIN_SCORES = SCRATCH / "sub_training_scores.pkl"
CALIB_JSON = OUTDIR / "calibration.json"           # moved out of _scratch (Checkpoint B #4)
DET_PKL = SCRATCH / "sub_detector.pkl"
CURVE_CSV = OUTDIR / "validation_sweep_curve.csv"
MX_H = (L.MX_N - 1) * L.STRIDE          # rolling-max horizon: 30 windows = 60 s
BG_HOURS_TESTING_ASSUMED = 286.5        # ASSUMED portal fpr denominator (unconfirmed)


def load_state(sep):
    """Emitted-style alarm rows for validation runs (no floor applied:
    the sweep filters by metric >= thr, exactly as the portal would)."""
    det = pickle.load(open(DET_PKL, "rb"))
    tc = pickle.load(open(TRAIN_SCORES, "rb"))
    cal = json.load(open(CALIB_JSON))
    names = L.source_names(str(ROOT / "training_v4.3.h5"))
    alarms = {}
    for r in L.TEST_RUNS:
        c = tc[r]
        p = L.mx31_peaks(c["mx31"], sep=sep)
        v = c["mx31"][p]
        rows = []
        for k in range(len(p)):
            w = int(p[k])
            tid = int(det["ids"][int(c["am"][w])])
            rows.append(dict(aid=k, run=r, win=w,
                             end=L.plateau_end(c["mx31"], w),
                             metric=float(v[k]),
                             label=L.official_label(names[tid]),
                             is_bg=bool(c["dmin"][w] > L.R_EXCL)))
        alarms[r] = rows
    return det, tc, cal, names, alarms


def gt_encounters(tc, names):
    """Per-run GT encounters: list of (CA time s, true label, category)."""
    enc = {}
    for r in L.TEST_RUNS:
        c = tc[r]
        enc[r] = [(float(t), L.official_label(names[int(sid)]),
                   L.LABEL2CAT.get(L.official_label(names[int(sid)])))
                  for t, sid in zip(c["stime"], c["sid"])]
    return enc


def sweep_run(alarms_r, encs_r, thr, rule="plateau", tol=60.0):
    """Sweep at threshold thr for one run.

    Returns (n_enc, n_hits, sel, matched_aids, id_hit, cat_hit), where an
    alarm is 'matched' if it overlaps some detected encounter (±tol), and
    id/cat hits use the highest-metric matched alarm of the encounter.
    """
    sel = [a for a in alarms_r if a["metric"] >= thr]
    matched, id_hit, cat_hit, hits = set(), 0, 0, 0
    for (ca, lab, cat) in encs_r:
        cand = []
        for a in sel:
            ctr = a["win"] * L.STRIDE + L.DET_T / 2.0
            if rule == "plateau":
                ov = not (a["end"] * L.STRIDE + L.DET_T < ca - tol
                          or a["win"] * L.STRIDE > ca + tol)
            elif rule == "exact":
                # peak at window p dominates mx31[i] for every i in [p, p+30];
                # so "max mx31 >= thr within CA +- tol" holds iff some emitted
                # peak with value >= thr has its centre in [CA-tol-60, CA+tol]
                # (60 s = MX_N windows * STRIDE, the rolling-max horizon).
                ov = (ca - tol - MX_H <= ctr) and (ctr <= ca + tol)
            else:
                ov = abs(ctr - ca) <= tol
            if ov:
                cand.append(a)
        if cand:
            hits += 1
            best = max(cand, key=lambda a: (a["metric"], -a["win"]))
            if L.LABEL2CAT.get(best["label"]) == cat:
                cat_hit += 1
            if best["label"] == lab:
                id_hit += 1
            matched.update(a["aid"] for a in cand)
    return len(encs_r), hits, sel, matched, id_hit, cat_hit


def sweep_all(alarms, enc, thr, rule="plateau", tol=60.0):
    tot = hit = idh = cath = nal = fa_unm = fa_bg = 0
    for r in L.TEST_RUNS:
        n, h, sel, matched, ih, ch = sweep_run(alarms[r], enc[r], thr, rule, tol)
        tot += n; hit += h; idh += ih; cath += ch; nal += len(sel)
        fa_unm += sum(1 for a in sel if a["aid"] not in matched)
        fa_bg += sum(1 for a in sel if a["is_bg"])
    return dict(recall=hit / tot, alarms=nal, fa_unmatched=fa_unm,
                fa_bg=fa_bg, id_hits=idh, cat_hits=cath, n_enc=tot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["proof", "table"])
    ap.add_argument("--sep", type=int, default=15)
    ap.add_argument("--thr", type=float, default=None)
    args = ap.parse_args()

    det, tc, cal, names, alarms = load_state(args.sep)
    enc = gt_encounters(tc, names)
    farm = L.farm_of(tc, L.TEST_RUNS)
    bg_h = sum(int(m.sum()) * L.STRIDE / 3600 for m in farm.values())
    total_h = sum(tc[r]["wtime"][-1] / 3600 + L.STRIDE / 3600
                  for r in L.TEST_RUNS)
    ntot = sum(len(v) for v in enc.values())
    floor = cal["floor"]["thr"]
    print(f"sep={args.sep}w | validation: {len(L.TEST_RUNS)} runs, {ntot} "
          f"encounters | bg {bg_h:.1f} h, total {total_h:.1f} h | "
          f"floor={floor:.3f}")
    print("NOTE: fpr here = FA per validation BACKGROUND hour (dmin>150). "
          "Testing extrapolation assumes the portal denominator ~286.5 bg-h "
          "(leaderboard evidence; UNCONFIRMED).")

    NOTE = {1: (5.08, 25.5), 3: (4.42, 40.5), 10: (3.59, 69.9)}  # B_all table

    if args.mode == "proof":
        below = [t for t in NOTE if cal["targets"][str(t)]["pool_a"]["thr"] < floor]
        if below:
            print(f"WARNING: target thresholds {below} below floor -> "
                  f"sweep truncated; raise floor headroom")
        emax = np.concatenate([L.emax_mx(tc, r) for r in L.TEST_RUNS])
        print(f"GT check: EMAX encounters = {len(emax)} (README: 943)")
        print("\n-- sweep reproduction at pool-a thresholds (episode FAR for "
              "reference) --")
        maxd = 0.0
        for tgt, (thr_nb, rec_nb) in NOTE.items():
            thr = cal["targets"][str(tgt)]["pool_a"]["thr"]
            rec_ref = float((emax >= thr).mean()) * 100
            for rule in ("exact", "plateau", "center"):
                s = sweep_all(alarms, enc, thr, rule)
                rec = s["recall"] * 100
                d = abs(rec - rec_ref)
                maxd = max(maxd, d)
                print(f"  FAR={tgt:<2} thr={thr:5.2f} {rule:7}: "
                      f"sweep={rec:5.1f}% vs EMAX={rec_ref:5.1f}% "
                      f"(notebook={rec_nb}%) diff={d:.2f}pp | "
                      f"alarms={s['alarms']} ({s['alarms'] / bg_h:.1f}/bg-h) "
                      f"fa_bg={s['fa_bg']} ({s['fa_bg'] / bg_h:.1f}/bg-h)")
        print(f"max |sweep - notebook criterion| = {maxd:.2f} pp")

        print("\n-- center-rule tolerance sensitivity (FAR=3 thr) --")
        thr3 = cal["targets"]["3"]["pool_a"]["thr"]
        for tol in (30, 60, 120):
            s = sweep_all(alarms, enc, thr3, "center", tol)
            print(f"  tol=+-{tol:>3}s: recall={s['recall'] * 100:.1f}%")

        print("\n-- separation sensitivity (rows at floor) --")
        for sep in (5, 15, 31):
            if sep != args.sep:
                _, _, _, _, a2 = load_state(sep)
            else:
                a2 = alarms
            n = sum(1 for r in L.TEST_RUNS for a in a2[r] if a["metric"] >= floor)
            nb_ = sum(1 for r in L.TEST_RUNS for a in a2[r]
                      if a["metric"] >= floor and a["is_bg"])
            print(f"  sep={sep:>2}: rows={n} ({n / total_h:.1f}/run-h), "
                  f"bg rows={nb_} ({nb_ / bg_h:.1f}/bg-h)")

        print(f"\n-- full sweep curve (exact rule) -> {CURVE_CSV.name} --")
        cand_show = [c for c in cal["pool_a"]["cand"] if c >= floor]
        curve = []
        for t in cand_show:
            s = sweep_all(alarms, enc, t, "exact")
            curve.append((t, s))
        with open(CURVE_CSV, "w", encoding="utf-8") as fh:
            fh.write("thr,recall,alarms,fa_unmatched,fa_bg,alarms_per_bg_h,"
                     "fa_bg_per_bg_h\n")
            for t, s in curve:
                fh.write(f"{t:.6f},{s['recall']:.6f},{s['alarms']},"
                         f"{s['fa_unmatched']},{s['fa_bg']},"
                         f"{s['alarms'] / bg_h:.4f},{s['fa_bg'] / bg_h:.4f}\n")
        for want in (0.5, 1, 3, 10):
            b = min(curve, key=lambda x: abs(x[1]["fa_bg"] / bg_h - want))
            print(f"  fa_bg~{want}/h: thr={b[0]:.2f} recall={b[1]['recall'] * 100:5.1f}% "
                  f"fa={b[1]['fa_bg'] / bg_h:.2f}/h alarms={b[1]['alarms'] / bg_h:.1f}/h")
        thrs = [t for t, _ in curve]
        cnts = [s["alarms"] for _, s in curve]
        # strongest form: alarms(t_high) must be a SUBSET of alarms(t_low)
        sel_sets = []
        for t in thrs:
            sel_sets.append({a["aid"] for r in L.TEST_RUNS
                             for a in alarms[r] if a["metric"] >= t})
        nested = all(sel_sets[j] <= sel_sets[i]
                     for i in range(len(thrs)) for j in range(i))
        mono = (all(thrs[i] >= thrs[i + 1] for i in range(len(thrs) - 1)) and
                all(cnts[i] <= cnts[i + 1] for i in range(len(cnts) - 1)))
        print(f"  nestedness: strict set-inclusion {'PASS' if nested else 'FAIL'}; "
              f"count monotone with threshold {'PASS' if mono else 'FAIL'}")

    else:  # table
        thr = args.thr if args.thr is not None else \
            cal["targets"]["3"]["pool_a"]["thr"]
        print(f"\nper-category at thr={thr:.3f}, center rule tol=60 "
              "(hypotheses in docstring)")
        per = {c: dict(e=0, d=0, ci=0, ii=0) for c in L.OFFICIAL_CATEGORIES}
        fa_lab = defaultdict(int)
        nal = nun = 0
        for r in L.TEST_RUNS:
            n, h, sel, matched, *_ = sweep_run(alarms[r], enc[r], thr, "center")
            nal += len(sel)
            nun += sum(1 for a in sel if a["aid"] not in matched)
            for a in sel:
                if a["aid"] not in matched:
                    fa_lab[L.LABEL2CAT.get(a["label"], "?")] += 1
            for (ca, lab, cat) in enc[r]:
                cand = [a for a in sel
                        if abs(a["win"] * L.STRIDE + L.DET_T / 2 - ca) <= 60]
                p = per[cat]
                p["e"] += 1
                if cand:
                    p["d"] += 1
                    best = max(cand, key=lambda a: (a["metric"], -a["win"]))
                    if L.LABEL2CAT.get(best["label"]) == cat:
                        p["ci"] += 1
                    if best["label"] == lab:
                        p["ii"] += 1
        print(f"{'category':17}{'enc':>5}{'d_recall':>10}{'c_recall':>10}"
              f"{'id_recall':>10}")
        for cat, p in per.items():
            e = max(p["e"], 1)
            print(f"{cat:17}{p['e']:>5}{p['d'] / e * 100:9.1f}%"
                  f"{p['ci'] / e * 100:9.1f}%{p['ii'] / e * 100:9.1f}%")
        print(f"rows={nal} | unmatched FAs={nun} "
              f"({nun / bg_h:.2f}/bg-h, {nun / total_h:.2f}/run-h)")
        print("FA by alarm-label category: " +
              ", ".join(f"{k}={v}" for k, v in sorted(fa_lab.items())))


if __name__ == "__main__":
    main()
