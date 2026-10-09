# -*- coding: utf-8 -*-
"""
harness_freeze.py -- Stage 0: FROZEN lockbox / dev folds / shift slice.

Run exactly once (refuses to overwrite unless --force and a confirmation).
Stratification keys: background count rate (median cps of rate_all over the
first 600 s and overall, max as "high-bg stretch" proxy) and n_encounters.

Outputs (submission_v2/cache/):
  lockbox.json   60 run ids (never used for anything except the single
                 Stage-4 finalist evaluation)
  folds.json     240 dev runs -> 5 folds by run
  shift_slice.json  dev runs sorted by bg rate; top-20 % ids (HIGH-BG slice)
  freeze_report.md provenance: inputs, strata, rng seed, ids.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import v2_lib as V

SEED = 20261003
N_LOCK = 60


def strata_keys(bg, nenc):
    """(bg quartile, encounter-count bucket)."""
    q = pd.qcut(bg, 4, labels=False, duplicates="drop")
    b = pd.cut(nenc, [-1, 7, 9, 11, 10_000], labels=["<=7", "8-9", "10-11",
                                                     ">=12"])
    return q.astype(int).astype(str) + "_" + b.astype(str)


def main(force=False):
    for nm in ("lockbox.json", "folds.json", "shift_slice.json"):
        p = V.CACHE / nm
        if p.exists() and not force:
            raise SystemExit(f"{p} already exists -- frozen. Use --force only "
                             f"if nothing has been trained yet.")
    meta = V.run_meta("train")
    log = json.load(open(V.CACHE / "build_log_train.json"))
    bg = pd.Series({p["run"]: p["cps_med"] for p in log["per_run"]})
    p99 = pd.Series({p["run"]: p["cps_p99"] for p in log["per_run"]})
    nenc = pd.Series({r: m["n_enc"] for r, m in meta.items()})
    runs = np.array(sorted(meta))
    assert set(runs) == set(bg.index) == set(nenc.index), "cache/log mismatch"
    key = pd.Series(strata_keys(bg[runs], nenc[runs]), index=runs)
    rng = np.random.default_rng(SEED)

    # ---- lockbox: proportional stratified pick of 60 ------------------------
    lock = []
    groups = pd.Series(runs).groupby(key[runs].to_numpy())
    for gname, ids in groups:
        ids = np.sort(np.asarray(ids))
        frac = len(ids) / len(runs)
        want = int(round(frac * N_LOCK))
        if want:
            pick = rng.choice(ids, size=min(want, len(ids)), replace=False)
            lock.extend(sorted(pick.tolist()))
    if len(lock) > N_LOCK:                        # adjust in largest stratum
        lock = sorted(lock)[:N_LOCK]
    while len(lock) < N_LOCK:                     # fill from largest strata
        for gname, ids in sorted(groups, key=lambda kv: -len(kv[1])):
            cand = [r for r in np.sort(np.asarray(ids)) if r not in lock]
            if cand:
                lock.append(int(cand[0]))
                break
    lock = sorted(lock)
    dev = [r for r in runs if r not in set(lock)]
    assert len(dev) == 240 and len(lock) == 60

    # ---- 5 folds: stratified round-robin over strata ------------------------
    folds = [[] for _ in range(5)]
    for gname, ids in groups:
        ids = np.sort(np.asarray([r for r in ids if r in set(dev)]))
        perm = rng.permutation(len(ids))
        for j, r in enumerate(ids[perm]):
            folds[j % 5].append(int(r))
    folds = [sorted(f) for f in folds]
    sizes = [len(f) for f in folds]
    assert sorted(sum(folds, [])) == sorted(dev), "fold coverage broken"

    # ---- shift slice: top 20% dev runs by bg median rate --------------------
    dev_bg = bg[dev].sort_values(ascending=False, kind="stable")
    hi = sorted(int(r) for r in dev_bg.index[:max(1, round(0.2 * len(dev)))])

    V.CACHE.mkdir(exist_ok=True)
    (V.CACHE / "lockbox.json").write_text(json.dumps(dict(
        runs=lock, seed=SEED, when=datetime.now(timezone.utc).isoformat(),
        rule="60 runs, proportional stratified by (bg cps quartile, "
             "n_encounter bucket), rng=default_rng(SEED) permutation order"),
        indent=1))
    (V.CACHE / "folds.json").write_text(json.dumps(dict(
        folds=folds, sizes=sizes, seed=SEED,
        rule="240 dev runs (all train runs minus lockbox), stratified "
             "round-robin by same keys"), indent=1))
    (V.CACHE / "shift_slice.json").write_text(json.dumps(dict(
        high_bg_runs=hi, rule="top-20% dev runs by median cps (build_log)",
        bg_cps_dev_sorted=[[int(r), float(bg[r])] for r in dev_bg.index],
        seed=SEED), indent=1))
    rep = ["# Frozen splits (harness_freeze.py)",
           f"generated {datetime.now(timezone.utc).isoformat()} seed={SEED}",
           f"lockbox ({len(lock)}): {lock}",
           f"folds sizes: {sizes}",
           *[f"fold{i}: {f}" for i, f in enumerate(folds)],
           f"high-bg shift slice ({len(hi)}): {hi}",
           f"bg cps dev range: {bg[dev].min():.1f} .. {bg[dev].max():.1f}"]
    (V.CACHE / "freeze_report.md").write_text("\n".join(rep))
    print("\n".join(rep[:6]))
    print("FROZEN.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    main(a.force)
