# -*- coding: utf-8 -*-
"""
regression_check.py -- Step 1 gate for the RADAI submission pipeline.

Re-runs the frozen matched-filter detector (refactored into radai_lib.py) on
the 123 notebook training runs FROM THE RAW training_v4.3.h5 and verifies:

  1. fresh M/U fit shapes (7 bg components, 61 templates);
  2. per-run scores reproduce _scratch/nb_mf_cache.pkl bit-for-bit
     (bestA, mx31, wtime, dmin, bg_rate: max abs diff must be 0.0);
  3. the full notebook calibration protocol (CAND grid, onset matrices,
     monotone-branch calib, B_loo per-run thresholds) reproduces the
     headline validation recall 25.2 / 40.5 / 70.0 % at FAR 1/3/10 ep/h
     (tolerance 0.05 pp on the printed value).

Read-only on repo inputs; writes only _scratch/regression_detector.pkl.
Run:  .venv\\Scripts\\python.exe regression_check.py
"""
import sys
import time
from pathlib import Path

import numpy as np

import radai_lib as L

ROOT = Path(__file__).resolve().parent.parent   # repo root (script lives in submission/)
TRAIN_H5 = ROOT / "training_v4.3.h5"
NB_CACHE = ROOT / "_scratch" / "nb_mf_cache.pkl"
DET_CACHE = ROOT / "_scratch" / "regression_detector.pkl"

EXPECTED = {1: 25.2, 3: 40.5, 10: 70.0}
EXPECTED_THRB = {1: 5.08, 3: 4.42, 10: 3.59}     # B_all thresholds, notebook table
TOL = 0.05


def main():
    t0 = time.time()
    print("== Step 1 regression: refactored lib vs notebook ==")
    assert TRAIN_H5.exists(), TRAIN_H5
    assert NB_CACHE.exists(), NB_CACHE

    # ---- 1. fresh detector fit ------------------------------------------------
    det = L.fit_detector(str(TRAIN_H5), L.TRAIN_RUNS, cache_path=str(DET_CACHE))
    print(f"M: {det['M'].shape}, U: {det['U'].shape}, templates: {len(det['ids'])}")
    assert det["M"].shape == (7, L.NB), "expected 7 background components"
    assert det["U"].shape[0] == 61, "expected 61 templates"

    # ---- 2. rescore 123 runs, compare to notebook cache ----------------------
    import pickle
    nb = pickle.load(open(NB_CACHE, "rb"))
    assert sorted(nb) == sorted(L.ALL_RUNS), "cache run set mismatch"
    fresh = L.score_runs(str(TRAIN_H5), L.ALL_RUNS, det,
                         cache_path=None, with_gt=True, progress_every=25,
                         recompute=True)
    # nb_mf_cache.pkl predates mx31 (the notebook computes it after loading),
    # so compare mx31 against rollmax(cached bestA) with the same function.
    keys = ("bestA", "wtime", "dmin", "bg_rate")
    worst = {k: 0.0 for k in ("bestA", "mx31", "wtime", "dmin", "bg_rate")}
    bad = []
    for r in L.ALL_RUNS:
        for k in keys:
            a, b = fresh[r][k], nb[r][k]
            if a.shape != b.shape:
                bad.append((r, k, "shape", a.shape, b.shape))
                continue
            d = float(np.max(np.abs(a - b)))
            worst[k] = max(worst[k], d)
            if d != 0.0:
                bad.append((r, k, d))
        mx_ref = L.rollmax(nb[r]["bestA"], L.MX_N)
        d = float(np.max(np.abs(fresh[r]["mx31"] - mx_ref)))
        worst["mx31"] = max(worst["mx31"], d)
        if d != 0.0:
            bad.append((r, "mx31", d))
    print("max abs diff vs notebook cache:", dict(worst))
    if bad:
        print("NON-ZERO DIFFS (first 10):", bad[:10])
        sys.exit("FAIL: scores do not reproduce nb_mf_cache.pkl bit-for-bit")
    print("PASS: scores bit-identical to notebook cache (all runs, all arrays)")

    # ---- 3. calibration protocol + headline recall ---------------------------
    farm = L.farm_of(fresh, L.ALL_RUNS)
    hrs = L.hrs_of(farm)
    print(f"background hours: train {sum(hrs[r] for r in L.TRAIN_RUNS):.1f} h | "
          f"all {sum(hrs.values()):.1f} h | median test run "
          f"{np.median([hrs[r] for r in L.TEST_RUNS]):.2f} h")

    pool = np.concatenate([fresh[r]["mx31"][farm[r]] for r in L.ALL_RUNS])
    CAND = L.build_cand(pool)
    print(f"CAND grid: {len(CAND)} thresholds "
          f"(notebook: 133), span {CAND[-1]:.2f}..{CAND[0]:.2f}")
    assert len(CAND) == 133, "threshold grid size drifted"

    OM = L.onset_matrix(fresh, L.ALL_RUNS, farm, CAND, "mx31")
    IH = {r: i for i, r in enumerate(L.ALL_RUNS)}
    HV = np.array([hrs[r] for r in L.ALL_RUNS])
    EMAX = {r: L.emax_mx(fresh, r) for r in L.ALL_RUNS}
    nenc = sum(len(EMAX[r]) for r in L.TEST_RUNS)
    print(f"onset matrix: {OM.shape}; validation encounters: {nenc} "
          f"(notebook README: 943)")

    ok = True
    for target in (1, 3, 10):
        jB = L.calib_index(OM[[IH[r] for r in L.ALL_RUNS]], HV, target)
        jL = np.array([L.calib_index(
            OM[[IH[x] for x in L.ALL_RUNS if x != r]],
            HV[[IH[x] for x in L.ALL_RUNS if x != r]], target)
            for r in L.TEST_RUNS])
        # recw with per-run B_loo thresholds (notebook cell 20 verbatim)
        h = sum((EMAX[r] >= CAND[jL[i]]).sum() for i, r in enumerate(L.TEST_RUNS))
        recw = h / nenc * 100
        jj = np.array([jL[i] for i in range(len(L.TEST_RUNS))])
        ach = OM[[IH[r] for r in L.TEST_RUNS], jj].sum() / HV[[IH[r] for r in L.TEST_RUNS]].sum()
        exp = EXPECTED[target]
        hit = abs(recw - exp) < TOL
        ok &= hit
        print(f"FAR={target:>2}: B_all thr={CAND[jB]:5.2f} "
              f"(notebook {EXPECTED_THRB[target]:.2f}) | B_loo recall={recw:5.1f}% "
              f"(target {exp}) ach={ach:.1f} | {'OK' if hit else 'MISMATCH'}")

    print(f"\n{'REGRESSION PASS' if ok else 'REGRESSION FAIL'} "
          f"({time.time() - t0:.0f}s total)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
