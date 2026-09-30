"""Precision pass, item 3: does the R=150 vs R=200 recall difference at FAR~1 survive a MUCH
denser candidate grid? Onset counts are computed EXACTLY for any threshold via the interval
trick: a masked window i is an onset at thr t iff cv_i >= t > pv_i; for pv<cv pairs
count(t) = #{pv < t} - #{cv < t} with sorted arrays. FAR curve = exact staircase, no grid.
Calibration = same robust monotone-branch rule + closest-FAR<=target (ties -> higher thr)."""
import pickle
from collections import Counter
import numpy as np
import pandas as pd

cache = pickle.load(open("_scratch/cache10.pkl", "rb"))
TRAIN = [0] + list(range(3, 20)); VAL = [20, 21, 22, 23, 24]; TEST = list(range(25, 125))
ALLR = TRAIN + VAL + TEST
S = {r: pd.Series(c["bestA"]).rolling(31, min_periods=1).max().to_numpy(np.float32).astype(np.float64)
     for r, c in cache.items()}
DMIN = {r: cache[r]["dmin"] for r in S}
EMAX = {r: np.array([S[r][np.abs(cache[r]["wtime"] - t) < 60].max()
                     for t in cache[r]["stime"] if (np.abs(cache[r]["wtime"] - t) < 60).any()])
        for r in ALLR}
ENCN_TE = sum(len(EMAX[r]) for r in TEST)

def onset_intervals(r, R):
    m = DMIN[r] > R
    s = S[r]
    pv = np.r_[s[0] - 1.0, s[:-1]]
    p = pv[m]; c = s[m]
    keep = p < c
    return np.sort(p[keep]), np.sort(c[keep])

def far_curve_R(cacheR, R, runs, ts):
    tot = np.zeros(len(ts)); hrs = 0.0
    for r, k in Counter(runs).items():
        p, c = cacheR[r]
        tot += k * (np.searchsorted(p, ts, "left") - np.searchsorted(c, ts, "left"))
        hrs += k * ((DMIN[r] > R).sum() * 2 / 3600)
    return tot / hrs

def calib2(cacheR, R, runs, target):
    """exact-staircase calibration on all distinct pool window values (FAR changes only there)"""
    cands = np.unique(np.concatenate([S[r][DMIN[r] > R] for r in runs]))[::-1]
    fr = far_curve_R(cacheR, R, runs, cands)
    runmax = np.maximum.accumulate(fr)
    br = len(cands) - 1
    for k in range(1, len(cands)):
        if fr[k] >= runmax[k]: continue
        if fr[k] < runmax[k] - max(0.5, 0.2 * runmax[k]) and runmax[k] > target:
            br = k - 1; break
    ok = np.flatnonzero(fr[:br + 1] <= target)
    if len(ok) == 0: return float(cands[0]), float(fr[0])
    jbest = ok[int(np.argmax(fr[ok]))]          # max FAR <= target; argmax first index = highest thr
    return float(cands[jbest]), float(fr[jbest])

def recall_at(thr):
    return sum((EMAX[r] >= thr).sum() for r in TEST) / ENCN_TE

if __name__ == "__main__":
    print("coarse-grid reference (task2_radius.out @FAR1): R150 24.0 | R200 27.9 | R250 23.3 | R300 23.3")
    for R in (150, 200, 250, 300):
        cR = {r: onset_intervals(r, R) for r in ALLR}
        line = []
        for target in (0.5, 1.0, 2.0, 3.0):
            thr, calfar = calib2(cR, R, TRAIN, target)
            line.append(f"@{target}: thr={thr:5.2f} (trainFAR={calfar:4.2f}) rec={recall_at(thr)*100:5.1f}%")
        print(f"DENSE exact staircase, R={R}:  " + " | ".join(line))
    # best recall achievable by ANY threshold with train-FAR<=1 (feasible-set bound, ignores
    # the closest-to-target preference) -> quantifies pure grid quantization loss
    for R in (150, 200, 250):
        cR = {r: onset_intervals(r, R) for r in ALLR}
        cands = np.unique(np.concatenate([S[r][DMIN[r] > R] for r in TRAIN]))[::-1]
        fr = far_curve_R(cR, R, TRAIN, cands)
        feas = cands[fr <= 1.0]
        if len(feas):
            thr = feas.min()
            print(f"  R={R}: max-recall-over-feasible-set @train-FAR<=1: thr={thr:5.2f} "
                  f"rec={recall_at(thr)*100:5.1f}%")
