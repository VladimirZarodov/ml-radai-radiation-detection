"""Item 1c (fixed): which templates fire on TRAIN far-window alarms vs TEST detections?
Uses cache10 (bestA/whichA/mx31-equivalent via rollmax) - no EM recompute.
Also reports the rate distribution of FA-onset windows and of mx31@FAR10 detections."""
import pickle
import numpy as np
import pandas as pd
import h5py
from collections import Counter

cache = pickle.load(open("_scratch/cache10.pkl", "rb"))
f = h5py.File("training_v4.3.h5", "r")
names = [str(n).split("_shielding")[0] for n in f.attrs["source_names"]]
tmpl = [str(n) for n in f.attrs["source_names"]]  # includes shielding suffix, index-aligned with U
TRAIN_RUNS = [0] + list(range(3, 20))
TEST_RUNS = list(range(25, 125))
STRIDE = 2.0

def rollmax(v, n=31): return pd.Series(v).rolling(n, min_periods=1).max().to_numpy(np.float32)
M = {r: rollmax(cache[r]["bestA"]) for r in cache}

# train-far episode-onset FAR binary search -> thr @ FAR 1 and 10
def onset_idx(st, thr):
    a = st >= thr
    return np.flatnonzero(a & ~np.r_[False, a[:-1]])
def calib(target):
    H = sum((cache[r]["dmin"] > 150).sum() * STRIDE / 3600 for r in TRAIN_RUNS)
    pool = np.concatenate([M[r][cache[r]["dmin"] > 150] for r in TRAIN_RUNS])
    qs = np.linspace(99.9998, 50, 120)
    prev_f, branch = -1, []
    for t in sorted(np.unique(np.quantile(pool, qs / 100)), reverse=True):
        fr = sum(sum(1 for i in onset_idx(M[r], t) if cache[r]["dmin"][i] > 150) for r in TRAIN_RUNS) / H
        if fr < prev_f: break
        branch.append((t, fr)); prev_f = fr
    ok = [x for x in branch if x[1] <= target] or branch
    return min(ok, key=lambda x: abs(x[1] - target))[0]

for tgt in (1, 10):
    thr = calib(tgt)
    fa_t, det_t = [], []
    fa_rates, det_rates = [], []
    for r in TRAIN_RUNS:
        c = cache[r]
        for i in onset_idx(M[r], thr):
            if c["dmin"][i] > 150:
                # template at the argmax window inside the trailing 31
                lo = max(0, i - 30)
                j = lo + int(np.argmax(c["bestA"][lo:i + 1]))
                fa_t.append(int(c["whichA"][j])); fa_rates.append(c["bg_rate"][i])
    for r in TEST_RUNS:
        c = cache[r]
        for j in range(len(c["snr"])):
            m = np.abs(c["wtime"] - c["stime"][j]) < 60
            if not m.any(): continue
            if (M[r][m] >= thr).any():
                k = np.flatnonzero(m)[np.argmax(M[r][m])]
                lo = max(0, k - 30)
                q = lo + int(np.argmax(c["bestA"][lo:k + 1]))
                det_t.append(int(c["whichA"][q])); det_rates.append(c["bg_rate"][k])
    def pct(ids):
        cc = Counter(ids); tot = sum(cc.values())
        return "  ".join(f"{names[i]}:{k*100//tot}%" for i, k in cc.most_common(10))
    print(f"--- @FAR{tgt}: train-fa onsets n={len(fa_t)} vs test detections n={len(det_t)} ---")
    print(" FA templates :", pct(fa_t))
    print(" DET templates:", pct(det_t))
    print(f" FA window bg_rate: med={np.median(fa_rates):.0f} q75={np.quantile(fa_rates,.75):.0f} q90={np.quantile(fa_rates,.9):.0f}")
    print(f" DET window rate  : med={np.median(det_rates):.0f}")
    # isotope-level FA attribution: how many far FA onsets happen while dmin in (150,300]?
    d = [cache[r]["dmin"][i] for r in TRAIN_RUNS for i in onset_idx(M[r], thr) if cache[r]["dmin"][i] > 150]
    print(f" FA onsets with dmin in 150-300 s: {np.mean((np.array(d) > 150) & (np.array(d) <= 300))*100:.0f}% (wing overlap)")
