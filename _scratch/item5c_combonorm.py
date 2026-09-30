"""Item 5c: two-branch combo done right - rank-normalize TMF onto mx31's train-far
distribution before the max (item5b combo was degenerate: raw tmf values ~3x scale of mx31,
so max() was pure tmf)."""
import pickle
import numpy as np
import pandas as pd
from scipy.signal import fftconvolve

cache = pickle.load(open("_scratch/cache10.pkl", "rb"))
TRAIN_RUNS = [0] + list(range(3, 20))
TEST_RUNS = list(range(25, 125))
STRIDE = 2.0
TAUS = [10, 20, 40]

def kernel(tau):
    half = int(np.ceil(3 * tau / STRIDE))
    x = np.arange(-half, half + 1) * STRIDE
    k = 1.0 / (1.0 + (x / tau) ** 2)
    return k / np.linalg.norm(k)
def rollmax(v, n): return pd.Series(v).rolling(n, min_periods=1).max().to_numpy(np.float32)

S = {}
for r, c in cache.items():
    bA = c["bestA"].astype(np.float32)
    tA = np.zeros(len(bA), np.float32)
    for tau in TAUS:
        tA = np.maximum(tA, fftconvolve(bA, kernel(tau), mode="same"))
    S[r] = dict(mx31=rollmax(bA, 31), tmf=tA)

# rank-warp tmf -> mx31 scale: quantile-to-quantile map learned on TRAIN far windows
tp = np.sort(np.concatenate([S[r]["tmf"][cache[r]["dmin"] > 150] for r in TRAIN_RUNS]))
mp = np.sort(np.concatenate([S[r]["mx31"][cache[r]["dmin"] > 150] for r in TRAIN_RUNS]))
u = np.linspace(0, 1, 2000)
tq, mq = np.quantile(tp, u), np.quantile(mp, u)
def warp(x): return np.interp(x, tq, mq).astype(np.float32)
for r in S:
    S[r]["combo"] = np.maximum(S[r]["mx31"], warp(S[r]["tmf"]))

def onset_idx(st, thr):
    a = st >= thr
    return np.flatnonzero(a & ~np.r_[False, a[:-1]])
def curves(k):
    pool = np.concatenate([S[r][k][cache[r]["dmin"] > 150] for r in TRAIN_RUNS])
    rows = []
    for thr in np.unique(np.quantile(pool, np.concatenate([np.linspace(50, 99.5, 40), np.linspace(99.5, 99.9998, 60)]) / 100))[::-1]:
        fa_n = fa_h = hits = tot = 0
        for r in TEST_RUNS:
            st, c = S[r][k], cache[r]
            fa_n += sum(1 for i in onset_idx(st, thr) if c["dmin"][i] > 150)
            fa_h += (c["dmin"] > 150).sum() * STRIDE / 3600
            for j in range(len(c["snr"])):
                m = np.abs(c["wtime"] - c["stime"][j]) < 60
                if m.any(): hits += bool((st[m] >= thr).any()); tot += 1
        rows.append((thr, fa_n / fa_h, hits / tot))
    return pd.DataFrame(rows, columns=["thr", "far", "recall"]).sort_values("far").drop_duplicates("far")

res = {k: curves(k) for k in ["mx31", "combo"]}
print("recall @ achieved FAR (capped at curve reach):")
for tag, d in res.items():
    line = " ".join(f"@{t}:{(np.interp(t, d.far, d.recall) if t <= d.far.max() else float('nan'))*100:5.1f}" for t in (1, 2, 3, 5, 10))
    print(f"  {tag:6s} peakFAR={d.far.max():5.1f}{line}")
pickle.dump(res, open("_scratch/item5c_res.pkl", "wb"))
