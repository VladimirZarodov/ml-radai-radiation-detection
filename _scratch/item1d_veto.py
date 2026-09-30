"""Item 1d: pile-up VETO instead of modeling. At FAR1 the FA windows have median rate 15.2k
counts/2s (vs 5.9k for detections) - the pile-up regime. Rule: discard windows flagged as
pile-up (score := 0 inside the rolling max). Two flag styles: pure rate, and rate+pileup-projection.
Threshold calibrated by binary search on TRAIN far episode-onset rate (notebook protocol)."""
import pickle
import numpy as np
import pandas as pd

cache = pickle.load(open("_scratch/cache10.pkl", "rb"))
D = pickle.load(open("_scratch/item1_D.pkl", "rb")); Dp = D["Dp"] / D["Dp"].sum()
TRAIN_RUNS = [0] + list(range(3, 20))
TEST_RUNS = list(range(25, 125))
STRIDE = 2.0

def rollmax(v, n=31): return pd.Series(v).rolling(n, min_periods=1).max().to_numpy(np.float32)
def onset_idx(st, thr):
    a = st >= thr
    return np.flatnonzero(a & ~np.r_[False, a[:-1]])

H_TR = sum((cache[r]["dmin"] > 150).sum() * STRIDE / 3600 for r in TRAIN_RUNS)
def far_at(S, thr):
    return sum(sum(1 for i in onset_idx(S[r], thr) if cache[r]["dmin"][i] > 150) for r in TRAIN_RUNS) / H_TR
def calib(S, target):
    """quantile grid + monotone branch (episode FAR is non-monotone below saturation)"""
    pool = np.concatenate([S[r][cache[r]["dmin"] > 150] for r in TRAIN_RUNS])
    qs = np.linspace(99.9998, 50, 120)
    prev_f, branch = -1, []
    for t in sorted(np.unique(np.quantile(pool, qs / 100)), reverse=True):  # descending thresholds
        fr = far_at(S, t)
        if fr < prev_f: break                      # merged-episodes region
        branch.append((t, fr)); prev_f = fr
    ok = [x for x in branch if x[1] <= target] or branch
    return min(ok, key=lambda x: abs(x[1] - target))[0]

# pileup projection per window: V = (X-S).D+ / sqrt(S.D+^2) - need residuals -> recompute S from specs
specs = pickle.load(open("_scratch/item1_specs.pkl", "rb"))
exec(open("_scratch/detect6.py", encoding="utf-8").read().split("rows = []")[0])
V = {}
for r, sp in specs.items():
    X = sp.astype(np.float64)
    S = np.maximum(poisson_fit(X), 1.0)
    V[r] = ((X - S) @ Dp / np.sqrt(S @ (Dp ** 2) + 1e-9)).astype(np.float32)
# train-far quantiles for V
vfar = np.concatenate([V[r][(cache[r]["dmin"] > 150)] for r in TRAIN_RUNS])
VQ = {q: np.quantile(vfar, q) for q in (.9, .99, .999)}

def build(veto):
    S = {}
    for r in cache:
        b = cache[r]["bestA"].copy()
        b[veto[r]] = 0.0
        S[r] = rollmax(b)
    return S

rate = {r: specs[r].sum(1) for r in specs}
rows = []
for tag, veto in [
    ("none", {r: np.zeros(len(rate[r]), bool) for r in rate}),
    ("rate>16k", {r: rate[r] > 16000 for r in rate}),
    ("rate>14k", {r: rate[r] > 14000 for r in rate}),
    ("rate>12k", {r: rate[r] > 12000 for r in rate}),
    ("rate>14k&V>q99", {r: (rate[r] > 14000) & (V[r] > VQ[.99]) for r in rate}),
    ("rate>12k&V>q99.9", {r: (rate[r] > 12000) & (V[r] > VQ[.999]) for r in rate}),
    ("V>q99.9", {r: V[r] > VQ[.999] for r in rate}),
]:
    Sv = build(veto)
    line = f"{tag:18s} vetoed={(sum(v.sum() for v in veto.values())):6d} win |"
    for tgt in (1, 3, 10):
        thr = calib(Sv, tgt)
        hits = tot = 0
        fa_n = fa_h = 0
        for r in TEST_RUNS:
            c = cache[r]
            fa_n += sum(1 for i in onset_idx(Sv[r], thr) if c["dmin"][i] > 150)
            fa_h += (c["dmin"] > 150).sum() * STRIDE / 3600
            for j in range(len(c["snr"])):
                m = np.abs(c["wtime"] - c["stime"][j]) < 60
                if m.any(): hits += bool((Sv[r][m] >= thr).any()); tot += 1
        line += f"  FAR{tgt}: thr={thr:4.2f} rec={hits/tot*100:5.1f}% (test-far {fa_n/fa_h:4.1f})"
    rows.append(line)
    print(line)
