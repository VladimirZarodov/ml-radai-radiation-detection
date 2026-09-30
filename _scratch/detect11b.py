"""Item 2b: LONG-horizon temporal aggregation (N up to 31 windows = 62s).

Motivation: per-2s-window z is capped ~1.6 even at CA for strong encounters; the
encounter signal lasts ~60s. detect7 showed a single long-window Poisson fit makes
things WORSE (shape misfit grows with counts). But INTEGRATING per-template residual
z across windows (each with its own bg fit) should recover sqrt(N) gain for a
persistent shape anomaly. tzN = max_template sum z / sqrt(N); mxN/smN for reference.
Episode-onset FAR on train far-windows; TEST recall. Uses cache10.pkl.
"""
import pickle
import numpy as np
import pandas as pd

cache = pickle.load(open("_scratch/cache10.pkl", "rb"))
TRAIN_RUNS = [0] + list(range(3, 20))
TEST_RUNS = list(range(25, 125))
STRIDE = 2.0
SNR_BINS = [(0, 5), (5, 8), (8, 12), (12, 99)]
def snr_bin(x): return next(f"{lo}-{hi}" for lo, hi in SNR_BINS if lo <= x < hi)

def roll_sum_sqrt(zA, n):
    nw = zA.shape[0]
    cum = np.vstack([np.zeros((1, zA.shape[1]), np.float64), np.cumsum(zA, 0, dtype=np.float64)])
    idx = np.arange(nw)
    sm = cum[idx + 1] - cum[np.maximum(idx - n + 1, 0)]
    cnt = (idx - np.maximum(idx - n + 1, 0) + 1).reshape(-1, 1)
    return (sm / np.sqrt(cnt)).astype(np.float32)

def rollmax(v, n):
    return pd.Series(v).rolling(n, min_periods=1).max().to_numpy(np.float32)

NS = (1, 3, 5, 11, 21, 31)
STATS = {}
for r, c in cache.items():
    s = {"mx31": rollmax(c["bestA"], 31)}
    for n in NS:
        s[f"tz{n}"] = roll_sum_sqrt(c["zA"], n).max(1)
    for n in (11, 21, 31):
        s[f"sm{n}"] = pd.Series(c["bestA"]).rolling(n, min_periods=1).mean().to_numpy(np.float32)
    STATS[r] = s

def onsets(st, thr):
    a = st >= thr
    return np.flatnonzero(a & ~np.r_[False, a[:-1]])

train_hours = sum((cache[r]["dmin"] > 150).sum() for r in TRAIN_RUNS) * STRIDE / 3600
rows = []
for k in STATS[TRAIN_RUNS[0]]:
    pool = np.concatenate([STATS[r][k][cache[r]["dmin"] > 150] for r in TRAIN_RUNS])
    for far in (1, 3, 10, 30):
        lo, hi = pool.min(), pool.max() + 1e-9
        for _ in range(24):
            thr = (lo + hi) / 2
            n_on = sum(sum(1 for i in onsets(STATS[r][k], thr) if cache[r]["dmin"][i] > 150)
                       for r in TRAIN_RUNS)
            if n_on / train_hours > far: lo = thr
            else: hi = thr
        thr = (lo + hi) / 2
        hits = tot = 0; fa_n = fa_h = 0
        bysn = {f"{lo}-{hi}": [0, 0] for lo, hi in SNR_BINS}
        for r in TEST_RUNS:
            st, c = STATS[r][k], cache[r]
            fa_n += sum(1 for i in onsets(st, thr) if c["dmin"][i] > 150)
            fa_h += (c["dmin"] > 150).sum() * STRIDE / 3600
            for j in range(len(c["snr"])):
                m = np.abs(c["wtime"] - c["stime"][j]) < 60
                if not m.any(): continue
                hit = bool((st[m] >= thr).any())
                hits += hit; tot += 1
                b = snr_bin(c["snr"][j]); bysn[b][1] += 1; bysn[b][0] += hit
        rows.append(dict(stat=k, far=far, thr=thr, recall=hits/tot, ach=fa_n/fa_h,
                         **{f"snr{b}": (h/t if t else np.nan)
                            for b, (h, t) in bysn.items()}))
df = pd.DataFrame(rows)
print(df.pivot_table(index="stat", columns="far", values="recall").round(3).to_string())
print()
print(df[df.far == 3].round(3).to_string(index=False))
df.to_pickle("_scratch/detect11b_df.pkl")
