"""Item 2: temporal aggregation of the matched-filter score.

Compare window-level bestA vs aggregated statistics at matched FAR/hr:
  w1    : raw per-2s-window max-template z (baseline)
  mxN   : trailing rolling MAX of bestA over N windows
  smN   : trailing rolling MEAN of bestA over N windows
  tzN   : max over templates of (sum of that template's z over trailing N windows)/sqrt(N)
          (proper incoherent integration assuming template persistence)
FAR definition: alarm EPISODE ONSETS per hour on train far-windows (dmin>150):
  onset = alarm at t with no alarm at t-1. Recall: encounter hit if any alarm window
  inside CA+-60s. Evaluated on TEST runs 25-124.
"""
import pickle
import numpy as np
import pandas as pd

cache = pickle.load(open("_scratch/cache10.pkl", "rb"))
exec(open("_scratch/detect6.py", encoding="utf-8").read().split("rows = []")[0])
TRAIN_RUNS_L, TEST_RUNS = TRAIN_RUNS, list(range(25, 125))
SNR_BINS = [(0, 5), (5, 8), (8, 12), (12, 99)]
def snr_bin(x): return next(f"{lo}-{hi}" for lo, hi in SNR_BINS if lo <= x < hi)


def roll(v, n, fn):
    s = pd.Series(v)
    return (s.rolling(n, min_periods=1).agg(fn)).to_numpy(np.float32)


def roll_sum_sqrt(zA, n):
    """per-template sum of trailing N windows / sqrt(actual count), min_periods=1."""
    nw = zA.shape[0]
    cum = np.vstack([np.zeros((1, zA.shape[1]), np.float64), np.cumsum(zA, 0, dtype=np.float64)])
    idx = np.arange(nw)
    sm = cum[idx + 1] - cum[np.maximum(idx - n + 1, 0)]
    cnt = (idx - np.maximum(idx - n + 1, 0) + 1).reshape(-1, 1)
    return (sm / np.sqrt(cnt)).astype(np.float32)


def stats_for(c):
    zA = c["zA"]; best = c["bestA"]
    out = {"w1": best}
    for n in (2, 3, 5):
        out[f"mx{n}"] = roll(best, n, "max")
        out[f"sm{n}"] = roll(best, n, "mean")
    for n in (2, 3, 5):
        out[f"tz{n}"] = roll_sum_sqrt(zA, n).max(1)
    return out


STATS = {}
for r, c in cache.items():
    STATS[r] = stats_for(c)


def onsets(st, thr):
    a = st >= thr
    return np.flatnonzero(a & ~np.r_[False, a[:-1]])


def evaluate(name, far_list=(1, 3, 10, 30)):
    c0 = STATS[TRAIN_RUNS_L[0]]
    pool = {k: np.concatenate([
        (lambda s, m: s[m])(STATS[r][k], cache[r]["dmin"] > 150) for r in TRAIN_RUNS_L])
        for k in STATS[TRAIN_RUNS_L[0]]}
    # hours of far-window time in train
    train_hours = sum((cache[r]["dmin"] > 150).sum() for r in TRAIN_RUNS_L) * STRIDE / 3600
    rows = []
    for k, v in pool.items():
        for far in far_list:
            # calibrate by ONSET episodes: iterative thr so onset-count/hr == far
            lo, hi = v.min(), v.max() + 1e-9
            for _ in range(28):
                thr = (lo + hi) / 2
                n_on = sum(len([i for i in onsets(x, thr) if cache[r]["dmin"][i] > 150])
                           for r, x in [(r, STATS[r][k]) for r in TRAIN_RUNS_L])
                if n_on / train_hours > far:
                    lo = thr
                else:
                    hi = thr
            thr = (lo + hi) / 2
            # eval
            hits, tot, n_on_eval, fa_hours = 0, 0, 0, 0
            bysn = {f"{lo}-{hi}": [0, 0] for lo, hi in SNR_BINS}
            for r in TEST_RUNS:
                st, c = STATS[r][k], cache[r]
                ons = onsets(st, thr)
                onw = np.zeros(len(st), bool); onw[ons] = True
                n_on_eval += int(((c["dmin"] > 150) & onw).sum())
                fa_hours += (c["dmin"] > 150).sum() * STRIDE / 3600
                for j in range(len(c["snr"])):
                    m = np.abs(c["wtime"] - c["stime"][j]) < 60
                    if not m.any():
                        continue
                    hit = bool((st[m] >= thr).any())
                    hits += hit; tot += 1
                    b = snr_bin(c["snr"][j]); bysn[b][1] += 1; bysn[b][0] += hit
            rows.append(dict(stat=k, target_far=far, thr=thr,
                             recall=hits / tot,
                             **{f"snr{b}": (h / t if t else np.nan)
                                for b, (h, t) in bysn.items()},
                             n=tot, achieved_far=n_on_eval / fa_hours))
    df = pd.DataFrame(rows)
    piv = df.pivot_table(index="stat", columns="target_far", values="recall")
    print(f"\n### recall by stat x FAR: {name}")
    print(piv.round(3).to_string())
    print("\n### per SNR stratum at FAR=3:")
    print(df[df.target_far == 3].set_index("stat")[["recall", "achieved_far", "snr0-5", "snr5-8", "snr8-12", "snr12-99"]].round(3).to_string())
    print("\n### per SNR stratum at FAR=10:")
    print(df[df.target_far == 10].set_index("stat")[["recall", "achieved_far", "snr0-5", "snr5-8", "snr8-12", "snr12-99"]].round(3).to_string())
    return df


df = evaluate("TEST(25-124)")
df.to_pickle("_scratch/detect11_df.pkl")
