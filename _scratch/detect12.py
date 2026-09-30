"""Item 3: matched filter with 6s overlapping windows (stride 2s) vs 2s windows.

Same pipeline as bestA (per-window Poisson fit of 7 bg shapes, unweighted projection z,
max over 61 templates) but T=6s. Evaluation identical to detect10/detect11:
train-calibrated episode-onset FAR on dmin>150 windows, encounter recall at CA+-60s,
TEST runs 25-124.
"""
import pickle, time
import numpy as np
import pandas as pd

exec(open("_scratch/detect6.py", encoding="utf-8").read().split("rows = []")[0])
TEST_RUNS = list(range(25, 125))
SNR_BINS = [(0, 5), (5, 8), (8, 12), (12, 99)]
def snr_bin(x): return next(f"{lo}-{hi}" for lo, hi in SNR_BINS if lo <= x < hi)

T = 6.0
def det6(rid):
    spec, tagw, wtime, snr, stime, sid = run_windows(rid, T)
    S = np.maximum(poisson_fit(spec), 1.0)
    R = spec - S
    zA = (R @ U.T) / np.sqrt(S @ U2.T + 1e-9)
    dmin = np.min(np.abs(wtime[:, None] - stime[None, :]), 1)
    return dict(bestA=zA.max(1).astype(np.float32), whichA=zA.argmax(1).astype(np.int16),
                wtime=wtime, dmin=dmin.astype(np.float32), snr=snr, stime=stime, sid=sid)

t0 = time.time()
c6 = {}
for i, rid in enumerate(TRAIN_RUNS + TEST_RUNS):
    c6[rid] = det6(rid)
    if (i + 1) % 25 == 0:
        print(f"  {i+1}/{len(TRAIN_RUNS)+len(TEST_RUNS)} ({time.time()-t0:.0f}s)")
print(f"T=6 scores built in {time.time()-t0:.0f}s")
pickle.dump(c6, open("_scratch/cache12_T6.pkl", "wb"))

cache2 = pickle.load(open("_scratch/cache10.pkl", "rb"))

def roll(v, n, fn):
    return pd.Series(v).rolling(n, min_periods=1).agg(fn).to_numpy(np.float32)

def onsets(st, thr):
    a = st >= thr
    return np.flatnonzero(a & ~np.r_[False, a[:-1]])


def run_eval(caches, label, keys=("w1", "mx3")):
    train_hours = sum((caches[r]["dmin"] > 150).sum() for r in TRAIN_RUNS) * STRIDE / 3600
    out = []
    for k in keys:
        pool = np.concatenate([caches[r][k][caches[r]["dmin"] > 150] for r in TRAIN_RUNS]) \
            if k == "w1" else None
        if k == "mx3":
            pool = np.concatenate([roll(caches[r]["w1" if "w1" in caches[r] else "bestA"], 3, "max")[caches[r]["dmin"] > 150]
                                   for r in TRAIN_RUNS])
        for far in (1, 3, 10, 30):
            lo, hi = pool.min(), pool.max() + 1e-9
            for _ in range(26):
                thr = (lo + hi) / 2
                n_on = sum(len([i for i in onsets(
                    (caches[r][k] if k == "w1" else roll(caches[r]["bestA"], 3, "max")), thr)
                    if caches[r]["dmin"][i] > 150]) for r in TRAIN_RUNS)
                if n_on / train_hours > far: lo = thr
                else: hi = thr
            thr = (lo + hi) / 2
            hits = tot = 0; fa_n = fa_h = 0
            bysn = {f"{lo}-{hi}": [0, 0] for lo, hi in SNR_BINS}
            for r in TEST_RUNS:
                st = caches[r][k] if k == "w1" else roll(caches[r]["bestA"], 3, "max")
                c = caches[r]
                fa_n += sum(1 for i in onsets(st, thr) if c["dmin"][i] > 150)
                fa_h += (c["dmin"] > 150).sum() * STRIDE / 3600
                for j in range(len(c["snr"])):
                    m = np.abs(c["wtime"] - c["stime"][j]) < 60
                    if not m.any(): continue
                    hit = bool((st[m] >= thr).any())
                    hits += hit; tot += 1
                    b = snr_bin(c["snr"][j]); bysn[b][1] += 1; bysn[b][0] += hit
            out.append(dict(setup=label, stat=k, far=far, thr=thr, recall=hits / tot,
                            ach=fa_n / fa_h, **{f"snr{b}": (h / t if t else np.nan)
                                                for b, (h, t) in bysn.items()}))
    return pd.DataFrame(out)


r2 = run_eval({r: dict(dmin=cache2[r]["dmin"], wtime=cache2[r]["wtime"], snr=cache2[r]["snr"],
                       stime=cache2[r]["stime"], bestA=cache2[r]["bestA"], w1=cache2[r]["bestA"])
               for r in TRAIN_RUNS + TEST_RUNS}, "T=2s")
r6 = run_eval({**{r: {**c6[r], "w1": c6[r]["bestA"]} for r in c6}}, "T=6s")
cmp = pd.concat([r2, r6])
print("\n### recall: 2s vs 6s windows, episode-onset FAR calibrated on train")
print(cmp[["setup", "stat", "far", "thr", "recall", "ach"] + [f"snr{lo}-{hi}" for lo, hi in SNR_BINS]]
      .round(3).to_string(index=False))
cmp.to_pickle("_scratch/detect12_df.pkl")
