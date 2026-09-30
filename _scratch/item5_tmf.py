"""Item 5: temporal matched filtering instead of flat trailing max.

Physics: during a fly-by the source count rate ~ A/(1+((t-tc)/tau)^2) (tau = d_perp/v),
a peaked (Lorentzian-like) profile, NOT a flat box. So match the per-template z-series
with normalized peaked kernels k_tau (truncated at +-3 tau), then max over (template, tau).
Variants:
  tmf_z   : max_s max_tau  sum_j k_tau[j] * zA[t+j-center] / ||k_tau||   (matched filter on raw z)
  tmf_A   : same but on the bestA (max-over-s) series (aggregate-then-match)
  mx31    : current baseline (flat trailing 62 s max of bestA)
  mx11    : 22 s flat max (for reference)
All on cache10 zA/bestA, 2 s non-overlapping windows; FAR by episode onsets on train far pool.
"""
import pickle
import numpy as np
import pandas as pd
from scipy.signal import fftconvolve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

cache = pickle.load(open("_scratch/cache10.pkl", "rb"))
TRAIN_RUNS = [0] + list(range(3, 20))
VAL_RUNS = [20, 21, 22, 23, 24]
TEST_RUNS = list(range(25, 125))
STRIDE = 2.0
TAUS = [10, 20, 40, 80]  # seconds: d_perp/v ~ 10m..3.5 m/s -> 3..30+ s; sweep a few

def kernel(tau):
    half = int(np.ceil(3 * tau / STRIDE))
    x = np.arange(-half, half + 1) * STRIDE
    k = 1.0 / (1.0 + (x / tau) ** 2)
    return k / np.linalg.norm(k)

def rollmax(v, n): return pd.Series(v).rolling(n, min_periods=1).max().to_numpy(np.float32)

S = {}
for r, c in cache.items():
    zA = c["zA"].astype(np.float32); bA = c["bestA"].astype(np.float32)
    t_z = np.zeros(len(bA), np.float32)
    t_A = np.zeros(len(bA), np.float32)
    for tau in TAUS:
        k = kernel(tau)
        v = fftconvolve(zA, k[:, None], mode="same", axes=0)  # (nw,61) per-template TMF
        t_z = np.maximum(t_z, v.max(1))
        t_A = np.maximum(t_A, fftconvolve(bA, k, mode="same"))
    S[r] = dict(tmf_z=t_z.astype(np.float32), tmf_A=t_A.astype(np.float32),
                mx31=rollmax(bA, 31), mx11=rollmax(bA, 11))

SNR_BINS = [(0, 5), (5, 8), (8, 12), (12, 99)]
def snr_bin(x): return next(f"{lo}-{hi}" for lo, hi in SNR_BINS if lo <= x < hi)
def onset_idx(st, thr):
    a = st >= thr
    return np.flatnonzero(a & ~np.r_[False, a[:-1]])

def curves(k, grid_q=np.concatenate([np.linspace(50, 99.5, 40), np.linspace(99.5, 99.9998, 40)])):
    pool = np.concatenate([S[r][k][cache[r]["dmin"] > 150] for r in TRAIN_RUNS])
    rows = []
    for thr in np.unique(np.quantile(pool, grid_q / 100))[::-1]:
        fa_n = fa_h = hits = tot = 0
        bysn = {f"{lo}-{hi}": [0, 0] for lo, hi in SNR_BINS}
        for r in TEST_RUNS:
            st, c = S[r][k], cache[r]
            fa_n += sum(1 for i in onset_idx(st, thr) if c["dmin"][i] > 150)
            fa_h += (c["dmin"] > 150).sum() * STRIDE / 3600
            for j in range(len(c["snr"])):
                m = np.abs(c["wtime"] - c["stime"][j]) < 60
                if not m.any(): continue
                hit = bool((st[m] >= thr).any())
                hits += hit; tot += 1
                b = snr_bin(c["snr"][j]); bysn[b][1] += 1; bysn[b][0] += hit
        rows.append(dict(thr=thr, far=fa_n / fa_h, recall=hits / tot,
                         **{b: h / t for b, (h, t) in bysn.items()}))
    return pd.DataFrame(rows).sort_values("far").drop_duplicates("far")

res = {k: curves(k) for k in ["mx31", "mx11", "tmf_A", "tmf_z"]}
print("=== recall vs achieved episode-FAR (test, thr on train) ===")
for tag, d in res.items():
    line = " ".join(f"@{t}:{np.interp(t, d.far, d.recall)*100:5.1f}" for t in (1, 3, 10))
    print(f"{tag:6s} peakFAR={d.far.max():5.1f} {line}")
    for b in ["0-5", "5-8", "8-12", "12-99"]:
        t3 = d.loc[(d.far - 3).abs().idxmin()]
        print(f"        snr{b:6s}@FAR3={t3[b]*100:5.1f}")

fig, ax = plt.subplots(figsize=(8, 5))
for tag, d in res.items():
    ax.plot(d.far, d.recall * 100, marker=".", ms=4, label=tag)
ax.set_xscale("log"); ax.axvline(1, c="k", lw=.4); ax.axvline(10, c="k", lw=.4)
ax.set_xlabel("achieved false-alarm episodes / hour"); ax.set_ylabel("encounter recall % (CA±60s)")
ax.grid(alpha=.3); ax.legend(); fig.tight_layout(); fig.savefig("_scratch/item5_tmf.png", dpi=110)
pickle.dump(res, open("_scratch/item5_res.pkl", "wb"))
print("plot -> _scratch/item5_tmf.png")
