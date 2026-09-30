"""Item 1: background-model tail behavior.

Diagnosis so far: high-rate FA windows show a rate-scaled spectral distortion
(deficits at 15-50 / 662 / 2614 keV, excesses at ~2x X-ray and 2x609 sum energies)
= pulse pile-up, NOT a separate window class -> model it, don't exclude it.

Variants (threshold always calibrated on TRAIN far pool, recall on TEST encounters):
  base   : bestA (7-shape Poisson fit), mx31 rolling max                       [= current detector]
  kap05/1/2 : Poisson fit + fixed pile-up offset C_w = k * rate_w^2 * D+ (k in {0.5,1,2}x k_hat)
  tail   : base mx31 then rate-bin tail-normalized (div by per-bin q99 ratios) -> adaptive threshold
  tailk  : tail-normalization on top of the best kappa variant
Also: which templates produce train-FA vs test-detection max?
"""
import pickle, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

exec(open("_scratch/detect6.py", encoding="utf-8").read().split("rows = []")[0])
TRAIN_RUNS = [0] + list(range(3, 20))
VAL_RUNS = [20, 21, 22, 23, 24]
TEST_RUNS = list(range(25, 125))
ALL_RUNS = TRAIN_RUNS + VAL_RUNS + TEST_RUNS
cache = pickle.load(open("_scratch/cache10.pkl", "rb"))

# ---------- step 1: window spectra for all runs (cached) ----------
try:
    specs = pickle.load(open("_scratch/item1_specs.pkl", "rb"))
    print(f"loaded specs for {len(specs)} runs")
except FileNotFoundError:
    t0 = time.time()
    specs = {}
    for rid in ALL_RUNS:
        sp, tagw, wtime, snr, stime, sid = run_windows(rid, 2.0)
        specs[rid] = sp.astype(np.float32)
    pickle.dump(specs, open("_scratch/item1_specs.pkl", "wb"))
    print(f"built specs in {time.time()-t0:.0f}s")

# ---------- step 2: learn distortion shape D+ and kappa from TRAIN far windows ----------
Rsum = np.zeros(NB); Rrate2 = 0.0; nfar = 0
proj_num = 0.0
far_R, far_rate = [], []
for rid in TRAIN_RUNS:
    X = specs[rid].astype(np.float64)
    S0 = np.maximum(poisson_fit(X), 1.0)
    rate = X.sum(1)
    m = cache[rid]["dmin"] > 150
    R = X - S0
    far_R.append(R[m]); far_rate.append(rate[m])
print("far windows:", sum(len(x) for x in far_R))
far_R = np.concatenate(far_R); far_rate = np.concatenate(far_rate)

# rate^2-weighted mean residual -> smooth -> positive part -> unit sum
w = far_rate ** 2
D0 = (far_R * w[:, None]).sum(0) / w.sum()
D0 = pd.Series(D0).rolling(7, center=True, min_periods=1).mean().to_numpy()
Dp = np.clip(D0, 0, None)
Dp /= Dp.sum()
kappa_hat = (far_R @ Dp).sum() / ((far_rate ** 2) * (Dp @ Dp)).sum()
print(f"kappa_hat = {kappa_hat:.3e}; D+ top bins (keV / share):")
edges = (np.linspace(np.sqrt(15), np.sqrt(3000), NB + 1)) ** 2
for b in np.argsort(-Dp)[:8]:
    print(f"   {edges[b]:.0f}-{edges[b+1]:.0f} keV  {Dp[b]*100:.1f}%")
pickle.dump(dict(Dp=Dp, kappa_hat=kappa_hat), open("_scratch/item1_D.pkl", "wb"))

def score_run(rid, kappa):
    """Poisson 7-shape fit with fixed pile-up offset C=kappa*rate^2*D+; returns bestA'. """
    X = specs[rid].astype(np.float64)
    rate = X.sum(1)
    C = kappa * (rate[:, None] ** 2) * Dp[None, :] if kappa > 0 else None
    A = np.full((X.shape[0], M.shape[0]), X.sum(1, keepdims=True) / M.shape[0])
    norm = M.sum(1)[None, :]
    for _ in range(60):
        B = A @ M + (1e-9 if C is None else np.maximum(C, 1e-9))
        A *= ((X / B) @ M.T) / norm
    Stot = A @ M + (0.0 if C is None else C) + 1e-9
    z = (X @ U.T - Stot @ U.T) / np.sqrt(Stot @ U2.T + 1e-9)
    return z.max(1).astype(np.float32), z.argmax(1)

# ---------- step 3: compute bestA' per kappa, mx31, curves ----------
STRIDE = 2.0
SNR_BINS = [(0, 5), (5, 8), (8, 12), (12, 99)]
def snr_bin(x): return next(f"{lo}-{hi}" for lo, hi in SNR_BINS if lo <= x < hi)
EDGE = np.exp(np.linspace(np.log(200), np.log(60000), 13))
def rate_bin(r): return np.clip(np.searchsorted(EDGE, r) - 1, 0, len(EDGE) - 2)
def rollmax(v, n=31): return pd.Series(v).rolling(n, min_periods=1).max().to_numpy(np.float32)
def onset_idx(st, thr):
    a = st >= thr
    return np.flatnonzero(a & ~np.r_[False, a[:-1]])

def curves(get_score, tail_norm=False, tag=""):
    """get_score(rid)->(bestA argmax arrays). tail_norm: divide mx31 by per-bin q99 ratio."""
    S = {}
    argmx = {}
    for rid in ALL_RUNS:
        best, am = get_score(rid)
        S[rid] = rollmax(best); argmx[rid] = am
    if tail_norm:
        ra = np.concatenate([specs[r].sum(1)[cache[r]["dmin"] > 150] for r in TRAIN_RUNS])
        ma = np.concatenate([S[r][cache[r]["dmin"] > 150] for r in TRAIN_RUNS])
        ba = rate_bin(ra)
        q_all = np.quantile(ma, .99)
        rho = np.full(len(EDGE) - 1, np.nan)
        for j in range(len(EDGE) - 1):
            if (ba == j).sum() > 400:
                rho[j] = np.quantile(ma[ba == j], .99) / q_all
        ok = ~np.isnan(rho); idx = np.flatnonzero(ok)
        for j in np.flatnonzero(~ok):
            rho[j] = rho[idx[int(np.argmin(np.abs(idx - j)))]]
        for rid in ALL_RUNS:
            S[rid] = (S[rid] / rho[rate_bin(specs[rid].sum(1))]).astype(np.float32)
    pool = np.concatenate([S[r][cache[r]["dmin"] > 150] for r in TRAIN_RUNS])
    grid_q = np.concatenate([np.linspace(50, 99.5, 40), np.linspace(99.5, 99.9998, 40)])
    rows = []
    for thr in np.unique(np.quantile(pool, grid_q / 100))[::-1]:
        fa_n = fa_h = hits = tot = 0
        bysn = {f"{lo}-{hi}": [0, 0] for lo, hi in SNR_BINS}
        for r in TEST_RUNS:
            st, c = S[r], cache[r]
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
    df = pd.DataFrame(rows).sort_values("far").drop_duplicates("far")
    return df, S, argmx

KAPPAS = [("base", 0.0), ("kap05", 0.5 * kappa_hat), ("kap1", kappa_hat), ("kap2", 2 * kappa_hat)]
res = {}
t0 = time.time()
for tag, k in KAPPAS:
    df, S, am = curves(lambda rid, k=k: score_run(rid, k), tag=tag)
    res[tag] = df
    if tag == "base":
        pickle.dump({r: S[r] for r in ALL_RUNS}, open("_scratch/item1_S_base.pkl", "wb"))
print(f"4 kappa variants in {time.time()-t0:.0f}s")

# best kappa -> + tail variant; also tail on base
res["tail"] = curves(lambda rid: score_run(rid, 0.0), tail_norm=True)[0]
best_k = max(("kap05", "kap1", "kap2"), key=lambda t: np.interp(3.0, res[t].far, res[t].recall))
print("best kappa variant by recall@FAR3:", best_k)
res[f"tail_{best_k}"] = curves(lambda rid: score_run(rid, dict(KAPPAS)[best_k]), tail_norm=True)[0]

print("\n=== recall vs achieved episode-FAR (TEST, thr calibrated on TRAIN) ===")
tgt = [0.5, 1, 2, 3, 5, 10]
hdr = "variant   " + " ".join(f"@FAR{t:<4}" for t in tgt)
print(hdr)
for tag, d in res.items():
    r = [np.interp(t, d.far, d.recall) * 100 if t <= d.far.max() else np.nan for t in tgt]
    print(f"{tag:9s} " + " ".join(f"{x:6.1f}" for x in r))

fig, ax = plt.subplots(figsize=(8.5, 5.5))
for tag, d in res.items():
    ax.plot(d.far, d.recall * 100, marker=".", ms=4, label=tag)
ax.set_xscale("log"); ax.axvline(1, c="k", lw=.4); ax.axvline(10, c="k", lw=.4)
for t in tgt: ax.axvline(t, c="gray", lw=.3)
ax.set_xlabel("achieved false-alarm episodes / hour"); ax.set_ylabel("encounter recall % (CA±60s)")
ax.grid(alpha=.3); ax.legend(); fig.tight_layout(); fig.savefig("_scratch/item1_variants.png", dpi=110)
pickle.dump(res, open("_scratch/item1_res.pkl", "wb"))
print("plot -> _scratch/item1_variants.png")

# ---------- step 4: argmax-template diagnostics ----------
_, S_b, am_base = curves(lambda rid: score_run(rid, 0.0))
thr10 = res["base"][(res["base"].far - 10).abs().idxmin()].thr
fa_ids, det_ids = [], []
for rid in TRAIN_RUNS:
    m = (cache[rid]["dmin"] > 150) & (S_b[rid] >= thr10)
    fa_ids += list(am_base[rid][m])
for rid in TEST_RUNS:
    c = cache[rid]
    for j in range(len(c["snr"])):
        m = np.abs(c["wtime"] - c["stime"][j]) < 60
        if m.any() and (S_b[rid][m] >= thr10).any():
            det_ids.append(int(am_base[rid][np.flatnonzero(m)[np.argmax(S_b[rid][m])]]))
srcn = [names[i] for i in ids]
from collections import Counter
def top(ids_, n=8):
    cc = Counter(ids_); tot = sum(cc.values())
    return "  ".join(f"{srcn[i]}:{k*100//max(tot,1)}%" for i, k in ids_ and cc.most_common(n))
print("\nargmax template (mx31 peak window):")
print("  TRAIN FA windows:", top(fa_ids))
print("  TEST detections :", top(det_ids))
