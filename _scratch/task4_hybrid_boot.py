"""Task 4: bootstrap CIs for the hybrid extension (mx31 OR ML-fallback) at FAR20-100.

Layer 1 (test-side): item3's selected (thr_p, thr_f) pairs kept fixed; run-cluster bootstrap of
        test runs -> CI for recall AND achieved test FAR.
Layer 2 (full protocol, selection-free): each replication resamples the 18 train runs, recalibrates
        thr_p = mx31 train-branch saturation point and thr_f = deepest fallback threshold keeping
        combined train onset-FAR <= target; then resamples test runs -> CIs. (Item3's pair selection
        maximized TEST recall -> mild in-sample optimism; layer 2 quantifies the honest uncertainty.)
Series on the baselines grid (nw = cache len - 1), exactly as item3.
"""
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(101)
cache = pickle.load(open("_scratch/cache10.pkl", "rb"))
ML = pickle.load(open("_scratch/item3_ml_proba.pkl", "rb"))
SEL = pd.read_pickle("_scratch/item3_hybrid.pkl") if False else pickle.load(open("_scratch/item3_hybrid.pkl", "rb"))
t1 = pickle.load(open("_scratch/task1_pool.pkl", "rb")); CAND = t1["CAND"]
TRAIN = [0] + list(range(3, 20)); VAL = [20, 21, 22, 23, 24]; TEST = list(range(25, 125))
STRIDE = 2.0
def rollmax(v, n=31): return pd.Series(v).rolling(n, min_periods=1).max().to_numpy(np.float32).astype(np.float64)

SC = {}
for r in sorted(ML):
    L = ML[r]["nw"]
    SC[r] = dict(mx=rollmax(cache[r]["bestA"])[:L],
                 far=cache[r]["dmin"][:L] > 150,
                 logreg=ML[r]["logreg"][:L].astype(np.float64),
                 mlp=ML[r]["mlp"][:L].astype(np.float64),
                 wt=cache[r]["wtime"][:L], st=cache[r]["stime"])
    SC[r]["hours"] = SC[r]["far"].sum() * STRIDE / 3600
    ep, ef_l, ef_m = [], [], []
    for j in range(len(SC[r]["st"])):
        m = np.abs(SC[r]["wt"] - SC[r]["st"][j]) < 60
        if not m.any(): continue
        ep.append(SC[r]["mx"][m].max())
        ef_l.append(SC[r]["logreg"][m].max()); ef_m.append(SC[r]["mlp"][m].max())
    SC[r]["enc"] = (np.array(ep), np.array(ef_l), np.array(ef_m))

def onset_arr(mx, arm, far, tp, tf):
    a = (mx >= tp) | (arm >= tf)
    return int(np.count_nonzero((a & ~np.r_[False, a[:-1]])[far]))

# --- mx31 onset matrix on CAND grid (for saturation threshold) ---
IDXs = {r: i for i, r in enumerate(sorted(cache))}
O_mx = np.zeros((len(cache), len(CAND)), np.int32)
for r in cache:
    s, f = SC[r]["mx"], SC[r]["far"]
    for j, t in enumerate(CAND):
        a = s >= t
        O_mx[IDXs[r], j] = np.count_nonzero((a & ~np.r_[False, a[:-1]])[f])
H = np.array([SC[r]["hours"] for r in sorted(cache)])
IDXs = {r: i for i, r in enumerate(sorted(cache))}
selTR = np.array([IDXs[r] for r in TRAIN]); selTE = np.array([IDXs[r] for r in TEST]); selVA = np.array([IDXs[r] for r in VAL])
FALLBACK_T = 1e9
def sat_j(ii):
    """index of mx31 branch saturation (first argmax of train onset-FAR curve)"""
    fr = O_mx[ii].sum(0) / H[ii].sum()
    return int(np.argmax(fr))

# --- fallback candidate grids (ML far-pool quantiles) ---
MLC = {}
for arm in ("logreg", "mlp"):
    pool = np.concatenate([SC[r][arm][SC[r]["far"]] for r in TRAIN])
    dq = np.logspace(np.log10(0.01), np.log10(50), 90)          # ML proba lives mid-distribution, not extreme tail
    MLC[arm] = np.unique(np.quantile(pool, 1 - dq / 100))[::-1]

TARGETS = [20, 30, 50, 70, 100]
rows = []
for arm in ("logreg", "mlp"):
    ENC = np.concatenate([np.column_stack([SC[r]["enc"][0], SC[r]["enc"][1 if arm == 'logreg' else 2]]) for r in TEST])
    ERUN = np.concatenate([[i] * len(SC[r]["enc"][0]) for i, r in enumerate(TEST)])
    ENCN = np.bincount(ERUN, minlength=len(TEST)).astype(float)
    ENCVA = np.concatenate([np.column_stack([SC[r]["enc"][0], SC[r]["enc"][1 if arm == 'logreg' else 2]]) for r in VAL])
    ENCNTOT_VA = ENCVA.shape[0]

    # grid of OR onset counts on TEST / VAL / TRAIN per (cand j of mx thr, k of ML thr)
    # cache by distinct tp value encountered during bootstrap
    orTE_cache, orVA_cache, orTR_cache, nTE_H = {}, {}, {}, H[selTE]
    def get_or(jp):
        if jp not in orTE_cache:
            tp = CAND[jp]
            orTE_cache[jp] = np.array([[onset_arr(SC[r]["mx"], SC[r][arm], SC[r]["far"], tp, tf)
                                        for tf in MLC[arm]] for r in TEST], np.float64)
            orVA_cache[jp] = np.array([[onset_arr(SC[r]["mx"], SC[r][arm], SC[r]["far"], tp, tf)
                                        for tf in MLC[arm]] for r in VAL], np.float64)
            orTR_cache[jp] = np.array([[onset_arr(SC[r]["mx"], SC[r][arm], SC[r]["far"], tp, tf)
                                        for tf in MLC[arm]] for r in TRAIN], np.float64)
        return orTE_cache[jp], orVA_cache[jp], orTR_cache[jp]

    # ---- Layer 1: selected pair fixed (test-side bootstrap) ----
    for tgt in TARGETS:
        s = SEL[(SEL.hybrid == f"mx31+{arm}") & (SEL.target == tgt)].iloc[0]
        tp, tf = s.thr_p, s.thr_f
        hitc, ftc = [], []
        for r in TEST:
            e0, e1, e2 = SC[r]["enc"]
            hits = ((e0 >= tp) | ((e1 if arm == "logreg" else e2) >= tf))
            hitc.append(hits.sum())
            ftc.append(onset_arr(SC[r]["mx"], SC[r][arm], SC[r]["far"], tp, tf))
        hitc, ftc = np.array(hitc, float), np.array(ftc, float)
        rec_p = hitc.sum() / ENCN.sum(); far_p = ftc.sum() / nTE_H.sum()
        recs, fars = [], []
        for b in range(1000):
            cnt = np.bincount(rng.integers(0, len(TEST), len(TEST)), minlength=len(TEST)).astype(float)
            recs.append((cnt * hitc).sum() / (cnt * ENCN).sum())
            fars.append((cnt * ftc).sum() / (cnt * nTE_H).sum())
        rows.append(dict(arm=arm, target=tgt, layer="fixed-pair", thr_p=tp, thr_f=tf,
                         recall_pt=rec_p, far_pt=far_p, train_far_pt=s.train_far,
                         rec_lo=np.percentile(recs, 2.5), rec_hi=np.percentile(recs, 97.5),
                         far_lo=np.percentile(fars, 2.5), far_hi=np.percentile(fars, 97.5)))

    # ---- Layer 2: full protocol, selection-free (B=250) ----
    recs = {t: [] for t in TARGETS}; fars = {t: [] for t in TARGETS}
    valr = {t: [] for t in TARGETS}; tps, tfs = {t: [] for t in TARGETS}, {t: [] for t in TARGETS}
    for b in range(250):
        rs = rng.integers(0, len(TRAIN), len(TRAIN))
        ii = selTR[rs]
        jp = sat_j(ii); tp = CAND[jp]
        orTE, orVA, orTR = get_or(jp)
        mult = np.bincount(rs, minlength=len(TRAIN)).astype(float)
        fr_tr = (mult @ orTR) / H[ii].sum()   # resampling-weighted combined train onset FAR vs tf grid
        for tgt in TARGETS:
            ok = np.flatnonzero(fr_tr <= tgt)
            k = int(ok[-1]) if len(ok) else 0
            tf = MLC[arm][k]; tfs[tgt].append(tf); tps[tgt].append(tp)
            cnt = np.bincount(rng.integers(0, len(TEST), len(TEST)), minlength=len(TEST)).astype(float)
            hit = (ENC[:, 0] >= tp) | (ENC[:, 1] >= tf)
            hs = np.bincount(ERUN[hit], minlength=len(TEST)).astype(float)   # hits per run
            recs[tgt].append((cnt * hs).sum() / (cnt * ENCN).sum())
            fars[tgt].append((cnt * orTE[:, k]).sum() / (cnt * nTE_H).sum())
            hitv = (ENCVA[:, 0] >= tp) | (ENCVA[:, 1] >= tf)
            valr[tgt].append(hitv.mean())
    for tgt in TARGETS:
        r_, f_ = np.array(recs[tgt]), np.array(fars[tgt])
        rows.append(dict(arm=arm, target=tgt, layer="full-protocol", thr_p=np.median(tps[tgt]), thr_f=np.median(tfs[tgt]),
                         recall_pt=r_.mean(), far_pt=f_.mean(), train_far_pt=np.nan,
                         rec_lo=np.percentile(r_, 2.5), rec_hi=np.percentile(r_, 97.5),
                         far_lo=np.percentile(f_, 2.5), far_hi=np.percentile(f_, 97.5),
                         val_pt=np.mean(valr[tgt])))
        print(f"  [{arm} full @FAR{tgt}] recall mean={r_.mean()*100:.1f} 95%CI [{np.percentile(r_,2.5)*100:.1f},{np.percentile(r_,97.5)*100:.1f}]"
              f" | FAR mean={f_.mean():.1f} [{np.percentile(f_,2.5):.1f},{np.percentile(f_,97.5):.1f}] | val={np.mean(valr[tgt])*100:.1f}")

df = pd.DataFrame(rows)
pickle.dump(df, open("_scratch/task4_hybrid_boot.pkl", "wb"))
print(df.round(3).to_string())

fig, ax = plt.subplots(figsize=(8.5, 5))
xmap = {t: i for i, t in enumerate(TARGETS)}
for arm, col in [("logreg", "tab:blue"), ("mlp", "tab:red")]:
    fp = df[(df.arm == arm) & (df.layer == "fixed-pair")]
    ax.errorbar([xmap[t] for t in fp.target], fp.recall_pt * 100,
                yerr=[(fp.recall_pt - fp.rec_lo) * 100, (fp.rec_hi - fp.recall_pt) * 100],
                fmt="o-", capsize=3, color=col, label=f"{arm} selected-pair")
    fu = df[(df.arm == arm) & (df.layer == "full-protocol")]
    ax.errorbar([xmap[t] for t in fu.target], fu.recall_pt * 100,
                yerr=[(fu.recall_pt - fu.rec_lo) * 100, (fu.rec_hi - fu.recall_pt) * 100],
                fmt="s--", capsize=3, color=col, alpha=.55, label=f"{arm} full-protocol")
ax.axhline(82.2, c="gray", ls=":", lw=1.2); ax.text(0.05, 83.5, "mx31 saturation (82.2%)", color="gray", fontsize=8)
ax.set_xticks(list(xmap.values())); ax.set_xticklabels([f"FAR {t}" for t in TARGETS])
ax.set_ylabel("test recall %"); ax.set_ylim(70, 105); ax.grid(alpha=.3); ax.legend()
fig.tight_layout(); fig.savefig("_scratch/task4_hybrid_boot.png", dpi=110)
print("plot -> _scratch/task4_hybrid_boot.png")
