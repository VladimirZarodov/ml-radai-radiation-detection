"""Verification pass #3:
1) reproduce raw-count plateau (LogReg + XGBoost), 2) recall at realistic FAR,
3) per-window encounter-SNR stratification, 4) background: rate-scaling vs shape variation.
"""
import time
import h5py
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import xgboost as xgb

H5 = "training_v4.3.h5"
EMIN, EMAX, NB = 15.0, 3000.0, 256
WS = 2.0
TRAIN_RUNS = [0] + list(range(3, 20))
VAL_RUNS = [20, 21, 22, 23, 24]

sqrt_min, sqrt_max = np.sqrt(EMIN), np.sqrt(EMAX)

def bin_edges(nbins=NB):
    return (np.linspace(sqrt_min, sqrt_max, nbins + 1)) ** 2

def build_windows(g):
    dt = g["listmode/dt"][:]
    e = g["listmode/energy"][:]
    eid = g["listmode/id"][:]
    t = np.cumsum(dt, dtype=np.uint64) / 1e6
    dur = float(t[-1])
    nw = int(np.floor((dur - WS) / WS + 1e-9))
    wi = np.floor(t / WS).astype(np.int64)
    ok = wi < nw
    wi, e, eid, t = wi[ok], e[ok], eid[ok], t[ok]

    # spectrum histogram
    ei = np.floor((np.sqrt(np.clip(e, EMIN, EMAX - 1e-6)) - sqrt_min) / (sqrt_max - sqrt_min) * NB).astype(np.int64)
    ei = np.clip(ei, 0, NB - 1)
    spec = np.bincount(wi * NB + ei, minlength=nw * NB).reshape(nw, NB).astype(np.float64)
    tot = np.bincount(wi, minlength=nw).astype(np.float64)
    src = np.bincount(wi[eid != 0], minlength=nw).astype(np.float64)

    # encounter metadata: map window -> max SNR among tagged events in it
    snr = g["sources/snr/peak"][:]
    stime = g["sources/time"][:] / 1000.0
    sid = g["sources/id"][:]
    win_snr = np.zeros(nw)          # SNR of the encounter contributing most events to the window
    win_srcid = np.zeros(nw, dtype=np.int64)
    ev_src_pos = np.searchsorted(stime, t[eid != 0])   # nearest encounter index by time
    ev_src_pos = np.clip(ev_src_pos, 0, len(sid) - 1)
    ev_wi = wi[eid != 0]
    ev_id = eid[eid != 0]
    ev_pos = ev_src_pos
    # keep only events whose id matches the temporally nearest encounter (else search neighbors)
    # simple robust approach: for each encounter j, take events with id==sid[j] and |t-stime[j]|<=180s
    contrib = np.zeros((nw, len(sid)))
    snr_rep = np.zeros(nw)
    for j in range(len(sid)):
        if sid[j] == 0:
            continue
        m = (ev_id == sid[j]) & (np.abs(ev_pos - j) < 1e9) & (np.abs(t[eid != 0] - stime[j]) < 180)
        cw = np.bincount(ev_wi[m], minlength=nw)
        contrib[:, j] = cw
    nz = contrib.sum(axis=1) > 0
    if nz.any():
        jbest = np.argmax(contrib, axis=1)
        win_srcid[nz] = sid[jbest[nz]]
        snr_rep[nz] = snr[jbest[nz]]

    return dict(spec=spec, tot=tot, src=src, y=(src > 0).astype(np.uint8),
                win_snr=snr_rep, win_srcid=win_srcid)

f = h5py.File(H5, "r")

t0 = time.time()
tr = [build_windows(f[f"runs/run{r}"]) for r in TRAIN_RUNS]
va = [build_windows(f[f"runs/run{r}"]) for r in VAL_RUNS]
print(f"built in {time.time()-t0:.0f}s")

def stack(ws, mode):
    X = np.concatenate([np.log1p(w["spec"]) if mode == "log" else w["spec"] for w in ws])
    y = np.concatenate([w["y"] for w in ws])
    sc = np.concatenate([w["src"] for w in ws])
    sn = np.concatenate([w["win_snr"] for w in ws])
    return X.astype(np.float32), y, sc, sn

Xtr, ytr, sctr, sntr = stack(tr, "log")
Xva, yva, scva, snva = stack(va, "log")
print("train", Xtr.shape, f"{ytr.mean()*100:.1f}% pos | val", Xva.shape, f"{yva.mean()*100:.1f}% pos")

# how many positives actually carry the encounter SNR tag?
print(f"val positives with snr>0: {np.mean(snva[yva==1]>0)*100:.1f}%")

models = {}
# LogReg
lr = Pipeline([("s", StandardScaler()), ("m", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))])
lr.fit(Xtr, ytr)
p_lr = lr.predict_proba(Xva)[:, 1]
models["LogReg"] = p_lr

# XGBoost (same hparams as notebooks)
spw = (ytr == 0).sum() / (ytr == 1).sum()
xg = xgb.XGBClassifier(objective="binary:logistic", n_estimators=1000, learning_rate=0.03,
                       max_depth=5, min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                       reg_alpha=0.1, reg_lambda=1.0, scale_pos_weight=spw, eval_metric="aucpr",
                       early_stopping_rounds=30, tree_method="hist", random_state=42, n_jobs=-1)
xg.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
print("xgb best_iteration:", xg.best_iteration, "best_score:", round(xg.best_score, 4))
p_xg = xg.predict_proba(Xva)[:, 1]
models["XGBoost"] = p_xg

print("\n=== 1) plateau check (val = runs20-24) ===")
for name, p in models.items():
    print(f"{name:8s} ROC-AUC={roc_auc_score(yva, p):.4f}  PR-AUC={average_precision_score(yva, p):.4f}")

print("\n=== 2) recall at realistic FAR (XGBoost) ===")
for name, p in models.items():
    rows = []
    for far in [1, 5, 15, 30, 60]:
        fpr_t, tpr_t, thr = roc_curve(yva, p)
        target = far / (3600 / WS)
        idx = np.searchsorted(fpr_t, target, side="right") - 1
        idx = int(np.clip(idx, 0, len(thr) - 1))
        t = thr[idx] if idx < len(thr) else np.inf
        rows.append((far, t, tpr_t[idx]))
    print(name, " | ".join(f"FAR{fr}: thr={t:.3f} recall={r*100:.1f}%" for fr, t, r in rows))

print("\n=== 3) recall by SNR strata vs by raw count strata (XGBoost @ FAR=15/h) ===")
fpr_t, tpr_t, thr = roc_curve(yva, p_xg)
idx = int(np.clip(np.searchsorted(fpr_t, 15 / (3600 / WS), side="right") - 1, 0, len(thr) - 1))
thr15 = thr[idx]
pred = (p_xg >= thr15).astype(int)

def strat(col, edges, labels):
    out = []
    for lo, hi, lab in zip(edges[:-1], edges[1:], labels):
        m = (yva == 1) & (col >= lo) & (col < hi)
        if m.sum():
            out.append((lab, int(m.sum()), float(pred[m].mean())))
    return out

print("by raw tagged counts:")
for lab, n, r in strat(scva, [1, 5, 10, 20, 50, 100, 200, 1e9], ["1-4", "5-9", "10-19", "20-49", "50-99", "100-199", "200+"]):
    print(f"   {lab:>8s}: n={n:5d} recall={r*100:5.1f}%")
print("by encounter SNR (only positive windows that have a tag):")
for lab, n, r in strat(snva, [0, 4, 6, 8, 10, 13, 1e9], ["<4", "4-6", "6-8", "8-10", "10-13", "13+"]):
    print(f"   {lab:>8s}: n={n:5d} recall={r*100:5.1f}%")

# does true detectability scale with snr? check mean excess total_z by snr bucket for bg windows later
print("\n=== 4) background spectrum: rate scaling vs shape change (runs 0,20,21, positive-free bg windows) ===")
glob = None
for wi, w in enumerate(tr[:3] + va[:2]):
    bg = w["spec"][w["src"] == 0]
    tot_bg = bg.sum(axis=1)
    if glob is None:
        glob = bg.mean(axis=0)
    glob = glob / glob.sum() * bg.sum()   # rescale shape template
    # per-window best scalar fit a = <x, s>/<s,s>, residual variance explained
    a = (bg @ glob) / (glob @ glob)
    resid = bg - a[:, None] * glob[None, :]
    ve = 1 - resid.var() / bg.var()
    # also: pure poisson expectation of residual var
    print(f"  set{wi}: bg rate mean={tot_bg.mean():7.0f} std={tot_bg.std():6.0f} | "
          f"variance of bg spectra explained by rate-scaled global shape: {ve:.3f}")
f.close()
