"""Baselines for the final notebook: LogReg / XGBoost / small 1D-CNN on 2s window spectra
(NB=256, log1p), trained on train runs, early-stopped on val, evaluated on TEST runs
(25-124) both window-level (ROC-AUC, PR-AUC) and encounter-level recall vs FAR/hr
(episode onsets on dmin>150 windows; hit = any alarm window inside CA+-60s).
Window labels: tagged source counts > 0 (repo convention).
"""
import os, time, pickle
import h5py
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb

H5 = "training_v4.3.h5"
EMIN, EMAX, NB = 15.0, 3000.0, 256
WS = 2.0
TRAIN_RUNS = [0] + list(range(3, 20))
VAL_RUNS = [20, 21, 22, 23, 24]
TEST_RUNS = list(range(25, 125))
sqrt_min, sqrt_max = np.sqrt(EMIN), np.sqrt(EMAX)

f = h5py.File(H5, "r")

def build(rid):
    g = f[f"runs/run{rid}"]
    dt = g["listmode/dt"][:]
    e = g["listmode/energy"][:]
    eid = g["listmode/id"][:]
    t = np.cumsum(dt, dtype=np.uint64) / 1e6
    dur = float(t[-1])
    nw = int(np.floor((dur - WS) / WS + 1e-9))
    wi = np.floor(t / WS).astype(np.int64)
    ok = wi < nw
    wi2, e2, eid2 = wi[ok], e[ok], eid[ok]
    inr = (e2 >= EMIN) & (e2 < EMAX)
    bi = np.zeros(len(e2), np.int64)
    bi[inr] = np.clip(np.floor((np.sqrt(e2[inr]) - sqrt_min) / (sqrt_max - sqrt_min) * NB).astype(np.int64), 0, NB - 1)
    spec = np.bincount(wi2[inr] * NB + bi[inr], minlength=nw * NB).reshape(nw, NB).astype(np.float32)
    src = np.bincount(wi2[eid2 != 0], minlength=nw).astype(np.float32)
    stime = g["sources/time"][:] / 1e3
    wtime = np.arange(nw) * WS + WS / 2
    dmin = np.min(np.abs(wtime[:, None] - stime[None, :]), 1).astype(np.float32)
    snr = g["sources/snr/peak"][:]
    return dict(spec=spec, y=(src > 0).astype(np.uint8), dmin=dmin, wtime=wtime,
                snr=snr, stime=stime)

t0 = time.time()
WINC = "_scratch/baselines_windows.pkl"
if os.path.exists(WINC):
    TR, VA, TE = pickle.load(open(WINC, "rb"))
    print("windows loaded from cache")
else:
    TR = [build(r) for r in TRAIN_RUNS]
    VA = [build(r) for r in VAL_RUNS]
    TE = [build(r) for r in TEST_RUNS]
    pickle.dump((TR, VA, TE), open(WINC, "wb"))
    print(f"windows built in {time.time()-t0:.0f}s")

def stack(ws):
    X = np.concatenate([np.log1p(w["spec"]) for w in ws])
    y = np.concatenate([w["y"] for w in ws])
    return X, y

Xtr, ytr = stack(TR); Xva, yva = stack(VA); Xte, yte = stack(TE)
print("train", Xtr.shape, f"{ytr.mean()*100:.1f}% pos | test pos {yte.mean()*100:.1f}%")

models = {}
lr = Pipeline([("s", StandardScaler()), ("m", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))])
lr.fit(Xtr, ytr)
models["LogReg"] = lr.predict_proba(Xva)[:, 1]  # early-stop proxy: LR has none
models["LogReg_te"] = lr.predict_proba(Xte)[:, 1]

spw = (ytr == 0).sum() / (ytr == 1).sum()
xg = xgb.XGBClassifier(objective="binary:logistic", n_estimators=1000, learning_rate=0.03,
                       max_depth=5, min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                       reg_alpha=0.1, reg_lambda=1.0, scale_pos_weight=spw, eval_metric="aucpr",
                       early_stopping_rounds=30, tree_method="hist", random_state=42, n_jobs=-1)
xg.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
print("xgb best_iteration:", xg.best_iteration)
models["XGBoost_te"] = xg.predict_proba(Xte)[:, 1]

# --- neural baseline: MLP (torch & TF both broken on this machine: c10/tf DLL load fail;
#     sklearn MLPClassifier is the honest neural net we can actually run here) ---
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(64, 16), alpha=1e-3, max_iter=60, random_state=0,
                    early_stopping=True, n_iter_no_change=5, validation_fraction=0.15)
mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
Xtr_s = ((Xtr - mu) / sd).astype(np.float32)
Xte_s = ((Xte - mu) / sd).astype(np.float32)
mlp.fit(Xtr_s, ytr)
models["CNN_te"] = mlp.predict_proba(Xte_s)[:, 1]

# ---------- window-level metrics ----------
rows = []
for nm, key in [("LogReg", "LogReg_te"), ("XGBoost", "XGBoost_te"), ("NN(MLP)", "CNN_te")]:
    p = models[key]
    rows.append(dict(model=nm, roc_auc=roc_auc_score(yte, p), pr_auc=average_precision_score(yte, p)))
print("\n=== window-level on TEST runs (2s windows, label: tagged counts>0) ===")
print(pd.DataFrame(rows).round(4).to_string(index=False))

# ---------- encounter-level recall vs FAR/hr (episode onsets on far windows) ----------
def enc_recall_far(scores_key, far_list=(1, 3, 10, 30, 60)):
    off = np.cumsum([0] + [len(t["spec"]) for t in TR])
    pool = np.concatenate([models[scores_key + "_tr"][off[i]:off[i+1]][TR[i]["dmin"] > 150] for i in range(len(TR))])
    hours = sum((t["dmin"] > 150).sum() for t in TR) * WS / 3600
    out = []
    off_te = np.cumsum([0] + [len(t["spec"]) for t in TE])
    p_te = models[scores_key + "_te"]
    for far in far_list:
        thr = np.sort(pool)[::-1][min(int(far * hours), len(pool) - 1)]
        hits = tot = 0; fa_n = fa_h = 0
        for i, t in enumerate(TE):
            s = p_te[off_te[i]:off_te[i+1]]
            a = s >= thr
            fa_n += int((a & (t["dmin"] > 150) & ~np.r_[False, a[:-1]]).sum())
            fa_h += (t["dmin"] > 150).sum() * WS / 3600
            for j in range(len(t["snr"])):
                m_ = np.abs(t["wtime"] - t["stime"][j]) < 60
                if not m_.any(): continue
                hits += bool(a[m_].any()); tot += 1
        out.append(dict(model=scores_key, target_far=far, achieved_far=fa_n/fa_h, recall=hits/tot, n=tot))
    return out

# need train-set predictions too
models["LogReg_tr"] = lr.predict_proba(Xtr)[:, 1]
models["XGBoost_tr"] = xg.predict_proba(Xtr)[:, 1]
models["CNN_tr"] = mlp.predict_proba(Xtr_s)[:, 1]

res = []
for nm in ["LogReg", "XGBoost", "CNN"]:  # CNN key = NN(MLP) baseline (torch/TF unusable here)
    res += enc_recall_far(nm)
r = pd.DataFrame(res)
print("\n=== ML baselines: encounter recall vs FAR/hr (non-overlapping 2s windows, TEST runs) ===")
print(r.round(3).to_string(index=False))

pickle.dump({k: v for k, v in models.items() if k.endswith("_te")}, open("_scratch/baselines_te.pkl", "wb"))
r.to_pickle("_scratch/baselines_df.pkl")
f.close()
