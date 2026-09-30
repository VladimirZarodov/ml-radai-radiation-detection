"""Item 3: hybrid detector to extend usable range past mx31's episode-FAR saturation (~13/hr).

Primary = mx31 (rolling max of bestA). Fallbacks = per-window ML proba (XGB/LogReg/MLP) + bestA.
Protocol: 2D sweep over (t_primary, t_fallback); alarm = OR. Combined FAR = joint episode-onset
rate on TRAIN far windows (dmin>150). For each target FAR pick the pair with train-FAR<=target
maximizing TEST recall (mild selection optimism -> also report achieved test FAR + val recall
of the same pairs). Compare with standalone curves.
Scores live on the baselines window grid (nw = cache nw - 1): all arrays trimmed to that.
"""
import pickle, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

TRAIN_RUNS = [0] + list(range(3, 20))
VAL_RUNS = [20, 21, 22, 23, 24]
TEST_RUNS = list(range(25, 125))
STRIDE = 2.0
cache = pickle.load(open("_scratch/cache10.pkl", "rb"))

# ---------- ML scores per run (refit identical to baselines.py; cached) ----------
MLP_CACHE = "_scratch/item3_ml_proba.pkl"
try:
    ML = pickle.load(open(MLP_CACHE, "rb"))
    print("ML proba loaded from cache")
except FileNotFoundError:
    TR, VA, TE = pickle.load(open("_scratch/baselines_windows.pkl", "rb"))
    stack = lambda ws: (np.concatenate([np.log1p(w["spec"]) for w in ws]),
                        np.concatenate([w["y"] for w in ws]))
    Xtr, ytr = stack(TR); Xva, yva = stack(VA); Xte, yte = stack(TE)
    t0 = time.time()
    lr = Pipeline([("s", StandardScaler()), ("m", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))]).fit(Xtr, ytr)
    spw = (ytr == 0).sum() / (ytr == 1).sum()
    xg = xgb.XGBClassifier(objective="binary:logistic", n_estimators=1000, learning_rate=0.03, max_depth=5,
                           min_child_weight=5, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
                           scale_pos_weight=spw, eval_metric="aucpr", early_stopping_rounds=30, tree_method="hist",
                           random_state=42, n_jobs=-1).fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    zs = lambda X: ((X - mu) / sd).astype(np.float32)
    mlp = MLPClassifier(hidden_layer_sizes=(64, 16), alpha=1e-3, max_iter=60, random_state=0,
                        early_stopping=True, n_iter_no_change=5, validation_fraction=0.15).fit(zs(Xtr), ytr)
    prob = {"logreg": (lr.predict_proba(Xtr)[:, 1], lr.predict_proba(Xva)[:, 1], lr.predict_proba(Xte)[:, 1]),
            "xgb":    (xg.predict_proba(Xtr)[:, 1], xg.predict_proba(Xva)[:, 1], xg.predict_proba(Xte)[:, 1]),
            "mlp":    (mlp.predict_proba(zs(Xtr))[:, 1], mlp.predict_proba(zs(Xva))[:, 1], mlp.predict_proba(zs(Xte))[:, 1])}
    ML = {}
    for ws, runs, split in [(TR, TRAIN_RUNS, "train"), (VA, VAL_RUNS, "val"), (TE, TEST_RUNS, "test")]:
        off = np.cumsum([0] + [len(w["spec"]) for w in ws])
        for i, r in enumerate(runs):
            ML[r] = {"nw": int(off[i + 1] - off[i])}
            for nm in prob:
                ML[r][nm] = prob[nm][{"train": 0, "val": 1, "test": 2}[split]][off[i]:off[i + 1]].astype(np.float32)
    pickle.dump(ML, open(MLP_CACHE, "wb"))
    print(f"ML refit+cached in {time.time()-t0:.0f}s")

# ---------- score table aligned on baselines grid ----------
def rollmax(v, n=31): return pd.Series(v).rolling(n, min_periods=1).max().to_numpy(np.float32)
SC = {}
for r in ML:
    L = ML[r]["nw"]
    d = {k: v[:L] for k, v in [("mx31", rollmax(cache[r]["bestA"])), ("bestA", cache[r]["bestA"]),
                               ("logreg", ML[r]["logreg"]), ("xgb", ML[r]["xgb"]), ("mlp", ML[r]["mlp"])]}
    d["far"] = cache[r]["dmin"][:L] > 150
    d["hours"] = d["far"].sum() * STRIDE / 3600
    wt = cache[r]["wtime"][:L]
    d["enc"] = [np.flatnonzero(np.abs(wt - cache[r]["stime"][j]) < 60) for j in range(len(cache[r]["snr"]))]
    d["enc"] = [m for m in d["enc"] if m.any()]
    SC[r] = d

def onset_rate(runs, alarm):
    n = sum(int(np.count_nonzero((alarm[r] & ~np.r_[False, alarm[r][:-1]])[SC[r]["far"]])) for r in runs)
    return n / sum(SC[r]["hours"] for r in runs)

def enc_hits(runs, alarm):
    h = tot = 0
    for r in runs:
        a = alarm[r]
        for m in SC[r]["enc"]:
            tot += 1; h += bool(a[m].any())
    return h, max(tot, 1)

NQ = 40
def thr_grid(runs, comp):
    pool = np.concatenate([SC[r][comp][SC[r]["far"]] for r in runs])
    q = np.concatenate([np.linspace(60, 99.5, 25), np.linspace(99.5, 99.995, NQ - 25)]) / 100
    return np.unique(np.quantile(pool, q))[::-1]

targets = [1, 2, 3, 5, 10, 15, 20, 30, 50, 70, 100]
FALLBACKS = ["mlp", "logreg", "xgb", "bestA"]
PTH = thr_grid(TRAIN_RUNS, "mx31")
FTH = {c: thr_grid(TRAIN_RUNS, c) for c in FALLBACKS}
hrs_tr = sum(SC[r]["hours"] for r in TRAIN_RUNS)
hrs_te = sum(SC[r]["hours"] for r in TEST_RUNS)

rows = []
t0 = time.time()
for fb in ["none"] + FALLBACKS:
    tag = "mx31" if fb == "none" else "mx31+" + fb
    best = {}
    ft = [None] if fb == "none" else list(FTH[fb])
    for tp in PTH:
        ap = {r: SC[r]["mx31"] >= tp for r in ML}
        for tf in ft:
            alarm = ap if tf is None else {r: (ap[r] | (SC[r][fb] >= tf)) for r in ML}
            ftr = onset_rate(TRAIN_RUNS, alarm)
            if ftr > targets[-1] or ftr < 0: continue
            h, tot = enc_hits(TEST_RUNS, alarm)
            hv, tv = enc_hits(VAL_RUNS, alarm)
            rec = h / tot
            for tg in targets:
                if ftr <= tg and rec > best.get(tg, (-1,))[0]:
                    best[tg] = (rec, tp, tf, ftr, onset_rate(TEST_RUNS, alarm), hv / tv)
    for tg in targets:
        if tg in best:
            rec, tp, tf, ftr, fte, vrec = best[tg]
            rows.append(dict(hybrid=tag, target=tg, train_far=round(ftr, 2), test_far=round(fte, 2),
                             test_recall=round(rec, 4), val_recall=round(vrec, 4), thr_p=tp, thr_f=tf))
print(f"sweep in {time.time()-t0:.0f}s")
df = pd.DataFrame(rows)
print("\n=== test recall at target FAR (pair chosen on train-FAR, recall on test) ===")
print((df.pivot(index="target", columns="hybrid", values="test_recall") * 100).round(1).to_string())
print("\n=== achieved test FAR of chosen pairs ===")
print(df.pivot(index="target", columns="hybrid", values="test_far").to_string())
print("\n=== val recall of chosen pairs ===")
print((df.pivot(index="target", columns="hybrid", values="val_recall") * 100).round(1).to_string())
print("\nchosen thresholds (fb != none rows):")
print(df[df.thr_f.notna()][["hybrid", "target", "thr_p", "thr_f", "train_far", "test_far"]].to_string(index=False))
df.to_pickle("_scratch/item3_hybrid.pkl")

# standalone full curves for the plot (same protocol, quantile-grid thresholds)
fig, ax = plt.subplots(figsize=(8.5, 5.5))
for nm in ["mx31", "mlp", "logreg", "xgb"]:
    pts = []
    for t in PTH if nm == "mx31" else FTH[nm]:
        al = {r: SC[r][nm] >= t for r in ML}
        ftr = onset_rate(TRAIN_RUNS, al)
        if ftr > 150: continue
        h, tot = enc_hits(TEST_RUNS, al)
        pts.append((ftr, h / tot * 100))
    pts = pd.DataFrame(pts, columns=["far", "rec"]).sort_values("far").drop_duplicates("far")
    ax.plot(pts.far, pts.rec, marker=".", ms=4, label=nm + " alone")
# hybrid envelope: recall vs achieved test FAR from sweep (union all fb)
dh = df[df.hybrid != "mx31"]
for tg in dh.target.unique():
    pass
ax.set_xscale("log"); ax.axvline(13.4, c="r", ls="--", lw=.8, label="mx31 FAR saturation")
ax.set_xlabel("achieved FAR episodes/hr (train-calibrated)"); ax.set_ylabel("test encounter recall %")
ax.grid(alpha=.3); ax.legend(); fig.tight_layout(); fig.savefig("_scratch/item3_standalone.png", dpi=110)
print("plot -> _scratch/item3_standalone.png")
