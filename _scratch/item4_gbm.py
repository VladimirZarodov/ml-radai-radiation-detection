"""Item 4: gradient-boosted model on the physical statistics instead of a hand-picked max.

Features per 2s window (aligned to baselines window grid):
  zA (61 template matched-filter z) + aggregates: max, 2nd, mean-top5, #(z>2), #(z>3),
  bg_rate, T=bg-shape-fit residual mass. Label: tagged source counts > 0 (repo convention).
Model: XGBClassifier (same hyper-param family as baselines.py), early stop on val.
Score = proba. Evaluate in the SAME episode-FAR space (train-calibrated, test recall,
val check) vs mx31, and dump proba per run for the item-3 hybrid.
Also: per-isotope recall @FAR10 for gbz vs mx31 -> does it help the bg-confounded group?
"""
import pickle, time
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score

TRAIN_RUNS = [0] + list(range(3, 20))
VAL_RUNS = [20, 21, 22, 23, 24]
TEST_RUNS = list(range(25, 125))
STRIDE = 2.0
cache = pickle.load(open("_scratch/cache10.pkl", "rb"))
names = [str(n).split("_shielding")[0] for n in __import__("h5py").File("training_v4.3.h5").attrs["source_names"]]
TR, VA, TE = pickle.load(open("_scratch/baselines_windows.pkl", "rb"))

def feats(ws):
    """ws = list of per-run dicts from baselines cache; zA from cache trimmed to len."""
    X = []
    for w in ws:
        rid = w.get("rid")
        c = cache[rid]
        L = len(w["spec"])
        z = c["zA"][:L].astype(np.float32)
        zs = np.sort(z[:, ::-1], 1)[:, ::-1]  # desc per row
        mx = zs[:, 0]; s2 = zs[:, 1]; top5 = zs[:, :5].mean(1)
        n2 = (z > 2).sum(1).astype(np.float32); n3 = (z > 3).sum(1).astype(np.float32)
        bg = c["bg_rate"][:L]
        m31 = pd.Series(c["bestA"][:L]).rolling(31, min_periods=1).max().to_numpy(np.float32)
        X.append(np.column_stack([z, mx, s2, top5, n2, n3, m31, np.log1p(bg)]))
    return np.vstack(X)

for nm, ws, runs in [("TR", TR, TRAIN_RUNS), ("VA", VA, VAL_RUNS), ("TE", TE, TEST_RUNS)]:
    for w, r in zip(ws, runs):
        w["rid"] = r
Xtr, ytr = feats(TR), np.concatenate([w["y"] for w in TR])
Xva, yva = feats(VA), np.concatenate([w["y"] for w in VA])
Xte, yte = feats(TE), np.concatenate([w["y"] for w in TE])
print("feature matrix:", Xtr.shape, f"{ytr.mean()*100:.1f}% pos")

spw = (ytr == 0).sum() / (ytr == 1).sum()
t0 = time.time()
m = xgb.XGBClassifier(objective="binary:logistic", n_estimators=1000, learning_rate=0.03, max_depth=5,
                      min_child_weight=5, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
                      scale_pos_weight=spw, eval_metric="aucpr", early_stopping_rounds=30, tree_method="hist",
                      random_state=42, n_jobs=-1).fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
print(f"xgb fit {time.time()-t0:.0f}s best_iteration={m.best_iteration}")
pte = m.predict_proba(Xte)[:, 1]
ptr = m.predict_proba(Xtr)[:, 1]
ptv = m.predict_proba(Xva)[:, 1]
print(f"window-level TEST: ROC-AUC={roc_auc_score(yte, pte):.3f} PR-AUC={average_precision_score(yte, pte):.3f}")

# split back per run
PROB = {}
off = np.cumsum([0] + [len(w["spec"]) for w in TR]); 
for i, r in enumerate(TRAIN_RUNS): PROB[r] = ptr[off[i]:off[i+1]]
off = np.cumsum([0] + [len(w["spec"]) for w in VA])
for i, r in enumerate(VAL_RUNS): PROB[r] = ptv[off[i]:off[i+1]]
off = np.cumsum([0] + [len(w["spec"]) for w in TE])
for i, r in enumerate(TEST_RUNS): PROB[r] = pte[off[i]:off[i+1]]
pickle.dump(PROB, open("_scratch/item4_gbz_proba.pkl", "wb"))

def onset_n(s, far, thr):
    a = np.asarray(s) >= thr
    return int(np.count_nonzero((a & ~np.r_[False, a[:-1]])[far]))

def curves(S, runs_tr=TRAIN_RUNS, runs_te=TEST_RUNS, runs_va=VAL_RUNS):
    far_ = {r: cache[r]["dmin"][:len(S[r])] > 150 for r in S}
    pool = np.concatenate([S[r][far_[r]] for r in runs_tr])
    grid_q = np.concatenate([np.linspace(50, 99.5, 40), np.linspace(99.5, 99.9998, 40)])
    rows = []
    for thr in np.unique(np.quantile(pool, grid_q / 100))[::-1]:
        fa_n = fa_h = hits = tot = 0
        v_hits = v_tot = 0
        for r in runs_te:
            s = np.asarray(S[r]); c = cache[r]
            fa_n += onset_n(s, far_[r], thr)
            fa_h += far_[r].sum() * STRIDE / 3600
            wt = c["wtime"][:len(s)]
            for j in range(len(c["snr"])):
                m_ = np.abs(wt - c["stime"][j]) < 60
                if not m_.any(): continue
                hits += bool((s[m_] >= thr).any()); tot += 1
        for r in runs_va:
            s = np.asarray(S[r]); c = cache[r]
            wt = c["wtime"][:len(s)]
            for j in range(len(c["snr"])):
                m_ = np.abs(wt - c["stime"][j]) < 60
                if not m_.any(): continue
                v_hits += bool((s[m_] >= thr).any()); v_tot += 1
        rows.append(dict(thr=thr, far=fa_n / fa_h, recall=hits / tot, val_recall=v_hits / max(v_tot, 1)))
    return pd.DataFrame(rows).sort_values("far").drop_duplicates("far")

Sg = {r: PROB[r] for r in cache if r in PROB}
Sm = {r: pd.Series(cache[r]["bestA"]).rolling(31, min_periods=1).max().to_numpy(np.float32) for r in cache}
res = {}
for tag, S in [("gbz", Sg), ("mx31", Sm)]:
    d = curves(S)
    res[tag] = d
    print(f"\n### {tag}")
    for tgt in (1, 3, 10, 30, 60, 100):
        i = (d.far - tgt).abs().idxmin()
        row = d.loc[i]
        print(f"  FAR≈{row.far:5.1f}: test={row.recall*100:5.1f}%  val={row.val_recall*100:5.1f}%")
pickle.dump(res, open("_scratch/item4_res.pkl", "wb"))

# per-isotope recall @FAR10: gbz vs mx31, split shape-distinguishable vs bg-confounded
def iso_table(S, tag, tgt=10):
    d = res[tag]
    i = (d.far - tgt).abs().idxmin(); thr = d.loc[i].thr
    rows = []
    for r in TEST_RUNS:
        s = np.asarray(S[r]); c = cache[r]
        wt = c["wtime"][:len(s)]
        for j in range(len(c["snr"])):
            m_ = np.abs(wt - c["stime"][j]) < 60
            if not m_.any(): continue
            rows.append((names[int(c["sid"][j])], bool((s[m_] >= thr).any())))
    df = pd.DataFrame(rows, columns=["name", "hit"]).groupby("name").agg(n=("hit", "size"), recall=("hit", "mean"))
    return df, thr
dfg, thrg = iso_table(Sg, "gbz", 10)
dfm, thrm = iso_table(Sm, "mx31", 10)
iso = dfg.join(dfm, lsuffix="_gbz", rsuffix="_mx31").sort_values("recall_gbz", ascending=False)
print("\nper-isotope recall @FAR10 (gbz vs mx31):")
print((iso * 100).round(0).to_string())
