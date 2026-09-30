"""Item 6a: spatial background feasibility.

Routes repeat across runs (corr(distance)~0.995). Hypothesis: bg count RATE has a
position-dependent profile that the per-window spectral fit cannot use (it absorbs
amplitude freely -> NORM sources indistinguishable). Build rate profile R(x) pooled
from TRAIN far windows, then on TEST: does obs-bg/pred-R correlate with position
better than with anything else, and do NORM-family encounters show excess residual
near CA that the spectral z misses?
"""
import pickle
import h5py
import numpy as np
import pandas as pd

f = h5py.File("training_v4.3.h5", "r")
cache = pickle.load(open("_scratch/cache10.pkl", "rb"))
TRAIN_RUNS = [0] + list(range(3, 20))
VAL_RUNS = [20, 21, 22, 23, 24]
TEST_RUNS = list(range(25, 125))

def win_dist(rid, wtime):
    d = f[f"runs/run{rid}/detector/position/distance"][:]   # meters @10Hz
    ts = np.arange(len(d)) / 10.0
    return np.interp(wtime, ts, d).astype(np.float32)

DIST = {r: win_dist(r, cache[r]["wtime"]) for r in cache}

# --- how much does bg rate vary with position vs run? ---
NX = 150
xb = np.linspace(0, 15200, NX + 1)
rows = []
for r in TRAIN_RUNS + VAL_RUNS + TEST_RUNS:
    c = cache[r]
    m = c["dmin"] > 150
    rows.append(pd.DataFrame(dict(run=r, xbin=np.clip(np.digitize(DIST[r][m], xb) - 1, 0, NX - 1),
                                  rate=c["bg_rate"][m])))
rg = pd.concat(rows)
tab = rg.groupby(["run", "xbin"])["rate"].median().unstack()
tab = tab.reindex(columns=range(NX))
ok = tab.notna().sum(0) > 40
tab = tab.loc[:, ok.to_numpy()]
overall = tab.median(1).rename("run_median")
xmed = tab.median(0).rename("x_median")
gm = np.nanmean(tab.values)
ss_tot = np.nansum((tab.values - gm) ** 2)
ss_res = np.nansum((tab.values - xmed.values[None, :]) ** 2)
print(f"bg rate: position-bin variance share = {(ss_tot - ss_res)/ss_tot:.2%}; "
      f"run median spread {overall.min():.0f}..{overall.max():.0f}")
# per-run LEVEL-normalized profile shape: divide each run by its own median, then corr vs pooled
tn = tab.div(tab.median(1), axis=0)
xn = tn.median(0)
cor = tn.apply(lambda row: np.corrcoef(row.fillna(1.0), xn.fillna(1.0))[0, 1], axis=1)
print("corr(run NORMALIZED rate profile, pooled x-profile): median %.3f, q10 %.3f, q90 %.3f"
      % (cor.median(), cor.quantile(.1), cor.quantile(.9)))
# split-half of TRAIN runs: does the pooled x-profile of half A correlate with half B?
ha = tab.iloc[:9].median(0); hb = tab.iloc[9:].median(0)
m = ha.notna() & hb.notna()
print("split-half corr of x-profile (train runs):", np.corrcoef(ha[m], hb[m])[0, 1].round(3))
# same but normalized per run first
han = tn.iloc[:9].median(0); hbn = tn.iloc[9:].median(0)
m = han.notna() & hbn.notna()
print("split-half corr, per-run level-normalized:", np.corrcoef(han[m], hbn[m])[0, 1].round(3))

# --- residual excess near CA for NORM-family vs line-source encounters (TEST) ---
names = [str(n).split("_shielding")[0] for n in f.attrs["source_names"]]
NORM = None
enc = []
for r in TEST_RUNS:
    c = cache[r]
    keep_x = ((xb[:-1] + xb[1:]) / 2)[ok.to_numpy()]
    med_prof = tab.median(0).interpolate().bfill().ffill().to_numpy()
    pred = np.interp(DIST[r], keep_x, med_prof)
    for j in range(len(c["snr"])):
        m = np.abs(c["wtime"] - c["stime"][j]) < 60
        if not m.any(): continue
        nm = names[int(c["sid"][j])]
        far_m = np.abs(c["wtime"] - c["stime"][j]) > 240
        base = np.median((c["bg_rate"] - pred)[far_m])
        rel = ((c["bg_rate"] - pred)[m] - base) / np.maximum(pred[m], 500)
        enc.append(dict(name=nm, rel_excess=float(rel.mean()), snr=float(c["snr"][j]),
                        z=float(c["bestA"][m].max())))
e = pd.DataFrame(enc)
print("\nisotopes present:", sorted(e.name.unique()))
grp = e.assign(kind=np.where(e.name.str.contains("U|Th|K-40|Cs", regex=True), "norm/bg-like", "line"))
print(grp.groupby("kind")[["rel_excess", "z"]].agg(["count", "median", "mean"]).round(3).to_string())
print("\ntop/bottom isotopes by rel_excess median:")
print(e.groupby("name")["rel_excess"].agg(["count", "median", "mean"]).sort_values("median").round(3).to_string())
e.to_pickle("_scratch/item6a_enc.pkl")
pickle.dump(dict(tab=tab, DIST=DIST), open("_scratch/item6a_pos.pkl", "wb"))
