"""Quick probe: detector/position datasets + diagnostics content (for item 6 later),
and pileup-statistic vs rate on TRAIN far windows (to design a flag/exclude rule)."""
import h5py, pickle
import numpy as np
import pandas as pd

f = h5py.File("training_v4.3.h5", "r")
g = f["runs/run25"]
print("--- detector/position ---")
for k in g["detector/position"]:
    d = g[f"detector/position/{k}"][:]
    print(f"{k}: shape={d.shape} dtype={d.dtype} head={d[:5]} range=({d.min():.2f},{d.max():.2f})")
for k, v in g["detector/position"].attrs.items():
    print("pos attr:", k, v)
print("--- diagnostics ---")
for k in g["diagnostics"]:
    d = g[f"diagnostics/{k}"][:]
    print(f"{k}: shape={d.shape} head={d[:6]} min={d.min():.3g} max={d.max():.3g} mean={d.mean():.3g}")
print("dur:", g["listmode/dt"][:].sum()/1e6)

# does position repeat across runs? compare distance arrays of runs 25 vs 26 vs 100
d25 = g["detector/position/distance"][:]
for r2 in (26, 100, 124):
    d2 = f[f"runs/run{r2}/detector/position/distance"][:]
    n = min(len(d25), len(d2))
    print(f"corr distance run25 vs run{r2}: {np.corrcoef(d25[:n], d2[:n])[0,1]:.3f} (len {len(d25)} vs {len(d2)})")

# --- pileup statistic on train far windows ---
exec(open("_scratch/detect6.py", encoding="utf-8").read().split("rows = []")[0])
edges = (np.linspace(np.sqrt(15), np.sqrt(3000), NB + 1)) ** 2
def binof(e): return int(np.clip(np.floor((np.sqrt(e)-np.sqrt(15))/(np.sqrt(3000)-np.sqrt(15))*NB), 0, NB-1))
B662, B1218, B1461, B2922, BLOW = binof(662), binof(1218), binof(1461), binof(2922), binof(30)

cache = pickle.load(open("_scratch/cache10.pkl", "rb"))
rows = []
for rid in TRAIN_RUNS:
    spec, tagw, wtime, snr, stime, sid = run_windows(rid, 2.0)
    S = np.maximum(poisson_fit(spec), 1.0)
    z = ((spec - S) @ U.T) / np.sqrt(S @ U2.T + 1e-9)
    best = z.max(1)
    dmin = cache[rid]["dmin"]
    rate = spec.sum(1)
    # raw pileup diagnostics: sum-line to parent-line ratios (independent of fit)
    pile = (spec[:, B1218] + spec[:, B2922]) / (spec[:, B662] + 1.0)
    lowfrac = spec[:, BLOW] / (rate + 1.0)
    m = dmin > 150
    for i in np.flatnonzero(m):
        rows.append((rid, i, rate[i], best[i], pile[i], lowfrac[i]))
    # also CA windows: what rates do encounters actually reach at CA+-60?
    for j in range(len(snr)):
        mm = np.abs(wtime - stime[j]) < 60
        if mm.any():
            rows.append((rid, -1, rate[mm].max(), best[mm].max(), np.nan, np.nan))
df = pd.DataFrame(rows, columns=["run", "w", "rate", "z", "pile", "lowfrac"])
far = df[df.w >= 0].copy()
ca = df[df.w == -1]
far["grp"] = np.where(far.z >= 4.34, "FA", np.where(far.rate > far.rate.quantile(.75), "hi-rate", "norm"))
print("\n--- pileup stat (1218+2922)/(662+1) by group, train far windows ---")
print(far.groupby("grp")[["rate", "pile", "lowfrac"]].describe().round(3).to_string())
# correlation of pile with rate among far windows
print("\nspearman(pile, rate) far windows:", far[["pile", "rate"]].corr(method="spearman").iloc[0, 1].round(3))
# how many CA windows exceed candidate exclusion rates?
for thr in (8000, 10000, 12000, 15000):
    print(f"CA(max rate in ±60s) > {thr}: {(ca.rate > thr).mean()*100:.1f}% of encounters (n={len(ca)})")
print("CA rate quantiles:", ca.rate.quantile([.5, .75, .9, .99]).round(0).to_dict())
print("far-window rate quantiles:", far.rate.quantile([.5, .75, .9, .99, .999]).round(0).to_dict())
far.to_pickle("_scratch/pile_stats_train.pkl")
