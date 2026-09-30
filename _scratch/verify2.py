"""Verification pass #2: reproduce the claimed background non-stationarity,
check window<->encounter SNR mapping, and the count-vs-detectability mismatch."""
import h5py
import numpy as np
import pandas as pd

H5 = "training_v4.3.h5"
EMIN, EMAX, NB = 15.0, 3000.0, 256

def run_times(run):
    dt = run["listmode/dt"][:]
    t = np.cumsum(dt, dtype=np.uint64) / 1e6
    return t, run["listmode/energy"][:], run["listmode/id"][:]

def windowize(t, energy, eid, ws=2.0):
    dur = float(t[-1])
    nw = int(np.floor((dur - ws) / ws + 1e-9))
    wi = np.floor(t / ws).astype(np.int64)
    ok = wi < nw
    tot = np.bincount(wi[ok], minlength=nw).astype(np.float64)
    src = np.bincount(wi[ok & (eid != 0)], minlength=nw).astype(np.float64)
    return wi, tot, src, nw

f = h5py.File(H5, "r")

# ---------- A) background non-stationarity ----------
print("=== A) per-2s-window gross counts (all events, no energy cut) ===")
rows = []
for rid in [0, 1, 2, 3, 20, 50]:
    t, e, eid = run_times(f[f"runs/run{rid}"])
    wi, tot, src, nw = windowize(t, e, eid)
    bg = tot - src
    rows.append({
        "run": rid, "mean_all": tot.mean(), "std_all": tot.std(),
        "sqrt_mean": np.sqrt(tot.mean()), "ratio": tot.std() / np.sqrt(tot.mean()),
        "mean_bg": bg.mean(), "std_bg": bg.std(),
        "cv_bg_pct": 100 * bg.std() / bg.mean(),
        "p5_bg": np.percentile(bg, 5), "p95_bg": np.percentile(bg, 95),
    })
print(pd.DataFrame(rows).round(1).to_string(index=False))

# window-to-window variability: lag-1 autocorrelation of bg rate (smooth or jumpy?)
t, e, eid = run_times(f["runs/run0"])
wi, tot, src, nw = windowize(t, e, eid)
bg = (tot - src)
print("\nrun0 bg lag1 autocorr:", np.corrcoef(bg[:-1], bg[1:])[0, 1].round(3),
      " lag10 (20s):", np.corrcoef(bg[:-10], bg[10:])[0, 1].round(3),
      " lag150 (300s):", np.corrcoef(bg[:-150], bg[150:])[0, 1].round(3))

# ---------- B) encounter metadata vs tagged source counts ----------
print("\n=== B) SNR field vs raw tagged source counts per encounter ===")
recs = []
for rid in range(0, 25):
    g = f[f"runs/run{rid}"]
    t, e, eid = run_times(g)
    wi, tot, src, nw = windowize(t, e, eid)
    snr = g["sources/snr/peak"][:]
    stime = g["sources/time"][:] / 1000.0     # ms -> s, closest approach
    sid = g["sources/id"][:]
    sh = g["sources/shielding"][:]
    for j in range(len(sid)):
        # tagged events for this source id within +/-60 s of closest approach
        near = np.abs(t - stime[j]) < 60
        m = near & (eid == sid[j])
        cnt = int(m.sum())
        # also count all events for this id over the whole run
        cnt_all = int((eid == sid[j]).sum())
        recs.append({"run": rid, "src_id": sid[j], "shield": sh[j],
                     "snr": snr[j], "cnt_120s": cnt, "cnt_all": cnt_all})
df = pd.DataFrame(recs)
df = df[df.src_id != 0]
names = f.attrs["source_names"]
sh_names = f.attrs["source_shielding_names"]
df["name"] = df.src_id.map(lambda i: str(names[i]))
print("n encounters:", len(df))
print("corr(count_120s, snr) pearson:", np.corrcoef(df.cnt_120s, df.snr)[0, 1].round(3),
      " spearman:", df[["cnt_120s", "snr"]].corr(method="spearman").iloc[0, 1].round(3))
print("corr(count_all,  snr) spearman:", df[["cnt_all", "snr"]].corr(method="spearman").iloc[0, 1].round(3))
print("\nshielding effect (median cnt_all & snr by shielding):")
print(df.groupby("shield").agg(n=("snr", "size"), cnt_all=("cnt_all", "median"), snr=("snr", "median")).round(2))
print("\nexamples of count-vs-SNR disagreement (cnt_all>1500 but low snr, or cnt<400 but high snr):")
bad = df[(df.cnt_all > 1500) & (df.snr < 6)]
good = df[(df.cnt_all < 400) & (df.snr > 10)]
print(bad.sort_values("snr").head(8).to_string(index=False))
print(good.sort_values("snr", ascending=False).head(8).to_string(index=False))

# distribution of snr field
print("\nsnr field stats:", df.snr.describe().round(2).to_dict())

f.close()
