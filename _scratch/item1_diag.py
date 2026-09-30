"""Item 1a: diagnostics.
1) Inspect detector/ and diagnostics/ groups + all h5 attrs (pileup flags? position/speed?).
2) Characterize the SPECTRAL SIGNATURE of high-z false-alarm windows in the TRAIN far pool:
   mean Pearson-residual spectrum (X-S)/sqrt(S) for FAR windows split by bestA level and by
   count rate -> is the misfit a smooth continuum (shape-model bias, could add component)
   or peaks at pileup sum-energies (K40+K40=2922, Bi214 609+609=1218, Cs+K, etc.)?
"""
import h5py
import numpy as np
import pandas as pd

exec(open("_scratch/detect6.py", encoding="utf-8").read().split("rows = []")[0])
# provides f, names, M, U, ids, U2, poisson_fit, run_windows, NB, STRIDE, TRAIN_RUNS, VAL_RUNS

# bin energy edges (sqrt-keV 15..3000, NB=128)
edges = (np.linspace(np.sqrt(15), np.sqrt(3000), NB + 1)) ** 2
def binof(e):
    return int(np.clip(np.floor((np.sqrt(e) - np.sqrt(15)) / (np.sqrt(3000) - np.sqrt(15)) * NB), 0, NB - 1))
print("key bins: 662keV->", binof(662), " 1461->", binof(1461), " 2614->", binof(2614),
      " 1218(609+609)->", binof(1218), " 2922(K40+K40)->", binof(2922), " 1764->", binof(1764))

print("\n--- h5 structure: detector / diagnostics / attrs ---")
g = f["runs/run25"]
for grp in ("detector", "diagnostics"):
    print(grp, ":", {k: (v.shape, str(v.dtype)[:12]) for k, v in grp and g[grp].items() if isinstance(v, h5py.Dataset)})
    for k, v in g[grp].items():
        for ak, av in getattr(v, "attrs", {}).items():
            print(f"   {grp}/{k}.{ak} = {av}")
for grp in (f, g, g["listmode"]):
    for ak, av in grp.attrs.items():
        s = str(av)
        print("ATTR", s[:120] if len(s) < 120 else s[:120] + "...")

# --- residual-signature analysis on TRAIN far windows ---
import pickle
cache = pickle.load(open("_scratch/cache10.pkl", "rb"))

# recompute spec/S for train runs (needed for residual spectra)
res_rows = []   # (run, win_idx, rate, z, resid/sqrt(S) vector) stored per-group later
groups = {}     # label -> list of residual arrays
for rid in TRAIN_RUNS:
    spec, tagw, wtime, snr, stime, sid = run_windows(rid, 2.0)
    S = np.maximum(poisson_fit(spec), 1.0)
    R = spec - S
    z = ((R @ U.T) / np.sqrt(S @ U2.T + 1e-9)).max(1)
    dmin = cache[rid]["dmin"]
    m_far = dmin > 150
    rate = spec.sum(1)
    pear = R / np.sqrt(S)
    hi = m_far & (z >= 4.34)          # ~FA windows at FAR10 threshold
    lo = m_far & (z < 4.34)
    hi_rate = m_far & (rate > np.quantile(rate[m_far], .75)) & (z < 4.34)  # high-rate NON-alarm
    for lab, sel in [("FA(z>=4.34)", hi), ("far-normal", lo), ("high-rate-no-alarm", hi_rate)]:
        if sel.any():
            groups.setdefault(lab, []).append(pear[sel].sum(0))
            groups.setdefault(lab + "_n", []).append([sel.sum()])
            groups.setdefault(lab + "_rate", []).append([rate[sel].sum()])

print("\n--- mean Pearson-residual spectrum per group (counts per window, smoothed 5-bin) ---")
out = {}
for lab in ["FA(z>=4.34)", "far-normal", "high-rate-no-alarm"]:
    tot = np.sum(groups[lab], 0)
    n = sum(map(lambda x: x[0], groups[lab + "_n"]))
    r = sum(map(lambda x: x[0], groups[lab + "_rate"])) / n
    sm = pd.Series(tot / n).rolling(5, center=True).mean().to_numpy()
    out[lab] = sm
    print(f"  {lab}: n={n}, mean rate={r:.0f}")
df = pd.DataFrame(out)
# where do FA and high-rate residuals differ from normal far windows?
df["FA_minus_normal"] = df["FA(z>=4.34)"] - df["far-normal"]
df["HR_minus_normal"] = df["high-rate-no-alarm"] - df["far-normal"]
# print coarse profile: 16 blocks of 8 bins
blk = df.groupby(np.arange(NB) // 8).mean()
blk.index = [f"{edges[i*8]:.0f}-{edges[min((i+1)*8,NB-1)]:.0f}keV" for i in range(len(blk))]
print(blk[["far-normal", "FA(z>=4.34)", "high-rate-no-alarm", "FA_minus_normal", "HR_minus_normal"]]
      .round(3).to_string())
