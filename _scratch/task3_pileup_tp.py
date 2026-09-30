"""Task 3: does pile-up hurt RECALL on TRUE detections (not just cause false alarms)?
For every test encounter: peak window rate (bg_rate, counts/2s) inside CA +/-60 s, mx31 score max in
same interval, snr, isotope family.  Questions:
 (a) recall by peak-rate stratum at the three headline thresholds (protocol A, train pool, dmin>150);
 (b) same restricted to shape-distinguishable line sources (removes NORM-confounding), and matched on snr
     (within snr tertile: low-rate vs high-rate recall);
 (c) score-level: is the CA score margin (EMAX - thr) depressed at high peak rate after controlling snr?
 (d) veto-regime: encounters with CA peak >16k counts/2s - how much recall do we lose there?
CIs: run-cluster bootstrap.
"""
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(23)
cache = pickle.load(open("_scratch/cache10.pkl", "rb"))
t1 = pickle.load(open("_scratch/task1_pool.pkl", "rb"))
CAND = t1["CAND"]; THRS = {t: CAND[t1["res"][t]["jA"]] for t in (1, 3, 10)}
print("protocol-A thresholds:", {k: round(v, 2) for k, v in THRS.items()})
import h5py
names = [str(n).split("_shielding")[0] for n in h5py.File("training_v4.3.h5", "r").attrs["source_names"]]
TEST = list(range(25, 125))
S = {r: pd.Series(c["bestA"]).rolling(31, min_periods=1).max().to_numpy(np.float32).astype(np.float64)
     for r, c in cache.items()}

CONFOUNDED_RE = "U|Th|Ra|K-40|Cs|Pu|Sr"   # U-series (incl. Ra), Th, K-40, Cs-137, Pu (weak shldd X-ray), Sr-90 (bremsstrahlung)
rows = []
for r in TEST:
    c = cache[r]
    for j in range(len(c["snr"])):
        m = np.abs(c["wtime"] - c["stime"][j]) < 60
        if not m.any():
            continue
        nm = names[int(c["sid"][j])]
        rows.append(dict(run=r, name=nm, line=not bool(pd.Series([nm]).str.contains(CONFOUNDED_RE, regex=True)[0]),
                         snr=float(c["snr"][j]), peak=float(c["bg_rate"][m].max()),
                         emax=float(S[r][m].max())))
enc = pd.DataFrame(rows)
print(f"{len(enc)} test encounters, {enc.run.nunique()} runs; peak-rate quantiles: "
      f"{np.percentile(enc.peak, [10, 25, 50, 75, 90, 99]).round(0)}")
enc["stratum"] = pd.cut(enc.peak, [0, 4000, 6000, 8000, 11000, 16000, np.inf],
                        labels=["<4k", "4-6k", "6-8k", "8-11k", "11-16k", ">=16k"])

def boot_recall(sub, thr, B=1000):
    """run-cluster bootstrap CI of recall(sub)=P(emax>=thr)"""
    hit = (sub.emax.values >= thr).astype(float)
    ri, _ = pd.factorize(sub.run.values)
    cnt_r = np.bincount(ri, minlength=ri.max() + 1)
    runs = np.arange(ri.max() + 1)
    hs = np.bincount(ri, weights=hit, minlength=len(runs))
    recs = []
    for b in range(B):
        sel = rng.integers(0, len(runs), len(runs))
        w = np.bincount(sel, minlength=len(runs)).astype(float)
        if (w * cnt_r).sum() == 0:
            continue
        recs.append((w * hs).sum() / (w * cnt_r).sum())
    return np.percentile(recs, [2.5, 97.5])

print("\n(a) recall by CA peak-rate stratum (all 61 templates, thresholds from train calibration):")
tab = []
for tg in (1, 3, 10):
    for s, g in enc.groupby("stratum", observed=True):
        lo, hi = boot_recall(g, THRS[tg])
        rec = (g.emax >= THRS[tg]).mean()
        tab.append(dict(target=tg, stratum=s, n=len(g), recall=rec, lo=lo, hi=hi))
df = pd.DataFrame(tab)
for tg, gg in df.groupby("target"):
    print(f"  @FAR{tg}: " + " | ".join(f"{x.stratum}:{x.recall*100:4.1f}%[{x.lo*100:.0f}-{x.hi*100:.0f}](n={x.n})"
                                       for _, x in gg.iterrows()))

encL = enc[enc.line].copy()
print(f"\n(b) shape-distinguishable ONLY ({len(encL)} enc):")
for tg in (3, 10):
    print(f"  @FAR{tg}: " + " | ".join(
        f"{s}:{(g.emax >= THRS[tg]).mean()*100:4.1f}%(n={len(g)})" for s, g in encL.groupby("stratum", observed=True)))
encL["snr_t"] = pd.qcut(encL.snr, 3, labels=["snr-low", "snr-mid", "snr-high"])
encL["rate_g"] = np.where(encL.peak >= 8000, "hi(>=8k)", "lo(<8k)")
print("  snr-matched rate comparison @FAR3 (line sources):")
for st, g in encL.groupby("snr_t", observed=True):
    out = [f"{st}: snr med {g.snr.median():.1f}"]
    for rg, gg in g.groupby("rate_g"):
        lo, hi = boot_recall(gg, THRS[3])
        out.append(f"{rg} {(gg.emax >= THRS[3]).mean()*100:4.1f}%[{lo*100:.0f}-{hi*100:.0f}](n={len(gg)})")
    print("    " + " | ".join(out))

print("\n(c) score margin vs peak rate, line sources, controlling snr:")
encL = encL.assign(margin3=encL.emax - THRS[3])
encL["snr_b"] = pd.cut(encL.snr, [0, 6, 12, np.inf], labels=["snr<6", "6-12", ">12"])
for sb, g in encL.groupby("snr_b", observed=True):
    lo_hi = g[g.peak >= 8000].margin3.median() - g[g.peak < 8000].margin3.median()
    print(f"  {sb} (n={len(g)}): median margin low-rate {g[g.peak < 8000].margin3.median():+.2f} "
          f"high-rate {g[g.peak >= 8000].margin3.median():+.2f}  diff={lo_hi:+.2f}")
# partial spearman: residualize emax on snr (linear), correlate with log peak
x = encL.snr.values; y = encL.emax.values
b = np.polyfit(x, y, 1); res_y = y - np.polyval(b, x)
from scipy.stats import spearmanr
rho, p = spearmanr(res_y, np.log(encL.peak.values))
print(f"  partial spearman(score residual vs log peak-rate | snr): rho={rho:+.3f} p={p:.1g}")
# whichA at CA window by rate stratum: does distortion pull the winning template to wrong family?
print("\n(d) veto regime (peak>16k) recall for line sources @FAR3:",
      end=" ")
g16 = encL[encL.peak >= 16000]
lo, hi = boot_recall(g16, THRS[3]) if len(g16) else (np.nan, np.nan)
print(f"{(g16.emax >= THRS[3]).mean()*100:.1f}% [{lo*100:.0f}-{hi*100:.0f}] (n={len(g16)}); "
      f"11-16k: {(encL[(encL.peak >= 11000) & (encL.peak < 16000)].emax >= THRS[3]).mean()*100:.1f}% "
      f"(n={len(encL[(encL.peak >= 11000) & (encL.peak < 16000)])})")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for i, tg in enumerate((3, 10)):
    xs = [str(s) for s in enc.stratum.cat.categories]
    recs = [df[(df.target == tg) & (df.stratum.astype(str) == x)].recall.iloc[0] * 100 for x in xs]
    los = [df[(df.target == tg) & (df.stratum.astype(str) == x)].lo.iloc[0] * 100 for x in xs]
    his = [df[(df.target == tg) & (df.stratum.astype(str) == x)].hi.iloc[0] * 100 for x in xs]
    ns = [df[(df.target == tg) & (df.stratum.astype(str) == x)].n.iloc[0] for x in xs]
    ax[i].bar(range(len(xs)), recs, yerr=[np.array(recs) - los, np.array(his) - np.array(recs)],
              capsize=3, color="steelblue")
    ax[i].set_xticks(range(len(xs))); ax[i].set_xticklabels([f"{x}\nn={n}" for x, n in zip(xs, ns)], fontsize=8)
    ax[i].set_ylabel(f"recall %"); ax[i].set_title(f"all sources @FAR{tg} vs CA peak rate")
    ax[i].grid(alpha=.3)
fig.tight_layout(); fig.savefig("_scratch/task3_pileup_tp.png", dpi=110)
print("plot -> _scratch/task3_pileup_tp.png")
enc.to_pickle("_scratch/task3_enc.pkl")
