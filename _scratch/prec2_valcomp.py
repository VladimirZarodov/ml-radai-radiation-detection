"""Precision pass, item 2: is the persistent val<test recall gap composition, not overfitting?
Group split (bg-confounded vs shape-distinguishable) for VAL runs 20-24 vs TEST runs 25-124;
recall per group at the SAME protocol-A thresholds; decomposition of the val-test gap into
composition term + within-group term."""
import pickle
import numpy as np
import pandas as pd
import h5py

cache = pickle.load(open("_scratch/cache10.pkl", "rb"))
names = [str(n).split("_shielding")[0] for n in h5py.File("training_v4.3.h5", "r").attrs["source_names"]]
TRAIN = [0] + list(range(3, 20)); VAL = [20, 21, 22, 23, 24]; TEST = list(range(25, 125))
ALLR = TRAIN + VAL + TEST
S = {r: pd.Series(c["bestA"]).rolling(31, min_periods=1).max().to_numpy(np.float32).astype(np.float64)
     for r, c in cache.items()}
EMAX = {r: np.array([S[r][np.abs(cache[r]["wtime"] - t) < 60].max()
                     for t in cache[r]["stime"] if (np.abs(cache[r]["wtime"] - t) < 60).any()])
        for r in ALLR}
thrA = {1: 5.19, 3: 4.49, 10: 3.61}   # protocol-A fixed thresholds (task1_pool.out)

def enc_table(runs):
    rows = []
    for r in runs:
        c = cache[r]
        for j in range(len(c["snr"])):
            m = np.abs(c["wtime"] - c["stime"][j]) < 60
            if not m.any(): continue
            rows.append(dict(run=r, name=names[int(c["sid"][j])], snr=float(c["snr"][j]),
                             emax=float(S[r][m].max())))
    return pd.DataFrame(rows)

V = enc_table(VAL); T = enc_table(TEST)

grp = lambda s: np.where(s.str.contains("U|Th|Ra|K-40|Cs|Pu|Sr", regex=True), "bg-confounded", "shape-dist.")
V["grp"] = grp(V.name); T["grp"] = grp(T.name)
print(f"encounters: VAL n={len(V)}  TEST n={len(T)}")
fv = V.grp.value_counts() / len(V); ft = T.grp.value_counts() / len(T)
print("\nДоля группы encounters:")
for g in ("bg-confounded", "shape-dist."):
    print(f"  {g:15s} VAL {fv.get(g,0)*100:5.1f}%  ({ (V.grp==g).sum() })   TEST {ft.get(g,0)*100:5.1f}%  ({ (T.grp==g).sum() })")

print("\nrecall по группам при тех же thr (протокол A):")
for tg, thr in thrA.items():
    rv = V.emax >= thr; rt = T.emax >= thr
    print(f"  @FAR{tg:>2} thr={thr:5.2f}:  VAL all {rv.mean()*100:5.1f}%  TEST all {rt.mean()*100:5.1f}%"
          f"  | bg-conf: VAL {rv[V.grp=='bg-confounded'].mean()*100:5.1f} TEST {rt[T.grp=='bg-confounded'].mean()*100:5.1f}"
          f"  | shape: VAL {rv[V.grp=='shape-dist.'].mean()*100:5.1f} TEST {rt[T.grp=='shape-dist.'].mean()*100:5.1f}")
    comp = sum(((V.grp == g).mean() - (T.grp == g).mean()) * rt[T.grp == g].mean() for g in ("bg-confounded", "shape-dist."))
    withg = sum((V.grp == g).mean() * (rv[V.grp == g].mean() - rt[T.grp == g].mean()) for g in ("bg-confounded", "shape-dist."))
    print(f"        decomposition: gap {rv.mean()*100 - rt.mean()*100:+5.1f} pt = composition {comp*100:+5.1f} + within-group {withg*100:+5.1f}")
    mv = V[V.grp=='bg-confounded'].groupby('name').size(); mt = T[T.grp=='shape-dist.'].groupby('name').size()
    print(f"        val snr: bg-conf med {V[V.grp=='bg-confounded'].snr.median():.1f} vs test {T[T.grp=='bg-confounded'].snr.median():.1f};"
          f" shape VAL {V[V.grp=='shape-dist.'].snr.median():.1f} vs TEST {T[T.grp=='shape-dist.'].snr.median():.1f}")
