# -*- coding: utf-8 -*-
"""Task 2 final PD50 protocol: logistic PD50 @FAR=1 per paper style, plus FAR3/FAR10
context, with convergence guards and empirical binned-PD fallback."""
import pickle
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
O = open('_scratch/t2_pd50.out', 'w', encoding='utf-8'); P = lambda *a: print(*a, file=O)

D = pd.read_pickle('_scratch/t2_enc.pkl')   # has run, split, base, snr_peak, snr_dir, emax, thr(FAR1)
cache = pickle.load(open('_scratch/cache10.pkl', 'rb'))
import h5py
f = h5py.File('training_v4.3.h5', 'r')
names = [str(n).split('_shielding')[0] for n in f.attrs['source_names']]
# rebuild B_loo thr@FAR3/FAR10 for Co-60/Am-241 encounters: reuse t1_thr.pkl (CAND, thrL for targets 1,3,10 on TE runs)
t1 = pickle.load(open('_scratch/t1_thr.pkl', 'rb'))
CAND, thrL = t1['CAND'], t1['thrL']
TE = list(range(25, 125))
# add columns thr3/thr10 by run (test only; val/train rows NaN)
for t, col in [(3, 'thr3'), (10, 'thr10')]:
    m = {r: CAND[thrL[t][i]] for i, r in enumerate(TE)}
    D[col] = D.run.map(m)

def fit_pd50(sub, xcol, thr):
    s = sub.dropna(subset=[xcol, thr])
    x = np.log(np.clip(s[xcol].values.astype(float), 1e-2, None))
    y = (s.emax.values >= s[thr].values).astype(int)
    if len(y) < 6 or y.min() == y.max(): return None
    for C in (1e6, 1.0, 0.3):
        try:
            lr = LogisticRegression(C=C, max_iter=20000).fit(x.reshape(-1, 1), y)
        except Exception:
            continue
        if not np.isfinite(lr.intercept_[0]) or lr.coef_[0][0] <= 0: continue
        est = float(np.exp(-lr.intercept_[0] / lr.coef_[0][0]))
        if est < 100:
            rr = np.random.default_rng(11); boots = []
            ri, _ = pd.factorize(s.run.values)
            for _ in range(500):
                pickr = rr.integers(0, ri.max()+1, ri.max()+1)
                idx = np.concatenate([np.flatnonzero(ri == q) for q in pickr])
                s2 = s.iloc[idx]
                x2 = np.log(np.clip(s2[xcol].values.astype(float), 1e-2, None))
                y2 = (s2.emax.values >= s2[thr].values).astype(int)
                if y2.min() == y2.max(): continue
                l2 = LogisticRegression(C=C, max_iter=20000).fit(x2.reshape(-1, 1), y2)
                if l2.coef_[0][0] > 0:
                    e2 = np.exp(-l2.intercept_[0]/l2.coef_[0][0])
                    if e2 < 100: boots.append(e2)
            lo, hi = (np.percentile(boots, [2.5, 97.5]) if len(boots) > 25 else (np.nan, np.nan))
            return dict(C=C, pd50=est, lo=lo, hi=hi, n=len(y), hit=y.mean())
    return dict(C=None, pd50=np.nan, lo=np.nan, hi=np.nan, n=len(y), hit=y.mean())

P('=== PD50 fits ===')
for b in ('Co-60', 'Am-241'):
    for split in ('test', 'all'):
        for xcol in ('snr_peak', 'snr_dir'):
            sub = D[(D.base == b) & ((D.split == split) if split != 'all' else True)]
            for thr, tl in (('thr', 1), ('thr3', 3), ('thr10', 10)):
                r_ = fit_pd50(sub, xcol, thr)
                if r_ is None:
                    P(f'{b:7s} {split:5s} {xcol:9s} FAR{tl:>2}: n={len(sub):3d} degenerate (all hits/misses)')
                elif r_['C'] is None:
                    P(f'{b:7s} {split:5s} {xcol:9s} FAR{tl:>2}: n={r_["n"]:3d} hit={r_["hit"]*100:5.1f}%  PD50 NOT CONVERGED (no monotone logistic fit)')
                else:
                    P(f'{b:7s} {split:5s} {xcol:9s} FAR{tl:>2}: n={r_["n"]:3d} hit={r_["hit"]*100:5.1f}%  PD50={r_["pd50"]:6.2f} '
                      f'[{r_["lo"]:.2f},{r_["hi"]:.2f}] (C={r_["C"]})')

P('\n=== empirical binned PD (snr_peak, test, FAR1) ===')
for b in ('Co-60', 'Am-241'):
    s = D[(D.base == b) & (D.split == 'test')]
    hits = (s.emax.values >= s.thr.values).astype(int)
    P(f'{b}: by snr_peak quartile bins:')
    qs = np.quantile(s.snr_peak, [0, .33, .66, 1.0])
    for k in range(3):
        m = (s.snr_peak >= qs[k]) & (s.snr_peak <= qs[k+1]) if k == 2 else (s.snr_peak >= qs[k]) & (s.snr_peak < qs[k+1])
        P(f'  snr {qs[k]:5.2f}-{qs[k+1]:5.2f}: n={m.sum():2d} PD={hits[m.values].mean()*100 if m.sum() else np.nan:5.1f}%')
    # same on FAR10
    hits10 = (s.emax.values >= s.thr10.fillna(-1).values).astype(int)
    P(f'  (FAR10 overall PD={hits10.mean()*100:.1f}%)')
O.close(); print('done')
