# -*- coding: utf-8 -*-
"""Task 1d: val-vs-test composition test with CORRECTED groups (NORM / U-family / other).
Same B_loo protocol as t1_recon.py. Decompose overall val-test recall gap."""
import io, re, pickle
import numpy as np, pandas as pd, h5py

O = io.open('_scratch/t1b_valcomp.out', 'w', encoding='utf-8'); P = lambda *a: print(*a, file=O)
cache = pickle.load(open('_scratch/cache10.pkl', 'rb'))
f = h5py.File('training_v4.3.h5', 'r')
names = [str(n).split('_shielding')[0] for n in f.attrs['source_names']]
TR = [0]+list(range(3,20)); VA=[20,21,22,23,24]; TE=list(range(25,125)); ALL=TR+VA+TE
STRIDE = 2.0
S = {r: pd.Series(c['bestA']).rolling(31, min_periods=1).max().to_numpy(np.float32).astype(np.float64)
     for r, c in cache.items()}
FARm = {r: cache[r]['dmin'] > 150 for r in S}
IH = {r: i for i, r in enumerate(ALL)}
HRS = {r: FARm[r].sum()*STRIDE/3600 for r in ALL}
EMAX = {r: np.array([S[r][np.abs(cache[r]['wtime']-t) < 60].max()
                     for t in cache[r]['stime'] if (np.abs(cache[r]['wtime']-t) < 60).any()]) for r in ALL}

poolv = np.concatenate([S[r][FARm[r]] for r in ALL])
dq = np.logspace(np.log10(2e-4), np.log10(30), 320)
CAND = np.array(sorted(set(np.quantile(poolv, 1 - dq/100)), reverse=True)); NC = len(CAND)
OM = np.zeros((len(ALL), NC), np.int64)
for r in ALL:
    s, m = S[r], FARm[r]
    for j, t in enumerate(CAND):
        a = s >= t
        OM[IH[r], j] = np.count_nonzero((a & ~np.r_[False, a[:-1]])[m])
HV = np.array([HRS[r] for r in ALL])
def branch_end(fr, target):
    runmax, br = fr[0], 0
    for k in range(1, NC):
        if fr[k] > runmax: runmax, br = fr[k], k
        elif fr[k] < runmax - max(0.5, 0.2*runmax) and runmax > target: break
    return br
def pick(fr, br, target):
    d = np.where(fr[:br+1] <= target, target - fr[:br+1], np.inf)
    return int(np.argmin(d)) if np.isfinite(d).any() else 0
thrL = {}
for target in (1, 3, 10):
    jj = []
    for r in TE + VA:                     # B_loo for test AND val runs
        others = np.array([IH[q] for q in ALL if q != r])
        fr = OM[others].sum(0)/HV[others].sum()
        jj.append(pick(fr, branch_end(fr, target), target))
    thrL[target] = np.array(jj)            # order: TE then VA
IDX_TE = TE; IDX_VA = VA

def baseof(nm):
    m = re.match(r'^(.*)-(\d+(?:\.\d+)?kg)$', nm)
    return m.group(1) if m else nm
NORM = {'K-40', 'Th-232', 'Ra-226'}; UFAM = {'DU', 'LEU', 'NatU', 'RefinedU'}
def group(nm):
    b = baseof(nm)
    return 'NORM' if b in NORM else ('U-family' if b in UFAM else 'other')

def encdf(runs, offset):
    rows = []
    for oi, r in enumerate(runs):
        c = cache[r]; k = -1
        for j in range(len(c['snr'])):
            m = np.abs(c['wtime']-c['stime'][j]) < 60
            if not m.any(): continue
            k += 1
            rows.append(dict(run=r, name=names[int(c['sid'][j])], grp=group(names[int(c['sid'][j])]),
                             emax=float(EMAX[r][k]), thr_off=offset+oi))
    return pd.DataFrame(rows)
enc_te = encdf(TE, 0); enc_va = encdf(VA, len(TE))
enc = pd.concat([enc_te, enc_va], ignore_index=True)
for t in (1, 3, 10):
    enc[f'hit{t}'] = enc.emax.values >= CAND[thrL[t][enc.thr_off.values]]

P('=== group fractions val vs test ===')
for t in (1, 3, 10):
    vr = enc[enc.run.isin(VA)][f'hit{t}'].mean(); tr_ = enc[enc.run.isin(TE)][f'hit{t}'].mean()
    P(f'\n@FAR{t}: overall val={vr*100:.1f}%  test={tr_*100:.1f}%  gap(val-test)={(vr-tr_)*100:+.1f} pt')
    fv = enc[enc.run.isin(VA)].grp.value_counts(normalize=True)
    ft = enc[enc.run.isin(TE)].grp.value_counts(normalize=True)
    rv = enc[enc.run.isin(VA)].groupby('grp')[f'hit{t}'].mean()
    rt = enc[enc.run.isin(TE)].groupby('grp')[f'hit{t}'].mean()
    comps = {}
    for g in ('NORM', 'U-family', 'other'):
        P(f'  {g:9s} val n={int((enc.run.isin(VA)&(enc.grp==g)).sum()):3d} f={fv.get(g,0)*100:5.1f}% r={rv.get(g,np.nan)*100:5.1f}% | '
          f'test n={int((enc.run.isin(TE)&(enc.grp==g)).sum()):3d} f={ft.get(g,0)*100:5.1f}% r={rt.get(g,np.nan)*100:5.1f}%')
        comps[g] = ((fv.get(g,0)-ft.get(g,0))*rt.get(g,0) + fv.get(g,0)*(rv.get(g,0)-rt.get(g,0)))
    comp_term = sum((fv.get(g,0)-ft.get(g,0))*rt.get(g,0) for g in comps)
    with_term   = sum(fv.get(g,0)*(rv.get(g,0)-rt.get(g,0)) for g in comps)
    P(f'  decomposition: composition={comp_term*100:+.1f} pt  within-group={with_term*100:+.1f} pt  sum={(comp_term+with_term)*100:+.1f}')
P('\nval group cells n: ' + str(dict(enc[enc.run.isin(VA)].grp.value_counts())))
O.close(); print('done')
