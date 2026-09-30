# -*- coding: utf-8 -*-
"""Task 1 recon: per-base-isotope recall under corrected grouping, group memberships,
and what old regex mislabeled. Uses B_loo protocol thresholds identical to the notebook."""
import io, pickle
import numpy as np, pandas as pd, h5py

O = io.open('_scratch/t1_recon.out', 'w', encoding='utf-8'); P = lambda *a: print(*a, file=O)
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

# ---- replicate notebook CAND + B_loo thresholds (from published thrL medians, exact CAND via own grid)
poolv = np.concatenate([S[r][FARm[r]] for r in ALL])
dq = np.logspace(np.log10(2e-4), np.log10(30), 320)
CAND = np.array(sorted(set(np.quantile(poolv, 1 - dq/100)), reverse=True)); NC=len(CAND)
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
    for r in TE:
        others = np.array([IH[q] for q in ALL if q != r])
        fr = OM[others].sum(0)/HV[others].sum()
        jj.append(pick(fr, branch_end(fr, target), target))
    thrL[target] = np.array(jj)
P('B_loo median thr: ' + str({t: round(float(np.median(CAND[thrL[t]])), 3) for t in thrL}))
P('headline recall: ' + str({t: round(sum((EMAX[r] >= CAND[thrL[t][i]]).sum() for i, r in enumerate(TE))
                               / sum(len(EMAX[r]) for r in TE)*100, 1) for t in thrL}))

# ---- corrected grouping
NORM = {'K-40', 'Th-232', 'Ra-226'}
UFAM = {'DU', 'LEU', 'NatU', 'RefinedU'}
import re
BASES = {}
def baseof(nm):
    """strip only a '-<mass>kg' suffix (DU-25kg -> DU); keep isotope mass numbers (K-40, Th-232)."""
    m = re.match(r'^(.*)-(\d+(?:\.\d+)?kg)$', nm)
    return m.group(1) if m else nm
for nm in sorted(set(names)):
    BASES.setdefault(baseof(nm), set()).add(nm)
P('\n=== base-name map (regex strip of "-<digits>...") ===')
for b, s_ in sorted(BASES.items()): P(f'  {b:16s} <- {sorted(s_)}')

def group(nm):
    b = baseof(nm)
    if b in NORM: return 'NORM'
    if b in UFAM: return 'U-family'
    return 'other'
ALLG = set(group(n) for n in names)
P('\ngroups: ' + str(ALLG))
# assert each name -> exactly one group
for n in sorted(set(names)):
    assert group(n) in ('NORM','U-family','other')
P('names in NORM      : ' + ', '.join(sorted({baseof(n) for n in names if group(n)=='NORM'})))
P('names in U-family  : ' + ', '.join(sorted({baseof(n) for n in names if group(n)=='U-family'})))
P('names in other     : ' + ', '.join(sorted({baseof(n) for n in names if group(n)=='other'})))

# ---- what the OLD regex got wrong
OLD = re.compile('U|Th|Ra|K-40|Cs|Pu|Sr')
P('\n=== old-regex mislabels (base name: old grp | new grp) ===')
for b in sorted(BASES):
    new = group(b)
    sample = sorted(BASES[b])[0]
    og = 'bg-confounded' if OLD.search(sample) else 'shape-dist.'
    if (og=='bg-confounded') != (new in ('NORM','U-family')):
        P(f'  {sample:16s} old={og:14s} new={new}')

# ---- per-isotope table by BASE name (test runs), B_loo thresholds, run-cluster boot CI
rows=[]
for i, r in enumerate(TE):
    c = cache[r]
    for j in range(len(c['snr'])):
        m = np.abs(c['wtime']-c['stime'][j]) < 60
        if not m.any(): continue
        rows.append(dict(run=r, name=names[int(c['sid'][j])], base=baseof(names[int(c['sid'][j])]),
                         grp=group(names[int(c['sid'][j])]), snr=float(c['snr'][j]),
                         emax=float(EMAX[r][sum(1 for k in range(j+1)
                                if (np.abs(c['wtime']-c['stime'][k])<60).any())-1])))
enc = pd.DataFrame(rows)
def bootrec(sub, thrmap, B=600, seed=5):
    """run-cluster bootstrap CI of recall with per-run B_loo thresholds."""
    thr_e = sub.run.map(thrmap).values
    hit = (sub.emax.values >= thr_e).astype(float)
    ri,_=pd.factorize(sub.run.values)
    hs=np.bincount(ri,weights=hit,minlength=ri.max()+1); ns=np.bincount(ri,minlength=ri.max()+1)
    rr=np.random.default_rng(seed); recs=[]
    for _ in range(B):
        cnt=np.bincount(rr.integers(0,len(ns),len(ns)),minlength=len(ns)).astype(float)
        if (cnt*ns).sum()==0: continue
        recs.append((cnt*hs).sum()/(cnt*ns).sum())
    return np.percentile(recs,[2.5,97.5])
THR = {t: {r: CAND[thrL[t][i]] for i, r in enumerate(TE)} for t in (1,3,10)}
P('\n=== per-BASE-isotope recall (test, B_loo thr, run-cluster boot 95% CI) ===')
P(f"{'base':16s} {'n':>4} {'med_snr':>7} " + ' '.join(f'@FAR{t}  [lo-hi]' for t in (1,3,10)))
for b, g in enc.groupby('base'):
    line=f'{b:16s} {len(g):4d} {g.snr.median():7.1f} '
    for t in (1,3,10):
        lo,hi=bootrec(g,THR[t])
        hr=(g.emax.values>=g.run.map(THR[t]).values).mean()*100
        line += f'{hr:5.1f}[{lo*100:4.1f},{hi*100:4.1f}] '
    P(line)
P('\n=== per-GROUP recall (test) ===')
for gr, g in enc.groupby('grp'):
    line=f'{gr:10s} n={len(g):4d} med_snr={g.snr.median():5.1f} '
    for t in (1,3,10):
        lo,hi=bootrec(g,THR[t])
        hr=(g.emax.values>=g.run.map(THR[t]).values).mean()*100
        line += f'| @FAR{t}: {hr:5.1f}% [{lo*100:4.1f},{hi*100:4.1f}] '
    P(line)
enc.to_pickle('_scratch/t1_enc.pkl')
pickle.dump(dict(CAND=CAND,thrL=thrL,EMAX=EMAX), open('_scratch/t1_thr.pkl','wb'))
O.close()
print('done')
