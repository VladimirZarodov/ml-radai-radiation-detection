# -*- coding: utf-8 -*-
"""Task 2: Fig.6-style PD50 (SNR at 50% detection, FAR=1/hr) for Co-60 and Am-241.
(1) dataset sources/snr/peak, (2) direct s/sqrt(s+b) from listmode id/background_id tags."""
import io, re, pickle
import numpy as np, pandas as pd, h5py
from sklearn.linear_model import LogisticRegression

O = io.open('_scratch/t2_fig6.out', 'w', encoding='utf-8'); P = lambda *a: print(*a, file=O)
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
# B_loo thr @ FAR=1 for every run in ALL (train runs usable: detector is unsupervised)
thr1 = {}
for r in ALL:
    others = np.array([IH[q] for q in ALL if q != r])
    fr = OM[others].sum(0)/HV[others].sum()
    thr1[r] = CAND[pick(fr, branch_end(fr, 1), 1)]
P('thr@FAR1 medians: test %.3f  val %.3f  train %.3f' % (
    np.median([thr1[r] for r in TE]), np.median([thr1[r] for r in VA]), np.median([thr1[r] for r in TR])))

def baseof(nm):
    m = re.match(r'^(.*)-(\d+(?:\.\d+)?kg)$', nm)
    return m.group(1) if m else nm

# ---- encounters with dataset peak snr + emax, for ALL runs
rows = []
for r in ALL:
    c = cache[r]; k = -1
    for j in range(len(c['snr'])):
        m = np.abs(c['wtime']-c['stime'][j]) < 60
        if not m.any(): continue
        k += 1
        nm = names[int(c['sid'][j])]
        rows.append(dict(run=r, split=('test' if r in TE else 'val' if r in VA else 'train'),
                         base=baseof(nm), snr_peak=float(c['snr'][j]),
                         emax=float(EMAX[r][k]), thr=thr1[r]))
E = pd.DataFrame(rows)
E['hit'] = E.emax >= E.thr
P('\nencounter counts (ALL runs vs test only):')
for b in ('Co-60', 'Am-241'):
    g = E[E.base == b]
    P(f'  {b:7s}: all n={len(g):3d} ({len(g.run.unique())} runs)  test n={int((g.split=="test").sum()):3d}')

# ---- direct s/sqrt(s+b) from listmode for the runs that host Co-60 / Am-241 encounters
def direct_one_run(r, want_slots):
    """want_slots: dict sid -> CA time (s). Returns dict sid -> (snr60, s60, b60, snrfull, dur)."""
    g = f[f'runs/run{r}']
    lm = g['listmode']
    dt = lm['dt'][:].astype(np.int64)
    eid = lm['id'][:].astype(np.int32)
    ebg = lm['background_id'][:]
    tt = np.cumsum(dt, dtype=np.float64)*1e-6
    del dt
    out = {}
    for sid, tca in want_slots.items():
        w = (tt >= tca-60) & (tt <= tca+60)
        s_ = int(np.count_nonzero(eid[w] == sid)); b_ = int(np.count_nonzero(ebg[w] != 0))
        snr60 = s_/np.sqrt(s_+b_) if s_ > 0 else np.nan
        pres = np.flatnonzero(eid == sid)
        snrf, dur = np.nan, np.nan
        if len(pres):
            t0, t1 = tt[pres[0]], tt[pres[-1]]
            wf = (tt >= t0) & (tt <= t1)
            sf = int(np.count_nonzero(eid[wf] == sid)); bf = int(np.count_nonzero(ebg[wf] != 0))
            snrf = sf/np.sqrt(sf+bf) if sf else np.nan
            dur = float(t1-t0)
        out[sid] = (snr60, s_, b_, snrf, dur)
    return out

P('\n=== sources/id -> source_names check on run25 ===')
g25 = f['runs/run25']
sidv = g25['sources/id'][:].astype(int)
P('sources/id: ' + str(sidv) + ' -> names: ' + ', '.join(names[s] for s in sidv))

DIRS = {}
tgt = {}
for r in ALL:
    c = cache[r]
    for j in range(len(c['snr'])):
        sid = int(c['sid'][j])
        if baseof(names[sid]) in ('Co-60', 'Am-241'):
            st = float(c['stime'][j])
            if st > 1e5: st /= 1000.0        # ms -> s heuristic
            tgt.setdefault(r, {})[sid] = st
P(f'runs with Co-60/Am-241 encounters: {len(tgt)}')
for r, slots in tgt.items():
    DIRS.update({(r, sid): v for sid, v in direct_one_run(r, slots).items()})
P(f'direct-SNR slots computed: {len(DIRS)}')
for (r, sid), v in list(DIRS.items())[:5]:
    P(f'  run{r} slot {sid} ({names[sid]}): snr60={v[0]} s={v[1]} b={v[2]} snrfull={v[3]} dur={v[4]}')

# attach direct snr to encounters by (run, sid)
rows2 = []
for r in ALL:
    c = cache[r]; k = -1
    for j in range(len(c['snr'])):
        m = np.abs(c['wtime']-c['stime'][j]) < 60
        if not m.any(): continue
        k += 1
        sid = int(c['sid'][j])
        if baseof(names[sid]) not in ('Co-60', 'Am-241'): continue
        v = DIRS.get((r, sid))
        rows2.append(dict(run=r, split=('test' if r in TE else 'val' if r in VA else 'train'),
                          base=baseof(names[sid]), snr_peak=float(c['snr'][j]),
                          snr_dir=(v[0] if v else np.nan), snr_dir_full=(v[3] if v and len(v) > 3 else np.nan),
                          s=(v[1] if v else 0), b=(v[2] if v else 0),
                          emax=float(EMAX[r][k]), thr=thr1[r]))
D = pd.DataFrame(rows2)
D['hit'] = D.emax >= D.thr
D = D.dropna(subset=['snr_dir'])
P('\njoined direct-SNR encounters: ' + str(D.groupby(['base', 'split']).size()))
P('corr peak vs direct: ' + str({b: (lambda g: round(float(np.corrcoef(np.log(np.clip(g.snr_peak.values.astype(float),1e-3,None)), np.log(np.clip(g.snr_dir.values.astype(float),1e-3,None)))[0, 1]), 3))(g) for b, g in D.groupby('base')}))
P('median peak / direct ratio: ' + str({b: round(float(np.median(g.snr_peak/g.snr_dir)), 3) for b, g in D.groupby('base')}))

def pd50(sub, col):
    x = np.log(np.clip(sub[col].values.astype(float), 1e-2, None)); y = sub.hit.values.astype(int)
    if len(y) < 6 or y.min() == y.max(): return np.nan, np.nan, np.nan
    lr = LogisticRegression(C=1e6).fit(x.reshape(-1, 1), y)
    est = float(np.exp(-lr.intercept_[0]/lr.coef_[0][0]))
    rr = np.random.default_rng(7); boots = []
    ri, _ = pd.factorize(sub.run.values)
    for _ in range(400):
        pickr = rr.integers(0, ri.max()+1, ri.max()+1)
        idx = np.concatenate([np.flatnonzero(ri == q) for q in pickr])
        s2 = sub.iloc[idx]
        x2 = np.log(np.clip(s2[col].values.astype(float), 1e-2, None)); y2 = s2.hit.values.astype(int)
        if y2.min() == y2.max(): continue
        try:
            l2 = LogisticRegression(C=1e6).fit(x2.reshape(-1, 1), y2)
            boots.append(np.exp(-l2.intercept_[0]/l2.coef_[0][0]))
        except Exception: pass
    lo, hi = np.percentile(boots, [2.5, 97.5]) if len(boots) > 20 else (np.nan, np.nan)
    return est, lo, hi

P('\n=== PD50 (SNR at 50% Pdet, B_loo thr @ FAR=1/hr) ===')
P(f"{'iso':8s} {'split':6s} {'metric':10s} {'n':>3} {'PD50':>7} [95% CI]")
for b in ('Co-60', 'Am-241'):
    for split in ('test', 'all'):
        for col in ('snr_peak', 'snr_dir'):
            sub = D[(D.base == b) & ((D.split == split) if split != 'all' else True)]
            e, lo, hi = pd50(sub, col)
            P(f'{b:8s} {split:6s} {col:10s} {len(sub):3d} {e:7.2f} [{lo:.2f},{hi:.2f}]' if e == e else
              f'{b:8s} {split:6s} {col:10s} {len(sub):3d}  -- too few / degenerate')
D.to_pickle('_scratch/t2_enc.pkl')
O.close(); print('done')
