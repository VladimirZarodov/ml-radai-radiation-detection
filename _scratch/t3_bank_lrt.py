# -*- coding: utf-8 -*-
"""Task 3: two Adaptive-NMF-paper ideas as alternative detector statistics, tested with the
EXACT t1_recon protocol (same CAND grid, branch/pick, B_loo per-run thresholds, +-60 s encounters,
run-cluster bootstrap CIs). Arms:
  3b bank : matched-filter bank over sliding windows T in {2,4,8} s; per-scale rolling max to the
            same ~62 s footprint, then max across scales on the 2 s grid.
  3a lrt  : joint Poisson-EM fit (7 bg components + one isotope template) per window for the top-3
            templates by cached zA; stat = max_t 2*(lnL_{B+t} - lnL_B), floored at 0.
Control = frozen mx31 (rolling max of cached bestA, 31x2 s)."""
import io, time, pickle, re
import numpy as np, pandas as pd, h5py

O = io.open('_scratch/t3.out', 'w', encoding='utf-8')
def P(*a):
    print(*a, file=O); O.flush()

EMIN, EMAXE, NB, STRIDE = 15.0, 3000.0, 128, 2.0
smin, smax = np.sqrt(EMIN), np.sqrt(EMAXE)
f = h5py.File('training_v4.3.h5', 'r')
names = [str(n).split('_shielding')[0] for n in f.attrs['source_names']]
TR = [0] + list(range(3, 20)); VA = [20, 21, 22, 23, 24]; TE = list(range(25, 125)); ALL = TR + VA + TE
cache = pickle.load(open('_scratch/cache10.pkl', 'rb'))

def ev_arrays(gg):
    dt = gg['listmode/dt'][:]; t = np.cumsum(dt, dtype=np.uint64) / 1e6
    e = gg['listmode/energy'][:]; eid = gg['listmode/id'][:]; bid = gg['listmode/background_id'][:]
    inr = (e >= EMIN) & (e < EMAXE)
    bi = np.zeros(len(e), np.int64)
    bi[inr] = np.clip(np.floor((np.sqrt(e[inr]) - smin) / (smax - smin) * NB).astype(np.int64), 0, NB - 1)
    return t, e, eid, bid, inr, bi

def poisson_A(X, iters=60):
    A = np.full((X.shape[0], M.shape[0]), X.sum(1, keepdims=True) / M.shape[0])
    norm = M.sum(1)[None, :]
    for _ in range(iters):
        B = A @ M + 1e-9
        A *= ((X / B) @ M.T) / norm
    return A

# ---- M, U exactly as notebook S3.1/S3.2 (train runs only) ----
t0 = time.time()
comp_hist = np.zeros((8, NB)); src_hist = {}
for rid in TR:
    t, e, eid, bid, inr, bi = ev_arrays(f[f'runs/run{rid}'])
    for k in range(1, 8):
        comp_hist[k] += np.bincount(bi[inr & (bid == k) & (eid == 0)], minlength=NB)
    sm = inr & (eid != 0)
    for sidx in np.unique(eid[sm]):
        sidx = int(sidx)
        src_hist[sidx] = src_hist.get(sidx, np.zeros(NB)) + np.bincount(bi[sm & (eid == sidx)], minlength=NB)
keep = comp_hist.sum(1) > 100
M = comp_hist[keep] / comp_hist[keep].sum(1, keepdims=True)
ids = np.array(sorted(src_hist))
U = np.array([src_hist[i] for i in ids]); U = U / U.sum(1, keepdims=True); U2 = U ** 2
NT = U.shape[0]
P(f'M/U rebuilt: bg components {M.shape[0]}, templates {NT}  ({time.time()-t0:.0f}s)')

# sanity: rebuilt run25 zA must match cache
g25 = f['runs/run25']; t, e, eid, bid, inr, bi = ev_arrays(g25)
nfull = int(np.floor(t[-1] / STRIDE)); wi = np.floor(t / STRIDE).astype(np.int64)
okw = (wi < nfull) & inr
sp = np.bincount(wi[okw] * NB + bi[okw], minlength=nfull * NB).reshape(nfull, NB).astype(np.float64)
i2 = np.arange(nfull); X2 = sp[i2]  # T=2 sliding window == the 2 s slot itself
S2chk = np.maximum(poisson_A(X2) @ M, 1.0)
zchk = ((X2 - S2chk) @ U.T) / np.sqrt(S2chk @ U2.T + 1e-9)
zc = cache[25]['zA'].astype(np.float64); L = min(len(zchk), zc.shape[0])
P(f'control check run25: max|dz|={np.abs(zchk[:L]-zc[:L]).max():.3e}')
assert np.abs(zchk[:L] - zc[:L]).max() < 1e-3, 'template/bg reconstruction mismatch'

# ---- per-run stats ----
bank_all = {}; lrt_all = {}
import os
if os.path.exists('_scratch/t3_stats.pkl'):
    _d = pickle.load(open('_scratch/t3_stats.pkl', 'rb')); bank_all = _d['bank']; lrt_all = _d['lrt']
    P(f't3_stats.pkl loaded (skip rebuild), runs={len(bank_all)}')
for n_, rid in enumerate([] if bank_all else ALL):
    gg = f[f'runs/run{rid}']
    t, e, eid, bid, inr, bi = ev_arrays(gg)
    nfull = int(np.floor(t[-1] / STRIDE)); wi = np.floor(t / STRIDE).astype(np.int64)
    okw = (wi < nfull) & inr
    sp = np.bincount(wi[okw] * NB + bi[okw], minlength=nfull * NB).reshape(nfull, NB).astype(np.float64)
    cum = np.vstack([np.zeros((1, NB)), np.cumsum(sp, 0)])
    ns2 = nfull; idx = np.arange(ns2)   # T=2 sliding windows coincide with 2 s slots: length = nfull, matches cache10
    X2 = cum[idx + 1] - cum[idx]
    # 3b bank
    parts = [pd.Series(cache[rid]['bestA'].astype(np.float64)).rolling(31, min_periods=1).max().to_numpy()[:ns2]]
    for T in (4.0, 8.0):
        k = int(round(T / STRIDE)); ns = max(nfull - k + 1, 1); jdx = np.arange(ns)
        XT = cum[jdx + k] - cum[jdx]
        ST = np.maximum(poisson_A(XT) @ M, 1.0)
        bT = (((XT - ST) @ U.T) / np.sqrt(ST @ U2.T + 1e-9)).max(1)
        W = int(round((62 - T) / STRIDE)) + 1
        mxT = pd.Series(bT).rolling(W, min_periods=1).max().to_numpy()
        mfull = np.full(ns2, -np.inf); mfull[:len(mxT)] = mxT
        parts.append(pd.Series(mfull).rolling(k, min_periods=1).max().to_numpy())
    bank_all[rid] = np.maximum(np.maximum(parts[0], parts[1]), parts[2]).astype(np.float32)
    # 3a lrt
    A2 = poisson_A(X2); S2 = np.maximum(A2 @ M, 1.0)
    lnL_B = np.sum(X2 * np.log(S2) - S2, 1)
    zA = ((X2 - S2) @ U.T) / np.sqrt(S2 @ U2.T + 1e-9)
    top3 = np.argsort(-zA, axis=1)[:, :3]
    LRT = np.zeros(ns2); normM = M.sum(1)
    for ti in range(NT):
        rows = np.flatnonzero((top3 == ti).any(1))
        if len(rows) == 0: continue
        Xt = X2[rows]; u = U[ti]; u2 = U2[ti]
        A = A2[rows].copy()
        c = np.maximum(((Xt - S2[rows]) @ u) / (1e-9 + u @ u), 0.5)
        for _ in range(80):
            B = A @ M + c[:, None] * u + 1e-9
            r = Xt / B
            A *= (r @ M.T) / normM
            c = np.maximum(c * (r @ u), 0.0)
        B = np.maximum(A @ M + c[:, None] * u, 1.0)
        lam = 2.0 * (np.sum(Xt * np.log(B) - B, 1) - lnL_B[rows])
        LRT[rows] = np.maximum(LRT[rows], np.clip(lam, 0.0, None))
    lrt_all[rid] = LRT.astype(np.float32)
    if n_ % 10 == 0: P(f'  run {rid} ({n_+1}/{len(ALL)})  {time.time()-t0:.0f}s')
P(f'stats built in {time.time()-t0:.0f}s')
pickle.dump(dict(bank=bank_all, lrt=lrt_all), open('_scratch/t3_stats.pkl', 'wb'))

# ---- calibration + evaluation ----
STATS = {'mx31': {r: pd.Series(cache[r]['bestA'].astype(np.float64)).rolling(31, min_periods=1).max().to_numpy() for r in ALL},
         'bank': {r: bank_all[r].astype(np.float64) for r in ALL},
         'lrt': {r: lrt_all[r].astype(np.float64) for r in ALL}}
FARm = {r: cache[r]['dmin'] > 150 for r in ALL}
IH = {r: i for i, r in enumerate(ALL)}
HRS = np.array([FARm[r].sum() * STRIDE / 3600 for r in ALL])

NORM = {'K-40', 'Th-232', 'Ra-226'}; UFAM = {'DU', 'LEU', 'NatU', 'RefinedU'}
def baseof(nm):
    m = re.match(r'^(.*)-(\d+(?:\.\d+)?kg)$', nm); return m.group(1) if m else nm
def group(nm):
    b = baseof(nm)
    return 'NORM' if b in NORM else ('U-family' if b in UFAM else 'other')

# encounters (valid = at least one +-60 s window), ordered per run exactly as EMAX arrays
ENC = []
for r in ALL:
    c = cache[r]; wt = c['wtime']
    for j in range(len(c['snr'])):
        m = np.abs(wt - c['stime'][j]) < 60
        if not m.any(): continue
        ENC.append(dict(run=r, grp=group(names[int(c['sid'][j])]), base=baseof(names[int(c['sid'][j])]),
                        snr=float(c['snr'][j]), widx=m))
ENC = pd.DataFrame(ENC)
ENCt = ENC[ENC.run.isin(TE)].reset_index(drop=True)

dq = np.logspace(np.log10(2e-4), np.log10(30), 320)
def branch_end(fr, target):
    runmax, br = fr[0], 0
    for k in range(1, len(fr)):
        if fr[k] > runmax: runmax, br = fr[k], k
        elif fr[k] < runmax - max(0.5, 0.2 * runmax) and runmax > target: break
    return br
def pick(fr, br, target):
    d = np.where(fr[:br + 1] <= target, target - fr[:br + 1], np.inf)
    return int(np.argmin(d)) if np.isfinite(d).any() else 0
def bootrec(hit, runs, B=600, seed=5):
    ri, _ = pd.factorize(runs)
    hs = np.bincount(ri, weights=hit, minlength=ri.max() + 1); ns = np.bincount(ri, minlength=ri.max() + 1)
    rr = np.random.default_rng(seed); recs = []
    for _ in range(B):
        cnt = np.bincount(rr.integers(0, len(ns), len(ns)), minlength=len(ns)).astype(float)
        if (cnt * ns).sum() == 0: continue
        recs.append((cnt * hs).sum() / (cnt * ns).sum())
    return np.percentile(recs, [2.5, 97.5])

P('\n=== task-3 (test 25-124, B_loo, +-60 s, run-cluster boot CI) ===')
te_rows = ENC[ENC.run.isin(TE)].reset_index(drop=True)
res_rows = []
for stat in ('mx31', 'bank', 'lrt'):
    S = STATS[stat]
    poolv = np.concatenate([S[r][FARm[r]] for r in ALL])
    CAND = np.array(sorted(set(np.quantile(poolv, 1 - dq / 100)), reverse=True)); NC = len(CAND)
    OM = np.zeros((len(ALL), NC), np.int64)
    for r in ALL:
        s, m = S[r], FARm[r]
        for j, t in enumerate(CAND):
            a = s >= t
            OM[IH[r], j] = np.count_nonzero((a & ~np.r_[False, a[:-1]])[m])
    # encounter emax (aligned to te_rows index; groupby keeps within-run order)
    em = []
    for r, sub in te_rows.groupby('run', sort=False):
        em.append(pd.Series([S[r][np.asarray(widx)].max() for widx in sub.widx], index=sub.index))
    te_rows['emax'] = pd.concat(em).reindex(te_rows.index).values
    P(f'\n--- stat = {stat} ---')
    for target in (1, 3, 10):
        jj = []
        for r in TE:
            others = np.array([IH[q] for q in ALL if q != r])
            fr = OM[others].sum(0) / HRS[others].sum()
            jj.append(pick(fr, branch_end(fr, target), target))
        thrmap = {r: CAND[jj[i]] for i, r in enumerate(TE)}
        hit = (te_rows.emax.values >= te_rows.run.map(thrmap).values).astype(float)
        runs_vec = te_rows.run.values
        ov = hit.mean() * 100
        lo, hi = bootrec(hit, runs_vec)
        res_rows.append(dict(stat=stat, far=target, thr_med=float(np.median([thrmap[r] for r in TE])),
                             recall=ov, lo=lo * 100, hi=hi * 100))
        line = f'@FAR{target:2d}: thr med={np.median([thrmap[r] for r in TE]):6.3f}  overall={ov:5.1f}% [{lo*100:4.1f},{hi*100:4.1f}]'
        for gr in ('NORM', 'U-family', 'other'):
            m_ = (te_rows.grp.values == gr)
            l2, h2 = bootrec(hit[m_], runs_vec[m_])
            line += f' | {gr} {hit[m_].mean()*100:5.1f}%[{l2*100:4.1f},{h2*100:4.1f}]'
        P(line)
    te_rows['emax_' + stat] = te_rows.emax.values
R = pd.DataFrame(res_rows)
ctl = R[R.stat == 'mx31'].set_index('far').recall.round(1).to_dict()
P(f"\ncontrol mx31 overall: {ctl}  (ожидаем {{1.0: 25.2, 3.0: 40.5, 10.0: 70.0}})")
P('\n=== сводка: дельта к mx31 (overall) ===')
for target in (1, 3, 10):
    base = R[(R.stat == 'mx31') & (R.far == target)].iloc[0]
    for stat in ('bank', 'lrt'):
        row = R[(R.stat == stat) & (R.far == target)].iloc[0]
        P(f'  FAR{target:>2d} {stat:5s}: {row.recall:5.1f}% vs {base.recall:5.1f}%  Δ={row.recall-base.recall:+5.1f} п.п.')
te_rows.to_pickle('_scratch/t3_enc.pkl')
R.to_csv('_scratch/t3_summary.csv', index=False)
O.close()
print('done')
