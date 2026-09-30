# -*- coding: utf-8 -*-
"""TASK 1: Tikhonov/L2-regularized joint-fit LRT, lambda sweep, frozen protocol.
Same top-3 joint Poisson-EM fit as the raw LRT (_scratch/t3_bank_lrt.py) but the template
coefficient a_t=c enters with penalty lambda*c^2 in the NLL (Adaptive-NMF paper, Sec. III-B).
EM update for c:  c <- c * (r@u) / (sum(u) + 2*lambda*c), sum(u)=1.
Statistic: Lambda_pen = 2*((LL_joint - lambda*c^2) - LL_B), floored at 0.
Grid: lambda=0 (must reproduce raw LRT) + logspace(-5,-1,9).
Evaluation identical to t1_recon/t3: CAND log-quantile grid, monotone branch/pick, B_loo
per-run thresholds, +-60 s encounters, run-cluster bootstrap CIs, corrected group map."""
import io, os, time, pickle, re
import numpy as np, pandas as pd, h5py

O = io.open('_scratch/t4.out', 'w', encoding='utf-8')
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

# ---- M/U from train (validated bit-exact vs cache10 builder in t3) ----
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
P(f'M/U rebuilt: {M.shape[0]} bg, {NT} templates ({time.time()-t0:.0f}s)')

LAMS = [0.0] + list(np.logspace(-5, -1, 9))
SPK = '_scratch/t4_l2lrt.pkl'
if os.path.exists(SPK):
    ser = pickle.load(open(SPK, 'rb'))
    P(f't4_l2lrt.pkl loaded, lam={len(ser)} runs x {len(next(iter(ser.values())))} x lam')
else:
    ser = {}
    for n_, rid in enumerate(ALL):
        gg = f[f'runs/run{rid}']
        t, e, eid, bid, inr, bi = ev_arrays(gg)
        nfull = int(np.floor(t[-1] / STRIDE)); wi = np.floor(t / STRIDE).astype(np.int64)
        okw = (wi < nfull) & inr
        sp = np.bincount(wi[okw] * NB + bi[okw], minlength=nfull * NB).reshape(nfull, NB).astype(np.float64)
        A2 = poisson_A(sp); S2 = np.maximum(A2 @ M, 1.0)
        lnL_B = np.sum(sp * np.log(S2) - S2, 1)
        zA = ((sp - S2) @ U.T) / np.sqrt(S2 @ U2.T + 1e-9)
        top3 = np.argsort(-zA, axis=1)[:, :3]
        normM = M.sum(1)
        per = {}
        for lam_ in LAMS:
            ST = np.zeros(nfull)
            for ti in range(NT):
                rows = np.flatnonzero((top3 == ti).any(1))
                if len(rows) == 0: continue
                Xt = sp[rows]; u = U[ti]
                A = A2[rows].copy()
                c = np.maximum(((Xt - S2[rows]) @ u) / (1e-9 + u @ u), 0.5)
                for _ in range(80):
                    B = A @ M + c[:, None] * u + 1e-9
                    r = Xt / B
                    A *= (r @ M.T) / normM
                    c = np.maximum(c * (r @ u) / (1.0 + 2.0 * lam_ * c), 0.0)
                B = np.maximum(A @ M + c[:, None] * u, 1.0)
                lam_stat = 2.0 * (np.sum(Xt * np.log(B) - B, 1) - lam_ * c * c - lnL_B[rows])
                ST[rows] = np.maximum(ST[rows], np.clip(lam_stat, 0.0, None))
            per[lam_] = ST.astype(np.float32)
        ser[rid] = per
        if n_ % 10 == 0: P(f'  run {rid} ({n_+1}/{len(ALL)})  {time.time()-t0:.0f}s')
    P(f'series built in {time.time()-t0:.0f}s')
    pickle.dump(ser, open(SPK, 'wb'))

# ---- evaluation, frozen protocol ----
FARm = {r: cache[r]['dmin'] > 150 for r in ALL}
IH = {r: i for i, r in enumerate(ALL)}
HRS = np.array([FARm[r].sum() * STRIDE / 3600 for r in ALL])
NORM = {'K-40', 'Th-232', 'Ra-226'}; UFAM = {'DU', 'LEU', 'NatU', 'RefinedU'}
def baseof(nm):
    m = re.match(r'^(.*)-(\d+(?:\.\d+)?kg)$', nm); return m.group(1) if m else nm
def group(nm):
    b = baseof(nm)
    return 'NORM' if b in NORM else ('U-family' if b in UFAM else 'other')
ENC = []
for r in ALL:
    c = cache[r]; wt = c['wtime']
    for j in range(len(c['snr'])):
        m = np.abs(wt - c['stime'][j]) < 60
        if not m.any(): continue
        ENC.append(dict(run=r, grp=group(names[int(c['sid'][j])]), widx=m))
ENC = pd.DataFrame(ENC)
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

def evaluate(S):
    """S: dict run->statistic series on 2 s grid. Returns CAND, thrmap per target, te_rows w/ emax."""
    poolv = np.concatenate([S[r][FARm[r]] for r in ALL])
    CAND = np.array(sorted(set(np.quantile(poolv, 1 - dq / 100)), reverse=True)); NC = len(CAND)
    OM = np.zeros((len(ALL), NC), np.int64)
    for r in ALL:
        s, m = S[r], FARm[r]
        for j, t in enumerate(CAND):
            a = s >= t
            OM[IH[r], j] = np.count_nonzero((a & ~np.r_[False, a[:-1]])[m])
    te = ENC[ENC.run.isin(TE)].reset_index(drop=True)
    te['emax'] = np.concatenate([np.array([S[r][np.asarray(w)].max() for w in sub.widx])
                                 for r, sub in te.groupby('run', sort=False)])
    thr = {}
    for target in (1, 3, 10):
        jj = []
        for r in TE:
            others = np.array([IH[q] for q in ALL if q != r])
            fr = OM[others].sum(0) / HRS[others].sum()
            jj.append(pick(fr, branch_end(fr, target), target))
        thr[target] = {r: CAND[jj[i]] for i, r in enumerate(TE)}
    return te, thr

# control: raw LRT (lam=0) should match t3 lrt; and mx31 headlines
P('=== control mx31 (frozen protocol) ===')
Smx = {r: pd.Series(cache[r]['bestA'].astype(np.float64)).rolling(31, min_periods=1).max().to_numpy() for r in ALL}
te0, thr0 = evaluate(Smx)
MXCI = {}
for target in (1, 3, 10):
    hit = (te0.emax.values >= te0.run.map(thr0[target]).values).astype(float)
    lo, hi = bootrec(hit, te0.run.values)
    MXCI[target] = (hit.mean() * 100, lo * 100, hi * 100)
    P(f'@FAR{target:2d}: {hit.mean()*100:5.1f}% [{lo*100:4.1f},{hi*100:4.1f}]')
assert abs(MXCI[1][0] - 25.2) < 0.051 and abs(MXCI[3][0] - 40.5) < 0.051 and abs(MXCI[10][0] - 70.0) < 0.051

P('\n=== raw LRT check: lam=0 through new code path (t3 gave 7.0/12.6/40.1) ===')
S0 = {r: ser[r][0.0].astype(np.float64) for r in ALL}
te0_, thr0_ = evaluate(S0)
for target in (1, 3, 10):
    hit = (te0_.emax.values >= te0_.run.map(thr0_[target]).values).astype(float)
    P(f'@FAR{target:2d}: {hit.mean()*100:5.1f}%')

P('\n=== TASK 1: L2-regularized joint-fit LRT, lambda sweep ===')
P('(threshold: B_loo; encounter +-60 s; CI: run-cluster bootstrap; groups: corrected map §4.1)')
rows = []
for lam_ in LAMS[1:]:
    S = {r: ser[r][lam_].astype(np.float64) for r in ALL}
    te, thr = evaluate(S)
    line = f'lam={lam_:8.1e} '
    rec = {}
    for target in (1, 3, 10):
        hit = (te.emax.values >= te.run.map(thr[target]).values).astype(float)
        lo, hi = bootrec(hit, te.run.values)
        rec[target] = (hit.mean() * 100, lo * 100, hi * 100)
        in_ci = hi >= MXCI[target][1] and lo <= MXCI[target][2]
        rows.append(dict(lam=lam_, far=target, recall=rec[target][0], lo=rec[target][1], hi=rec[target][2],
                         mx31=MXCI[target][0], overlaps_mx31_CI=bool(in_ci)))
        line += f'| @FAR{target:2d}: {rec[target][0]:5.1f}%[{rec[target][1]:4.1f},{rec[target][2]:4.1f}] '
        line += ('vs ' + f'{MXCI[target][0]:.1f} ' + ('CI-overlap!' if in_ci else ''))
    P(line + f'|| med thr {np.median(list(thr[1].values())):7.1f}')
    P('   per-group @FAR1: ' + ' '.join(
        f"{gr} {(lambda h: h.mean()*100)((te.emax.values >= te.run.map(thr[1]).values).astype(float)[te.grp.values == gr]):5.1f}%"
        for gr in ('NORM', 'U-family', 'other')) +
      '  @FAR10: ' + ' '.join(
        f"{gr} {(lambda h: h.mean()*100)((te.emax.values >= te.run.map(thr[10]).values).astype(float)[te.grp.values == gr]):5.1f}%"
        for gr in ('NORM', 'U-family', 'other')))
R = pd.DataFrame(rows)
R.to_csv('_scratch/t4_summary.csv', index=False)
P('\n=== вердикт ===')
for target in (1, 3, 10):
    sub = R[R.far == target]
    best = sub.sort_values('recall').iloc[-1]
    cand = best.recall >= MXCI[target][1]  # recall within or above mx31's CI
    P(f'@FAR{target}: best lam={best.lam:.0e} recall={best.recall:.1f}% [{best.lo:.1f},{best.hi:.1f}] '
      f'vs mx31 {MXCI[target][0]:.1f}% [{MXCI[target][1]:.1f},{MXCI[target][2]:.1f}] '
      f'{">>> в пределах/выше CI mx31 — реальный кандидат" if cand else "— вне CI mx31 (ухудшение)"}')
O.close()
print('done')
