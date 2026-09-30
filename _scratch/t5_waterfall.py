# -*- coding: utf-8 -*-
"""TASK 2b (exploratory, NOT part of the frozen detector, cannot change headlines):
waterfall-CNN probe on the hard groups NORM/U-family, following the PNNL idea
(arXiv 2607.00270): consecutive spectra as input CHANNELS + Focal Loss.
Representation: W=8 consecutive 2 s slots x 128 sqrt-keV bins (existing binning), stride 1,
centered window, edge-replicated; input log1p(counts). 1D-CNN over the energy axis.
Trained ONLY on TRAIN runs (positives: slots within +-60 s of a NORM/U-family source;
negatives: pure-background FARM slots, subsampled 4:1 "matched background" pool), epoch
selection on VAL runs. Evaluated on TEST runs through the FROZEN protocol: CAND log-quantile
grid, monotone branch/pick, per-run B_loo thresholds, +-60 s encounters, run-cluster bootstrap.
Report NORM / U-family recall separately, raw logits + rolling-31 (same 62 s integration as mx31).
"""
import io, os, pickle, re, time
import torch  # FIRST: на этой машине импорт torch после numpy/h5py падает (WinError 1114, конфликт DLL)
import numpy as np, pandas as pd, h5py
from sklearn.metrics import roc_auc_score

torch.manual_seed(7); np.random.seed(7)
O = io.open('_scratch/t5.out', 'w', encoding='utf-8')
def P(*a):
    print(*a, file=O); O.flush()

EMIN, EMAXE, NB, STRIDE, W = 15.0, 3000.0, 128, 2.0, 8
smin, smax = np.sqrt(EMIN), np.sqrt(EMAXE)
f = h5py.File('training_v4.3.h5', 'r')
names = [str(n).split('_shielding')[0] for n in f.attrs['source_names']]
TRAIN = [0] + list(range(3, 20)); VAL = [20, 21, 22, 23, 24]; TEST = list(range(25, 125)); ALL = TRAIN + VAL + TEST
try:
    cache = pickle.load(open('_scratch/cache10.pkl', 'rb'))          # сырой кэш detect6.py
except FileNotFoundError:
    cache = pickle.load(open('_scratch/nb_mf_cache.pkl', 'rb'))      # или кэш ноутбука §3.2 (те же поля)
NORM = {'K-40', 'Th-232', 'Ra-226'}; UFAM = {'DU', 'LEU', 'NatU', 'RefinedU'}
def baseof(nm):
    m = re.match(r'^(.*)-(\d+(?:\.\d+)?kg)$', nm); return m.group(1) if m else nm
def group(nm):
    b = baseof(nm)
    return 'NORM' if b in NORM else ('U-family' if b in UFAM else 'other')

def get_spec(rid):
    gg = f[f'runs/run{rid}']
    dt = gg['listmode/dt'][:]; t = np.cumsum(dt, dtype=np.uint64) / 1e6
    e = gg['listmode/energy'][:]; bid = gg['listmode/background_id'][:]; eid = gg['listmode/id'][:]
    inr = (e >= EMIN) & (e < EMAXE)
    bi = np.zeros(len(e), np.int64)
    bi[inr] = np.clip(np.floor((np.sqrt(e[inr]) - smin) / (smax - smin) * NB).astype(np.int64), 0, NB - 1)
    nfull = int(np.floor(t[-1] / STRIDE)); wi = np.floor(t / STRIDE).astype(np.int64)
    okw = (wi < nfull) & inr
    return np.bincount(wi[okw] * NB + bi[okw], minlength=nfull * NB).reshape(nfull, NB).astype(np.float64)

def group_dmin(spec_n, rid, grp):
    c = cache[rid]; js = [j for j in range(len(c['sid'])) if group(names[int(c['sid'][j])]) == grp]
    if not js: return np.full(len(spec_n), np.inf)
    st = c['stime'][js]
    return np.abs(c['wtime'][:, None] - st[None, :]).min(1)

def windows(spec):
    n = len(spec)
    idx = np.clip(np.arange(n)[:, None] + np.arange(-W // 2, W // 2)[None, :], 0, n - 1)
    return np.log1p(spec[idx]).astype(np.float32)          # (n, W, NB)

FARM = {r: cache[r]['dmin'] > 150 for r in ALL}
IH = {r: i for i, r in enumerate(ALL)}
HRS = {r: FARM[r].sum() * STRIDE / 3600 for r in ALL}
HV = np.array([HRS[r] for r in ALL])

# ---------- per-run masks ----------
t0 = time.time()
masks = {}
for rid in ALL:
    c = cache[rid]; n = len(c['bestA'])
    dmN, dmU = group_dmin(c['bestA'], rid, 'NORM'), group_dmin(c['bestA'], rid, 'U-family')
    masks[rid] = dict(pos=(np.minimum(dmN, dmU) < 60), dn=dmN, du=dmU)
P(f'masks built ({time.time()-t0:.0f}s)')

SPK = '_scratch/t5_wf.pkl'
if os.path.exists(SPK):
    D = pickle.load(open(SPK, 'rb'))
    P('t5_wf.pkl loaded: skip training/scoring')
else:
    # ---------- train/val pools ----------
    rng = np.random.default_rng(11)
    def pool(runs):
        Xs, Ys, Gs = [], [], []
        for rid in runs:
            Xw = windows(get_spec(rid)); m = masks[rid]
            pos = np.flatnonzero(m['pos']); neg = np.flatnonzero(FARM[rid])
            take = rng.choice(neg, size=min(len(neg), 4 * max(len(pos), 1)), replace=False)
            Xs.append(Xw[pos]); Ys.append(np.ones(len(pos), np.int64)); Gs.append(m['pos'][pos].astype(np.int64) & (m['dn'][pos] <= m['du'][pos]))
            Xs.append(Xw[take]); Ys.append(np.zeros(len(take), np.int64)); Gs.append(np.zeros(len(take), np.int64) - 1)
        return np.concatenate(Xs), np.concatenate(Ys), np.concatenate(Gs)
    Xtr, Ytr, Gtr = pool(TRAIN); Xva, Yva, Gva = pool(VAL)
    P(f'train pool: {Xtr.shape[0]} win (pos {int(Ytr.sum())}, bg {int((Ytr==0).sum())}) | val {Xva.shape[0]} win (pos {int(Yva.sum())})')

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.c = torch.nn.Sequential(
                torch.nn.Conv1d(W, 16, 5, padding=2), torch.nn.BatchNorm1d(16), torch.nn.ReLU(), torch.nn.MaxPool1d(4),
                torch.nn.Conv1d(16, 32, 5, padding=2), torch.nn.BatchNorm1d(32), torch.nn.ReLU(),
                torch.nn.AdaptiveMaxPool1d(1), torch.nn.Flatten(), torch.nn.Linear(32, 1))
        def forward(self, x): return self.c(x).squeeze(-1)

    def focal(logit, y, gamma=2.0):
        ce = torch.nn.functional.binary_cross_entropy_with_logits(logit, y.float(), reduction='none')
        p = torch.sigmoid(logit)
        wt = torch.where(y > 0, 1 - p, p) ** gamma
        return (wt * ce).mean()

    Xt = torch.from_numpy(Xtr); Yt = torch.from_numpy(Ytr)
    Xv = torch.from_numpy(Xva); Yv = torch.from_numpy(Yva)
    net = Net(); opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    best, state, best_ep = -1, None, 0
    N = len(Xt)
    for ep in range(1, 31):
        net.train(); perm = torch.randperm(N)
        for i in range(0, N, 256):
            b = perm[i:i+256]
            opt.zero_grad(); loss = focal(net(Xt[b]), Yt[b]); loss.backward(); opt.step()
        if ep % 3 == 0 or ep == 1:
            net.eval()
            with torch.no_grad():
                lv = torch.cat([net(Xv[i:i+4096]) for i in range(0, len(Xv), 4096)]).numpy()
            auc = roc_auc_score(Yv, lv)
            P(f'  epoch {ep:2d}: val AUC {auc:.4f}')
            if auc > best: best, best_ep, state = auc, ep, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(state); net.eval()
    with torch.no_grad():
        lv = torch.cat([net(Xv[i:i+4096]) for i in range(0, len(Xv), 4096)]).numpy()
    # per-group vs bg on val: positives of the group vs all val background windows
    def gauc(gval):
        sel = (Gva == gval) | (Yva == 0)
        if ((Gva[sel] == gval).sum() < 2): return float('nan')
        return roc_auc_score((Gva[sel] == gval).astype(int), lv[sel])
    a_N, a_U = gauc(1), gauc(0)
    P(f'best epoch {best_ep}: val AUC {best:.4f} | NORM-vs-bg {a_N:.4f} | U-family-vs-bg {a_U:.4f}')

    # ---------- score every run ----------
    logits = {}
    for i, rid in enumerate(ALL):
        Xw = torch.from_numpy(windows(get_spec(rid)))
        with torch.no_grad():
            lg = torch.cat([net(Xw[j:j+8192]) for j in range(0, len(Xw), 8192)]).numpy()
        logits[rid] = lg.astype(np.float32)
        if (i + 1) % 25 == 0: P(f'  scored {i+1}/123 ({time.time()-t0:.0f}s)')
    D = dict(logits=logits, val_auc=float(best), best_epoch=best_ep, auc_norm=float(a_N), auc_ufam=float(a_U),
             n_train_pos=int(Ytr.sum()), n_train_bg=int((Ytr == 0).sum()))
    pickle.dump(D, open(SPK, 'wb'))
    P(f'training+scoring done in {time.time()-t0:.0f}s')

# ---------- frozen-protocol evaluation ----------
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
    poolv = np.concatenate([S[r][FARM[r]] for r in ALL])
    CAND = np.array(sorted(set(np.quantile(poolv, 1 - dq / 100)), reverse=True)); NC = len(CAND)
    OM = np.zeros((len(ALL), NC), np.int64)
    for r in ALL:
        s, m = S[r], FARM[r]
        for j, t in enumerate(CAND):
            a = s >= t
            OM[IH[r], j] = np.count_nonzero((a & ~np.r_[False, a[:-1]])[m])
    te = ENC[ENC.run.isin(TEST)].reset_index(drop=True)
    te['emax'] = np.concatenate([np.array([S[r][np.asarray(w)].max() for w in sub.widx])
                                 for r, sub in te.groupby('run', sort=False)])
    thr, ach = {}, {}
    for target in (1, 3, 10):
        jj = []
        for r in TEST:
            others = np.array([IH[q] for q in ALL if q != r])
            fr = OM[others].sum(0) / HV[others].sum()
            jj.append(pick(fr, branch_end(fr, target), target))
        thr[target] = {r: CAND[jj[i]] for i, r in enumerate(TEST)}
        ach[target] = float(np.mean([OM[IH[r], jj[i]] * 1.0 / HRS[r] for i, r in enumerate(TEST)]))
    return te, thr, ach

P('\n=== контроль: mx31 (тот же протокол) ===')
Smx = {r: pd.Series(cache[r]['bestA'].astype(np.float64)).rolling(31, min_periods=1).max().to_numpy() for r in ALL}
te0, thr0, _ = evaluate(Smx)
MXCI, MXG = {}, {}
for target in (1, 3, 10):
    hit = (te0.emax.values >= te0.run.map(thr0[target]).values).astype(float)
    lo, hi = bootrec(hit, te0.run.values); MXCI[target] = (hit.mean() * 100, lo * 100, hi * 100)
    MXG[target] = {g: hit[te0.grp.values == g].mean() * 100 for g in ('NORM', 'U-family', 'other')}
    P(f'@FAR{target:2d}: {hit.mean()*100:5.1f}% [{lo*100:4.1f},{hi*100:4.1f}] | ' +
      ' '.join(f'{g} {MXG[target][g]:5.1f}%' for g in ('NORM', 'U-family', 'other')))
assert abs(MXCI[1][0]-25.2) < 0.051 and abs(MXCI[3][0]-40.5) < 0.051 and abs(MXCI[10][0]-70.0) < 0.051

P('\n=== TASK 2b: waterfall-CNN (val AUC {v:.3f}, best epoch {e}, NORM-vs-bg {an:.3f}, U-vs-bg {au:.3f}) ==='.format(
    v=D['val_auc'], e=D['best_epoch'], an=D['auc_norm'], au=D['auc_ufam']))
P(f"train pool: pos {D['n_train_pos']} / bg {D['n_train_bg']} (train runs only; epoch pick on val)")
Slg = {r: D['logits'][r].astype(np.float64) for r in ALL}
Sroll = {r: pd.Series(D['logits'][r].astype(np.float64)).rolling(31, min_periods=1).max().to_numpy() for r in ALL}
for tag, S in (('raw logit', Slg), ('roll31 logit', Sroll)):
    te, thr, ach = evaluate(S)
    P(f'\n--- {tag} ---')
    for target in (1, 3, 10):
        hit = (te.emax.values >= te.run.map(thr[target]).values).astype(float)
        lo, hi = bootrec(hit, te.run.values)
        gg = ' '.join(f'{g} {hit[te.grp.values == g].mean()*100:5.1f}%' for g in ('NORM', 'U-family', 'other'))
        P(f'@FAR{target:2d} (ach {ach[target]:4.1f}): all {hit.mean()*100:5.1f}%[{lo*100:4.1f},{hi*100:4.1f}] | {gg}'
          f' | mx31 per-group: ' + ' '.join(f'{MXG[target][g]:5.1f}%' for g in ('NORM', 'U-family', 'other')))
O.close()
print('done')
