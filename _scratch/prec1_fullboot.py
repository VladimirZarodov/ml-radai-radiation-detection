"""Precision pass, item 1: FULL-protocol bootstrap side by side.
Resample BOTH the calibration pool (threshold recalibrated each replicate) AND the test
evaluation, on (a) the old 7 h train-only pool with the FIXED branch rule, (b) the same pool
with the OLD stop-at-first-decrease rule (isolates the bug, same pool+grid), (c) the 50 h
leave-one-run-out pool. B=300, seed fixed. Notebook cell reproduces this verbatim."""
import pickle
import numpy as np
import pandas as pd

cache = pickle.load(open("_scratch/cache10.pkl", "rb"))
TRAIN = [0] + list(range(3, 20)); VAL = [20, 21, 22, 23, 24]; TEST = list(range(25, 125))
ALLR = TRAIN + VAL + TEST
STRIDE = 2.0
S = {r: pd.Series(c["bestA"]).rolling(31, min_periods=1).max().to_numpy(np.float32).astype(np.float64)
     for r, c in cache.items()}
FARm = {r: cache[r]["dmin"] > 150 for r in S}
IDX = {r: i for i, r in enumerate(ALLR)}
H = np.zeros(len(ALLR))
for r in ALLR: H[IDX[r]] = FARm[r].sum() * STRIDE / 3600
EMAX = {r: np.array([S[r][np.abs(cache[r]["wtime"] - t) < 60].max()
                     for t in cache[r]["stime"] if (np.abs(cache[r]["wtime"] - t) < 60).any()])
        for r in ALLR}
poolv = np.concatenate([S[r][FARm[r]] for r in ALLR])
dq = np.logspace(np.log10(2e-4), np.log10(30), 320)
CAND = np.array(sorted(set(np.quantile(poolv, 1 - dq / 100)), reverse=True))
NC = len(CAND)
O = np.zeros((len(ALLR), NC), np.int32)
for r in ALLR:
    s, f = S[r], FARm[r]
    for j, t in enumerate(CAND):
        a = s >= t
        O[IDX[r], j] = np.count_nonzero((a & ~np.r_[False, a[:-1]])[f])
print(f"grid {NC} pts | pool {H.sum():.1f} h | train {H[[IDX[r] for r in TRAIN]].sum():.1f} h")

def branch_end(fr, target):
    runmax, br = fr[0], 0
    for k in range(1, NC):
        if fr[k] > runmax: runmax, br = fr[k], k
        elif fr[k] < runmax - max(0.5, 0.2 * runmax) and runmax > target: break
    return br

def branch_end_old(fr, target):          # old rule: stop at the FIRST decrease
    runmax, br = fr[0], 0
    for k in range(1, NC):
        if fr[k] > runmax: runmax, br = fr[k], k
        elif fr[k] < runmax: break
    return br

def pick(fr, br, target):
    d = np.where(fr[:br + 1] <= target, target - fr[:br + 1], np.inf)
    return int(np.argmin(d)) if np.isfinite(d).any() else 0

selTR = np.array([IDX[r] for r in TRAIN])
selTE = np.array([IDX[r] for r in TEST])
selALL = np.arange(len(ALLR))
ENCN = np.array([len(EMAX[r]) for r in TEST], float)
HM = np.array([[(EMAX[r] >= t).sum() for t in CAND] for r in TEST])

def full_boot(pool_sel, rule, target, B, rng, loo=False):
    recs, fars, thrl = [], [], []
    nP = len(pool_sel)
    for b in range(B):
        cnt = np.bincount(rng.integers(0, len(TEST), len(TEST)), minlength=len(TEST)).astype(float)
        if loo:
            # per-evaluated-run threshold from a with-replacement resample of ALL other runs
            jj = np.empty(len(TEST), int)
            for i, r in enumerate(TEST):
                others = np.delete(pool_sel, np.where(pool_sel == IDX[r])[0])
                ii = others[rng.integers(0, len(others), len(others))]
                fr = O[ii].sum(0) / H[ii].sum()
                jj[i] = pick(fr, rule(fr, target), target)
            hvec = HM[np.arange(len(TEST)), jj]
            fa_vec = O[selTE, jj]
        else:
            ii = pool_sel[rng.integers(0, nP, nP)]
            fr = O[ii].sum(0) / H[ii].sum()
            j = pick(fr, rule(fr, target), target)
            jj = np.full(len(TEST), j); hvec = HM[:, j]; fa_vec = O[selTE, j]
            thrl.append(CAND[j])
        recs.append((cnt * hvec).sum() / (cnt * ENCN).sum())
        fars.append((cnt * fa_vec).sum() / (cnt * H[selTE]).sum())
    return (np.percentile(recs, [2.5, 50, 97.5]), np.percentile(fars, [2.5, 50, 97.5]),
            np.percentile(thrl, [2.5, 50, 97.5]) if thrl else None)

B = 300
out = []
for target in (1, 3, 10):
    rngA = np.random.default_rng(7); rngB = np.random.default_rng(7); rngL = np.random.default_rng(7)
    recA_old, faA_old, thrA_old = full_boot(selTR, branch_end_old, target, B, rngA)
    recA_new, faA_new, _        = full_boot(selTR, branch_end,     target, B, rngB)
    recL,     faL,     _        = full_boot(selALL, branch_end,    target, B, rngL, loo=True)
    for tag, rec, fa in [("A 7h  OLD-rule ", recA_old, faA_old),
                         ("A 7h  fixed    ", recA_new, faA_new),
                         ("B_loo 50h fixed", recL, faL)]:
        print(f"@FAR{target:>2} {tag}: recall 95% CI [{rec[0]*100:5.1f}, {rec[2]*100:5.1f}] med {rec[1]*100:5.1f}"
              f" | achieved-FAR CI [{fa[0]:5.2f}, {fa[2]:5.2f}]")
    print(f"        [A OLD-rule thr dist p2.5/50/97.5 = {thrA_old.round(2)}]")
    out.append((target, recA_old, faA_old, recA_new, faA_new, recL, faL))
pickle.dump(out, open("_scratch/prec1_fullboot.pkl", "wb"))
