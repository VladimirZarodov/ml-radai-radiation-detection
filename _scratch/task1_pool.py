"""Task 1: widen the FAR-calibration background pool; which CI actually narrows?

Protocols (detector frozen = mx31 = rollmax(bestA,31), far mask dmin>150):
  A      : threshold from TRAIN-run far windows  [~7 h]                    = current
  B_all  : threshold from far windows of ALL runs (train+val+test, ~49 h)  (in-sample caveat!)
  B_loo  : per evaluated test run, threshold from ALL OTHER runs           (honest wide pool)
  C_self : per run, threshold from its OWN far windows                     (online self-calibration)
CIs: test-side cluster bootstrap (B=500, thresholds fixed) for all protocols;
full-protocol (recalibrate on resampled calibration pool, B=300) for A and B_all.
Implementation: onset-count matrix O[run, cand_thr] precomputed -> every calibration is a colsum.
"""
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)
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
        for r in ALLR}   # per-encounter max of the ROLLMAX series = notebook hit rule (alarms trailing into +-60s count)

# candidate grid: global quantiles of the widest pool (superset of any sub-pool's useful thr)
poolv = np.concatenate([S[r][FARm[r]] for r in ALLR])
dq = np.logspace(np.log10(2e-4), np.log10(30), 320)          # dense in the tail where FAR 1-100 lives
CAND = np.array(sorted(set(np.quantile(poolv, 1 - dq / 100)), reverse=True))
NC = len(CAND)
O = np.zeros((len(ALLR), NC), np.int32)
for r in ALLR:
    s, f = S[r], FARm[r]
    for j, t in enumerate(CAND):
        a = s >= t
        O[IDX[r], j] = np.count_nonzero((a & ~np.r_[False, a[:-1]])[f])
print(f"onset matrix built: {O.shape}, total far-hours {H.sum():.1f} (train {H[:len(TRAIN)].sum():.1f})")

def branch_end(fr, target):
    """monotone-branch end: run down the FAR curve until it falls >0.5/hr (or >20%) below its
    running max AND the peak already exceeded the target (real merged-episode rollover, not a
    1-episode wobble in the sparse-count tail)."""
    runmax, br = fr[0], 0
    for k in range(1, NC):
        if fr[k] > runmax:
            runmax, br = fr[k], k
        elif fr[k] < runmax - max(0.5, 0.2 * runmax) and runmax > target:
            break
    return br

def calib(runs, target):
    """thr on the monotone branch closest to target FAR with FAR <= target. Returns (cand idx, ceiling)."""
    ii = [IDX[r] for r in runs]
    fr = O[ii].sum(0) / H[ii].sum()
    br = branch_end(fr, target)
    d = np.where(fr[:br + 1] <= target, target - fr[:br + 1], np.inf)  # prefer FAR<=target, closest first (ties -> higher thr)
    j = int(np.argmin(d)) if np.isfinite(d).any() else 0
    return j, fr[br]

# ---- test-side helpers ----
selTE = np.array([IDX[r] for r in TEST])
ECAT = np.concatenate([EMAX[r] for r in TEST])
ERUN = np.concatenate([[i] * len(EMAX[r]) for i, r in enumerate(TEST)])
ENCN = np.bincount(ERUN, minlength=len(TEST)).astype(float)
# hit counts per (test-run, candidate thr)
HM = np.zeros((len(TEST), NC))
for i, r in enumerate(TEST):
    HM[i] = [(EMAX[r] >= t).sum() for t in CAND]

def eval_test(jvec):
    """jvec: candidate index per TEST run -> (recall, achieved far on test far windows)"""
    fa = O[selTE, jvec].sum(); hh = H[selTE].sum()
    hits = HM[np.arange(len(TEST)), jvec].sum()
    return hits / ENCN.sum(), fa / hh

res = {}
print("pool hours: train %.1f | all %.1f | test-run median %.2f"
      % (H[:18].sum(), H.sum(), np.median(H[selTE])))
for target in (1, 3, 10):
    jA, capA = calib(TRAIN, target)
    jB, capB = calib(ALLR, target)
    rA, fA = eval_test(np.full(len(TEST), jA))
    rB, fB = eval_test(np.full(len(TEST), jB))
    jL, capsL = np.zeros(len(TEST), int), []
    for i, r in enumerate(TEST):
        jL[i], cp = calib([x for x in ALLR if x != r], target); capsL.append(cp)
    rL, fL = eval_test(jL)
    jS, capsS = np.zeros(len(TEST), int), []
    for i, r in enumerate(TEST):
        jS[i], cp = calib([r], target); capsS.append(cp)
    rS, fS = eval_test(jS)
    thrL, thrS = CAND[jL], CAND[jS]
    print(f"\n### target FAR={target}/hr   (branch ceilings: train {capA:.1f}, all {capB:.1f})")
    print(f"  A     : thr={CAND[jA]:6.2f}  recall={rA*100:5.1f}%  achieved-test-far={fA:5.1f}")
    print(f"  B_all : thr={CAND[jB]:6.2f}  recall={rB*100:5.1f}%  achieved-test-far={fB:5.1f}")
    print(f"  B_loo : thr med={np.median(thrL):6.2f}  recall={rL*100:5.1f}%  achieved-test-far={fL:5.1f}"
          f"  (thr p10/50/90 {np.percentile(thrL,[10,50,90]).round(2)}, capped {np.mean(np.array(capsL)<target)*100:.0f}%)")
    print(f"  C_self: thr med={np.median(thrS):6.2f}  recall={rS*100:5.1f}%  achieved-test-far={fS:5.1f}"
          f"  (thr p10/50/90 {np.percentile(thrS,[10,50,90]).round(2)}, capped {np.mean(np.array(capsS)<target)*100:.0f}%)")
    res[target] = dict(jA=jA, jB=jB, jL=jL.copy(), jS=jS.copy(),
                       rA=rA, rB=rB, rL=rL, rS=rS, fA=fA, fB=fB, fL=fL, fS=fS)

# ---------- bootstrap CIs ----------
print("\n### bootstrap CIs  (test-side B=500 fixed thr; full-protocol B=300 recalibrated)")
rows = []
for target in (1, 3, 10):
    d = res[target]
    for ptag, jv in [("A", np.full(len(TEST), d["jA"])), ("B_all", np.full(len(TEST), d["jB"])),
                     ("B_loo", d["jL"]), ("C_self", d["jS"])]:
        recs, fars = [], []
        for b in range(500):
            cnt = np.bincount(rng.integers(0, len(TEST), len(TEST)), minlength=len(TEST)).astype(float)
            recs.append((cnt * HM[np.arange(len(TEST)), jv]).sum() / (cnt * ENCN).sum())
            fars.append((cnt * O[selTE, jv]).sum() / (cnt * H[selTE]).sum())
        rows.append(dict(protocol=ptag, target=target, kind="test-side",
                         rec=np.percentile(recs, [2.5, 97.5]), far=np.percentile(fars, [2.5, 97.5])))
    for ptag, pool in [("A", TRAIN), ("B_all", ALLR)]:
        sel = np.array([IDX[r] for r in pool])
        recs, fars, thrl = [], [], []
        for b in range(300):
            ii = sel[rng.integers(0, len(sel), len(sel))]
            fr = O[ii].sum(0) / H[ii].sum()
            br = branch_end(fr, target)
            d = np.where(fr[:br + 1] <= target, target - fr[:br + 1], np.inf)
            j = int(np.argmin(d)) if np.isfinite(d).any() else 0
            thrl.append(CAND[j])
            cnt = np.bincount(rng.integers(0, len(TEST), len(TEST)), minlength=len(TEST)).astype(float)
            recs.append((cnt * HM[:, j]).sum() / (cnt * ENCN).sum())
            fars.append((cnt * O[selTE, j]).sum() / (cnt * H[selTE]).sum())
        rows.append(dict(protocol=ptag, target=target, kind="full",
                         rec=np.percentile(recs, [2.5, 97.5]), far=np.percentile(fars, [2.5, 97.5])))
        print(f"  [{ptag} full @{target}] recalibrated thr dist p2.5/50/97.5 = {np.percentile(thrl,[2.5,50,97.5]).round(2)}")

for _, x in pd.DataFrame(rows).iterrows():
    print(f"  {x.protocol:6s} @{x.target:>2} ({x.kind:9s}): recall CI [{x.rec[0]*100:.1f}, {x.rec[1]*100:.1f}]  "
          f"far CI [{x.far[0]:.1f}, {x.far[1]:.1f}]")
pickle.dump(dict(res=res, boot=rows, CAND=CAND), open("_scratch/task1_pool.pkl", "wb"))

fig, ax = plt.subplots(figsize=(8, 5))
for ptag, runs in [("train pool (7 h)", TRAIN), ("all-runs pool (~49 h)", ALLR)]:
    ii = [IDX[r] for r in runs]
    ax.plot(CAND, O[ii].sum(0) / H[ii].sum(), label=ptag)
ax.set_yscale("log"); ax.invert_xaxis()
ax.set_xlabel("threshold (mx31 score)"); ax.set_ylabel("episode-onset FAR / hr on its own pool")
for y in (1, 3, 10): ax.axhline(y, c="gray", lw=.4)
ax.grid(alpha=.3); ax.legend(); fig.tight_layout(); fig.savefig("_scratch/task1_pool.png", dpi=110)
print("plot -> _scratch/task1_pool.png")
