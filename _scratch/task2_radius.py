"""Task 2: is dmin>150s clean background? Re-calibrate FAR at exclusion radii R=150/200/250/300/400.
For each R: train-pool threshold (robust monotone-branch rule), test recall, achieved test FAR
on the SAME mask AND on a common yardstick mask (dmin>400) so the numbers are comparable across R.
Also report wing contamination: fraction of test far-onset episodes (at each R's thr) that sit in
[R,300s) i.e. source wings, and the onset rate on the pure background (dmin>400) alone.
"""
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(11)
cache = pickle.load(open("_scratch/cache10.pkl", "rb"))
TRAIN = [0] + list(range(3, 20)); VAL = [20, 21, 22, 23, 24]; TEST = list(range(25, 125))
ALLR = TRAIN + VAL + TEST
STRIDE = 2.0
S = {r: pd.Series(c["bestA"]).rolling(31, min_periods=1).max().to_numpy(np.float32).astype(np.float64)
     for r, c in cache.items()}
DMIN = {r: cache[r]["dmin"] for r in S}
EMAX = {r: np.array([S[r][np.abs(cache[r]["wtime"] - t) < 60].max()
                     for t in cache[r]["stime"] if (np.abs(cache[r]["wtime"] - t) < 60).any()])
        for r in ALLR}
IDX = {r: i for i, r in enumerate(ALLR)}
poolv = np.concatenate([S[r][DMIN[r] > 150] for r in ALLR])
dq = np.logspace(np.log10(2e-4), np.log10(30), 320)
CAND = np.array(sorted(set(np.quantile(poolv, 1 - dq / 100)), reverse=True))
NC = len(CAND)

def onset_mask(r, mask, thr):
    a = S[r] >= thr
    return int(np.count_nonzero((a & ~np.r_[False, a[:-1]])[mask]))

def make_table(R):
    """O[run,thr] onset counts + far-hours for mask dmin>R"""
    O = np.zeros((len(ALLR), NC), np.int32)
    H = np.zeros(len(ALLR))
    for r in ALLR:
        m = DMIN[r] > R
        H[IDX[r]] = m.sum() * STRIDE / 3600
        for j, t in enumerate(CAND):
            O[IDX[r], j] = onset_mask(r, m, t)
    return O, H

def branch_end(fr, target):
    runmax, br = fr[0], 0
    for k in range(1, NC):
        if fr[k] > runmax:
            runmax, br = fr[k], k
        elif fr[k] < runmax - max(0.5, 0.2 * runmax) and runmax > target:
            break
    return br

def calib(O, H, runs, target):
    ii = [IDX[r] for r in runs]
    fr = O[ii].sum(0) / H[ii].sum()
    br = branch_end(fr, target)
    d = np.where(fr[:br + 1] <= target, target - fr[:br + 1], np.inf)
    j = int(np.argmin(d)) if np.isfinite(d).any() else 0
    return j, fr[br]

selTR = np.array([IDX[r] for r in TRAIN]); selTE = np.array([IDX[r] for r in TEST])
EN = {R: {r: DMIN[r] > R for r in ALLR} for R in (150, 200, 250, 300, 400)}

# pure-background yardstick: onset counts at any thr on dmin>400 for test runs
Os = {}
print("building onset tables for R=150..400 ...")
for R in (150, 200, 250, 300, 400):
    Os[R] = make_table(R)

ENCN = np.array([len(EMAX[r]) for r in TEST], float)
rows = []
for R in (150, 200, 250, 300, 400):
    O, H = Os[R]
    O4, H4 = Os[400]
    hrs_tr = H[selTR].sum(); hrs_te = H[selTE].sum(); hrs_te4 = H4[selTE].sum()
    print(f"\n===== exclusion R={R}s | train far-hours {hrs_tr:.1f} | test far-hours {hrs_te:.1f} | ultra-far(>400) test {hrs_te4:.1f}")
    for target in (1, 3, 10):
        j, cap = calib(O, H, TRAIN, target)
        thr = CAND[j]
        rec = sum((EMAX[r] >= thr).sum() for r in TEST) / ENCN.sum()
        far_own = O[selTE, j].sum() / hrs_te                      # achieved on its OWN mask
        far_pure = O4[selTE, j].sum() / hrs_te4                   # achieved on clean >400s yardstick
        # wing share: episodes in [R,300) / all episodes beyond R  (only meaningful if R<300)
        n_all = int(O[selTE, j].sum())
        n_ring = 0
        if R < 300:
            for r in TEST:
                a = S[r] >= thr
                m = a & ~np.r_[False, a[:-1]] & (DMIN[r] > R)
                n_ring += int(np.count_nonzero(m & (DMIN[r] < 300)))
        wing = n_ring / n_all * 100 if n_all else float("nan")
        print(f"  @FAR{target:>2}: thr={thr:5.2f}  cap={cap:5.1f}  recall={rec*100:5.1f}%  "
              f"far(own mask)={far_own:5.2f}  far(pure>400)={far_pure:5.2f}  wing-share[150-300s]={wing:5.1f}%")
        rows.append(dict(R=R, target=target, thr=thr, cap=cap, recall=rec, far_own=far_own,
                         far_pure=far_pure, wing=wing, hrs_tr=hrs_tr, hrs_te=hrs_te))
df = pd.DataFrame(rows)
df.to_pickle("_scratch/task2_radius.pkl")

# how many test-run FAR hours vanish between masks: purity check on episodes themselves
print("\ntest onsets at R=150/thr(FAR3) by dmin ring:")
j3, _ = calib(*Os[150][:2], [IDX and r for r in TRAIN], 3) if False else calib(Os[150][0], Os[150][1], TRAIN, 3)
thr3 = CAND[j3]
bins = [(150, 200), (200, 250), (250, 300), (300, 400), (400, 1e9)]
for lo, hi in bins:
    n = 0
    for r in TEST:
        a = S[r] >= thr3
        m = (DMIN[r] > lo) & (DMIN[r] <= hi) & ~np.r_[False, a[:-1]]
        n += int(np.count_nonzero(a & m))
    print(f"  dmin {lo:>3}-{hi if hi < 1e8 else 'inf':>4} s: {n:4d} episodes ({n / (hrs_te / hrs_te) if False else ''}")
# val-side confirmation of recall shift with R
print("\nval-run recall (5 runs) at each R's thr@FAR3:")
ENval = [r for r in VAL]
nv = sum(len(EMAX[r]) for r in ENval)
for R in (150, 200, 250, 300, 400):
    j, _ = calib(Os[R][0], Os[R][1], TRAIN, 3)
    rv = sum((EMAX[r] >= CAND[j]).sum() for r in ENval) / nv
    print(f"  R={R}: thr={CAND[j]:.2f} val recall={rv*100:.1f}%")

fig, ax = plt.subplots(figsize=(8, 5))
for R in (150, 200, 250, 300, 400):
    O, H = Os[R]
    ax.plot(CAND, O[selTR].sum(0) / H[selTR].sum(), label=f"train pool, dmin>{R}s")
ax.set_yscale("log"); ax.invert_xaxis(); ax.set_xlim(right=5.8)
ax.set_xlabel("threshold (mx31)"); ax.set_ylabel("episode-onset FAR / hr (train pool)")
for y in (1, 3, 10): ax.axhline(y, c="gray", lw=.4)
ax.grid(alpha=.3); ax.legend(); fig.tight_layout(); fig.savefig("_scratch/task2_radius.png", dpi=110)
print("plot -> _scratch/task2_radius.png")
