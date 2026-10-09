# CHECKPOINT 1 — Stage 1 learned baseline, 4-config comparison + 5 diagnostics

Date: 2026-10-04 15:05. Protocol: frozen PLAN.md + two AMENDED addenda
(lockbox power; C1 diagnostics). **Configs tried: 4** (the pre-registered
Stage-1 matrix): c64h / c64s / c128h / c128s = context ±{64,128} s × target
{hard window-label, soft CA-peaked sigma=6 s}. One architecture for all
(547,325 params). Selection: pooled dev OOF at pre-registered fpr targets ONLY.
Lockbox never evaluated. Shift folds report-only.

---

## 1. Setup (as pre-registered)

- Dev pool: 240 annotated runs (of the 300-run "train" split), 229.4 official
  background-hours; folds frozen `cache/folds.json` (57/49/46/44/44);
  lockbox 60 runs sealed (evaluated exactly once at C4, never here).
- Features: 128 ch sqrt(total / per-run MEDIAN of own totals) + 4 robust
  log-rate/rise channels, F=132, label-free. Iso vocab = 24 official labels.
- Training: 12 epochs, patience 3 on inner-val focal (inner split = last 15 %
  of fold-train runs by id), focal(g2,a.75)+.5CEcat+.5CEiso(sqrt-inv-freq),
  AdamW 1.5e-3 cosine, wd 1e-5, batch 192/96, AMP, NEG_RATE 0.12, deterministic
  seeds per cfg/fold. Alarms: Gaussian sigma 1.5 s, NMS sep 15 s, PMIN 0.02,
  cap 300/run, span ±2 s (fixed before results).
- Cost per config (measured): train sum-of-folds c64h 37.6 min, c64s 47.0,
  c128h 67.8, c128s 104.1; eval ~4-7 min; +SYNTH step ~5-10 min. Peak VRAM
  3153-3205 MiB (6 GB card). Params 547,325 (all configs).

## 2. Pooled dev OOF — selected config c64s vs v1 comparator

v1 reproduced IN this harness (parity-verified at 0.0 max diff, E028).
Global, thresholds = dev-self-calibrated (thr_at_fpr on pooled OOF):

| target | thr | d | d CI95 | c | id | fpr | FP | v1 d | paired diff CI95 |
|---|---|---|---|---|---|---|---|---|---|
| 0.085 | 0.8617 | 0.4435 | [0.420,0.466] | 0.4273 | 0.2880 | 0.0828 | 19 | 0.1199 | [+0.299,+0.348] |
| 0.17  | 0.8449 | 0.4926 | [0.468,0.515] | 0.4750 | 0.3171 | 0.1657 | 38 | 0.1287 | [+0.340,+0.390] |
| 0.35  | 0.8234 | 0.5347 | [0.512,0.555] | 0.5157 | 0.3421 | 0.3488 | 80 | 0.1481 | [+0.361,+0.412] |
| 0.70  | 0.7961 | 0.5736 | [0.553,0.595] | 0.5509 | 0.3676 | 0.6976 | 160 | 0.1657 | [+0.383,+0.432] |

Alarms emitted at thresholds: 4.31 / 4.90 / 5.58 / 6.44 per bg-hour.

Per-category (c64s vs v1 at same thr), d/c/id — full grid in
`reports/c64s_oof.md`; @0.085: NORM 0.064/0.064/0.064 (v1 0.005/0/0),
Medical 0.260/0.242/0.226 (v1 0.219/0.204/0.193), Industrial 0.599/0.517/0.513
(v1 0.115/0.026/0.026), NuclearMaterial 0.503/0.497/0.289 (v1 0.119/0.104/0.074).
c64s wins every category at every target.

## 3. The 4 configs (all tried, all reported)

| cfg | d@.085 | d@.17 | d@.35 | d@.70 | diff-CI lo @.085 | NORM d@.085 | hb d@.085 | synth infl @.085 |
|---|---|---|---|---|---|---|---|---|
| **c64s** | **0.4435** | **0.4926** | **0.5347** | **0.5736** | +0.299 | 0.064 | 0.4225 | 5.89 |
| c64h | 0.4324 | 0.4606 | 0.5028 | 0.5454 | +0.287 | 0.049 | 0.4157 | 4.79 |
| c128s | 0.3245 | 0.3676 | 0.4148 | 0.4630 | +0.175 | 0.078 | 0.2921 | 0.63 |
| c128h | 0.2481 | 0.2866 | 0.3282 | 0.3644 | +0.096 | 0.015 | 0.2539 | 1.05 |

**Selected: c64s.** Best at all four targets; margin over c64h at 0.085 is
within single-CI noise (+0.011) but c64s dominates at .17/.35/.70 (+0.032 each,
near-non-overlapping CIs) and on NORM recall. ±128 s context is clearly
counterproductive (dev-OOF d down 0.12-0.19); its low synth "inflation" is an
artifact of uniformly weak scores, not robustness (self-cal synth d 0.069 vs
c64s 0.062 at .085 — comparable floor, far less signal).

Best epoch per fold (of 12, patience 3): c64h 2/2/3/2/4; c64s 5/4/3/6/2;
c128h 1/2/4/**0/1**; c128s 5/6/3/7/4.
*Comment (observation only; nothing changed):* early best-epochs are expected
with patience-3 on a ~50-run inner-val focal surface — the val curve is flat
within run-cluster noise after ~epoch 3, so argmin lands early and cosine/LR
details do not plausibly matter at this horizon (at ep 2-4 the cosine is still
at 0.91-0.75x peak LR; a longer/annealed schedule would mostly move *which*
noise-plateau epoch wins). c128h folds 0/3 selecting epoch 0-1 (val focal 0.050)
indicate genuine optimization instability at ±128 s, consistent with their OOF
deficit. If Stage 2+ wants a knob, the target-softness axis mattered (+0.011
..+0.032 for soft) far more than LR/schedule would.

## 4. Diagnostics (C1 addendum, run after chain, no tuning impact)

**1. Time-shift control (c64s pooled OOF, ±400 s, out-of-run DROPPED):**
d@.085 0.4435 -> 0.0181 (+400 s) / 0.0208 (−400 s); fpr 0.0828 -> 3.92 / 3.51
(FP 898 / 804). Residual d = ~400 s encounter spacing inside multi-encounter
runs matching by accident. **Detection collapses -> no matching/scoring
artifact; alarms genuinely encode CA-relative timing.** PASS.

**2. Speed/duration sensitivity (|v| at CA from training_v4.3.h5; fixed dev
thrs):** d/c/id@.35 by speed tercile: 0.654/0.631/0.419 [0.6-4.4 m/s] ->
0.532/0.515/0.344 [4.4-6.8] -> **0.418/0.401/0.263 [6.8-8.0]**; width proxy
(official window): 0.374 -> 0.526 -> 0.690 (short windows hurt); distance
mild (0.550->0.515). **The shift axis is fast/short encounters.** Testing
platforms go to 13.4 m/s vs training max 8.0 — i.e. testing extrapolates BELOW
the worst dev tercile. This is the number to beat in Stage 2, not the dev mean.

**3. Unlabeled testing diagnostic (c64s fold-ensemble, 300 testing runs,
count stream only, no labels/CSV; testing-derived — disclose in README):**
test pool 301.4 h (1.005 h/run) vs dev 229.4 h. Alarms/h at the four dev
thresholds: **1.93/2.21/2.58/2.98 vs dev OOF 4.31/4.90/5.58/6.44 = ratio
0.45-0.46×.** @.35-thr per-run: mean 2.59, sd 1.42, max 7, zero-alarm runs
21/300. Score quantiles (p50/90/99/99.9/max): test 0.329/0.469/0.790/0.918/
0.973 vs dev OOF 0.347/0.576/0.864/0.951/0.993 (medians match, tails shrink).
**Not the FP-inflation warning** (would be ratios ≫1.2-1.3 per v1 precedent):
the opposite — the model *under-fires* ~2.2× less on testing at fixed dev
thresholds, consistent with diagnostic 2 (shorter/faster encounters; +17 % rate
/ +20 % NORM also change the normalized feature profile). Caveat: testing used
the 5-model ensemble mean, dev OOF single own-fold models — some tail
compression is ensembling; the median alignment says most is not. **Dev FP/hr
targets will NOT transfer to testing as "calibration": Stage-4 must threshold
on testing-side score statistics (disclosed aggregate use), and Stage-2 must
recover the lost detection.**

**4. Leakage audit** (reports/leak_audit.md): (a) inputs = per-second
total/rate_all/rate_in only; no listmode id / background_id / sources/* / AK
in feature code; normalization = run's own median. (b) windows never leave
their own run's padded block (runtime bounds check PASS on real stores).
(c) per fold, scored runs ∩ fold-train = 0, ∩ inner-val = 0, ∩ lockbox = 0
(all five folds, frozen lists). (d) thresholds from pooled dev OOF only
(thr_at_fpr over runs240). CLEAN.

**5. Per-config SYNTH + HIGH-BG + per-category + best-epoch:** in
`reports/c1_perconfig.md`. Summary — HIGH-BG (20 % highest-bg dev runs, dev
thrs): c64s d 0.4225 vs v1 0.1124 at .085 (wins at every target, and its
slice-fpr 0.087 < v1's 0.109). SYNTH (frozen transform): c64s FP inflation
5.89× vs v1 25.79× on the same fold; self-cal d on synth fold c64s 0.0616 vs
v1 0.0546 @.085 (ours >= comparator at all 4 targets, +0.007 at .085 is within
noise — the synth fold is pessimistic vs portal band by design, E-staged for
relative comparison only).

## 5. FP calibration + failure examples (c64s)

- A4 calibration: measured fpr/target = 0.974 / 0.975 / 0.997 / 0.997 — all in
  [0.8, 1.2] (by construction of thr_at_fpr + audit confirms).
- FP distribution @.35: 178/240 runs have ZERO FPs; max 5/run; top-5 runs hold
  14 FPs (18 %); top-3 share 0.14 — not pathological concentration.
- Misses @.35: 1005/2160 encounters missed; worst isotopes (missed/total):
  NatU 155/253, FGPu 96/253, Ra-226 72/84, LEU 71/253, Th-232 62/85,
  RefinedU 62/253, WGPu 59/252, DU 58/249 — weak/low-energy and near-background
  classes. Missed-SNR p10/50/90 = 2.55/6.48/9.08 vs detected 3.55/7.46/13.6:
  misses are low-SNR, as expected. Sample FP alarms: real-looking spectra on
  background seconds (F-18 medical waste, K-40 nature bumps, FGPu/HEU
  mislabels on bg peaks) — the iso head is the FP story at these thresholds.

## 6. Acceptance A1-A4 (c64s vs v1, pooled dev OOF) — plain verdict

- **A1 PASS**: paired Δd CI lower bound > 0 at all targets (+0.299/+0.340/
  +0.361/+0.383); effect is ~3× the CI width — not noise.
- **A2 PASS** (evidence class noted): HIGH-BG slice Δd ≈ +0.31..+0.38 everywhere
  and slice fpr below v1 at 3/4 targets; SYNTH fold: ours ≥ v1 at all 4 targets
  (self-cal and dev-thr) with 4.4× less FP inflation. No bootstrap CIs on the
  shift folds (report-only per amendment) — margins are far larger than any
  plausible CI except the +0.007 synth self-cal point at .085, which is a tie.
- **A3 PASS**: NORM d 0.064 vs v1 0.0049 (above, not below; NORM still the
  hard class — 13/204 detected); NuclearMaterial 0.503 vs 0.119 (far above).
- **A4 PASS**: measured FP/hr within [0.8, 1.2]× of every target; distribution
  audited above.

## 7. Honest reservations

1. Dev OOF 0.4435 at .085 is optimistic for testing: diagnostic 3 says alarm
   emission at the SAME thresholds is 0.45× on testing; diagnostic 2 says the
   fastest dev tercile is already at 0.42. Expect a materially lower testing d
   unless Stage 2 closes the speed/short-window gap. v1 precedent (0.65×
   detection on testing) is the floor to beat, not dev numbers.
2. FP inflation 5.89× under synth (and rising d-vs-fpr flattening) means
   dev-fpr calibration ≠ testing fpr; Stage 4 threshold work is mandatory.
3. id_recall 0.288 at .085 — identification lags detection; fine vs v1
   (0.076) but the iso head is FP-prone on background bumps.
4. Selection noise: c64s vs c64h at 0.085 within CI; decision rests on the
   3/4-target sweep dominance + NORM. No further config hunting planned.
5. The c10.dll SYNTH crash (E036/37) was environmental; all numbers here come
   from completed, exit-0 steps with the launcher (identical code paths).

## 8. Proposed next (ONLY on user "Proceed")

Stage 2 augmentation (the main lever): speed-compress + thin to testing SNR
range + NORM/bg shifts, gated by G1-G3 realism + re-run of diagnostics 2/3
logic; SYNTH strength calibrated so v1 FP inflation ≈ 7.3× (portal, already
disclosed). Target: lift the fast/short-encounter tercile and testing emission
ratio. C2 checkpoint after gate results. No lockbox, no testing CSV before C4
approval.
