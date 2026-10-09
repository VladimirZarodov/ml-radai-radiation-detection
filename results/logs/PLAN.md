# RADAI v2 — PLAN: pre-registered protocol & acceptance criteria

FROZEN at Stage 0 (before any learned model is trained). Changes after this
point require an explicit note here marked AMENDED with date + reason, and any
amendment invalidates prior acceptances that depended on the changed criterion.

## Comparators (always reported together)

1. **v1-mx31 reproduced in this harness** (identical AK, alarm NMS, metric
   code) on the same folds — the baseline every change must beat.
2. Previous accepted v2 configuration (roving baseline).

## Primary metric

Global **d_recall at matched official fpr** on POOLED out-of-fold development
predictions (240 runs), thresholds swept on the pooled OOF score axis; the
comparison point is a small fpr grid {0.085, 0.17, 0.35, 0.7} (per-bucket
targets + midpoint). Secondary: c_recall, id_recall, per-category
d_recall (NORM / Medical / Industrial / NuclearMaterial) at matched fpr.

## Confidence intervals

Run-cluster bootstrap over runs, B = 1000, percentile 95 % CI. For
*differences* between two configurations, paired bootstrap (same resampled
run sets for both), CI of the difference.

## Acceptance rule (every model/augmentation/ensemble change)

ACCEPT iff ALL hold:
- A1. Δd_recall (ours − comparator) at the fpr grid point closest to the
  bucket we target (primary 0.085; also checked at 0.17 and 0.7) has paired
  bootstrap CI lower bound > 0 on pooled OOF; improvements smaller than the
  CI band are NOISE → reject or keep as tie.
- A2. No degradation on shift evaluations: on HIGH-BG slice and SYNTH-SHIFT
  folds, Δd at matched fpr CI lower bound > −0.005 (practical non-inferiority
  margin 0.5 pp global).
- A3. Per-category guard: NORM d_recall must not decrease below comparator's
  CI; NuclearMaterial (largest class) must not degrade beyond CI either.
- A4. FPR calibration audit: measured FP/hr at chosen threshold on OOF within
  [0.8×, 1.2×] of target (else threshold search redone, not model rejected).

REJECT → keep comparator; log the number in STATE.md (no silent discards).

## Validation hygiene (hard)

- Lockbox (60 runs, `cache/lockbox.json`): touched exactly once per finalist
  at Stage 4; no hyperparameter, threshold, fold model, or early-stopping
  decision may reference it. Any accidental use → that finalist is flagged
  contaminated in STATE.md and re-run clean.
- 5-fold dev CV by run (`cache/folds.json`); predictions strictly out-of-fold;
  per-fold models may not see other folds' labels or scores.
- Early stopping: inner split WITHIN the fold's training runs only
  (deterministic: last 15 % of runs sorted by run_id, same rule for all folds;
  recorded in fold config).
- Developer file GT may be used for training/augmentation only; its runs are
  never in dev/lockbox metrics (different campaign).
- Testing GT: does not exist; nothing infers or stores per-run testing labels.
  Unlabeled testing aggregate stats usable only in Stage 4 with disclosure.
- Alarm extraction params (NMS sep, smoothing) fixed per configuration before
  looking at shift/lockbox results; tuning only on pooled OOF with inner
  selection.

## Realism gates for augmentations (Stage 2)

An augmentation is TRUSTED only if, evaluated on held-out real data:
- G1. v1 mx31 detection-PD vs SNR curve of thinned/developer-source encounters
  lies inside the bootstrap band of real training encounters matched in SNR
  (logistic fit on SNR bins, B = 300).
- G2. Window geometry statistics (width, start−CA, stop−CA) of AK rebuilt on
  augmented synthetic runs stay within ±10 % of real means; rate statistics
  (median cps, p99 cps) shift no more than the designed factor.
- G3. Sum-to-total decomposition identities still hold after transform.
Failed gates → augmentation removed from the keep-set and reported.

## Stage-4 threshold protocol

- Target official Global fpr ≈ 0.085 / 0.17 / 0.7 (bucket edges 0.125/0.25/1.0
  with margin), plus safe variants (edge worst-case).
- Derived from pooled OOF FP/hr curve + measured FP inflation ratio on
  SYNTH-SHIFT folds (inflation multiplier applied with its CI), sanity
  anchored by portal measurement #1 (v1: predicted FP_val 21 → actual 153,
  inflation 7.29× per-file at thr 5.7492; factor band [5.0, 11.7]).
- Low-fpr thresholds explicitly reported as noisy: show point + band +
  alternative safe threshold for each bucket.
- ONE first upload proposed as a measurement; only after user approval write
  testing CSVs via `submission/csv_format.py`.

## Deliverables per checkpoint (stop-and-report)

C0: caches + verification + harness numbers for v1 (OOF/shift), lockbox ids.
C1: learned baseline OOF vs v1 (+shifts, per-category, CIs) + the 5
    diagnostics of the 2026-10-03 C1 amendment (time-shift control, speed
    terciles, unlabeled testing ratio, written leakage audit, per-config
    SYNTH/HIGH-BG/per-cat/best-epochs). Diagnostics run only AFTER the chain.
C2: augmentation ablation table (each item a–f alone + combo), realism gates.
C3: ensemble/stacking OOF numbers; final candidate list for lockbox.
C4: lockbox once; threshold proposal + first-upload file choice.

## AMENDED 2026-10-03 (pre-training, per user) — Lockbox statistical power

Measured on the frozen harness with the v1 comparator's alarms (numbers from
`submission_v2/plan_stats.py`, run 2026-10-03):

- dev pooled = 240 runs, **229.37 background-hours** (official denominator;
  corrects my earlier ~110 h estimate);
- lockbox = 60 runs, **57.45 background-hours**.

Expected FP count at a target = target × pool bg-hours; observed v1 FP counts
(dev self-selected thresholds; lockbox evaluated at the same dev-selected
thresholds):

| target fpr | dev FP | lockbox FP (v1 @ dev thr) | lockbox expected |
|---|---|---|---|
| 0.085 | 19 | 4 | ~4.3 |
| 0.17 | 38 | 12 | ~9.8 |
| 0.35 | 80 | 22 | ~20.1 |
| 0.70 | 160 | 45 | ~40.2 |

RULES (apply to the C4 lockbox comparison of finalists; A1–A4 on pooled OOF
unchanged):
- Lockbox **selection** uses only targets where the lockbox has ≳30 FPs:
  **0.70 is primary (45 FP observed); 0.35 reported as secondary**
  (~20–22 FP — borderline, treated as indicative only).
- Lockbox results at **0.085 and 0.17 are REPORTED but MUST NOT drive
  selection** (≤ ~4–12 FPs: a single alarm moves the metric; CI bands, if
  given, are wide and advisory).
- Acceptance on lockbox = same direction + non-inferiority logic as A1/A2 but
  evaluated at the selection targets, with the paired run-cluster bootstrap
  over lockbox runs (B = 1000).
- Any finalist comparison that would have flipped by using 0.085/0.17 lockbox
  numbers is explicitly invalid.

This amendment does not invalidate prior acceptances (no model trained yet;
C0 involved no lockbox evaluation).

## AMENDED 2026-10-03 (during Stage-1 chain, per user) — Checkpoint-1 diagnostics

Selection protocol UNCHANGED (pooled dev OOF at pre-registered targets decides
A1–A4; lockbox rules as above). The user added five **diagnostics-only** items
to the C1 package, to be run only AFTER the training chain completes (no GPU
work concurrently with the chain; no tuning, no config changes):

1. Time-shift control: pooled OOF alarms re-scored with every alarm time
   shifted +/-400 s (out-of-run alarms DROPPED, not wrapped - stated policy);
   report d and fpr at the four dev thresholds; detection must collapse
   (guards matching/scoring artefacts).
2. Speed/duration sensitivity: dev OOF encounters split into terciles by
   platform |velocity| at closest approach (exists in the h5 as
   detector/position/velocity) - proxies: official window width_s, distance
   at CA; report Global d/c/id per tercile at the fixed 0.35/0.70 dev
   thresholds. Recall drop vs speed/short-window = proxy for the testing
   shift; informs WHICH Stage-2 augmentation matters (decided at C2, not now).
3. Unlabeled testing diagnostic (fold-ensemble of the best config on the 300
   testing runs): alarms/hour testing vs dev-OOF at the four dev thresholds,
   per-run dispersion, score quantiles; v1 precedent ratio ~1.2-1.3x; much
   larger = warning. Uses only the unlabeled count stream; no labels/answer
   keys/CSV; MUST be disclosed in README as testing-derived, diagnostics only.
4. Written leakage audit: (a) no input/normalisation uses sources/*, listmode
   id/background_id or answer keys (per-run median normalisation = run's own
   totals only); (b) post-E030 fix, no window reads outside its own run;
   (c) per fold, by run-id lists: OOF-scored runs absent from that fold's
   training AND its inner-validation split; (d) fpr-target thresholds derive
   from pooled OOF only.
5. Per config: SYNTH + HIGH-BG (report-only), per-category d/c/id, best epoch
   per fold; comment on the early best-epochs (2-4 of 12) incl. whether
   cosine schedule/LR plausibly matter - change NOTHING.

Corrected caveat (user): +/-64 s same-run context is legitimate (available on
testing, uses only the run's own counts) and does NOT flatter dev. The real
risks: distribution shift - testing platform speeds up to 13.4 m/s vs 8.0
training (encounters shorter than anything seen), +17% count rate, +20% NORM -
and FP inflation (v1: 7.29x FP, 0.65x detection on testing).

Implemented in `submission_v2/s1_diag.py` (modes timeshift/speed/testing/
leakaudit); prepared during the chain, executed only after CHAIN2 COMPLETE.

## AMENDED 2026-10-04 (post-training, portal measurement #2) — SYNTH/shift calibration RETARGETED

User decision after measurement #2 ("second_try" = c64s single-model file,
scheme r%5, thr 0.8617; portal row recorded in `submission_v2/portal_results.md`):

- **RETIRED** for the network: "calibrate the synthetic shift so v1's FP
  inflation ~7.3x is reproduced" (that anchor is v1-method-specific; the
  portal shows the NETWORK's measured inflation is 1.17x [0.74, 1.61] with
  detection retention 0.570 [0.525, 0.616]).
- **NEW calibration target (hard gate before any Stage-2 training):** tune
  the existing synthetic-shift generator on the FROZEN c64s fold models
  (eval-only, single-model scoring on dev OOF, no training) so that BOTH
  portal numbers reproduce simultaneously:
    retention  d_synith/d_plain  in [0.52, 0.62]   (point 0.570), AND
    inflation  fpr_synith/fpr_plain in [0.80, 1.60] (point 1.17).
  Parameters (grid, pre-declared): speed-compression factor
  kappa_max in {1.3, 1.5, 1.7} applied to a random half of encounters
  (kappa ~ U[1, kappa_max]; testing platform ratio 13.4/8.0 = 1.68 bounds
  the top cell); source-count thinning theta in {1.0, 0.7, 0.5};
  background rate x1.17 and NORM-component x1.2 FIXED from the organizer
  README. If no grid cell hits both, closest retention wins subject to
  inflation in [0.7, 1.8]; chosen cell + measured pair are logged BEFORE
  training starts. Only testing-derived inputs to this calibration are the
  portal aggregate row (disclosed, README section E).
- **Primary shift metric** for Stage-2 selection: detection retention under
  the calibrated shift (global + per-category), replacing SYNTH inflation vs
  v1. Gates A1-A4 unchanged; A2's SYNTH clause now means the CALIBRATED
  synth (both numbers above as acceptance bands for the trained configs,
  tolerances widened by each config's CI).
- Threshold-transfer rule (Stage 4): single-model scoring, run_id % 5, dev
  thresholds meaningful within retention/inflation bands pinned by this
  anchor; any additional testing stat must be disclosed first.


## AMENDED 2026-10-06 (pre-Stage-2-training, per user brief) - One-factor augmentation ablation

Protocol for Stage 2 (supersedes the 4-config ladder of STAGE2_DESIGN.md; its G1-G3
gates stay in force). Full working spec in S2_NOTES.md. Five learned configs, c64s
recipe identical (arch/feat/targets/soft sigma 6 s/loss/folds/12 ep/patience 3/batch
192/NEG_RATE .12/frozen alarm extraction); only the training pool differs:
  s2-0 (new seed, no aug - noise floor), s2-spd (+time-compress kappa~U[1,1.7] on
  Bernoulli(.5) of encounters), s2-thin (+binomial source thinning t~U[theta_aug,1]),
  s2-bg (+bg Poisson-redraw f~U[1.0,1.34] mean 1.17, NORM comps x1.2), s2-all (all 3).
theta_aug = Step-0 calibrated theta if <1, else 0.5 (fallback logged).
Augmentation is applied per-run (own counts only) to FOLD-TRAIN runs including their
inner-val 15 % subset; fold-val runs and OOF scoring use REAL data; lockbox excluded
from every build by the leak guard. Multi-instance source ids (15 same-id CA gaps
<301 s) are transformed per-instance on nearest-instance-masked segments so energy
hoods and all-energy sec series stay consistently assigned.
Seeds: SEED2=20261006; seed(cfg,fold)=(20261006*7919+fold*101+4*7907+31337)%2**31 for
ALL s2 configs (identical init across configs; differs from c64s only by base seed).
Aug RNG: default_rng([20261006, cfg_slot, run]); eval-synth RNG: default_rng([20261007,
split_id, run]) - disjoint streams (anti-circularity).

Step 0 (eval-only, <=1.5 GPU-h): calibrate synth generator (same functional form,
kappa_max{1.3,1.5,1.7} x theta{1.0,0.7,0.5}, bg 1.17/NORM 1.2 fixed) on FROZEN c64s,
single-model r%5 scoring at fixed thr 0.8617: screen 9 pts on folds 0+1 (106 runs),
confirm chosen point on all 240; target retention in [0.52,0.62] AND inflation in
[0.80,1.60]; if none: closest retention with inflation in [0.7,1.8], log, CONTINUE.
Out-of-range point kappa_max=2.0 (calibrated theta) always evaluated for reporting.

Unit tests before ANY augmented training: (T1) identity params (kappa=1,t=1,f=1 fast
path = verbatim arrays) reproduce feat.h5 X/y/ys/tcat/tiso AND window rows
BIT-IDENTICALLY on >=30 pool runs; (T2) transformed runs: rebuilt windows by official
ak_window on the augmented all-energy SNR profile, y == official TP rule on rebuilt
windows, ys geometry consistent, in-view identities exact (bg sum, total sum, rate
definitions).

Metrics per config (pooled dev OOF, own thresholds at .085/.17/.35/.70): Global+per-cat
d/c/id; REAL fast/slow speed-tercile d and ratio at 4 targets (tercile edges frozen
4.4/6.8 m/s from E039); retention + FP inflation under calibrated shift (in-range AND
kappa 2.0) at own .085 thr; Medical c/d & id/d watch; cost (min/fold, peak VRAM, best
epochs). Paired run-cluster bootstrap B=1000 vs c64s and vs s2-0 at each target.
CLAIM rule: effect is claimed only if paired CI excludes 0 AND the point diff exceeds
the noise floor NF(target)=|d(c64s)-d(s2-0)| at that target.

SELECTION (pre-registered; no rule shopping after results): candidate iff paired diff
vs s2-0 at targets 0.35 AND 0.70 has CI lower bound >= -0.01. Among candidates pick
max mean of REAL fast-tercile d over the four targets; tie-break retention@.085 under
the calibrated shift. If no candidate passes, the finding is "augmentation did not
help" and is reported as such. Reserve (only if GPU ledger <=12 h projection and the
ablation is interpretable): second seed of the selected config OR WGT (x2 loss on
short windows) on top of it; total learned configs <=6.

End-of-stage: exactly ONE measurement CSV for the selected config (its own dev .085
threshold, single-model r%5 scoring, frozen extraction, csv_format.py, +-2 s, quoted
'#' header) + full verification suite + predicted-outcome table vs portal anchor #2;
NO upload, no other CSVs. Lockbox: script + dev dry run prepared only; run requires
separate user authorization. Budget: BUDGET.md ledger, HARD STOP 15 GPU-h.
