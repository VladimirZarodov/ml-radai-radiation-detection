# C1 follow-up task 3 — portal MEASUREMENT-file scenario table (c64s)

Recommended scoring (task 2): **one fold model per testing run, assignment
run_id % 5** (fixed rule, disclosed). Under this scheme the testing mixture
behaves like the pooled dev OOF calibration distribution (single model scoring
runs it never saw), so rows/h at a dev threshold ≈ the **mean per-model**
testing rate measured in reports/testing_diag2.md.

Denominators: testing raw recording 301.4 h (unlabeled -> no window
subtraction). Estimated testing official-style bg-hours ~= 288 h (dev pool
shows windows consume ~4.6 % of raw hours; assumption disclosed; note the
scenario fpr below is denominator-insensitive because FP/hour and fpr differ
only by that ~4.6 %).

Encounters estimate for the d upper bound: dev density 2160/240 = 9.0/run
=> ~2700 (user's 2750 within estimate noise); bounds shown vs 2750 AND 2700.

Scenario FP inflation: 1x = FP/h as measured on dev OOF at that threshold;
3x; 7.3x = v1's measured testing inflation (worst case). FP count =
dev_fpr(thr) x mult x 288 h. fpr_est ~= dev_fpr(thr) x mult.

| thr | dev fpr | rows/h | rows total | d upper (rows/2750 | /2700) | FP/fpr @1x | @3x | @7.3x | fpr<1.0 ALL scenarios? |
|---|---|---|---|---|---|---|---|---|
| 0.8800 | 0.0392 | 1.93 | 582 | 0.212 / 0.216 | 11 / 0.039 | 34 / 0.118 | 83 / 0.286 | YES |
| **0.8617 (dev .085 thr)** | 0.0828 | 2.40 | 723 | **0.263 / 0.268** | 24 / 0.083 | 71 / 0.248 | 174 / 0.604 | **YES (worst 0.60)** |
| ~0.855 (interp) | ~0.116 | ~2.56 | ~771 | ~0.280 / ~0.286 | ~34 / ~0.116 | ~103 / ~0.35 | ~250 / ~0.85 | YES (borderline 0.85) |
| ~0.851 (bucket ceiling) | ~0.136 | ~2.63 | ~792 | ~0.288 / ~0.293 | ~39 / ~0.136 | ~117 / ~0.41 | ~285 / ~0.99 | AT LIMIT |
| 0.8449 (dev .17 thr) | 0.1657 | 2.79 | 841 | 0.306 / 0.311 | 48 / 0.166 | 143 / 0.497 | 348 / 1.210 | NO @7.3x |
| 0.8234 (dev .35 thr) | 0.3488 | 3.29 | 992 | 0.361 / 0.367 | 100 / 0.349 | 301 / 1.046 | 734 / 2.546 | NO @3x,7.3x |
| 0.7961 (dev .70 thr) | 0.6976 | 4.02 | 1212 | 0.441 / 0.449 | 201 / 0.698 | 603 / 2.093 | 1467 / 5.093 | NO |

(interp rows: linear between the two computed grid points above/below; the
computed grid in testing_diag2.md is exact.)

## Reading

- Bucket-safe (fpr < 1.0 in ALL THREE scenarios): thr >= ~0.851. The clean
  pre-registered choice **0.8617** is safely inside even at 7.3x inflation
  (worst-case fpr 0.60, ~174 FP, ~723 rows, d <= ~0.26).
- **Most informative safe choice**: thr ~= 0.855 (d upper ~0.28, ~768 rows,
  worst-case fpr 0.88) — buys +45 rows over 0.8617 but eats margin; a 0.851
  file would leave NO room for error.
- Information content: portal returns d/c/id + fpr + per-category. matched
  ~= rows - FP. At 0.8617 the matched count will discriminate the scenarios:
  fpr measured ~0.08 vs ~0.25 vs ~0.60 pins the inflation multiplier to
  within a factor ~1.3 (N(FP)~24/71/174 -> sd~5/8/13), and d_recall tells us
  how much of the 0.56 emission ratio is real detection loss vs threshold
  position. 2750-encounter denominator => 1 encounter = 0.0004 of d.
- Expected NORM informativeness: testing NORM encounters ~ 300x(204/240)x1.2
  ~ 306; dev NORM d=0.064 x emission-retention 0.56 => ~11 expected NORM
  detections — measurable but thin; a lower threshold mainly helps NORM
  (~20 at 0.8234, d=0.118 there) at the cost of bucket risk.
- If inflation comes back ~1-3x, the natural next measurement file uses a
  lower thr (0.8234 band) — but that is a decision FOR AFTER this upload,
  not now.

## Not done (per instruction)
No CSV written. Scoring assignment/threshold for any real upload will be
re-stated in README + STATE before generation, and generation goes through
the frozen csv_format path only on explicit approval.
