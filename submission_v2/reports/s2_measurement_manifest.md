# Stage-2 measurement manifest -- submission_v2_s2-spd_a_thr0p8536.csv

SELECTED CONFIG (pre-registered rule, selection.json): **s2-spd** -- rationale: max mean real fast-tercile d among ['s2-spd']

Scheme (a) r%5 single model; thr 0.8536 = own dev .085 target (frozen pre-testing); frozen alarm extraction; csv_format.py; +-2 s spans. Suggested portal name: `submission_v2_s2-spd_a_thr0p8536`.

## (1) assignment + leakage
300 testing runs, per-model counts {0: 60, 1: 60, 2: 60, 3: 60, 4: 60}; feat.h5 asserted == 240 pool + 60 sealed lockbox of DEV file; testing features only from feat_test_diag.h5; fold0.pt md5 8abd096bde7ecd9b...; model trained Aug-store folds per s2 protocol (leakguard.log).

## (2) content stats
rows 683 (2.28/run); rows/run min 1 max 7; zero-alarm runs 29; time ms min/max 24500/3598500; all inside run: True
NaN required 0, NaN metric_2/3 0; md5 c04e558df82940c3881b2f34e8de04bf

## (3) official-reader emulation
read comment='#' OK; identical columns without comment= (quoted header survives): True; dtypes #:int64, run_id:int64, time:int64, time_start:int64, time_stop:int64, label_1:object, metric_1:float64, label_2:float64, metric_2:int64, label_3:float64, metric_3:int64

## (4) nestedness
t=0.8616: True (627 rows); t=0.8736: True (546 rows)

## (5) determinism
two passes: alarms equal=True, csv md5 equal=True

## (6) predicted outcome vs portal anchor #2 (c64s)
dev @ .085: d=0.4481 fpr=0.0828 | REAL fast-tercile d@.085=0.3606 | calibrated-synth @.085: ret=0.306 infl=0.47

| scenario | expected FP | expected fpr | expected d |
|--|--|--|--|
| 1x (dev-like) | 24 | 0.083 | 0.448 |
| portal-#2-like (x1.17 FP, x0.57 d) | 28 | 0.097 | 0.255 |
| row cap | 683 rows -> d <= 0.247 | | |

Anchor #2 actuals (c64s @ .8617): rows 727, d 0.253, retention 0.57, inflation 1.17.

NOT uploaded. This is the ONLY Stage-2 measurement CSV.