# Measurement file manifest -- submission_v2_c64s_a_thr0p8617.csv

Suggested submission NAME (portal field): `submission_v2_c64s_a_thr0p8617` (file: submission_v2_c64s_a_thr0p8617.csv).

Scheme (a): run r scored ONLY by fold model `r % 5` (best_state checkpoint); features = cached X from feat_test_diag.h5 (unlabeled count stream of testing_v4.3.h5). Threshold 0.8617 = dev-OOF .085 target, frozen before any testing was touched. Alarms: frozen extraction (gauss sigma 1.5 s, NMS sep 15 s, PMIN 0.02, cap 300/run); rows = alarms with metric >= 0.8617.

**metric_1** = gauss_smooth(sigmoid(detector-head logit), sigma=1.5 s) at the alarm second (the exact quantity thresholded; monotone in confidence). **time** = window centre (j+0.5) s in ms; **time_start/stop** = time -+ 2000 ms clamped to [0, nb*1000] (per-second grid end; detector-coverage convention of v1 run_lengths_ms). label_1 = that model's 24-name iso-head argmax at the alarm second; label_2/3 empty, metric_2/3 = 0.

## (1) assignment + leakage
assignment table: reports/measurement_assignment.txt (300 rows), per-model run counts {0: 60, 1: 60, 2: 60, 3: 60, 4: 60}.
provenance: feat.h5 = dev-file cache, asserted to hold exactly the 300 dev-file runs (md5 f66671ac6a3d4ab2...) = 240 pool + 60 SEALED lockbox; models trained on pool-240 splits only (fold leakguard.log + check-4 audit: scored/inner-train/lockbox disjointness 5/5 PASS); testing scoring reads ONLY feat_test_diag.h5 (testing-file features). Numerical run-id overlap testing-vs-pool ids: 300 ids collide by number -- they are DIFFERENT recordings in a different file (id = index within file); no testing recording is in any training split. Nothing in this pipeline loads testing into training (s1_train/s1_data take feat.h5 only).

## (2) content stats
rows 727 (2.42/run); rows/run min 1 max 7; runs with 0 alarms: 29 [0, 69, 116, 125, 136, 150, 155, 161, 162, 165, 170, 174]...
time ms min 23500 max 3587500; run_len_ms min 3600000 max 3717000; all rows inside run: True; time_start clamped 0, time_stop clamped 0
metric_1 min 0.861889 max 0.983618; NaN required cols 0, NaN metric_2/3 0; empty label_2/3 cells 1454 (only allowed empties)
label histogram: FGPu=119, HEU=115, DU=97, LEU=70, RefinedU=53, WGPu=52, Cs-137=42, Ir-192=33, Co-60=32, Ba-133=29, Co-57=17, NatU=17, Am-241=13, F-18=12, I-131=11, K-40=11, Xe-133=2, Th-232=1, Tl-201=1
md5: e41c68e31905223ef26bbd274a3f8f49

## (3) official-reader emulation
pd.read_csv(path, comment='#') OK; dtypes: #:int64, run_id:int64, time:int64, time_start:int64, time_stop:int64, label_1:object, metric_1:float64, label_2:float64, metric_2:int64, label_3:float64, metric_3:int64
read WITHOUT comment= gives identical columns/header (quoted '"#"' survives): True

## (4) nestedness (single file: subset property vs higher thrs)
rows of this file at metric >= t == alarms generated directly at >= t: t=0.87: equal=True, rows=667; t=0.88: equal=True, rows=592 (NMS is threshold-independent -> lower-thr file is a superset)

## (5) determinism
two INDEPENDENT scoring passes in-process: alarm lists equal=True (PMIN-level counts 60129/60129), CSV md5 equal: True (e41c68e31905223ef26bbd274a3f8f49). cudnn deterministic flags set. Cross-process evidence: this script's per-model scoring is the same code path s1_diag2 ran, whose ensemble reproduced check-3 counts EXACTLY in a separate process (1.93/2.21/2.58/2.98) -- no nondeterminism observed anywhere.

## (6) predicted outcome vs portal return
rows 727 -> d_recall <= 0.264 (enc ~2750, user's) / 0.269 (enc ~2700, my dev-density estimate)

| scenario | expected FP | fpr (=FP/286.5 portal den) | bucket |
|---|---|---|---|
| 1x (dev-like) | 24 | 0.083 | <0.125 |
| 3x | 71 | 0.249 | <0.25 |
| 7.3x (v1 inflation) | 173 | 0.605 | <1.0 |

Reading: returned fpr ~0.08 => dev-like FP; ~0.25 => 3x; ~0.60 => v1-level 7.3x inflation (counts 24/71/174 discriminate to ~x1.3). Returned d vs the 0.263 row-cap shows how much of the 0.56 emission ratio is true detection loss vs threshold position.

NOT uploaded by this script. No other threshold files written.
dev measured fpr at thr: 0.0828; PMIN-level alarms pass1 60129, rows>=thr 727