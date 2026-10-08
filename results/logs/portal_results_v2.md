# Portal measurement #2 -- c64s single-model file, thr 0.8617
(uploaded by USER 2026-10-04 as "second_try"; row pasted by user)

## 1. Record + integrity
File on disk: submission_v2_c64s_a_thr0p8617.csv | md5 `e41c68e31905223ef26bbd274a3f8f49` | rows 727 | zero-alarm runs 29.
Identity note: portal name was 'second_try' (renamed at upload), so byte-identity with the disk file cannot be cryptographically confirmed; arithmetic below is consistent ONLY with this file: an integer denominator fit gives FP counts {'Global': 28, 'Industrial': 6, 'Medical': 1, 'NORM': 1, 'NuclearMaterial': 20} (sum check: 28 == 28 global) with residuals [0.0, -0.014, 0.007, 0.007, 0.0] on den ~287.77 h (admissible window [287.6, 288.0] h). TP = rows - FP = 699; d=0.2530 (4dp) forces N_enc =
**unique integer [2763]** (2762 rounds to 0.2531) -- matches the ~2750-2763 band from dev encounter density.

Portal row (categories x metrics): table as provided by user: Global d .2530 c .2278 id .1625 fpr .0973 | Industrial .3370/.2932/.2910/.0208 | Medical .1555/.0670/.0605/.0035 | NORM .0482/.0482/.0482/.0035 | NM .2866/.2816/.1716/.0695. Leaderboard: 3rd in low_false_alarms (<participant-A> .3662@.0868, baseline_nmf .2979@.1222).

## 2. Derived quantities (with uncertainties)
- FP counts (fit): Global 28 (+-5.3 Poisson), Ind 6, Med 1, NORM 1, NM 20
- matched rows = 727 - 28 = 699; implied precision = 0.961 (+-0.014 95%) vs dev OOF precision 958/987 = 0.971
- implied testing encounters = 699/0.2530 = 2763 (+-95% ~ +-180 via binomial on d and the 4dp rounding; unique integer consistent with rounding is 2763)
- detection retention = 0.2530/0.4435(dev OOF @ same thr) = **0.570** 95% CI [0.525, 0.616]
- FP inflation = 0.0973/0.0828 = **1.17x** 95% CI [0.74, 1.61] (testing-Poisson only; adding dev FP=19 sampling widens to [0.49, 1.86])

## 3. vs scenario table (predicted FP/fpr: 24/.083, 71/.249, 173/.605)
observed 28/0.0973 -> 1x scenario z = +0.8 sigma; 3x z = 8.2; 7.3x z = 27.6. PLAIN WORDS: FP inflation is ~1x (dev-like); the 3x and 7.3x v1-style scenarios are REJECTED for this network at this threshold. Detection emission/retention ~0.56-0.57 (alarm ratio 2.40/4.31=0.557 vs matched-d retention 0.570): the testing loss is ALMOST ENTIRELY missed true encounters, not FP flood; precision held (0.961 vs dev 0.971).

## 4. Per-category vs dev OOF @0.8617 (dev: single own-fold model)
| cat | dev d | test d | retention | dev c/d | test c/d | dev id/d | test id/d | dev FP | test FP |
|---|---|---|---|---|---|---|---|---|---|
| Global | 0.444 | 0.2530 | 0.57 | 0.96 | 0.90 | 0.65 | 0.64 | 19 | 28 |
| Industrial | 0.599 | 0.3370 | 0.56 | 0.86 | 0.87 | 0.86 | 0.86 | 0 | 6 |
| Medical | 0.260 | 0.1555 | 0.60 | 0.93 | 0.43 | 0.87 | 0.39 | 1 | 1 |
| NORM | 0.064 | 0.0482 | 0.76 | 1.00 | 1.00 | 1.00 | 1.00 | 1 | 1 |
| NuclearMaterial | 0.503 | 0.2866 | 0.57 | 0.99 | 0.98 | 0.57 | 0.60 | 17 | 20 |

Retention is UNIFORM (~0.56-0.75): detection loss scales everything; the ONE qualitative break is Medical c/d 0.93->0.43 (id/d 0.87->0.39): medical detections happen but land on wrong labels/categories. Industrial id/c 0.99 both sides; NM id/c 0.58/0.61. NORM: d=c=id (both single-ish label world), retention 0.75.

## 5. Predicted-label mix: testing CSV vs dev OOF @same thr (unlabeled, disclosed)
dev alarms 988 vs test rows 727. (labels absent from TRAINING AK: Cu-67, Lu-177 [never seen]; also compare rare ones)
| label | cat | dev pool enc | dev alarms | dev % | test alarms | test % | test%/dev% |
|---|---|---|---|---|---|---|---|
| Am-241 | Industrial | 65 | 10 | 1.0 | 13 | 1.8 | 1.77 |
| Ba-133 | Industrial | 32 | 19 | 1.9 | 29 | 4.0 | 2.07 |
| Co-57 | Medical | 65 | 28 | 2.8 | 17 | 2.3 | 0.83 |
| Co-60 | Industrial | 33 | 18 | 1.8 | 32 | 4.4 | 2.42 |
| Cs-137 | Industrial | 72 | 56 | 5.7 | 42 | 5.8 | 1.02 |
| Cu-67 | -(untrained) | 0 | 0 | 0.0 | 0 | 0.0 | DEV-ZERO |
| DU | NuclearMaterial | 196 | 138 | 14.0 | 97 | 13.3 | 0.96 |
| F-18 | Medical | 31 | 11 | 1.1 | 12 | 1.7 | 1.48 |
| FGPu | NuclearMaterial | 211 | 137 | 13.9 | 119 | 16.4 | 1.18 |
| HEU | NuclearMaterial | 197 | 130 | 13.2 | 115 | 15.8 | 1.20 |
| I-131 | Medical | 30 | 24 | 2.4 | 11 | 1.5 | 0.62 |
| Ir-192 | Industrial | 67 | 45 | 4.6 | 33 | 4.5 | 1.00 |
| K-40 | NORM | 67 | 14 | 1.4 | 11 | 1.5 | 1.07 |
| LEU | NuclearMaterial | 207 | 140 | 14.2 | 70 | 9.6 | 0.68 |
| Lu-177 | -(untrained) | 0 | 0 | 0.0 | 0 | 0.0 | DEV-ZERO |
| NatU | NuclearMaterial | 203 | 34 | 3.4 | 17 | 2.3 | 0.68 |
| Ra-226 | NORM | 72 | 0 | 0.0 | 0 | 0.0 | DEV-ZERO |
| RefinedU | NuclearMaterial | 206 | 68 | 6.9 | 53 | 7.3 | 1.06 |
| Sr-90 | Medical | 33 | 0 | 0.0 | 0 | 0.0 | DEV-ZERO |
| Tc-99m | Medical | 33 | 0 | 0.0 | 0 | 0.0 | DEV-ZERO |
| Th-232 | NORM | 65 | 0 | 0.0 | 1 | 0.1 | DEV-ZERO |
| Tl-201 | Medical | 36 | 1 | 0.1 | 1 | 0.1 | 1.36 |
| WGPu | NuclearMaterial | 202 | 110 | 11.1 | 52 | 7.2 | 0.64 |
| Xe-133 | Medical | 37 | 5 | 0.5 | 2 | 0.3 | 0.54 |

| predicted category share | dev % | test % |
|---|---|---|
| Medical | 7.0 | 5.9 |
| Industrial | 15.0 | 20.5 |
| NuclearMaterial | 76.6 | 71.9 |
| NORM | 1.4 | 1.7 |
| ? | 0.0 | 0.0 |

What this CAN show: (i) LABEL BLINDNESS THAT ALREADY EXISTS ON DEV: Tc-99m, Sr-90, Ra-226 (and near-zero Th-232) produce NO alarms on dev OOF despite 33/33/72/65 pool encounters - Sr-90 is a beta emitter (invisible to photon counts), and the 140-keV Tc-99m / weak-line Ra-226 were effectively never learned as detections; dev Medical TP come from Co-57/F-18/I-131/Tl-201/Xe-133 only. (ii) The testing OUTPUT mix shifts at category level only mildly (Medical-labeled alarms 7.0%->5.9%), but individual industrial labels over-fire 1.8-2.4x relative to dev (Am-241 1.8, Ba-133 2.1, Co-60 2.4, Cs-137 flat) while LEU/WGPu/NatU under-fire (0.64-0.68) - consistent with a different true source population. Combined with portal (Medical test c/d 0.43 vs dev 0.93 at d-retention 0.60) it proves matched Medical detections on testing are largely LABELED OUTSIDE medical classes. What it CANNOT show: the true testing composition or per-isotope ground truth (no labels); whether Cu-67/Lu-177 (absent from training entirely) dominate testing Medical - leading hypothesis, not a measurement.

## 6. Task-3 re-pin: predictions at lower dev thresholds (single-model scoring, r%5)
Assumptions: ONE anchor point; retention and inflation treated as CONSTANT across thresholds (tail behaviour at lower thrs NOT verified; v1 taught that inflation can grow at the low tail); row rates are MEASURED (diag2), only FP/d are modelled.
| thr (dev target) | rows (measured) | fpr pred [95%] | bucket central | d pred [95%] | matched check |
|---|---|---|---|---|---|
| 0.8449 (0.17) | 841 | 0.195 [0.123, 0.267] | <0.25 | 0.281 [0.241, 0.321] | 0.284 |
| 0.8234 (0.35) | 992 | 0.410 [0.258, 0.561] | <1.0 | 0.305 [0.263, 0.347] | 0.317 |
| 0.7961 (0.7) | 1212 | 0.819 [0.516, 1.123] | <1.0 | 0.327 [0.283, 0.371] | 0.353 |

vs leaderboard: even the dev-0.70 threshold projects d ~0.33 (< <participant-A> 0.3662) and fpr ~0.82 (worst bucket). To BEAT <participant-A> in the LOW bucket (fpr<0.125) needs dev-OOF d >= 0.366/0.57 = 0.64 at the .085 threshold (now 0.4435): Stage-2/3 must raise intrinsic dev d by ~+0.20 abs -- threshold games cannot close that.

## 7. Runtimes (measured, for Stage-2 budget)
c64s: 40 fold-epochs x ~66 s = 47 min train GPU; fold .done wall-clock spans per config (min, includes OOF eval): c64s~69.9, c64h~31.2, c128s~84.3, c128h~56.9 (sequential-chain estimate).


# Portal measurement #3 -- s2-spd single-model file, thr 0.8536
(user uploaded 2026-10-06 as "third_try"; FILE IDENTITY CONFIRMED by user;
aggregate row parsed from `leaderboard.html` fetched by user 2026-10-06 23:32,
username <participant-B>, submission_time 2026-10-06T19:24:05)

## 1. Record + integrity
File on disk: submission_v2_s2-spd_a_thr0p8536.csv | md5
`c04e558df82940c3881b2f34e8de04bf` | rows 683 | re-verified this session by
direct read (identical to Stage-2 manifest, reports/s2_measurement_manifest.md,
and to the two byte-identical regenerations of E056).
Unlike #2, identity rests on BOTH user confirmation and the arithmetic fit
below; not cryptographically confirmable from the portal side.
Portal row: Global d .2402 c .2150 id .1464 fpr .0730 | Industrial
.2713/.2254/.2144/.0035 | Medical .1533/.0842/.0778/.0035 | NORM
.0088/.0088/.0088/.0069 | NuclearMaterial .2898/.2797/.1666/.0591.

## 2. Derived quantities (integer fit, same method as #2; den 287.77 h)
Implied FP counts (unique integer candidates at displayed rounding):
**Global 21, Industrial 1, Medical 1, NORM 2, NuclearMaterial 17** --
partition exact (1+1+2+17 = 21), no residual slack: strong corroboration of
file identity. TP = 683 - 21 = 662; implied testing encounters = unique
integer **2756** at d = 0.2402 (662/2756 = 0.240203).
DISCLOSED INCONSISTENCY vs #2: measurement #2 fit forced N_enc = 2763; at
2763 no integer TP gives 0.2402 (663/2763 = 0.2400, 664/2763 = 0.2403).
The two rows disagree by 7 encounters (0.25 % of N) -- organizer-side
rounding/matching policy at boundaries, not explainable from our side; both
retentions computed with each row's own fit (difference is << sampling noise).
Implied precision = 662/683 = **0.969** (dev OOF s2-spd 1075/1094 = 0.983;
#2 test 0.961). Emission 683/300 test-runs = 2.28/run vs dev 4.56/run ->
ratio 0.50 ~= d-retention 0.536: loss is missed detections, not FP flood
(same pattern as #2).
Retention vs s2-spd dev OOF @.085 (d 0.4481): **0.5360, 95 % CI [0.491,
0.585]** (binomial-on-testing x normal-on-dev parametric bootstrap, B=20k,
seed 20261006). c-retention 0.498, id-retention 0.493.
FP inflation = 0.0730/0.08283 = **0.881, CI [0.503, 1.259]** -- testing FP
rate at or BELOW dev; 1x scenario again supported; 3x/7.3x rejected
(Poisson p of >=21 FP under 3x (63 FP expected) is astronomically small --
same z argument as #2 section 3).

## 3. Comparison with second_try (c64s) -- honest noise statement
d: 0.2402 vs 0.2530 -> diff -0.0128; binomial SEs 0.0081/0.0083 -> even
treating the files as independent, |diff| = 1.1 sigma; the two models share
the same features/recipe and their portal scores are positively correlated,
so the effective SE of the difference is SMALLER -> **indistinguishable**.
fpr: 0.0730 vs 0.0973 -> -0.0243, ~1.0 sigma (independent upper bound) ->
noise. TP: 662 vs 699 (-37); rows: 683 vs 727 (-44); FP: 21 vs 28 (-7).
Neither the small d deficit nor the lower FP rate of third_try is a
measurable portal effect at 2756-encounter power. The Stage-2 fast-tercile
hypothesis (spd should retain MORE under the speed shift) got its direct
test here: it did NOT show up in the only real measurement of the shift;
at portal power a retention difference of <= ~0.07 in either direction
cannot be excluded (CI overlap [0.491,0.585] vs [0.525,0.616]).

## 4. Per-category test vs s2-spd dev OOF @.085 (thr 0.8536; dev cats from cache/s2/metrics_s2-spd.json)
| cat | dev d | test d | retention | dev c/d | test c/d | dev id/d | test id/d | dev FP | test FP |
|---|---|---|---|---|---|---|---|---|---|
| Global | 0.448 | 0.2402 | 0.536 | 0.964 | 0.895 | 0.663 | 0.610 | 19 | 21 |
| Industrial | 0.528 | 0.2713 | 0.514 | 0.873 | 0.830 | 0.873 | 0.790 | 0 | 1 |
| Medical | 0.275 | 0.1533 | 0.557 | 0.945 | 0.549 | 0.918 | 0.507 | 2 | 1 |
| NORM | 0.039 | 0.0088 | 0.224 | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 2 |
| NuclearMaterial | 0.524 | 0.2898 | 0.553 | 0.983 | 0.965 | 0.595 | 0.575 | 17 | 17 |
Same qualitative picture as #2: retention ~0.51-0.56 everywhere except NORM
(0.22 on 1/51-ish counts -- low-stat, dev NORM d was already 0.039) and the
REPEATED Medical label-accuracy break (c/d 0.945 -> 0.549, id/d 0.918 ->
0.507; #2: 0.93 -> 0.43). Two independent files, same failure mode: medical
encounters are detected but labeled outside Medical on the testing campaign.
This REINFORCES the #2 finding (not an artifact of one threshold/model).

## 5. Leaderboard placement (bucket low_false_alarms, fpr < 0.125, by d)
<participant-A> .3662@.0868 > <participant-C> .2979@.1222 > **second_try
.2530@.0973** > **third_try .2402@.0730** > baseline rows... third_try is
4th in the bucket and holds the lowest fpr among the top four. high bucket:
first_try (v1) .0946@.535. (Portal rows from leaderboard.html parsed this
session; full json in reports/lockbox/third_try.json.)

## 6. What #3 proves / does not prove
PROVES: the s2-spd file performs on the real testing campaign within the
same envelope as c64s (both ~0.54-0.57 retention, FP inflation ~0.9-1.2);
the 683-row/21-FP arithmetic is consistent ONLY with our prepared file.
DOES NOT PROVE: any ordering between s2-spd and c64s under shift (1-sigma);
anything about per-isotope testing truth (no labels); testing N_enc exactly
(2756 vs 2763 row-inconsistency disclosed above).
All three portal returns are now disclosed (README section G); none was used
for selection or tuning anywhere in Stages 0-2 or the lockbox step.
