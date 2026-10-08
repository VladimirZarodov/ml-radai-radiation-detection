# STAGE 2 FINAL -- one-factor augmentation ablation (c64s recipe)

## Step 0: synthetic-shift calibration on FROZEN c64s (thr 0.8617)

targets: retention [0.52,0.62], inflation [0.80,1.60] (portal #2: 0.570 / 1.17).

| kmax | theta | ret | infl | pass |
|--|--|--|--|--|
| 1.3 | 1.0 | 0.269 | 41.00 | n |
| 1.3 | 0.7 | 0.173 | 41.67 | n |
| 1.3 | 0.5 | 0.128 | 41.00 | n |
| 1.5 | 1.0 | 0.253 | 42.00 | n |
| 1.5 | 0.7 | 0.165 | 41.67 | n |
| 1.5 | 0.5 | 0.114 | 41.33 | n |
| 1.7 | 1.0 | 0.237 | 41.67 | n |
| 1.7 | 0.7 | 0.165 | 41.33 | n |
| 1.7 | 0.5 | 0.136 | 42.33 | n |

**chosen k=1.3, theta=1.0 (source: fallback-nearest)** -- all-240 confirm: ret 0.262, infl 6.74; out-of-range k=2.0: ret 0.223, infl 6.68.

DISCLOSED: no grid cell reproduced the portal FP-inflation band; the bg/NORM-dominated synthetic shift inflates FP ~6.7x at any tested strength while the portal showed ~1.2x. (That Step-0 figure used the pre-fix stacked-NORM generator; the per-config rows below use the corrected generator per E052.) Retention/inflation rows below are therefore SECONDARY (relative, same-for-all-configs stress; note the synth FP response is itself model-unstable: c64s 4.32x vs s2-0 0.26x at the same stress, so cross-config reads at this axis are weak); the PRIMARY shift metric is the REAL fast-tercile d (speed >= 6.8 m/s, edges frozen at 4.4/6.8 from E039).

## Gates (per config; ~60-run subset; rules pre-registered in s2_gates.py)

| cfg | G1 (v1 PD-vs-SNR) | G2 geometry/rate | G3 identities | outcome |
|--|--|--|--|--|
| s2-0 | True | True | True | PASS |
| s2-spd | True | True | True | PASS |
| s2-thin | True | True | True | PASS |
| s2-bg | True | True | True | PASS |
| s2-all | False | True | True | DROPPED |

## Pooled dev OOF at each config's OWN thresholds (B=1000 run-cluster CI on d)

| cfg | target | thr | d | CI95 | c | id | fpr |
|--|--|--|--|--|--|--|--|
| c64s | 0.085 | 0.8617 | 0.4435 | [0.420,0.466] | 0.427 | 0.288 | 0.0828 |
| c64s | 0.17 | 0.8449 | 0.4926 | [0.468,0.515] | 0.475 | 0.317 | 0.1657 |
| c64s | 0.35 | 0.8234 | 0.5347 | [0.512,0.555] | 0.516 | 0.342 | 0.3488 |
| c64s | 0.7 | 0.7961 | 0.5736 | [0.553,0.595] | 0.551 | 0.368 | 0.6976 |
| s2-0 | 0.085 | 0.8508 | 0.3824 | [0.356,0.410] | 0.364 | 0.247 | 0.0828 |
| s2-0 | 0.17 | 0.8225 | 0.4528 | [0.428,0.479] | 0.429 | 0.293 | 0.1657 |
| s2-0 | 0.35 | 0.8004 | 0.5023 | [0.479,0.526] | 0.475 | 0.320 | 0.3488 |
| s2-0 | 0.7 | 0.7750 | 0.5486 | [0.524,0.570] | 0.517 | 0.349 | 0.6976 |
| s2-spd | 0.085 | 0.8536 | 0.4481 | [0.423,0.472] | 0.432 | 0.297 | 0.0828 |
| s2-spd | 0.17 | 0.8313 | 0.5014 | [0.478,0.525] | 0.484 | 0.333 | 0.1657 |
| s2-spd | 0.35 | 0.8045 | 0.5440 | [0.518,0.567] | 0.521 | 0.359 | 0.3488 |
| s2-spd | 0.7 | 0.7797 | 0.5806 | [0.557,0.602] | 0.555 | 0.381 | 0.6976 |
| s2-thin | 0.085 | 0.8289 | 0.3588 | [0.336,0.381] | 0.345 | 0.211 | 0.0828 |
| s2-thin | 0.17 | 0.8166 | 0.3810 | [0.358,0.404] | 0.364 | 0.220 | 0.1657 |
| s2-thin | 0.35 | 0.7935 | 0.4245 | [0.402,0.446] | 0.403 | 0.244 | 0.3488 |
| s2-thin | 0.7 | 0.7707 | 0.4634 | [0.440,0.486] | 0.438 | 0.266 | 0.6976 |
| s2-bg | 0.085 | 0.8692 | 0.1898 | [0.167,0.214] | 0.184 | 0.106 | 0.0828 |
| s2-bg | 0.17 | 0.8518 | 0.2231 | [0.199,0.247] | 0.216 | 0.119 | 0.1657 |
| s2-bg | 0.35 | 0.8247 | 0.2875 | [0.264,0.313] | 0.274 | 0.146 | 0.3488 |
| s2-bg | 0.7 | 0.7957 | 0.3315 | [0.306,0.356] | 0.312 | 0.166 | 0.6976 |

## REAL speed terciles (fast = |v|>=6.8 m/s: PRIMARY shift metric)

| cfg | fast@.085 | fast@.17 | fast@.35 | fast@.7 | fast_mean | slow_mean | ratio@.35 |
|--|--|--|--|--|--|--|--|
| c64s | 0.3245 | 0.3731 | 0.4189 | 0.4646 | 0.3953 | 0.6301 | 0.639 |
| s2-0 | 0.2607 | 0.3259 | 0.3745 | 0.4327 | 0.3485 | 0.6182 | 0.577 |
| s2-spd | 0.3606 | 0.4133 | 0.4508 | 0.4965 | 0.4303 | 0.6231 | 0.699 |
| s2-thin | 0.2497 | 0.2732 | 0.3107 | 0.3509 | 0.2961 | 0.5231 | 0.573 |
| s2-bg | 0.0985 | 0.1221 | 0.1720 | 0.2067 | 0.1498 | 0.3836 | 0.410 |

## Shift-stress rows (calibrated k/theta and k=2.0, OWN thrs; secondary, out-of-band generator - see Step-0 disclosure)

| cfg | ret@.085 cal | infl@.085 cal | ret@.085 k2.0 | infl@.085 k2.0 | ret@.35 cal | infl@.35 cal |
|--|--|--|--|--|--|--|
| c64s | 0.353 | 4.32 | 0.316 | 4.32 | 0.396 | 2.46 |
| s2-0 | 0.299 | 0.26 | 0.264 | 0.26 | 0.333 | 0.40 |
| s2-spd | 0.306 | 0.47 | 0.274 | 0.37 | 0.374 | 0.41 |
| s2-thin | 0.316 | 0.32 | 0.266 | 0.42 | 0.356 | 0.29 |
| s2-bg | 0.515 | 1.11 | 0.446 | 1.05 | 0.564 | 0.78 |

## Paired bootstrap vs comparators (field d, own thrs, B=1000). NF(target)=|d(c64s)-d(s2-0)| = 0.085:0.0611, 0.17:0.0398, 0.35:0.0324, 0.7:0.0250

| pair | target | diff | CI95 | claim |
|--|--|--|--|--|
| s2-spd_vs_c64s | 0.085 | +0.0046 | [-0.0176,+0.0278] | no |
| s2-spd_vs_c64s | 0.17 | +0.0088 | [-0.0103,+0.0297] | no |
| s2-spd_vs_c64s | 0.35 | +0.0093 | [-0.0107,+0.0289] | no |
| s2-spd_vs_c64s | 0.7 | +0.0069 | [-0.0128,+0.0235] | no |
| s2-spd_vs_s2-0 | 0.085 | +0.0657 | [+0.0375,+0.0961] | YES |
| s2-spd_vs_s2-0 | 0.17 | +0.0486 | [+0.0213,+0.0759] | YES |
| s2-spd_vs_s2-0 | 0.35 | +0.0417 | [+0.0135,+0.0684] | YES |
| s2-spd_vs_s2-0 | 0.7 | +0.0319 | [+0.0061,+0.0558] | YES |
| s2-thin_vs_c64s | 0.085 | -0.0847 | [-0.1083,-0.0626] | YES |
| s2-thin_vs_c64s | 0.17 | -0.1116 | [-0.1333,-0.0893] | YES |
| s2-thin_vs_c64s | 0.35 | -0.1102 | [-0.1311,-0.0916] | YES |
| s2-thin_vs_c64s | 0.7 | -0.1102 | [-0.1298,-0.0925] | YES |
| s2-thin_vs_s2-0 | 0.085 | -0.0236 | [-0.0545,+0.0032] | no |
| s2-thin_vs_s2-0 | 0.17 | -0.0718 | [-0.1002,-0.0463] | YES |
| s2-thin_vs_s2-0 | 0.35 | -0.0778 | [-0.1045,-0.0545] | YES |
| s2-thin_vs_s2-0 | 0.7 | -0.0852 | [-0.1114,-0.0604] | YES |
| s2-bg_vs_c64s | 0.085 | -0.2537 | [-0.2810,-0.2257] | YES |
| s2-bg_vs_c64s | 0.17 | -0.2694 | [-0.2962,-0.2406] | YES |
| s2-bg_vs_c64s | 0.35 | -0.2472 | [-0.2718,-0.2197] | YES |
| s2-bg_vs_c64s | 0.7 | -0.2421 | [-0.2684,-0.2160] | YES |
| s2-bg_vs_s2-0 | 0.085 | -0.1926 | [-0.2202,-0.1657] | YES |
| s2-bg_vs_s2-0 | 0.17 | -0.2296 | [-0.2556,-0.2024] | YES |
| s2-bg_vs_s2-0 | 0.35 | -0.2148 | [-0.2417,-0.1879] | YES |
| s2-bg_vs_s2-0 | 0.7 | -0.2171 | [-0.2428,-0.1905] | YES |

## Medical watch (category rows at own thrs: c/d and id/d)

| cfg | target | med_d | med_c/d | med_id/d |
|--|--|--|--|--|
| c64s | 0.085 | 0.260 | 0.928 | 0.870 |
| c64s | 0.17 | 0.291 | 0.922 | 0.857 |
| c64s | 0.35 | 0.328 | 0.920 | 0.839 |
| c64s | 0.7 | 0.366 | 0.907 | 0.835 |
| s2-0 | 0.085 | 0.257 | 0.926 | 0.853 |
| s2-0 | 0.17 | 0.317 | 0.917 | 0.821 |
| s2-0 | 0.35 | 0.351 | 0.914 | 0.796 |
| s2-0 | 0.7 | 0.385 | 0.912 | 0.794 |
| s2-spd | 0.085 | 0.275 | 0.945 | 0.918 |
| s2-spd | 0.17 | 0.313 | 0.940 | 0.916 |
| s2-spd | 0.35 | 0.366 | 0.918 | 0.876 |
| s2-spd | 0.7 | 0.400 | 0.915 | 0.868 |
| s2-thin | 0.085 | 0.158 | 0.762 | 0.690 |
| s2-thin | 0.17 | 0.185 | 0.714 | 0.653 |
| s2-thin | 0.35 | 0.226 | 0.667 | 0.617 |
| s2-thin | 0.7 | 0.264 | 0.629 | 0.586 |
| s2-bg | 0.085 | 0.079 | 0.714 | 0.524 |
| s2-bg | 0.17 | 0.091 | 0.708 | 0.500 |
| s2-bg | 0.35 | 0.113 | 0.700 | 0.533 |
| s2-bg | 0.7 | 0.140 | 0.622 | 0.459 |

## Training cost (from models/<cfg>/history.jsonl)

| cfg | total train min | peak VRAM MiB | best epochs (folds) |
|--|--|--|--|
| c64s | 47.0 | 3205 | 5,4,3,6,2 |
| s2-0 | 46.5 | 3206 | 5,4,5,2,3 |
| s2-spd | 45.5 | 3206 | 5,3,4,4,3 |
| s2-thin | 46.7 | 3206 | 3,5,3,4,4 |
| s2-bg | 51.5 | 3206 | 3,3,6,3,8 |

## Selection (pre-registered rule)

**selected: s2-spd** -- max mean real fast-tercile d among ['s2-spd']

| cfg | candidate | losing@s2-0@.35/.70 | fast_mean | retention cal@.085 |
|--|--|--|--|--|
| s2-spd | True | [] | 0.4303 | 0.3057851239669422 |
| s2-thin | False | ['0.35', '0.7'] | 0.2961 | 0.31612903225806455 |
| s2-bg | False | ['0.35', '0.7'] | 0.1498 | 0.5146341463414633 |

## Sealed-eval readiness (DEV dry-run only -- lockbox NOT opened)

- `s2_lockbox.py --cfg s2-spd --dry-run`: 240 DEV-240 runs (fold-true OOF, single model r%5 per the threshold-transfer rule); d reproduced the metrics json at all 4 targets, assertion max|diff| = 0.000000 PASS. Rehearsed v1-paired diff @.085: our d 0.4481 vs v1 0.1199 -> delta 0.3282 CI [0.3023,0.3555]. The real 60-run lockbox eval needs explicit user authorization (`--yes`).

- Measurement prepared, NOT uploaded: `submission_v2_s2-spd_a_thr0p8536.csv` (683 rows, md5 c04e558df82940c3881b2f34e8de04bf); manifest reports/s2_measurement_manifest.md -- all 6 protocol checks PASS (assignment/leakage, content, reader emulation, nestedness, byte-determinism x2 passes, predicted-outcome table).
