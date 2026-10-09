# FINAL LOCKBOX EVALUATION -- three finalists (60 sealed runs)

Rule frozen in LOCKBOX_LOG.md. lockbox = 60 runs, 57.45 bg-h: CIs and terciles are WIDE - low power; per PLAN 2026-10-03 only 0.70 (primary, ~45 v1 FP) and 0.35 (secondary/indicative, ~22) may drive conclusions; 0.085/.17 are REPORT-ONLY.

pool: 60 runs, bg 60.44 h (PLAN power table said 57.45 h).

## Per finalist x target (Global at OWN dev thresholds)

| cfg | target | role | thr | d | CI95 | c | id | fpr | FP | exp.FP |
|--|--|--|--|--|--|--|--|--|--|--|
| v1 | 0.085 | report-only | 6.2266 | 0.0981 | [0.0734,0.1238] | 0.0907 | 0.0667 | 0.0870 | 5 | ~4.3 |
| v1 | 0.17 | report-only | 5.9117 | 0.1130 | [0.0856,0.1407] | 0.1056 | 0.0796 | 0.1741 | 10 | ~9.8 |
| v1 | 0.35 | indicative | 5.6020 | 0.1389 | [0.1101,0.1686] | 0.1278 | 0.0963 | 0.2959 | 17 | ~20.1 |
| v1 | 0.7 | SELECTION | 5.2640 | 0.1593 | [0.1309,0.1903] | 0.1463 | 0.1148 | 0.7137 | 41 | ~40.2 |
| c64s | 0.085 | report-only | 0.8617 | 0.4519 | [0.4049,0.5000] | 0.4333 | 0.3278 | 0.1044 | 6 | ~4.3 |
| c64s | 0.17 | report-only | 0.8449 | 0.4870 | [0.4428,0.5332] | 0.4648 | 0.3481 | 0.1393 | 8 | ~9.8 |
| c64s | 0.35 | indicative | 0.8234 | 0.5278 | [0.4831,0.5738] | 0.5056 | 0.3796 | 0.2959 | 17 | ~20.1 |
| c64s | 0.7 | SELECTION | 0.7961 | 0.5722 | [0.5311,0.6131] | 0.5463 | 0.4074 | 0.6963 | 40 | ~40.2 |
| s2-spd | 0.085 | report-only | 0.8536 | 0.4407 | [0.3892,0.4948] | 0.4241 | 0.2852 | 0.1393 | 8 | ~4.3 |
| s2-spd | 0.17 | report-only | 0.8313 | 0.5000 | [0.4485,0.5509] | 0.4796 | 0.3333 | 0.2263 | 13 | ~9.8 |
| s2-spd | 0.35 | indicative | 0.8045 | 0.5519 | [0.5027,0.6015] | 0.5278 | 0.3648 | 0.4526 | 26 | ~20.1 |
| s2-spd | 0.7 | SELECTION | 0.7797 | 0.5833 | [0.5358,0.6296] | 0.5593 | 0.3852 | 1.0096 | 58 | ~40.2 |

## Per-category (d / c / id with CIs, FP counts)

### v1 @ 0.085 (report-only)
| cat | n_enc | d | CI | c | CI | id | CI | fp | fpr CI |
|--|--|--|--|--|--|--|--|--|--|
| NORM | 51 | 0.000 [0.000,0.000] | x | 0.000 [0.000,0.000] | x | 0.000 [0.000,0.000] | x | 0 | [0.000,0.000] |
| Medical | 74 | 0.230 [0.139,0.324] | x | 0.230 [0.139,0.324] | x | 0.216 [0.125,0.319] | x | 3 | [0.000,0.133] |
| Industrial | 69 | 0.058 [0.014,0.122] | x | 0.014 [0.000,0.050] | x | 0.014 [0.000,0.050] | x | 1 | [0.000,0.050] |
| NuclearMaterial | 346 | 0.092 [0.062,0.124] | x | 0.090 [0.060,0.119] | x | 0.055 [0.033,0.080] | x | 1 | [0.000,0.050] |
| Global | 540 | 0.098 [0.073,0.124] | x | 0.091 [0.066,0.116] | x | 0.067 [0.048,0.086] | x | 5 | [0.017,0.182] |

### v1 @ 0.17 (report-only)
| cat | n_enc | d | CI | c | CI | id | CI | fp | fpr CI |
|--|--|--|--|--|--|--|--|--|--|
| NORM | 51 | 0.000 [0.000,0.000] | x | 0.000 [0.000,0.000] | x | 0.000 [0.000,0.000] | x | 0 | [0.000,0.000] |
| Medical | 74 | 0.257 [0.167,0.356] | x | 0.257 [0.167,0.356] | x | 0.243 [0.147,0.348] | x | 5 | [0.017,0.182] |
| Industrial | 69 | 0.058 [0.014,0.122] | x | 0.014 [0.000,0.050] | x | 0.014 [0.000,0.050] | x | 1 | [0.000,0.050] |
| NuclearMaterial | 346 | 0.110 [0.077,0.144] | x | 0.107 [0.076,0.141] | x | 0.069 [0.043,0.096] | x | 4 | [0.017,0.133] |
| Global | 540 | 0.113 [0.086,0.141] | x | 0.106 [0.079,0.133] | x | 0.080 [0.058,0.102] | x | 10 | [0.083,0.281] |

### v1 @ 0.35 (indicative)
| cat | n_enc | d | CI | c | CI | id | CI | fp | fpr CI |
|--|--|--|--|--|--|--|--|--|--|
| NORM | 51 | 0.000 [0.000,0.000] | x | 0.000 [0.000,0.000] | x | 0.000 [0.000,0.000] | x | 0 | [0.000,0.000] |
| Medical | 74 | 0.297 [0.203,0.403] | x | 0.297 [0.203,0.403] | x | 0.284 [0.187,0.392] | x | 9 | [0.050,0.298] |
| Industrial | 69 | 0.072 [0.016,0.140] | x | 0.014 [0.000,0.050] | x | 0.014 [0.000,0.050] | x | 1 | [0.000,0.050] |
| NuclearMaterial | 346 | 0.139 [0.104,0.176] | x | 0.133 [0.100,0.170] | x | 0.087 [0.059,0.117] | x | 7 | [0.033,0.215] |
| Global | 540 | 0.139 [0.110,0.169] | x | 0.128 [0.101,0.156] | x | 0.096 [0.074,0.118] | x | 17 | [0.149,0.446] |

### v1 @ 0.7 (SELECTION)
| cat | n_enc | d | CI | c | CI | id | CI | fp | fpr CI |
|--|--|--|--|--|--|--|--|--|--|
| NORM | 51 | 0.020 [0.000,0.067] | x | 0.000 [0.000,0.000] | x | 0.000 [0.000,0.000] | x | 0 | [0.000,0.000] |
| Medical | 74 | 0.311 [0.213,0.419] | x | 0.311 [0.213,0.419] | x | 0.297 [0.194,0.413] | x | 23 | [0.215,0.563] |
| Industrial | 69 | 0.072 [0.016,0.140] | x | 0.014 [0.000,0.050] | x | 0.014 [0.000,0.050] | x | 2 | [0.000,0.083] |
| NuclearMaterial | 346 | 0.165 [0.129,0.202] | x | 0.159 [0.123,0.196] | x | 0.113 [0.082,0.146] | x | 16 | [0.149,0.397] |
| Global | 540 | 0.159 [0.131,0.190] | x | 0.146 [0.119,0.177] | x | 0.115 [0.090,0.139] | x | 41 | [0.464,0.926] |

### c64s @ 0.085 (report-only)
| cat | n_enc | d | CI | c | CI | id | CI | fp | fpr CI |
|--|--|--|--|--|--|--|--|--|--|
| NORM | 51 | 0.118 [0.041,0.200] | x | 0.118 [0.041,0.200] | x | 0.118 [0.041,0.200] | x | 0 | [0.000,0.000] |
| Medical | 74 | 0.324 [0.220,0.433] | x | 0.270 [0.169,0.372] | x | 0.257 [0.162,0.362] | x | 1 | [0.000,0.050] |
| Industrial | 69 | 0.594 [0.464,0.726] | x | 0.536 [0.400,0.672] | x | 0.536 [0.400,0.672] | x | 2 | [0.000,0.083] |
| NuclearMaterial | 346 | 0.500 [0.446,0.553] | x | 0.494 [0.440,0.546] | x | 0.332 [0.281,0.384] | x | 3 | [0.000,0.099] |
| Global | 540 | 0.452 [0.405,0.500] | x | 0.433 [0.387,0.480] | x | 0.328 [0.287,0.371] | x | 6 | [0.017,0.198] |

### c64s @ 0.17 (report-only)
| cat | n_enc | d | CI | c | CI | id | CI | fp | fpr CI |
|--|--|--|--|--|--|--|--|--|--|
| NORM | 51 | 0.118 [0.041,0.200] | x | 0.118 [0.041,0.200] | x | 0.118 [0.041,0.200] | x | 0 | [0.000,0.000] |
| Medical | 74 | 0.338 [0.232,0.449] | x | 0.284 [0.181,0.386] | x | 0.270 [0.169,0.377] | x | 1 | [0.000,0.050] |
| Industrial | 69 | 0.638 [0.509,0.761] | x | 0.565 [0.431,0.698] | x | 0.565 [0.431,0.698] | x | 2 | [0.000,0.083] |
| NuclearMaterial | 346 | 0.543 [0.487,0.599] | x | 0.535 [0.479,0.589] | x | 0.355 [0.301,0.411] | x | 5 | [0.017,0.149] |
| Global | 540 | 0.487 [0.443,0.533] | x | 0.465 [0.421,0.510] | x | 0.348 [0.304,0.391] | x | 8 | [0.050,0.232] |

### c64s @ 0.35 (indicative)
| cat | n_enc | d | CI | c | CI | id | CI | fp | fpr CI |
|--|--|--|--|--|--|--|--|--|--|
| NORM | 51 | 0.157 [0.068,0.259] | x | 0.157 [0.068,0.259] | x | 0.157 [0.068,0.259] | x | 1 | [0.000,0.050] |
| Medical | 74 | 0.351 [0.243,0.467] | x | 0.297 [0.189,0.409] | x | 0.284 [0.175,0.391] | x | 2 | [0.000,0.083] |
| Industrial | 69 | 0.696 [0.575,0.811] | x | 0.623 [0.493,0.743] | x | 0.623 [0.493,0.743] | x | 2 | [0.000,0.083] |
| NuclearMaterial | 346 | 0.587 [0.530,0.640] | x | 0.578 [0.523,0.630] | x | 0.384 [0.328,0.440] | x | 12 | [0.083,0.331] |
| Global | 540 | 0.528 [0.483,0.574] | x | 0.506 [0.461,0.551] | x | 0.380 [0.335,0.424] | x | 17 | [0.133,0.446] |

### c64s @ 0.7 (SELECTION)
| cat | n_enc | d | CI | c | CI | id | CI | fp | fpr CI |
|--|--|--|--|--|--|--|--|--|--|
| NORM | 51 | 0.157 [0.068,0.259] | x | 0.157 [0.068,0.259] | x | 0.157 [0.068,0.259] | x | 2 | [0.000,0.083] |
| Medical | 74 | 0.486 [0.373,0.600] | x | 0.405 [0.286,0.522] | x | 0.365 [0.253,0.480] | x | 5 | [0.017,0.165] |
| Industrial | 69 | 0.710 [0.588,0.828] | x | 0.638 [0.516,0.754] | x | 0.638 [0.516,0.754] | x | 4 | [0.017,0.133] |
| NuclearMaterial | 346 | 0.624 [0.570,0.675] | x | 0.616 [0.561,0.664] | x | 0.408 [0.353,0.459] | x | 29 | [0.248,0.762] |
| Global | 540 | 0.572 [0.531,0.613] | x | 0.546 [0.505,0.588] | x | 0.407 [0.364,0.449] | x | 40 | [0.414,0.943] |

### s2-spd @ 0.085 (report-only)
| cat | n_enc | d | CI | c | CI | id | CI | fp | fpr CI |
|--|--|--|--|--|--|--|--|--|--|
| NORM | 51 | 0.020 [0.000,0.070] | x | 0.020 [0.000,0.070] | x | 0.020 [0.000,0.070] | x | 0 | [0.000,0.000] |
| Medical | 74 | 0.311 [0.206,0.427] | x | 0.284 [0.186,0.400] | x | 0.284 [0.186,0.400] | x | 0 | [0.000,0.000] |
| Industrial | 69 | 0.565 [0.449,0.676] | x | 0.493 [0.380,0.603] | x | 0.493 [0.380,0.603] | x | 0 | [0.000,0.000] |
| NuclearMaterial | 346 | 0.506 [0.449,0.565] | x | 0.500 [0.443,0.559] | x | 0.283 [0.233,0.339] | x | 8 | [0.033,0.265] |
| Global | 540 | 0.441 [0.389,0.495] | x | 0.424 [0.373,0.476] | x | 0.285 [0.241,0.329] | x | 8 | [0.033,0.265] |

### s2-spd @ 0.17 (report-only)
| cat | n_enc | d | CI | c | CI | id | CI | fp | fpr CI |
|--|--|--|--|--|--|--|--|--|--|
| NORM | 51 | 0.020 [0.000,0.070] | x | 0.020 [0.000,0.070] | x | 0.020 [0.000,0.070] | x | 1 | [0.000,0.050] |
| Medical | 74 | 0.365 [0.258,0.479] | x | 0.311 [0.203,0.431] | x | 0.311 [0.203,0.431] | x | 1 | [0.000,0.066] |
| Industrial | 69 | 0.638 [0.524,0.750] | x | 0.565 [0.449,0.678] | x | 0.565 [0.449,0.678] | x | 1 | [0.000,0.050] |
| NuclearMaterial | 346 | 0.572 [0.511,0.627] | x | 0.566 [0.506,0.622] | x | 0.338 [0.285,0.392] | x | 10 | [0.050,0.298] |
| Global | 540 | 0.500 [0.449,0.551] | x | 0.480 [0.429,0.532] | x | 0.333 [0.288,0.378] | x | 13 | [0.083,0.364] |

### s2-spd @ 0.35 (indicative)
| cat | n_enc | d | CI | c | CI | id | CI | fp | fpr CI |
|--|--|--|--|--|--|--|--|--|--|
| NORM | 51 | 0.059 [0.000,0.122] | x | 0.059 [0.000,0.122] | x | 0.059 [0.000,0.122] | x | 2 | [0.000,0.083] |
| Medical | 74 | 0.419 [0.314,0.526] | x | 0.365 [0.250,0.486] | x | 0.351 [0.243,0.478] | x | 3 | [0.000,0.116] |
| Industrial | 69 | 0.696 [0.586,0.794] | x | 0.609 [0.492,0.720] | x | 0.609 [0.492,0.720] | x | 2 | [0.000,0.099] |
| NuclearMaterial | 346 | 0.624 [0.565,0.680] | x | 0.616 [0.559,0.670] | x | 0.364 [0.308,0.421] | x | 19 | [0.182,0.479] |
| Global | 540 | 0.552 [0.503,0.601] | x | 0.528 [0.480,0.576] | x | 0.365 [0.321,0.406] | x | 26 | [0.265,0.613] |

### s2-spd @ 0.7 (SELECTION)
| cat | n_enc | d | CI | c | CI | id | CI | fp | fpr CI |
|--|--|--|--|--|--|--|--|--|--|
| NORM | 51 | 0.078 [0.019,0.157] | x | 0.078 [0.019,0.157] | x | 0.078 [0.019,0.157] | x | 2 | [0.000,0.083] |
| Medical | 74 | 0.446 [0.338,0.562] | x | 0.392 [0.277,0.514] | x | 0.378 [0.269,0.500] | x | 7 | [0.050,0.215] |
| Industrial | 69 | 0.696 [0.586,0.794] | x | 0.609 [0.492,0.720] | x | 0.609 [0.492,0.720] | x | 3 | [0.000,0.132] |
| NuclearMaterial | 346 | 0.665 [0.608,0.721] | x | 0.656 [0.601,0.710] | x | 0.387 [0.331,0.442] | x | 46 | [0.513,1.028] |
| Global | 540 | 0.583 [0.536,0.630] | x | 0.559 [0.513,0.604] | x | 0.385 [0.339,0.430] | x | 58 | [0.662,1.293] |

## Paired differences (Global, own thrs, run-cluster B=1000)

| pair | target | role | d diff | CI95 | fpr diff | CI95 |
|--|--|--|--|--|--|--|
| s2-spd_minus_v1 | 0.085 | report-only | +0.3426 | [+0.2884,+0.3982] | +0.0522 | [-0.0870,+0.2084] |
| s2-spd_minus_v1 | 0.17 | report-only | +0.3870 | [+0.3333,+0.4405] | +0.0522 | [-0.1219,+0.2440] |
| s2-spd_minus_v1 | 0.35 | secondary | +0.4130 | [+0.3631,+0.4640] | +0.1567 | [-0.0698,+0.4012] |
| s2-spd_minus_v1 | 0.7 | PRIMARY | +0.4241 | [+0.3716,+0.4751] | +0.2959 | [-0.1561,+0.7469] |
| c64s_minus_v1 | 0.085 | report-only | +0.3537 | [+0.3071,+0.3978] | +0.0174 | [-0.0873,+0.1393] |
| c64s_minus_v1 | 0.17 | report-only | +0.3741 | [+0.3288,+0.4185] | -0.0348 | [-0.1741,+0.1053] |
| c64s_minus_v1 | 0.35 | secondary | +0.3889 | [+0.3421,+0.4340] | +0.0000 | [-0.2090,+0.2262] |
| c64s_minus_v1 | 0.7 | PRIMARY | +0.4130 | [+0.3693,+0.4568] | -0.0174 | [-0.3665,+0.3467] |
| s2-spd_minus_c64s | 0.085 | report-only | -0.0111 | [-0.0578,+0.0356] | +0.0348 | [-0.1045,+0.1917] |
| s2-spd_minus_c64s | 0.17 | report-only | +0.0130 | [-0.0258,+0.0530] | +0.0870 | [-0.0697,+0.2608] |
| s2-spd_minus_c64s | 0.35 | secondary | +0.0241 | [-0.0180,+0.0675] | +0.1567 | [-0.0174,+0.3483] |
| s2-spd_minus_c64s | 0.7 | PRIMARY | +0.0111 | [-0.0295,+0.0499] | +0.3133 | [-0.0522,+0.6963] |

## REAL speed terciles on lockbox (edges 4.4/6.8 m/s, LOW POWER at 60 runs)

| cfg | tier | n_enc | d@.085 | d@.17 | d@.35 | d@.7 | fast-tier CI@.7 |
|--|--|--|--|--|--|--|--|
| v1 | slow | 150 | 0.1533 | 0.1867 | 0.2267 | 0.2333 |  |
| v1 | mid | 197 | 0.0660 | 0.0711 | 0.0914 | 0.1269 |  |
| v1 | FAST | 193 | 0.0881 | 0.0984 | 0.1192 | 0.1347 | [0.091,0.179] |
| v1 | FAST mean |  | 0.1101 | | | | |
| c64s | slow | 150 | 0.6000 | 0.6333 | 0.6400 | 0.6800 |  |
| c64s | mid | 197 | 0.4365 | 0.4873 | 0.5635 | 0.6041 |  |
| c64s | FAST | 193 | 0.3523 | 0.3731 | 0.4041 | 0.4560 | [0.375,0.540] |
| c64s | FAST mean |  | 0.3964 | | | | |
| s2-spd | slow | 150 | 0.5667 | 0.6200 | 0.6667 | 0.6800 |  |
| s2-spd | mid | 197 | 0.4162 | 0.4822 | 0.5482 | 0.5990 |  |
| s2-spd | FAST | 193 | 0.3679 | 0.4249 | 0.4663 | 0.4922 | [0.407,0.580] |
| s2-spd | FAST mean |  | 0.4378 | | | | |

## Dev OOF vs lockbox (d; optimism of dev-based selection)

| cfg | 0.085: dev/lock/delta | 0.17: dev/lock/delta | 0.35: dev/lock/delta | 0.7: dev/lock/delta |
|--|--|--|--|--|
| v1 | 0.120/0.098/-0.022 | 0.129/0.113/-0.016 | 0.148/0.139/-0.009 | 0.166/0.159/-0.006 |
| c64s | 0.444/0.452/+0.008 | 0.493/0.487/-0.006 | 0.535/0.528/-0.007 | 0.574/0.572/-0.001 |
| s2-spd | 0.448/0.441/-0.007 | 0.501/0.500/-0.001 | 0.544/0.552/+0.008 | 0.581/0.583/+0.003 |

## Portal returns vs lockbox (LOCKBOX IS TRAINING-DISTRIBUTION -- it does NOT measure the testing shift; the portal does)

| try | model | thr | d | fpr | rows | md5 | note |
|--|--|--|--|--|--|--|--|
| first_try | v1 | 5.7492 | 0.0946 | 0.535 | - | - | disclosed PLAN/STATE anchor; FP x2.4, d x0.65 vs dev |
| second_try | c64s | 0.8617 | 0.253 | 0.0973 | 727 | e41c68e31905 |  |
| third_try | s2-spd | 0.8536 | - | - | - | - | PENDING USER CONFIRMATION |


---

# APPENDIX (written after frozen generation, 2026-10-06 ~23:5x): portal #3 confirmed + plain verdict. All numbers ABOVE are unchanged and frozen.

## A1. Power correction (measured at LOCKBOX_LOG record)
True lockbox background = **60.44 h**, not the 57.45 h assumed in the frozen
header. Expected FP at own dev thresholds rescales to **5.1 / 10.3 / 21.2 /
42.3** at {.085,.17,.35,.7}. Observed FP counts:
v1 5/10/17/41 (on target); c64s 6/8/17/40 (on target);
s2-spd 8/13/26/58 -- **over budget at .17/.35/.7**: dev-threshold transfer is
noisy at 60 runs / small FP counts (dev fpr 0.0828 -> lockbox 0.1393 at .085;
dev 0.0828+ -> 1.010 vs budget 0.7 at .7).

## A2. Portal #3 -- third_try CONFIRMED (user, 2026-10-06)
third_try **IS** `submission_v2_s2-spd_a_thr0p8536.csv` (user-confirmed; disk
re-verified this session: **683 rows, md5 c04e558df82940c3881b2f34e8de04bf**
-- identical to the Stage-2 manifest). Return parsed from
`leaderboard.html` (user-fetched 23:32; username <participant-B>, 2026-10-06T19:24:05):

| try | model | thr | d | c | id | fpr | rows | md5(12) |
|--|--|--|--|--|--|--|--|--|
| first_try | v1 | 5.7492 | 0.0946 | 0.0737 | 0.0544 | 0.535 | - | - |
| second_try | c64s | 0.8617 | 0.2530 | 0.2278 | 0.1625 | 0.0973 | 727 | e41c68e31905 |
| third_try | s2-spd | 0.8536 | 0.2402 | 0.2150 | 0.1464 | 0.0730 | 683 | c04e558df829 |

third_try per-category: Industrial .2713/.2254/.2144/.0035; Medical
.1533/.0842/.0778/.0035; NORM .0088/.0088/.0088/.0069; NuclearMaterial
.2898/.2797/.1666/.0591. (Supersedes the PENDING row in the frozen table.)

Integer fit (same method as #2, den 287.77 h): FP **21** = Ind 1 + Med 1 +
NORM 2 + NM 17 (exact partition, unique candidates for every category);
TP = 683-21 = **662**; unique N_enc consistent with d=0.2402 = **2756**.
DISCLOSED INCONSISTENCY: #2's fit gave N_enc = 2763; at N=2763 no integer TP
reproduces 0.2402 (663 -> 0.2400, 664 -> 0.2403). The two portal rows are
mutually inconsistent by ~7 encounters (0.25 % of N) under the naive
rows-FP model -- rounding-policy or matching-policy detail at the organizer
side; immaterial for conclusions (far inside sampling noise).
Retention vs s2-spd dev OOF @.085 (d 0.4481, fpr 0.0828): **0.5360
[0.491, 0.585]**; c 0.215/0.4319 = 0.498; id 0.1464/0.2972 = 0.493.
FP inflation 0.0730/0.08283 = **0.88 [0.50, 1.26]** (testing FP rate BELOW
dev). Emission 683/300 = 2.28 alarms/test-run vs dev 4.56 -> ratio 0.50,
matching d-retention: same story as #2, the loss is missed detections, not
FP flood. Precision 662/683 = 0.969 (dev 0.983; #2 test 0.961).
vs second_try: d -0.0128 = ~1.1 sigma (independent upper bound; the two
models are correlated so effective sigma is SMALLER); fpr -0.0243 = ~1.0
sigma. **STATISTICALLY INDISTINGUISHABLE on the real testing distribution.**
Leaderboard low_false_alarms bucket (fpr<0.125), by d: <participant-A> .3662 >
<participant-C> .2979 > second_try .2530 > third_try .2402 -- our two
network files are 3rd/4th; third_try has the lowest fpr (0.073) of these
four. Stage-2 manifest prediction for this file: row-cap d <= 0.247
(respected: 0.2402), portal-#2-like point ~0.255 (actual slightly below,
inside the forecast's own noise).

## A3. PLAIN VERDICT (pre-registered roles honored; no re-selection)
1. **Deep detector >> physics baseline, sealed and same-distribution**: at
   PRIMARY target 0.70 both nets beat v1 by +0.4130 [0.3693,0.4568] (c64s)
   and +0.4241 [0.3716,0.4751] (s2-spd); SECONDARY 0.35 +0.3889/+0.4130.
   The ML approach is confirmed outside the dev pool. v1's lockbox FP (41)
   landed exactly on the power-table expectation (40-45).
2. **s2-spd vs c64s on the lockbox: no separation** -- PRIMARY +0.0111
   [-0.0295,+0.0499], SECONDARY +0.0241 [-0.0180,+0.0675]; fast-tercile
   mean 0.4378 vs 0.3964 nominally favors s2-spd but the 60-run CIs overlap
   ([0.407,0.580] vs [0.375,0.540]) -- LOW POWER, disclosed. This is what
   the dev non-inferiority analysis predicted (diffs below the 0.061/0.032
   seed-noise floor).
3. **On the ONLY real measure of the shift (the portal): NO GAIN from the
   speed-compression augmentation is demonstrated.** s2-spd retention 0.536
   [0.491,0.585] vs c64s 0.570 [0.525,0.616] -- overlapping CIs, point
   estimate slightly lower d, lower FP (inflation 0.88 vs 1.17). Equally, no
   harm is demonstrated (~1 sigma). The Stage-2 hypothesis (aug closes the
   fast-tier gap under shift) is neither confirmed nor refuted at portal
   power; what IS confirmed: it beat its own same-seed control everywhere
   in-distribution and matches the Stage-1 pick within noise.
4. **Winner's-curse estimate**: dev->lockbox |delta| <= 0.008 for all three
   (c64s +0.008/-0.006/-0.007/-0.001; s2-spd -0.007/-0.001/+0.008/+0.003;
   v1 -0.022/-0.016/-0.009/-0.006) -- far below the 0.061 seed-noise floor
   and below 60-run CI width: within-distribution selection optimism is NOT
   measurable here (|true delta| bounded by ~0.04). The ~45 % portal loss is
   DISTRIBUTION SHIFT, not dev overfitting.
5. **Threshold transfer caveat**: dev-calibrated thresholds do not carry
   exact FP budgets across 60-run pools (s2-spd over-fired on lockbox,
   under-fired on portal); any future lockbox-based acceptance rule should
   keep targets .35/.70 only (PLAN power amendment re-applied).

STOP: no re-selection, no tuning, no further lockbox touches. s2-spd remains
the Stage-2 selection, c64s the Stage-1 selection, both per pre-registered
rules; portal #3 and lockbox numbers were used for reporting only.
