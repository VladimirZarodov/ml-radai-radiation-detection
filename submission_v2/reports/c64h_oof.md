# Stage-1 c64h: pooled dev OOF, 240 runs, den 229.4 bg-h; alarm NMS sep 15.0s, sigma 1.5s

| target | thr | d_G | d CI95 | c_G | id_G | fpr | FP | v1 d_G | d - v1 | CI(diff) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.085 | 0.8240 | 0.4324 | [0.409,0.455] | 0.4102 | 0.2713 | 0.0828 | 19 | 0.1199 | +0.3125 | [+0.2869,+0.3372] |
| 0.17 | 0.8074 | 0.4606 | [0.437,0.482] | 0.4352 | 0.2866 | 0.1657 | 38 | 0.1287 | +0.3319 | [+0.3062,+0.3569] |
| 0.35 | 0.7821 | 0.5028 | [0.481,0.524] | 0.4736 | 0.3116 | 0.3488 | 80 | 0.1481 | +0.3546 | [+0.3292,+0.3794] |
| 0.7 | 0.7567 | 0.5454 | [0.523,0.566] | 0.5106 | 0.3375 | 0.6976 | 160 | 0.1657 | +0.3796 | [+0.3536,+0.4049] |

### c64h @ target fpr 0.085 (thr 0.8240)
                 n_enc   tp   tc  tid  fp  d_recall  c_recall  id_recall    fpr
category                                                                       
NORM               204   10    9    9   0    0.0490    0.0441     0.0441 0.0000
Medical            265   82   76   64   4    0.3094    0.2868     0.2415 0.0174
Industrial         269  134  116  116   3    0.4981    0.4312     0.4312 0.0131
NuclearMaterial   1422  708  685  397  12    0.4979    0.4817     0.2792 0.0523
Global            2160  934  886  586  19    0.4324    0.4102     0.2713 0.0828
v1 comparator @ target fpr 0.085 (thr 6.2266)
                 n_enc   tp   tc  tid  fp  d_recall  c_recall  id_recall    fpr
category                                                                       
NORM               204    1    0    0   0    0.0049    0.0000     0.0000 0.0000
Medical            265   58   54   51  13    0.2189    0.2038     0.1925 0.0567
Industrial         269   31    7    7   2    0.1152    0.0260     0.0260 0.0087
NuclearMaterial   1422  169  148  105   4    0.1188    0.1041     0.0738 0.0174
Global            2160  259  209  163  19    0.1199    0.0968     0.0755 0.0828

### c64h @ target fpr 0.17 (thr 0.8074)
                 n_enc   tp   tc  tid  fp  d_recall  c_recall  id_recall    fpr
category                                                                       
NORM               204   11   10   10   0    0.0539    0.0490     0.0490 0.0000
Medical            265   86   80   68   7    0.3245    0.3019     0.2566 0.0305
Industrial         269  142  120  119   6    0.5279    0.4461     0.4424 0.0262
NuclearMaterial   1422  756  730  422  25    0.5316    0.5134     0.2968 0.1090
Global            2160  995  940  619  38    0.4606    0.4352     0.2866 0.1657
v1 comparator @ target fpr 0.17 (thr 5.9117)
                 n_enc   tp   tc  tid  fp  d_recall  c_recall  id_recall    fpr
category                                                                       
NORM               204    1    0    0   0    0.0049    0.0000     0.0000 0.0000
Medical            265   61   57   54  22    0.2302    0.2151     0.2038 0.0959
Industrial         269   31    7    7   3    0.1152    0.0260     0.0260 0.0131
NuclearMaterial   1422  185  164  116  13    0.1301    0.1153     0.0816 0.0567
Global            2160  278  228  177  38    0.1287    0.1056     0.0819 0.1657

### c64h @ target fpr 0.35 (thr 0.7821)
                 n_enc    tp    tc  tid  fp  d_recall  c_recall  id_recall    fpr
category                                                                         
NORM               204    12    11   11   0    0.0588    0.0539     0.0539 0.0000
Medical            265    96    89   75  10    0.3623    0.3358     0.2830 0.0436
Industrial         269   158   131  130   9    0.5874    0.4870     0.4833 0.0392
NuclearMaterial   1422   820   792  457  61    0.5767    0.5570     0.3214 0.2659
Global            2160  1086  1023  673  80    0.5028    0.4736     0.3116 0.3488
v1 comparator @ target fpr 0.35 (thr 5.6020)
                 n_enc   tp   tc  tid  fp  d_recall  c_recall  id_recall    fpr
category                                                                       
NORM               204    1    0    0   0    0.0049    0.0000     0.0000 0.0000
Medical            265   72   68   63  42    0.2717    0.2566     0.2377 0.1831
Industrial         269   36    9    9   8    0.1338    0.0335     0.0335 0.0349
NuclearMaterial   1422  211  189  131  30    0.1484    0.1329     0.0921 0.1308
Global            2160  320  266  203  80    0.1481    0.1231     0.0940 0.3488

### c64h @ target fpr 0.7 (thr 0.7567)
                 n_enc    tp    tc  tid   fp  d_recall  c_recall  id_recall    fpr
category                                                                          
NORM               204    16    15   15    3    0.0784    0.0735     0.0735 0.0131
Medical            265   116   108   92   22    0.4377    0.4075     0.3472 0.0959
Industrial         269   173   143  142   15    0.6431    0.5316     0.5279 0.0654
NuclearMaterial   1422   873   837  480  120    0.6139    0.5886     0.3376 0.5232
Global            2160  1178  1103  729  160    0.5454    0.5106     0.3375 0.6976
v1 comparator @ target fpr 0.7 (thr 5.2640)
                 n_enc   tp   tc  tid   fp  d_recall  c_recall  id_recall    fpr
category                                                                        
NORM               204    2    0    0    0    0.0098    0.0000     0.0000 0.0000
Medical            265   78   74   69   79    0.2943    0.2792     0.2604 0.3444
Industrial         269   39   10   10   13    0.1450    0.0372     0.0372 0.0567
NuclearMaterial   1422  239  213  150   68    0.1681    0.1498     0.1055 0.2965
Global            2160  358  297  229  160    0.1657    0.1375     0.1060 0.6976

## HIGH-BG slice (OOF, thresholds from dev selection)
| target | model d | model fpr | v1 d | v1 fpr |
|---|---|---|---|---|
| 0.085 | 0.4157 | 0.0872 | 0.1124 | 0.1090 |
| 0.17 | 0.4404 | 0.1526 | 0.1169 | 0.2398 |
| 0.35 | 0.4697 | 0.3489 | 0.1393 | 0.5669 |
| 0.7 | 0.5101 | 0.7413 | 0.1506 | 1.2210 |

## FP distribution @0.35: zero-FP runs 178/240; max/run 4; top5 {63: 4, 129: 3, 79: 2, 76: 2, 248: 2}; top3 share 0.11

## Encounters @0.35: detected 1086 / missed 1074 of 2160
missed by isotope (missed/total, top): NatU 178/253, FGPu 94/253, LEU 76/253, Ra-226 72/84, HEU 71/255, WGPu 64/252, Th-232 64/85, RefinedU 63/253, K-40 56/86, DU 56/249
snr_window quantiles missed p10/50/90: [2.72 6.43 9.24]
snr_window quantiles detected p10/50/90: [ 3.33  7.54 13.51]

sample FP alarms @0.35 (run, t_s, metric, label):
  r76 t=2586s s=0.930 HEU
  r79 t=3000s s=0.924 Tl-201
  r248 t=3456s s=0.913 LEU
  r76 t=1280s s=0.888 HEU
  r274 t=400s s=0.887 Co-60
  r248 t=1492s s=0.882 Xe-133
  r163 t=2506s s=0.879 Tc-99m
  r133 t=360s s=0.871 HEU

## SYNTH-SHIFT (frozen params, report only) OOF-scores
| target | d self-cal | fpr self-cal | d @dev-thr | fpr @dev-thr | FP inflation |
|---|---|---|---|---|---|
| 0.085 | 0.0769 | 0.8887 thr | 0.1481 | 0.3967 | 4.79 |
| 0.17 | 0.0968 | 0.8712 thr | 0.1611 | 0.5624 | 3.39 |
| 0.35 | 0.1361 | 0.8334 thr | 0.1856 | 0.8807 | 2.52 |
| 0.7 | 0.1694 | 0.7966 thr | 0.2102 | 1.2905 | 1.85 |