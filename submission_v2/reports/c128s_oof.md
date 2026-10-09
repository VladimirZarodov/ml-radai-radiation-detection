# Stage-1 c128s: pooled dev OOF, 240 runs, den 229.4 bg-h; alarm NMS sep 15.0s, sigma 1.5s

| target | thr | d_G | d CI95 | c_G | id_G | fpr | FP | v1 d_G | d - v1 | CI(diff) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.085 | 0.8675 | 0.3245 | [0.297,0.352] | 0.3120 | 0.2056 | 0.0828 | 19 | 0.1199 | +0.2046 | [+0.1754,+0.2324] |
| 0.17 | 0.8471 | 0.3676 | [0.339,0.396] | 0.3500 | 0.2296 | 0.1657 | 38 | 0.1287 | +0.2389 | [+0.2080,+0.2662] |
| 0.35 | 0.8260 | 0.4148 | [0.386,0.442] | 0.3912 | 0.2565 | 0.3488 | 80 | 0.1481 | +0.2667 | [+0.2362,+0.2945] |
| 0.7 | 0.7966 | 0.4630 | [0.435,0.491] | 0.4338 | 0.2773 | 0.6976 | 160 | 0.1657 | +0.2972 | [+0.2671,+0.3253] |

### c128s @ target fpr 0.085 (thr 0.8675)
                 n_enc   tp   tc  tid  fp  d_recall  c_recall  id_recall    fpr
category                                                                       
NORM               204   16   16   15   3    0.0784    0.0784     0.0735 0.0131
Medical            265   50   44   39   2    0.1887    0.1660     0.1472 0.0087
Industrial         269  121  113  108   1    0.4498    0.4201     0.4015 0.0044
NuclearMaterial   1422  514  501  282  13    0.3615    0.3523     0.1983 0.0567
Global            2160  701  674  444  19    0.3245    0.3120     0.2056 0.0828
v1 comparator @ target fpr 0.085 (thr 6.2266)
                 n_enc   tp   tc  tid  fp  d_recall  c_recall  id_recall    fpr
category                                                                       
NORM               204    1    0    0   0    0.0049    0.0000     0.0000 0.0000
Medical            265   58   54   51  13    0.2189    0.2038     0.1925 0.0567
Industrial         269   31    7    7   2    0.1152    0.0260     0.0260 0.0087
NuclearMaterial   1422  169  148  105   4    0.1188    0.1041     0.0738 0.0174
Global            2160  259  209  163  19    0.1199    0.0968     0.0755 0.0828

### c128s @ target fpr 0.17 (thr 0.8471)
                 n_enc   tp   tc  tid  fp  d_recall  c_recall  id_recall    fpr
category                                                                       
NORM               204   19   19   18   5    0.0931    0.0931     0.0882 0.0218
Medical            265   57   50   44   5    0.2151    0.1887     0.1660 0.0218
Industrial         269  143  129  122   5    0.5316    0.4796     0.4535 0.0218
NuclearMaterial   1422  575  558  312  23    0.4044    0.3924     0.2194 0.1003
Global            2160  794  756  496  38    0.3676    0.3500     0.2296 0.1657
v1 comparator @ target fpr 0.17 (thr 5.9117)
                 n_enc   tp   tc  tid  fp  d_recall  c_recall  id_recall    fpr
category                                                                       
NORM               204    1    0    0   0    0.0049    0.0000     0.0000 0.0000
Medical            265   61   57   54  22    0.2302    0.2151     0.2038 0.0959
Industrial         269   31    7    7   3    0.1152    0.0260     0.0260 0.0131
NuclearMaterial   1422  185  164  116  13    0.1301    0.1153     0.0816 0.0567
Global            2160  278  228  177  38    0.1287    0.1056     0.0819 0.1657

### c128s @ target fpr 0.35 (thr 0.8260)
                 n_enc   tp   tc  tid  fp  d_recall  c_recall  id_recall    fpr
category                                                                       
NORM               204   20   19   18  13    0.0980    0.0931     0.0882 0.0567
Medical            265   68   57   51   5    0.2566    0.2151     0.1925 0.0218
Industrial         269  163  145  135  13    0.6059    0.5390     0.5019 0.0567
NuclearMaterial   1422  645  624  350  49    0.4536    0.4388     0.2461 0.2136
Global            2160  896  845  554  80    0.4148    0.3912     0.2565 0.3488
v1 comparator @ target fpr 0.35 (thr 5.6020)
                 n_enc   tp   tc  tid  fp  d_recall  c_recall  id_recall    fpr
category                                                                       
NORM               204    1    0    0   0    0.0049    0.0000     0.0000 0.0000
Medical            265   72   68   63  42    0.2717    0.2566     0.2377 0.1831
Industrial         269   36    9    9   8    0.1338    0.0335     0.0335 0.0349
NuclearMaterial   1422  211  189  131  30    0.1484    0.1329     0.0921 0.1308
Global            2160  320  266  203  80    0.1481    0.1231     0.0940 0.3488

### c128s @ target fpr 0.7 (thr 0.7966)
                 n_enc    tp   tc  tid   fp  d_recall  c_recall  id_recall    fpr
category                                                                         
NORM               204    22   20   19   20    0.1078    0.0980     0.0931 0.0872
Medical            265    75   61   53   10    0.2830    0.2302     0.2000 0.0436
Industrial         269   172  151  139   24    0.6394    0.5613     0.5167 0.1046
NuclearMaterial   1422   731  705  388  106    0.5141    0.4958     0.2729 0.4621
Global            2160  1000  937  599  160    0.4630    0.4338     0.2773 0.6976
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
| 0.085 | 0.2921 | 0.1308 | 0.1124 | 0.1090 |
| 0.17 | 0.3258 | 0.2398 | 0.1169 | 0.2398 |
| 0.35 | 0.3640 | 0.3489 | 0.1393 | 0.5669 |
| 0.7 | 0.4022 | 0.6541 | 0.1506 | 1.2210 |

## FP distribution @0.35: zero-FP runs 173/240; max/run 2; top5 {293: 2, 39: 2, 7: 2, 161: 2, 182: 2}; top3 share 0.07

## Encounters @0.35: detected 896 / missed 1264 of 2160
missed by isotope (missed/total, top): NatU 173/253, FGPu 132/253, LEU 104/253, WGPu 103/252, RefinedU 90/253, HEU 88/255, DU 87/249, Ra-226 72/84, Th-232 62/85, K-40 50/86
snr_window quantiles missed p10/50/90: [2.62 6.7  9.54]
snr_window quantiles detected p10/50/90: [ 3.74  7.57 14.16]

sample FP alarms @0.35 (run, t_s, metric, label):
  r57 t=318s s=0.931 K-40
  r7 t=1550s s=0.929 K-40
  r117 t=544s s=0.917 RefinedU
  r110 t=2460s s=0.911 Cs-137
  r15 t=180s s=0.904 RefinedU
  r239 t=3510s s=0.903 DU
  r70 t=2316s s=0.897 NatU
  r182 t=2614s s=0.893 LEU

## SYNTH-SHIFT (frozen params, report only) OOF-scores
| target | d self-cal | fpr self-cal | d @dev-thr | fpr @dev-thr | FP inflation |
|---|---|---|---|---|---|
| 0.085 | 0.1088 | 0.8508 thr | 0.0926 | 0.0523 | 0.63 |
| 0.17 | 0.1296 | 0.8305 thr | 0.1120 | 0.1003 | 0.61 |
| 0.35 | 0.1583 | 0.7939 thr | 0.1324 | 0.1787 | 0.51 |
| 0.7 | 0.1792 | 0.7629 thr | 0.1574 | 0.3313 | 0.48 |