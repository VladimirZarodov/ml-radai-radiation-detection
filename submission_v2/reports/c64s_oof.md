# Stage-1 c64s: pooled dev OOF, 240 runs, den 229.4 bg-h; alarm NMS sep 15.0s, sigma 1.5s

| target | thr | d_G | d CI95 | c_G | id_G | fpr | FP | v1 d_G | d - v1 | CI(diff) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.085 | 0.8617 | 0.4435 | [0.420,0.466] | 0.4273 | 0.2880 | 0.0828 | 19 | 0.1199 | +0.3236 | [+0.2990,+0.3478] |
| 0.17 | 0.8449 | 0.4926 | [0.468,0.515] | 0.4750 | 0.3171 | 0.1657 | 38 | 0.1287 | +0.3639 | [+0.3396,+0.3897] |
| 0.35 | 0.8234 | 0.5347 | [0.512,0.555] | 0.5157 | 0.3421 | 0.3488 | 80 | 0.1481 | +0.3866 | [+0.3612,+0.4123] |
| 0.7 | 0.7961 | 0.5736 | [0.553,0.595] | 0.5509 | 0.3676 | 0.6976 | 160 | 0.1657 | +0.4079 | [+0.3831,+0.4321] |

### c64s @ target fpr 0.085 (thr 0.8617)
                 n_enc   tp   tc  tid  fp  d_recall  c_recall  id_recall    fpr
category                                                                       
NORM               204   13   13   13   1    0.0637    0.0637     0.0637 0.0044
Medical            265   69   64   60   1    0.2604    0.2415     0.2264 0.0044
Industrial         269  161  139  138   0    0.5985    0.5167     0.5130 0.0000
NuclearMaterial   1422  715  707  411  17    0.5028    0.4972     0.2890 0.0741
Global            2160  958  923  622  19    0.4435    0.4273     0.2880 0.0828
v1 comparator @ target fpr 0.085 (thr 6.2266)
                 n_enc   tp   tc  tid  fp  d_recall  c_recall  id_recall    fpr
category                                                                       
NORM               204    1    0    0   0    0.0049    0.0000     0.0000 0.0000
Medical            265   58   54   51  13    0.2189    0.2038     0.1925 0.0567
Industrial         269   31    7    7   2    0.1152    0.0260     0.0260 0.0087
NuclearMaterial   1422  169  148  105   4    0.1188    0.1041     0.0738 0.0174
Global            2160  259  209  163  19    0.1199    0.0968     0.0755 0.0828

### c64s @ target fpr 0.17 (thr 0.8449)
                 n_enc    tp    tc  tid  fp  d_recall  c_recall  id_recall    fpr
category                                                                         
NORM               204    18    18   18   2    0.0882    0.0882     0.0882 0.0087
Medical            265    77    71   66   5    0.2906    0.2679     0.2491 0.0218
Industrial         269   171   149  148   1    0.6357    0.5539     0.5502 0.0044
NuclearMaterial   1422   798   788  453  30    0.5612    0.5541     0.3186 0.1308
Global            2160  1064  1026  685  38    0.4926    0.4750     0.3171 0.1657
v1 comparator @ target fpr 0.17 (thr 5.9117)
                 n_enc   tp   tc  tid  fp  d_recall  c_recall  id_recall    fpr
category                                                                       
NORM               204    1    0    0   0    0.0049    0.0000     0.0000 0.0000
Medical            265   61   57   54  22    0.2302    0.2151     0.2038 0.0959
Industrial         269   31    7    7   3    0.1152    0.0260     0.0260 0.0131
NuclearMaterial   1422  185  164  116  13    0.1301    0.1153     0.0816 0.0567
Global            2160  278  228  177  38    0.1287    0.1056     0.0819 0.1657

### c64s @ target fpr 0.35 (thr 0.8234)
                 n_enc    tp    tc  tid  fp  d_recall  c_recall  id_recall    fpr
category                                                                         
NORM               204    24    24   24   4    0.1176    0.1176     0.1176 0.0174
Medical            265    87    80   73   8    0.3283    0.3019     0.2755 0.0349
Industrial         269   178   154  153   4    0.6617    0.5725     0.5688 0.0174
NuclearMaterial   1422   866   856  489  64    0.6090    0.6020     0.3439 0.2790
Global            2160  1155  1114  739  80    0.5347    0.5157     0.3421 0.3488
v1 comparator @ target fpr 0.35 (thr 5.6020)
                 n_enc   tp   tc  tid  fp  d_recall  c_recall  id_recall    fpr
category                                                                       
NORM               204    1    0    0   0    0.0049    0.0000     0.0000 0.0000
Medical            265   72   68   63  42    0.2717    0.2566     0.2377 0.1831
Industrial         269   36    9    9   8    0.1338    0.0335     0.0335 0.0349
NuclearMaterial   1422  211  189  131  30    0.1484    0.1329     0.0921 0.1308
Global            2160  320  266  203  80    0.1481    0.1231     0.0940 0.3488

### c64s @ target fpr 0.7 (thr 0.7961)
                 n_enc    tp    tc  tid   fp  d_recall  c_recall  id_recall    fpr
category                                                                          
NORM               204    30    30   30    7    0.1471    0.1471     0.1471 0.0305
Medical            265    97    88   81   20    0.3660    0.3321     0.3057 0.0872
Industrial         269   189   163  162   18    0.7026    0.6059     0.6022 0.0785
NuclearMaterial   1422   923   909  521  115    0.6491    0.6392     0.3664 0.5014
Global            2160  1239  1190  794  160    0.5736    0.5509     0.3676 0.6976
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
| 0.085 | 0.4225 | 0.0872 | 0.1124 | 0.1090 |
| 0.17 | 0.4652 | 0.2398 | 0.1169 | 0.2398 |
| 0.35 | 0.5056 | 0.5015 | 0.1393 | 0.5669 |
| 0.7 | 0.5303 | 0.8503 | 0.1506 | 1.2210 |

## FP distribution @0.35: zero-FP runs 178/240; max/run 5; top5 {215: 5, 24: 3, 122: 3, 74: 2, 206: 2}; top3 share 0.14

## Encounters @0.35: detected 1155 / missed 1005 of 2160
missed by isotope (missed/total, top): NatU 155/253, FGPu 96/253, Ra-226 72/84, LEU 71/253, Th-232 62/85, RefinedU 62/253, WGPu 59/252, DU 58/249, HEU 55/255, K-40 46/86
snr_window quantiles missed p10/50/90: [2.55 6.48 9.08]
snr_window quantiles detected p10/50/90: [ 3.55  7.46 13.6 ]

sample FP alarms @0.35 (run, t_s, metric, label):
  r148 t=1470s s=0.918 F-18
  r24 t=1760s s=0.900 FGPu
  r160 t=2420s s=0.898 HEU
  r141 t=1046s s=0.897 LEU
  r7 t=1552s s=0.897 K-40
  r24 t=1244s s=0.889 FGPu
  r229 t=1114s s=0.886 DU
  r163 t=2818s s=0.884 FGPu

## SYNTH-SHIFT (frozen params, report only) OOF-scores
| target | d self-cal | fpr self-cal | d @dev-thr | fpr @dev-thr | FP inflation |
|---|---|---|---|---|---|
| 0.085 | 0.0616 | 0.9091 thr | 0.1338 | 0.4883 | 5.89 |
| 0.17 | 0.0866 | 0.8914 thr | 0.1537 | 0.7455 | 4.50 |
| 0.35 | 0.1222 | 0.8719 thr | 0.1847 | 1.0332 | 2.96 |
| 0.7 | 0.1486 | 0.8483 thr | 0.2167 | 1.5477 | 2.22 |