## 1. Parity: harness scorer vs official_scorer.py
runs 25-124 @ thr 5.7492: rows=158
| source | d | fpr |
|---|---|---|
| official_scorer | 0.1453 | 0.2198 |
| harness        | 0.1453 | 0.2198 |
max diffs: d 0.00e+00, fpr 1.94e-16 -> PASS

### target fpr 0.085  (thr=6.2266)
                 n_enc   tp   tc  tid  fp  d_recall  c_recall  id_recall    fpr
category                                                                       
NORM               204    1    0    0   0    0.0049    0.0000     0.0000 0.0000
Medical            265   58   54   51  13    0.2189    0.2038     0.1925 0.0567
Industrial         269   31    7    7   2    0.1152    0.0260     0.0260 0.0087
NuclearMaterial   1422  169  148  105   4    0.1188    0.1041     0.0738 0.0174
Global            2160  259  209  163  19    0.1199    0.0968     0.0755 0.0828
Global d CI [0.1062, 0.1324]  fpr CI [0.0479, 0.122]

### target fpr 0.17  (thr=5.9117)
                 n_enc   tp   tc  tid  fp  d_recall  c_recall  id_recall    fpr
category                                                                       
NORM               204    1    0    0   0    0.0049    0.0000     0.0000 0.0000
Medical            265   61   57   54  22    0.2302    0.2151     0.2038 0.0959
Industrial         269   31    7    7   3    0.1152    0.0260     0.0260 0.0131
NuclearMaterial   1422  185  164  116  13    0.1301    0.1153     0.0816 0.0567
Global            2160  278  228  177  38    0.1287    0.1056     0.0819 0.1657
Global d CI [0.1143, 0.142]  fpr CI [0.1177, 0.2224]

### target fpr 0.35  (thr=5.6020)
                 n_enc   tp   tc  tid  fp  d_recall  c_recall  id_recall    fpr
category                                                                       
NORM               204    1    0    0   0    0.0049    0.0000     0.0000 0.0000
Medical            265   72   68   63  42    0.2717    0.2566     0.2377 0.1831
Industrial         269   36    9    9   8    0.1338    0.0335     0.0335 0.0349
NuclearMaterial   1422  211  189  131  30    0.1484    0.1329     0.0921 0.1308
Global            2160  320  266  203  80    0.1481    0.1231     0.0940 0.3488
Global d CI [0.1332, 0.1625]  fpr CI [0.27, 0.4412]

### target fpr 0.7  (thr=5.2640)
                 n_enc   tp   tc  tid   fp  d_recall  c_recall  id_recall    fpr
category                                                                        
NORM               204    2    0    0    0    0.0098    0.0000     0.0000 0.0000
Medical            265   78   74   69   79    0.2943    0.2792     0.2604 0.3444
Industrial         269   39   10   10   13    0.1450    0.0372     0.0372 0.0567
NuclearMaterial   1422  239  213  150   68    0.1681    0.1498     0.1055 0.2965
Global            2160  358  297  229  160    0.1657    0.1375     0.1060 0.6976
Global d CI [0.1499, 0.1812]  fpr CI [0.5798, 0.8421]

### v1 at portal-measured thr 5.7492 (dev set, reference)
                 n_enc   tp   tc  tid  fp  d_recall  c_recall  id_recall    fpr
category                                                                       
NORM               204    1    0    0   0    0.0049    0.0000     0.0000 0.0000
Medical            265   66   62   59  31    0.2491    0.2340     0.2226 0.1352
Industrial         269   33    8    8   4    0.1227    0.0297     0.0297 0.0174
NuclearMaterial   1422  200  178  125  21    0.1406    0.1252     0.0879 0.0916
Global            2160  300  248  192  56    0.1389    0.1148     0.0889 0.2441

## 3. HIGH-BG shift slice (20 % dev, thr from dev selection)
| target fpr | thr | d | fpr |
|---|---|---|---|
| 0.085 | 6.2266 | 0.1124 | 0.1090 |
| 0.17 | 5.9117 | 0.1169 | 0.2398 |
| 0.35 | 5.6020 | 0.1393 | 0.5669 |
| 0.7 | 5.2640 | 0.1506 | 1.2210 |

## 4. SYNTH-SHIFT folds (frozen shift_synth params)
| target | thr (self-cal) | d (self-cal) | d @ dev-thr | fpr @ dev-thr | FP inflation @ dev-thr |
|---|---|---|---|---|---|
| 0.085 | 7.8652 | 0.0546 | 0.1116 | 2.1362 | 25.79 |
| 0.17 | 7.5791 | 0.0588 | 0.1287 | 3.3308 | 20.11 |
| 0.35 | 7.2546 | 0.0681 | 0.1509 | 4.9308 | 14.14 |
| 0.7 | 6.8463 | 0.0852 | 0.1736 | 7.2632 | 10.41 |

## 5. v1 baseline summary (dev pool; per-category at targets)
| target | thr | d_G | c_G | id_G | fpr | d NORM | d Med | d Ind | d NM |
|---|---|---|---|---|---|---|---|---|---|
| 0.085 | 6.2266 | 0.1199 | 0.0968 | 0.0755 | 0.0828 | 0.0049 | 0.2189 | 0.1152 | 0.1188 |
| 0.17 | 5.9117 | 0.1287 | 0.1056 | 0.0819 | 0.1657 | 0.0049 | 0.2302 | 0.1152 | 0.1301 |
| 0.35 | 5.6020 | 0.1481 | 0.1231 | 0.0940 | 0.3488 | 0.0049 | 0.2717 | 0.1338 | 0.1484 |
| 0.7 | 5.2640 | 0.1657 | 0.1375 | 0.1060 | 0.6976 | 0.0098 | 0.2943 | 0.1450 | 0.1681 |