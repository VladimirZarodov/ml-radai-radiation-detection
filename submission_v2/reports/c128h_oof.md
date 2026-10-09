# Stage-1 c128h: pooled dev OOF, 240 runs, den 229.4 bg-h; alarm NMS sep 15.0s, sigma 1.5s

| target | thr | d_G | d CI95 | c_G | id_G | fpr | FP | v1 d_G | d - v1 | CI(diff) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.085 | 0.8577 | 0.2481 | [0.220,0.279] | 0.2398 | 0.1667 | 0.0828 | 19 | 0.1199 | +0.1282 | [+0.0961,+0.1586] |
| 0.17 | 0.8378 | 0.2866 | [0.256,0.319] | 0.2773 | 0.1843 | 0.1657 | 38 | 0.1287 | +0.1579 | [+0.1233,+0.1897] |
| 0.35 | 0.8054 | 0.3282 | [0.296,0.363] | 0.3162 | 0.2042 | 0.3488 | 80 | 0.1481 | +0.1801 | [+0.1458,+0.2125] |
| 0.7 | 0.7731 | 0.3644 | [0.332,0.399] | 0.3500 | 0.2208 | 0.6976 | 160 | 0.1657 | +0.1986 | [+0.1623,+0.2330] |

### c128h @ target fpr 0.085 (thr 0.8577)
                 n_enc   tp   tc  tid  fp  d_recall  c_recall  id_recall    fpr
category                                                                       
NORM               204    3    3    3   0    0.0147    0.0147     0.0147 0.0000
Medical            265   45   38   33   4    0.1698    0.1434     0.1245 0.0174
Industrial         269   76   70   70   6    0.2825    0.2602     0.2602 0.0262
NuclearMaterial   1422  412  407  254   9    0.2897    0.2862     0.1786 0.0392
Global            2160  536  518  360  19    0.2481    0.2398     0.1667 0.0828
v1 comparator @ target fpr 0.085 (thr 6.2266)
                 n_enc   tp   tc  tid  fp  d_recall  c_recall  id_recall    fpr
category                                                                       
NORM               204    1    0    0   0    0.0049    0.0000     0.0000 0.0000
Medical            265   58   54   51  13    0.2189    0.2038     0.1925 0.0567
Industrial         269   31    7    7   2    0.1152    0.0260     0.0260 0.0087
NuclearMaterial   1422  169  148  105   4    0.1188    0.1041     0.0738 0.0174
Global            2160  259  209  163  19    0.1199    0.0968     0.0755 0.0828

### c128h @ target fpr 0.17 (thr 0.8378)
                 n_enc   tp   tc  tid  fp  d_recall  c_recall  id_recall    fpr
category                                                                       
NORM               204    4    4    4   2    0.0196    0.0196     0.0196 0.0087
Medical            265   48   40   34   7    0.1811    0.1509     0.1283 0.0305
Industrial         269   80   73   73   8    0.2974    0.2714     0.2714 0.0349
NuclearMaterial   1422  487  482  287  21    0.3425    0.3390     0.2018 0.0916
Global            2160  619  599  398  38    0.2866    0.2773     0.1843 0.1657
v1 comparator @ target fpr 0.17 (thr 5.9117)
                 n_enc   tp   tc  tid  fp  d_recall  c_recall  id_recall    fpr
category                                                                       
NORM               204    1    0    0   0    0.0049    0.0000     0.0000 0.0000
Medical            265   61   57   54  22    0.2302    0.2151     0.2038 0.0959
Industrial         269   31    7    7   3    0.1152    0.0260     0.0260 0.0131
NuclearMaterial   1422  185  164  116  13    0.1301    0.1153     0.0816 0.0567
Global            2160  278  228  177  38    0.1287    0.1056     0.0819 0.1657

### c128h @ target fpr 0.35 (thr 0.8054)
                 n_enc   tp   tc  tid  fp  d_recall  c_recall  id_recall    fpr
category                                                                       
NORM               204    4    4    4   3    0.0196    0.0196     0.0196 0.0131
Medical            265   53   44   37  12    0.2000    0.1660     0.1396 0.0523
Industrial         269   88   78   78  17    0.3271    0.2900     0.2900 0.0741
NuclearMaterial   1422  564  557  322  48    0.3966    0.3917     0.2264 0.2093
Global            2160  709  683  441  80    0.3282    0.3162     0.2042 0.3488
v1 comparator @ target fpr 0.35 (thr 5.6020)
                 n_enc   tp   tc  tid  fp  d_recall  c_recall  id_recall    fpr
category                                                                       
NORM               204    1    0    0   0    0.0049    0.0000     0.0000 0.0000
Medical            265   72   68   63  42    0.2717    0.2566     0.2377 0.1831
Industrial         269   36    9    9   8    0.1338    0.0335     0.0335 0.0349
NuclearMaterial   1422  211  189  131  30    0.1484    0.1329     0.0921 0.1308
Global            2160  320  266  203  80    0.1481    0.1231     0.0940 0.3488

### c128h @ target fpr 0.7 (thr 0.7731)
                 n_enc   tp   tc  tid   fp  d_recall  c_recall  id_recall    fpr
category                                                                        
NORM               204    6    5    5    5    0.0294    0.0245     0.0245 0.0218
Medical            265   58   49   41   15    0.2189    0.1849     0.1547 0.0654
Industrial         269   92   82   81   23    0.3420    0.3048     0.3011 0.1003
NuclearMaterial   1422  631  620  350  117    0.4437    0.4360     0.2461 0.5101
Global            2160  787  756  477  160    0.3644    0.3500     0.2208 0.6976
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
| 0.085 | 0.2539 | 0.0872 | 0.1124 | 0.1090 |
| 0.17 | 0.2854 | 0.1308 | 0.1169 | 0.2398 |
| 0.35 | 0.3146 | 0.4361 | 0.1393 | 0.5669 |
| 0.7 | 0.3573 | 0.8067 | 0.1506 | 1.2210 |

## FP distribution @0.35: zero-FP runs 181/240; max/run 3; top5 {112: 3, 99: 3, 180: 3, 1: 2, 257: 2}; top3 share 0.11

## Encounters @0.35: detected 709 / missed 1451 of 2160
missed by isotope (missed/total, top): NatU 175/253, FGPu 148/253, WGPu 133/252, HEU 111/255, RefinedU 101/253, LEU 96/253, DU 94/249, Ra-226 72/84, Th-232 65/85, K-40 63/86
snr_window quantiles missed p10/50/90: [2.6  6.57 9.64]
snr_window quantiles detected p10/50/90: [ 4.22  7.83 14.77]

sample FP alarms @0.35 (run, t_s, metric, label):
  r45 t=460s s=0.933 Co-57
  r184 t=2s s=0.924 Co-57
  r215 t=3522s s=0.921 Co-57
  r53 t=908s s=0.918 NatU
  r99 t=3324s s=0.913 NatU
  r137 t=0s s=0.908 Co-57
  r294 t=790s s=0.907 Co-60
  r53 t=168s s=0.900 NatU

## SYNTH-SHIFT (frozen params, report only) OOF-scores
| target | d self-cal | fpr self-cal | d @dev-thr | fpr @dev-thr | FP inflation |
|---|---|---|---|---|---|
| 0.085 | 0.0690 | 0.8580 thr | 0.0690 | 0.0872 | 1.05 |
| 0.17 | 0.0880 | 0.8268 thr | 0.0810 | 0.1134 | 0.68 |
| 0.35 | 0.1106 | 0.7925 thr | 0.1009 | 0.2659 | 0.76 |
| 0.7 | 0.1227 | 0.7609 thr | 0.1185 | 0.5275 | 0.76 |