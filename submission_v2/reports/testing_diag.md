
# check 3 testing diagnostic -- c64s (fold-ensemble)
DISCLOSURE: computed on the 300 TESTING runs. Only the unlabeled count stream is used (features + model scores). No labels, no answer keys, not in any selection; diagnostics only.
test runs 300, total 301.4 h (1.005 h/run) vs dev pool 229.4 h

| dev thr | test alarms/h | dev OOF alarms/h | ratio |
|---|---|---|---|
| 0.8617 | 1.93 | 4.31 | 0.45 |
| 0.8449 | 2.21 | 4.90 | 0.45 |
| 0.8234 | 2.58 | 5.58 | 0.46 |
| 0.7961 | 2.98 | 6.44 | 0.46 |

@0.35-thr alarms/run: mean 2.59 sd 1.42 max 7 zero-runs 21/300
test ens score quantiles p50/90/99/99.9/max: [0.3286 0.4692 0.7904 0.9183 0.9734]
dev OOF p quantiles (single own-fold model per run): [0.3474 0.5762 0.8638 0.9507 0.9932]
v1 precedent: testing rows/h vs val ~1.2-1.3x at comparable thr (Stage-0). Ratios above ~1.5x would be a warning sign.