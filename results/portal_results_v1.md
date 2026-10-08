# Portal results log

## Measurement #1 -- 2026-10-02, submission name "first_try"

Uploaded file: `submission_mx31_v1_fpr_mid.csv` -- **ACCEPTED** by the portal.

| property | value |
|---|---|
| threshold (metric_1 >=) | 5.7492 |
| rows | 413 |
| md5 | `1683df759008bc46438029bd0f45659e` |
| header | quoted `"#"` first cell (post official-reader fix) |

Portal metrics:

| category | d_recall | c_recall | id_recall | fpr |
|---|---|---|---|---|
| Global | 0.0946 | 0.0737 | 0.0544 | 0.535 |
| Industrial | 0.0394 | 0.0088 | 0.0088 | 0.0347 |
| Medical | 0.1685 | 0.1166 | 0.0972 | 0.2223 |
| NORM | 0.0 | 0.0 | 0.0 | 0.0 |
| NuclearMaterial | 0.1025 | 0.0905 | 0.0629 | 0.2779 |

Interpretation (denominator ~286.5 testing background-hours):
~153 false positives (NM ~80, Medical ~64, Industrial ~10, NORM 0);
~260 rows matched answer-key windows.  Global fpr = SUM of category fprs ->
the portal's "fpr" is `FPR[Global]` in `answers_to_metrics` (ALL FP labels);
the NuclearMaterial-only sum is the separate `far` field (earlier note here
corrected).

Derived cross-checks (see `official_scorer.py` / README):

* Implied testing encounter count: tp/d_recall = 260/0.0946 ~= 2748
  (~9.2/run) -- matches validation answer key 943/100 = 9.4/run.
* Implied testing encounter-window time: 301.4 - 286.5 ~= 14.9 h -> mean
  window ~19.6 s -- validation answer-key mean width 19.5 s (median 14 s).
  Window geometry confirmed: time_start ~= CA-5 s, time_stop ~= CA+8 s.
* FP model error of the pre-upload calibration: expected 48, actual 153
  (x3.2).  Cause: pool-(b) "background" peaks (GT dmin>150 s) undercount
  official FPs, AND the testing background FP tail runs ~2.4x the training
  rate at this threshold (validation at the same thr: FP 21 / 95.5 h =
  0.22/h vs testing 0.535/h).
* Recall: d 0.0946 vs 0.145 predicted from validation under the faithful
  rules -> testing sources effectively weaker/harder than validation.
  Transfer factor r = d_test/d_val ~= 0.65 at equal threshold.
* NORM d_recall = 0 reproduced exactly by the faithful local scorer (frozen
  detector never alarms on K-40/Ra-226/Th-232 encounters).
* Alarm-timing diagnostics (validation): winner-alarm offset from CA is
  median +0.3 s (q05..q95 -2.0..+2.5 s) -- timing is already inside the
  window; smoothing/centroid re-timing options were tested and are HARMFUL
  (see README "time_options" section); consolidation (sep 60/90 s) is
  neutral.  The FP problem is threshold-anchoring, not localisation.

Status of the other four written files: **STALE -- do not upload** (thresholds
were calibrated with the disproven pool-(b) FA model; see README).  The next
upload set will be re-solved against the faithful scorer anchored to this
measurement (Checkpoint D proposal in README).
