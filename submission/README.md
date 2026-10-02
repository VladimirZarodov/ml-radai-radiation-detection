# RADAI submission pipeline (mx31 v1)

Everything in this folder is NEW (added Oct 2026) for running the frozen
matched-filter detector (`Matched_Filter_Detection.ipynb`, untouched) on the
official blind test file `testing_v4.3.h5` and producing a leaderboard
submission CSV. No pre-existing repo file was modified.

## Hard rules honoured

* Detector is **frozen**: 7 background components + 61 templates fit from the
  training file only, `TRAIN_RUNS = [0, 3..19]` (notebook cells 17, verbatim).
* Testing data is used for **inference only**: `score_run` on testing reads
  only `listmode/dt` + `listmode/energy` (those are the only listmode fields
  the testing file has anyway). All calibration pools, thresholds, masks and
  evaluation use the **training** file exclusively.
* The notebook's validation results (runs 25-124) must keep reproducing:
  `regression_check.py` re-derives all 123 notebook-run scores from the raw
  h5 and asserts bit-identical arrays AND the headline recall
  25.2 / 40.5 / 70.0 % at FAR 1/3/10 ep/h (passed: all diffs 0.0).
* Alarms are emitted as **mx31 local maxima (rise onsets) with a
  min-separation**, threshold-free, then filtered by a fixed emission floor:
  raising the `metric_1` sweep cut always yields a strict SUBSET of the
  emitted rows (required by the portal's sweep semantics).

## Files

| file | role |
|---|---|
| `radai_lib.py` | frozen detector library (streaming h5 reader, EM fit, zA/mx31 scoring, calibration protocol, mx31 peak extraction), parameterised by h5 path + run list; formulas/dtype casts verbatim from the notebook |
| `regression_check.py` | Step-1 gate: refactored code vs `_scratch/nb_mf_cache.pkl` (bit-exact) + headline numbers |
| `make_submission.py` | staged pipeline: `score` / `calib` / `emit` / `csv` / `report` |
| `local_scorer.py` | validation-only: sweep-reproduction proof + per-category d/c/id/fpr table (hypotheses documented in its docstring) |
| `verify_submission.py` | final CSV checklist: format, nestedness, no-leakage provenance, git cleanliness, end-to-end re-score |
| `label_mapping.csv` | 61 internal template names -> 22 official labels |
| `validation_dryrun_placeholder.csv` | validation dry-run; satisfies every rule of the official validator (only the portal UI template is unconfirmed) |
| `validation_sweep_curve.csv` | recall/alarms/FA vs threshold, validation, plateau rule |
| `csv_format.py` | SINGLE config place for the CSV header/row layout (`HEADER` writes the first cell quoted as `"#"` — required by the official `comment="#"` reader, see below) |
| `calibration.json` | FAR 1/3/10 targets + emission floor (2.1593, 25 bg peaks/h) |
| `file_thresholds.json` | per-bucket threshold, FA expected/Poisson/bootstrap/drift scenarios, SAFE stats — source for threshold-only re-adjustment after portal feedback |
| `report_submission.py` | per-file evidence: md5, pandas dtypes, histograms, nestedness |
| `validate_official_emulation.py` | 1:1 emulation of the organiser's `validate_submission`/`calculate_fp_info` rules (from `radai/radai/evaluation/`; package needs numba, hence emulation) |
| `official_scorer.py` | **FAITHFUL local scorer**: rebuilds the validation answer key exactly as `build_answerkey()` (SNR-window geometry) and replicates `calculate_fp_info` + `answers_to_metrics`; stages `ak` / `score` / `diag`; supersedes `local_scorer.py` for portal-metric emulation |
| `time_options.py` | Checkpoint-D task 5: alarm-time / consolidation options evaluated with the faithful scorer on validation at equal official fpr, run-cluster bootstrap (result: keep base timing) |
| `portal_results.md` | log of real portal measurements (measurement #1 = mid file, accepted) |
| `submission_mx31_v1_fpr_mid.csv` | uploaded & ACCEPTED 2026-10-02 ("first_try", fpr 0.535) -- see `portal_results.md` |
| `submission_mx31_v1_fpr_{low,high,low_safe,mid_safe}.csv` | **STALE -- DO NOT UPLOAD** (2026-10-02): thresholds were calibrated with the pool-(b) FA model, which portal measurement #1 disproved (~3x FP underestimation; see `portal_results.md`). Replace only after Checkpoint-D approval with the re-anchored thresholds below |

Caches land in `_scratch/` (gitignored): `sub_detector.pkl`,
`sub_training_scores.pkl` (300 runs, with GT fields), `sub_testing_scores.pkl`
(300 runs, blind), `sub_calibration.json`, `sub_validation_alarms.pkl`,
`sub_testing_alarms.pkl`.

## Run order (from repo root, in `.venv`)

```
.venv\Scripts\python.exe submission\regression_check.py
.venv\Scripts\python.exe submission\make_submission.py score --split training
.venv\Scripts\python.exe submission\make_submission.py score --split testing
.venv\Scripts\python.exe submission\make_submission.py calib --floor-peak-h 25
.venv\Scripts\python.exe submission\make_submission.py emit --split validation
.venv\Scripts\python.exe submission\make_submission.py emit --split testing
.venv\Scripts\python.exe submission\local_scorer.py proof --sep 15
.venv\Scripts\python.exe submission\make_submission.py csv --split validation
.venv\Scripts\python.exe submission\verify_submission.py submission\validation_dryrun_placeholder.csv --split validation
# final testing bucket CSVs (header rules confirmed from radai/evaluation;
# --i-confirm-template is the explicit go-ahead flag of the writer):
.venv\Scripts\python.exe submission\make_submission.py csv --split testing --bucket low --i-confirm-template   # mid|high|low_safe|mid_safe likewise
.venv\Scripts\python.exe submission\verify_submission.py submission\submission_mx31_v1_fpr_low.csv --split testing
.venv\Scripts\python.exe submission\validate_official_emulation.py
.venv\Scripts\python.exe submission\report_submission.py
```

Upload naming (per portal JS rules, `[A-Za-z0-9-_]`, name ends `.csv`):
file `submission_mx31_v1.csv`, submission name `submission-mx31-v1`.

## v1 identification-coverage limitations (accepted for v1 — Checkpoint B decision 3)

* The 61-template argmax that produces `label_1` **essentially never selects
  NORM templates** on real NORM sources, and rarely Industrial ones: on the
  validation hold-out, the highest-metric peak inside CA+-60 s (at ANY
  height, threshold-free) is labelled within the true category for
  Medical 66 % and Nuclear Material 67 % of encounters, but
  **NORM 0.0 %** (best-peak labels are HEU/Co-57/Tl-201/... instead) and
  **Industrial 16.7 %**.  Any-height ceilings: c_recall ~46.9 %, id_recall
  ~20.0 % overall.  This is honest frozen-detector behaviour (NORM spectra
  are close to the fitted background components); the detector was NOT
  changed for v1.  A v1.1 relabelling proposal will follow once the portal's
  real c/id numbers are visible.
* Official labels with **no template** (never appear in training at all):
  **Cu-67, Lu-177** (and `Ir-192_industrial` exists but is shielded variants
  absent from training -> maps to `Ir-192`, template present).
* Three training combos without templates (untemplatized shield variants):
  `RefinedU-2.5kg_sh2`, `RefinedU-10kg_sh2`, `HEU-10kg_sh2` -- their base
  labels CAN still be emitted from other variants.
* Adding these is a detector change -> v1.1 proposal, not done here.

## Three-file bucket strategy (Checkpoint B decision 2 — **FA model superseded at Checkpoint D**)

Leaderboard evidence suggests portal `fpr = false-alarms / ~286.5 testing
background hours` (fpr values are integers/286.5), one fpr per upload, portal
threshold sweep "coming soon".  So we prepare files containing ONLY peaks
above a per-file threshold, targeting ~30 % margin under bucket edges:

| file | target FA | bucket edge | threshold | name | status |
|---|---|---|---|---|---|
| fpr_low | ~24 (0.085/h) | 35 (0.125) | 6.0470 | `submission_mx31_v1_fpr_low.csv` | **STALE** |
| fpr_mid | ~49 (0.17/h) | 71 (0.25) | 5.7492 | `submission_mx31_v1_fpr_mid.csv` | uploaded 0.535 (accepted; thresholds wrong by x3) |
| fpr_high | ~200 (0.70/h) | 285 (1.0) | 5.2796 | `submission_mx31_v1_fpr_high.csv` | **STALE** |
| fpr_low_safe | worst-case <= edge | 35 | 6.5666 | `submission_mx31_v1_fpr_low_safe.csv` | **STALE** |
| fpr_mid_safe | worst-case <= edge | 71 | 6.0023 | `submission_mx31_v1_fpr_mid_safe.csv` | **STALE** |

**STALE = do not upload.**  The pool-(b) "background peak/hour" FA model
behind these thresholds is empirically dead: the mid file produced 153 FPs
where the model predicted ~48 (x3.2).  Two independent reasons, both proven
in the faithful scorer (see next sections): (1) official FPs include alarms
in the narrow wings of real encounters (windows are only ~15-20 s wide, not
the +-150 s we excluded), and (2) the testing background FP tail runs ~2.4x
the training rate at this threshold.  Replacement thresholds are PROPOSED in
"Checkpoint D" below and gated on user approval.

All five files were WRITTEN (Checkpoint C decisions 1–3: target thresholds
chosen; first upload = `mid`, its revealed fpr used as a *measurement*, later
files adjust by threshold only — never labels or detector). The `_wide` floor
file was skipped by decision. Full per-file evidence (md5, dtypes, histograms,
strict nestedness low subset-of mid subset-of high, safe subset-of base) from
`report_submission.py` (latest run logged to `_scratch/report_sub2.out`, after
the 2026-10-02 quoted-header fix).

Suggested portal submission names ([A-Za-z0-9-_], 5–100 chars, unique):
`mx31_frozen_v1_low`, `mx31_frozen_v1_mid`, `mx31_frozen_v1_high`,
`mx31_frozen_v1_low_safe`, `mx31_frozen_v1_mid_safe`.

Thresholds (old, superseded): calibrated on TRAINING pool (b) background
peaks + rows-per-hour drift band (`file_thresholds.json` scenarios) — the FA
model behind them is invalid per portal measurement #1; regenerate only from
the re-anchored values in the "Checkpoint D" section below with
`make_submission.py csv --split testing --bucket X --i-confirm-template`
(the bucket thresholds in `file_thresholds.json` must be UPDATED there
first, once approved).

**CSV format: the organiser's evaluation code is IN THIS REPO (`radai/`).**
`submission/validate_official_emulation.py` emulates it 1:1 (the package
itself needs numba, unavailable here).  Everything below comes from
`radai/radai/evaluation/{evaluation.py, visualization.py, tools.py}`:

* Required columns: `run_id, time, label_1, metric_1` (+ our numeric
  `time_start/time_stop`, which `compute_metrics` uses for variation lookup
  and the duration penalty; extra `metric_2/3`, `label_2/3` are ignored).
* **The official reader is `pd.read_csv(path, comment="#")`.**  An UNQUOTED
  leading `#` header cell makes pandas discard the whole header row (columns
  lost -> validation FAILS, and the first data row is swallowed).  Discovered
  by the emulation on 2026-10-02; fixed by writing the header as
  `"#",run_id,...` (quoted cell survives with AND without `comment="#"`).
* Matching: an alarm is a true positive iff its `time` lies INSIDE
  `[time_start, time_stop]` of an answer-key encounter (same run).  The
  matched alarm with the HIGHEST `metric_1` sets that encounter's label
  (detection/category/id); additional matched alarms are free.  No
  tolerance-radius rule exists -- the ±30/60/120 s hypotheses in
  `local_scorer.py` are superseded by this window rule (our center±60 s
  estimates bracket it; the window is the SNR>5%-of-peak span).
* False positive iff the time is in NO encounter window; penalty =
  `1 + floor(duration/160 s)` -> our 4 s alarms count exactly 1 each.
* `fpr = FP count / (total run hours - sum of encounter-window hours)`,
  i.e. false alarms per background-only hour -- **this matches the
  leaderboard evidence (~286.5 testing background hours) and the definition
  adopted at Checkpoint B/C**.
* `metrics.loc["Global", "far"]` sums fpr over `alarm_categories =
  ["NuclearMaterial"]` ONLY -- a separate field.  **Portal measurement #1
  settles which field the leaderboard shows**: its Global `fpr` 0.535 equals
  the SUM of the four category fprs, so the portal reports
  `FPR[Global] = ALL false positives / bg-hours` (by FP label category), not
  `far`.  Earlier note here claiming Global might count NM-only: corrected.
* Label vocabulary (exact membership): our 24 labels all pass.  Note the
  official mapping puts **Am-241 in Industrial** (challenge text grouped it
  with Nuclear Material); `DepletedU`, `Background`, `BKG` are extra legal
  strings we never emit.  Category tables quoted earlier used the challenge
  grouping; submission files are unaffected (label strings identical).
* Remaining validation risk: `time <= ak.time.max() + 122000 ms` (hard-coded
  for v4.3) can't be checked locally; our max time is 3,607,000 ms and any
  testing encounter later than 3,485,000 ms clears it -- near-certain for
  300 one-hour runs, but unverified until the server sees the file.

The format remains a **placeholder** in the sense that the portal UI may want
a slightly different template, but every rule the scoring code itself checks
is now satisfied locally.  Config still lives in ONE place: `csv_format.py`.

## Checkpoint D — official encounter-window semantics, faithful scorer, re-anchored thresholds

### D.1 How the answer-key windows are defined (task 2, quoted from `radai/`)

The repo DOES contain code that builds an answer key from any RADAI h5
(including a training-style one): `radai/radai/evaluation/answerkey.py`,
`build_answerkey(path, snr_relative_thresh=0.05, time_step_size=1000)`.
For each GT encounter (run `sources/id`+`sources/time`, CA time `ts`):

```python
# answerkey.py
 92:  center_idx = int(round((ts - time_step_size / 2) / time_step_size))  # CA 1-s bin
100:  while n_below_thresh < 25:          # greedy bounding window: grow left/right
103:      window_snrs = snrs.iloc[center_idx - width_left : center_idx + width_right + 1]
105:      n_below_thresh = (window_snrs < 0.01).sum()
140:  snr_peak = window_snrs.max()
141:  snrs_above_thresh = window_snrs > (snr_peak * snr_relative_thresh)   # STRICT > 5% of peak
143:  time_start = snrs_above_thresh.idxmax() - data.time_step_size / 2      # first above bin, left edge
144:  time_stop  = snrs_above_thresh.iloc[::-1].idxmax() + ... / 2           # last above bin, right edge
145:  time_max   = window_snrs.idxmax()
```

The SNR series itself is GT-attributed counts, not measured:
`data_set.py:960-1152 get_snr_for_run()` histogrammes each listmode event by
(`time`, `id`) into 1-s bins (`time = np.cumsum(dt, uint32)/1e3` ms,
`data_set.py:484`; bin edges `arange(0, get_end_time, 1000, uint32)`;
`get_end_time = (end_timestamp - start_timestamp)*1e3`, `data_set.py:207-220`)
and returns `snr = S / sqrt(S + B)` per source name (method default
`s/sqrt(s+b)`), `B` = the id-0 background column.  No energy cut.  Labels are
tier-0 raw source names (`tools.py:117-128` + `label_manager.py` tier map).

FP-hours denominator (both variants in `evaluation.py`):

```python
303-305: total_time_hr = sum(data.get_end_time(run_id)/1000/3600 for run in ak.run_ids)
374-376: total_time_bkg_only_hr = total_time_hr \
             - (ak.time_stop - ak.time_start).sum()/1000.0/3600.0
211-213: answers_to_metrics: total_time_hr = ak.run_id.max()*1.0 \
             - (ak.time_stop - ak.time_start).sum()/1000/3600     # pseudo-total variant
```

Matching (`visualization.py:978-1039`, quoted in the section above): alarm
`time` inside `[time_start, time_stop]` -> TP for that encounter; else FP
(penalty `1 + floor((time_stop-time_start)/160 s)` = 1 for our 4 s rows).
**`time_start/time_stop` of the ALARM are never used for matching** -- only
for the FP duration penalty and for the count/shape variation diagnostics
(`evaluation.py:329+`, not in the core metrics).  So alarm span must simply
stay <= 160 s; our ±2 s is fine and needs no change.

### D.2 Validation answer key rebuilt faithfully (`official_scorer.py ak`)

Run: `.venv\Scripts\python.exe submission\official_scorer.py ak` (cached to
`_scratch/sub_val_answerkey.csv` + `..._stats.json`).  Validation split =
training runs 25–124 (`radai_lib.TEST_RUNS`), matching the user's
"training runs 25–124".  Stats (943 encounters / 100 runs = 9.43/run):

| quantity | validation | portal-consistent implication |
|---|---|---|
| window width | mean **19.5 s**, median 14, p90 38, max 98, none >160 | testing Σwin = 301.4−286.5 = 14.9 h over ~2750 enc → mean 19.5 s ✓ |
| window vs CA | start median **−5.2 s**, stop median **+7.7 s** | user's "~±10 s around CA" hypothesis CONFIRMED |
| Σ window hours | 5.098 h of 100.64 run-hours | ×3 ≈ 15.3 h ≈ 14.9 h → portal denominator = **`compute_metrics` variant** (Σ run lengths − Σ windows), not `ak.run_id.max()` |
| overlap violations | 0 | official RuntimeError never triggers |
| encounters per category | NORM 93, Medical 110, Industrial 118, NM 622 | NM-dominated like the portal row counts suggest |

### D.3 Faithful scorer vs portal (task 3) -- validation at thr 5.7492

`.venv\Scripts\python.exe submission\official_scorer.py score` (or `diag`) —
158 rows (21 FP, 137 matched):

| category | enc | d_recall | c_recall | id_recall | fpr |
|---|---|---|---|---|---|
| NORM | 93 | 0.0000 | 0.0 | 0.0 | 0.000 |
| Medical | 110 | 0.2273 | 0.2182 | 0.2091 | 0.126 |
| Industrial | 118 | 0.0932 | 0.0169 | 0.0169 | 0.010 |
| NuclearMaterial | 622 | 0.1624 | 0.1511 | 0.1077 | 0.084 |
| **Global** | 943 | **0.1453** | **0.1273** | **0.0976** | **0.220** |

vs portal testing (same threshold): d 0.0946, c 0.0737, id 0.0544, fpr 0.535.
**Honest verdict**: the replication reproduces the portal *structure*
exactly (NORM d = 0.0; per-category ordering Medical > NM > Industrial >
NORM; Global = sum; c/d 0.88 vs portal 0.78; id/d 0.67 vs 0.58) and the
window geometry (D.2 identities), but NOT the level: testing is **~3x harder
on FP** (FP_test/FP_val = 153/21 = 7.29 per file, i.e. rate ratio 2.43) and
**~1.5x harder on recall** (d ratio 0.65).  That is a real train→test drift
in the background FP tail and source strength, invisible to the old
row-count-based correction (rows/h were nearly equal: 1.65 vs 1.44/bgh).
Transfer assumptions used below: `FP_test(thr) ≈ 7.29 × FP_val(thr)`
(single anchor; run-cluster bootstrap gives FP_val CI [13,31] at 5.75 →
factor band [5.0, 11.7]) and `d_test(thr) ≈ 0.65 × d_val(thr)`.

### D.4 Timing diagnostics (task 4)

`official_scorer.py diag`: alarm time is the mx31 rising-plateau FIRST index
= window centre at which `bestA` (max template z-score, 2 s windows) reached
its episode argmax (`radai_lib.mx31_peaks`).  Winner-alarm offset from CA:
**median +0.3 s, q05–q95 −2.0…+2.5 s** — timing is already centred inside
the [−5,+8] s windows.  At thr 5.75 on validation: 158 rows = 137 in-window +
13 wing (≤150 s of a CA but outside every window) + 8 far.  The wings sit at
offsets −60…−130 s (secondary episode peaks), only 3/13 are suppressable by a
higher-metric neighbour.  Blind proxy on testing rows (no labels used): 28/413
have a higher-metric row within 150 s (7 %) vs 4 % on validation.

### D.5 Time-localisation options — EVALUATED AND REJECTED (task 5)

`time_options.py` (same official rules, per-run exact scoring, run-cluster
bootstrap B=300).  Compared at equal official fpr (FP_val = 21 ≈ portal
0.535) and at fixed thr 5.75:

| option | thr(FPv≈21) | d_val @FPv21 | FP_val @5.75 | d_val @5.75 (CI) |
|---|---|---|---|---|
| **base** (keep) | 5.75 | **0.1453** | **21** | **0.1453** [0.124,0.168] |
| smooth5 (σ=5 s bestA argmax) | 6.05 | 0.1188 | 36 | 0.1294 [0.109,0.151] |
| smooth10 (σ=10 s) | 6.50 | 0.0912 | 51 | — |
| centroid (bestA−P5)+ over episode | none ≤21 | — | 109 | 0.0520 |
| sep30 (60 s NMS) | 5.75 | 0.1453 | 21 | 0.1453 (identical peaks) |
| sep45 (90 s NMS) | 5.70 | 0.1463 | 19 | 0.1442 [0.123,0.166] |

Conclusion: because winner offsets are already ≈0, every time-shift variant
MOVES alarms out of the narrow windows and multiplies FPs; consolidation
(sep45) is within noise (−2 FP, d unchanged; bootstrap CIs overlap fully).
**Proposed: keep v1 timing exactly as-is (no option implemented).**  Expected
gain from timing/consolidation ≤ 0; the FP problem is threshold anchoring.

### D.6 Re-anchored bucket thresholds (task 6 — PROPOSED, awaiting approval)

Solved on the validation FP curve (base timing) with
`FP_test = 7.29 × FP_val` (band [5.0, 11.7]×):

| bucket | FA target (fpr×286.5) | proposed thr | FP_test point | FP_test band | fpr point [band] | d_test ≈ 0.65×d_val |
|---|---|---|---|---|---|---|
| high (≤1.0) | 200 | **5.65** | 182 | 125–293 | 0.64 [0.44–1.02] | 0.097 |
| mid (≤0.25) | 49 | **6.40** | 44 | 30–70 | 0.15 [0.10–0.25] | 0.073 |
| low (≤0.125) | 24 | **6.60** | 29 | 20–47 | 0.10 [0.07–0.16] | 0.068 |

Uncertainty statement: the FP scale factor rests on ONE portal anchor point;
the band above is from the run-cluster bootstrap of FP_val only — it does not
cover regime differences in the *shape* of the testing FP tail (e.g. at
6.5666 the two defensible extrapolations disagree wildly: 29 vs ~4 FPs).
Therefore: upload the new **mid (6.40) first** — its revealed fpr pins the
factor once more and lets low/high be re-solved before upload.  Recall at
these thresholds is modest (0.07–0.10 global) — that is the real cost of the
narrow official windows; it is reported, not hidden.  If the approved plan
wants stricter bucket-edge safety at low, 6.70–6.80 (FP_val 2–3) is the
edge-safe corner at d_test ≈ 0.062–0.065.

### D.7 Run order (faithful scorer, validation-only; no testing CSVs written)

```
.venv\Scripts\python.exe submission\official_scorer.py ak
.venv\Scripts\python.exe submission\official_scorer.py score
.venv\Scripts\python.exe submission\official_scorer.py diag
.venv\Scripts\python.exe submission\official_scorer.py score --curve 5.2796,5.7492,6.047,6.5666,7.0
.venv\Scripts\python.exe submission\time_options.py
```

