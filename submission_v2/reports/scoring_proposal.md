# C1 follow-up task 2 — how testing scores should be produced so OOF-derived
thresholds are meaningful (decision record; recommended before any upload)

Background: dev thresholds were calibrated on the POOLED DEV OOF distribution
= for each run, its OWN single fold model's per-second scores (that model
never saw the run). Testing scoring must reproduce that generative regime or
the threshold numbers lose their meaning. Diagnostic (reports/testing_diag2.md)
measured the 5-model MEAN fires 0.80x of a single model at the same threshold.

## Candidates

(a) ONE fold model per testing run, fixed pseudo-random assignment (run_id % 5).
- Why thresholds transfer: deployment scoring distribution IS the calibration
  distribution by construction (same 5 models, same single-model-per-unseen-run
  regime, same mixture weights up to fold-size 57/49/46/44/44 vs 60x5).
- Checkable on dev without leakage: everything — per-fold thr_at_fpr spread,
  per-fold alarms/h at fixed pooled thr (measured: 3.32-5.09/h at the .085
  thr -> model-to-model spread 1.5x; per-model testing/dev emission band
  0.52-0.58 -> tight), mixture identity.
- NOT checkable on dev: the testing data shift itself (measured separately on
  testing aggregates, disclosed; that is the point of the measurement file).
- Cost: zero (models exist); deterministic and auditable.

(b) All five separately + alarm union / voting.
- Dev-checkable: essentially NOTHING honestly — the union over fold models
  requires models to score runs they were trained on (80% leak); no
  leakage-free dev proxy exists (inner-val gives only per-model subsets).
- On testing we could only measure unlabeled take-rates, not the d/fpr
  trade-off the union creates.
- Effect: more rows AND more FPs; dev thresholds lose calibration meaning.
- Verdict: right idea for Stage-3 ensemble-score gains, wrong tool for
  threshold transfer.

(c) Ensemble (mean) + quantile mapping of testing scores onto the pooled dev
OOF score distribution (testing-side, label-free).
- Procedure is mechanical and monotone; identity on dev (where it can't be
  validated because the ensemble-vs-single relation is not OOF-honest there).
- The compression correction it would apply (factor 0.80 / tail quantiles) is
  TESTING-derived -> using it for calibration = tuning on testing (disclosed,
  but adds a contamination surface and an assumption).
- Verdict: workable for a FINAL file if ever needed; strictly inferior to (a)
  for the measurement goal (less clean causal readout, more disclosure).

(d) Final single model trained on all 240 (or all 300) pool runs.
- No uncontaminated dev calibration possible at all: its OOF scores ARE
  training-distribution scores; fold-model thresholds transfer only
  approximately; the only honest holdout left is the LOCKBOX (sealed, one use
  at C4, must not be spent on calibration).
- Verdict: possible deployment model for Stage 3/4 finals; excluded as the
  calibration-transfer scheme now, per frozen PLAN hygiene rules.

## RECOMMENDATION: (a), assignment run_id % 5, fold = run_id mod 5 (fixed,
documented in README before any generation).
- Only scheme where "OOF-derived threshold" remains literally true on testing.
- Its per-run model assignment is pre-declarable and unchanging -> a later
  Stage-3/4 rescore with ensembles/union stays comparable.
- Known limits to state on upload: fold models saw unequal dev data (folds
  57..44 val sizes); their dev-side take rates differ 1.5x at fixed thr — so
  per-model testing take rates carry that spread (measured 0.52-0.58 of each
  model's own dev rate); assignment-by-modulo keeps each model's testing slice
  independent of run content (verify slice homogeneity with label-free run
  lengths before upload, optional).
