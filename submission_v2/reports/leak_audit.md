# check 4 leakage audit (config c64s) -- checklist with evidence
(a) inputs/normalisation:
  D.features signature: (total, rate_all, rate_in) -- only per-second totals + rate channels; label/answer-key tokens found in source: NONE
  normalisation = sqrt(total / per-run MEDIAN of the run's own total counts) + robust log-rate channels (run-local); no listmode id, no background_id, no sources/*, no AK.
  EVIDENCE: grep s1_data.py cache ->
    hits: NONE (per-run median uses run's own totals only)
(b) window gather inside own run:
  Store concatenates per-run padded blocks; windows() gathers j+arange(-ctx,ctx+1); inference j limited to [j0+ctx, j0+ctx+nb) -> gathered rows in [j0, j0+nb+2ctx) = exactly that run's block; E030 synthetic-store assert (centered, no inter-run rows); re-verified at runtime here:
    runtime bounds check on 3 runs: PASS
(c) fold separation (OOF scoring runs absent from that fold's model training AND its inner validation):
  fold0: |val|=57 inter(inner-train,val)=0 inter(inner-val,val)=0 lockbox∩val=0
  fold1: |val|=49 inter(inner-train,val)=0 inter(inner-val,val)=0 lockbox∩val=0
  fold2: |val|=46 inter(inner-train,val)=0 inter(inner-val,val)=0 lockbox∩val=0
  fold3: |val|=44 inter(inner-train,val)=0 inter(inner-val,val)=0 lockbox∩val=0
  fold4: |val|=44 inter(inner-train,val)=0 inter(inner-val,val)=0 lockbox∩val=0
  NOTE: other pool runs are IN fold-k training by design (cross-fitting); each scored run is held out of ITS OWN model. All lists derived from frozen cache/folds.json, unchanged.
(d) threshold provenance:
  thr values in reports/*_oof.json come from AlarmCurve.thr_at_fpr(target, runs240, meta) on POOLED OOF alarms only (runs240, dev pool; lockbox excluded -- leakguard overlap=0 logged per run of train/eval/eval-table). No test data involved. v1 comparator thresholds likewise pooled-AK-only.