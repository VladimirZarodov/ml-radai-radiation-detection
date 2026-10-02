# -*- coding: utf-8 -*-
"""
make_submission.py -- staged pipeline for the RADAI mx31 v1 submission.

Stages (each idempotent, caches under _scratch/, new outputs under submission/):

  score  --split training|testing   blind-score runs with the frozen detector
                                     (training cache also gets GT fields, used
                                     for calibration/eval ONLY; testing stays blind)
  calib                             build threshold grids for pool (a) notebook
                                     123-run and pool (b) all-300 training runs;
                                     report ep/h and peak/h, pick emission floor
  emit   --split validation|testing extract mx31 peaks (threshold-free), write
                                     alarm tables (subset-safe under a
                                     metric_1 sweep); validation also runs the
                                     sweep-reproduction proof
  csv    --split validation|testing write CSVs in the single-
                                     source-of-truth placeholder format (csv_format.py; user-confirmed
                                     at Checkpoint C; format itself still UNOFFICIAL -- the server
                                     validates after upload).  Testing needs --i-confirm-template;
                                     --bucket low|mid|high|low_safe|mid_safe picks the per-file
                                     threshold from file_thresholds.json
  report                            background/rate drift: testing vs training

No file on the testing split is ever used to fit, calibrate or tune anything.

Usage examples:
  .venv\\Scripts\\python.exe make_submission.py score --split testing
  .venv\\Scripts\\python.exe make_submission.py calib --floor-peak-h 25
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

import radai_lib as L

ROOT = Path(__file__).resolve().parent.parent   # repo root (script lives in submission/)
TRAIN_H5 = ROOT / "training_v4.3.h5"
TEST_H5 = ROOT / "testing_v4.3.h5"
SCRATCH = ROOT / "_scratch"
SUBDIR = ROOT / "submission"

NB_CACHE = SCRATCH / "nb_mf_cache.pkl"
DET_PKL = SCRATCH / "sub_detector.pkl"
TRAIN_SCORES = SCRATCH / "sub_training_scores.pkl"
TEST_SCORES = SCRATCH / "sub_testing_scores.pkl"
CALIB_JSON = SUBDIR / "calibration.json"          # moved from _scratch (Checkpoint B decision 4)
VAL_ALARM_PKL = SCRATCH / "sub_validation_alarms.pkl"
TEST_ALARM_PKL = SCRATCH / "sub_testing_alarms.pkl"
FILE_JSON = SUBDIR / "file_thresholds.json"

N_TRAIN_RUNS, N_TEST_RUNS = 300, 300
SEP = 15            # peak min separation, windows (30 s) -- validated in emit/sweep

from csv_format import HEADER as PLACEHOLDER_HEADER, format_row


def _log(msg):
    print(msg, flush=True)


def load_detector():
    det = L.fit_detector(str(TRAIN_H5), L.TRAIN_RUNS, cache_path=str(DET_PKL))
    assert det["M"].shape == (7, L.NB) and det["U"].shape[0] == 61
    return det


def template_labels(det):
    """Per-template-id official label + internal name (61 entries)."""
    names = L.source_names(str(TRAIN_H5))
    rows = []
    for tid in det["ids"]:
        nm = names[int(tid)]
        rows.append((int(tid), nm, L.official_label(nm)))
    return rows


# ---------------------------------------------------------------------------
# stage: score
# ---------------------------------------------------------------------------

def cmd_score(args):
    det = load_detector()
    if args.split == "training":
        rids = list(range(N_TRAIN_RUNS))
        cache = L.score_runs(str(TRAIN_H5), rids, det,
                             cache_path=str(TRAIN_SCORES), with_gt=True,
                             recompute=args.force)
        # sanity: 123 notebook runs must match nb_mf_cache bit-for-bit
        nb = pickle.load(open(NB_CACHE, "rb"))
        worst = 0.0
        for r in L.ALL_RUNS:
            for k, ref in (("bestA", nb[r]["bestA"]),
                           ("dmin", nb[r]["dmin"]),
                           ("mx31", L.rollmax(nb[r]["bestA"], L.MX_N))):
                worst = max(worst, float(np.max(np.abs(cache[r][k] - ref))))
        _log(f"training cache: {len(cache)} runs; vs nb_mf_cache max|diff|={worst}")
        assert worst == 0.0, "training scores diverge from notebook cache"
    else:
        assert TEST_H5.exists(), TEST_H5
        rids = list(range(N_TEST_RUNS))
        cache = L.score_runs(str(TEST_H5), rids, det,
                             cache_path=str(TEST_SCORES), with_gt=False,
                             recompute=args.force)
        _log(f"testing cache: {len(cache)} runs (blind: dt+energy only)")


# ---------------------------------------------------------------------------
# stage: calib  (pools are TRAINING-only)
# ---------------------------------------------------------------------------

def build_pool_stats(cache, run_ids, cand):
    """episode/h and peak/h of background alarms vs threshold for a pool."""
    farm = L.farm_of(cache, run_ids)
    hrs = np.sum([L.hrs_of(farm)[r] for r in run_ids])
    n = len(run_ids)
    ep = np.zeros(len(cand))
    pk = np.zeros(len(cand))
    peak_vals = []
    for r in run_ids:
        p = L.mx31_peaks(cache[r]["mx31"], sep=SEP)
        dmin = cache[r]["dmin"]
        keep_bg = p[dmin[p] > L.R_EXCL]          # peak window far from any CA
        peak_vals.append(cache[r]["mx31"][keep_bg])
    pv = np.concatenate(peak_vals)
    for j, t in enumerate(cand):
        ep[j] = sum(L.onset_at(cache[r]["mx31"], t, farm[r]) for r in run_ids) / hrs
        pk[j] = (pv >= t).sum() / hrs
    return ep, pk, hrs, pv


def floor_from_peaks(pv, hrs, target_h):
    """Exact background-peak threshold giving target_h peaks/hour on the pool.

    pv: all background peak mx31 values of the pool (training only).
    Returns thr = k-th largest pv with k = ceil(target_h * hrs), so
    count(pv >= thr)/hrs <= target_h by construction (grid-independent,
    unlike CAND which bottoms out at the pool's 70th percentile).
    """
    k = int(np.ceil(target_h * hrs))
    if k >= len(pv):
        return float(pv.min())
    return float(np.sort(pv)[::-1][k - 1])


def cmd_calib(args):
    det = load_detector()
    tc = pickle.load(open(TRAIN_SCORES, "rb"))
    farm123 = L.farm_of(tc, L.ALL_RUNS)
    pool_a = np.concatenate([tc[r]["mx31"][farm123[r]] for r in L.ALL_RUNS])
    CAND_A = L.build_cand(pool_a)
    CAND_B = CAND_A
    _log(f"pool (a): 123 runs, CAND size {len(CAND_A)} (notebook: 133)")
    assert len(CAND_A) == 133

    all300 = list(range(N_TRAIN_RUNS))
    ep_a, pk_a, hrs_a, _ = build_pool_stats(tc, L.ALL_RUNS, CAND_A)
    _log(f"pool (a) background hours: {hrs_a:.1f} (notebook: 50.2)")

    farm300 = L.farm_of(tc, all300)
    pool_b = np.concatenate([tc[r]["mx31"][farm300[r]] for r in all300])
    CAND_B = L.build_cand(pool_b)
    OM_B = L.onset_matrix(tc, all300, farm300, CAND_B, "mx31")
    HV_B = np.array([L.hrs_of(farm300)[r] for r in all300])
    ep_b, pk_b, hrs_b, pv_b = build_pool_stats(tc, all300, CAND_B)
    np.savez(SCRATCH / "sub_pool_b_peaks.npz", pv=pv_b, hrs=hrs_b)
    _log(f"pool (b): {len(all300)} runs, {hrs_b:.1f} background h, "
         f"CAND size {len(CAND_B)}  [raw bg peaks -> sub_pool_b_peaks.npz]")

    OM_A = L.onset_matrix(tc, L.ALL_RUNS, farm123, CAND_A, "mx31")
    HV_A = np.array([L.hrs_of(farm123)[r] for r in L.ALL_RUNS])
    out = {"pool_a": {"cand": CAND_A.tolist(), "ep_h": ep_a.tolist(),
                      "peak_h": pk_a.tolist(), "bg_hours": hrs_a},
           "pool_b": {"cand": CAND_B.tolist(), "ep_h": ep_b.tolist(),
                      "peak_h": pk_b.tolist(), "bg_hours": hrs_b},
           "sep": SEP, "targets": {}}

    # thresholds at FAR targets via notebook protocol (episode/h on monotone branch)
    for target in (1, 3, 10):
        jB = L.calib_index(OM_B, HV_B, target)
        jA = L.calib_index(OM_A, HV_A, target)
        rec = {"target": target,
               "pool_a": {"thr": float(CAND_A[jA]), "ep_h": float(ep_a[jA]),
                          "peak_h": float(pk_a[jA])},
               "pool_b": {"thr": float(CAND_B[jB]), "ep_h": float(ep_b[jB]),
                          "peak_h": float(pk_b[jB])}}
        out["targets"][str(target)] = rec
        _log(f"FAR={target:>2}/h  pool(a): thr={CAND_A[jA]:6.3f} "
             f"ep/h={ep_a[jA]:5.1f} peak/h={pk_a[jA]:5.1f}   "
             f"pool(b): thr={CAND_B[jB]:6.3f} ep/h={ep_b[jB]:5.1f} "
             f"peak/h={pk_b[jB]:5.1f}")

    # emission floor: exact training-pool threshold for the requested peak/h
    floor = floor_from_peaks(pv_b, hrs_b, args.floor_peak_h)
    jf = int(np.searchsorted(-CAND_B, -floor, side="right"))
    pk_at = float((pv_b >= floor).sum() / hrs_b)
    ep_at = float(ep_b[min(jf, len(ep_b) - 1)])
    out["floor"] = {"thr": floor, "peak_h": pk_at,
                    "requested_peak_h": args.floor_peak_h,
                    "cand_index_near": jf, "ep_h_near_grid": ep_at}
    _log(f"EMISSION FLOOR (pool b): thr={floor:.4f} -> "
         f"bg peak/h={pk_at:.1f} (requested <= {args.floor_peak_h})")
    json.dump(out, open(CALIB_JSON, "w"), indent=1)
    _log(f"-> {CALIB_JSON}")


# ---------------------------------------------------------------------------
# stage: emit (alarms as mx31 peaks above the floor)
# ---------------------------------------------------------------------------

def extract_alarms(cache, run_ids, det, names, floor=None):
    """Alarm rows for ALL mx31 peaks (floor=None) or those >= floor.
    Rows: (run_id, peak window, plateau end, time_ms, metric, template id,
    name, official label[, is_bg])."""
    rows = []
    for rid in run_ids:
        c = cache[rid]
        p = L.mx31_peaks(c["mx31"], sep=SEP)
        val = c["mx31"][p]
        sel = p if floor is None else p[val >= floor]
        v = c["mx31"][sel]
        tid = det["ids"][c["am"][sel]].astype(int)
        for k in range(len(sel)):
            w = int(sel[k])
            nm = names[tid[k]]
            rows.append(dict(
                run=int(rid), win=w,
                win_end=L.plateau_end(c["mx31"], w),
                time_ms=float(c["wtime"][w]) * 1e3,
                metric=float(v[k]),
                sid=int(tid[k]), name=nm, label=L.official_label(nm),
                is_bg=(bool(c["dmin"][w] > L.R_EXCL)
                       if "dmin" in c else None),
            ))
    return rows


def cmd_emit(args):
    det = load_detector()
    cal = json.load(open(CALIB_JSON))
    floor = cal["floor"]["thr"]
    names = L.source_names(str(TRAIN_H5))
    unlabeled = [(t, n) for t, n, lab in template_labels(det) if lab is None]
    if unlabeled:
        _log(f"WARNING templates with no official label: {unlabeled}")
    if args.split == "validation":
        tc = pickle.load(open(TRAIN_SCORES, "rb"))
        rows = extract_alarms(tc, L.TEST_RUNS, det, names)
        pickle.dump(rows, open(VAL_ALARM_PKL, "wb"))
        hrs = sum(tc[r]["wtime"][-1] / 3600 for r in L.TEST_RUNS)
        nf = sum(1 for r in rows if r["metric"] >= floor)
        _log(f"validation: {len(rows)} peaks total over {hrs:.1f} run-hours; "
             f"{nf} rows >= floor {floor:.3f} ({nf / len(L.TEST_RUNS):.1f}/run)")
    else:
        cc = pickle.load(open(TEST_SCORES, "rb"))
        rows = extract_alarms(cc, range(N_TEST_RUNS), det, names)
        pickle.dump(rows, open(TEST_ALARM_PKL, "wb"))
        hrs = sum(cc[r]["wtime"][-1] / 3600 for r in range(N_TEST_RUNS))
        nf = sum(1 for r in rows if r["metric"] >= floor)
        _log(f"TESTING: {len(rows)} peaks over {hrs:.1f} run-hours; "
             f"{nf} rows >= floor {floor:.3f} "
             f"({nf / N_TEST_RUNS:.1f} rows/run, {nf / hrs:.1f}/h)")
        import collections
        cnt = collections.Counter(r["label"] for r in rows if r["metric"] >= floor)
        _log("emitted label histogram: " +
             ", ".join(f"{k}={v}" for k, v in cnt.most_common()))
        none_lab = sum(1 for r in rows if r["label"] is None
                       and r["metric"] >= floor)
        _log(f"emitted rows with unmappable label: {none_lab}")


# ---------------------------------------------------------------------------
# stage: csv (validation dry-run only until official template confirmed)
# ---------------------------------------------------------------------------

def run_lengths_ms(cache):
    """Detector coverage end per run, ms from run start.  wtime holds window
    CENTRES, so the last analysed window ends at wtime[-1] + DET_T/2."""
    return {int(r): int(round((cache[r]["wtime"][-1] + L.DET_T / 2) * 1000))
            for r in cache}


def write_csv(rows, path, run_len):
    path = Path(path)
    path.parent.mkdir(exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        fh.write(PLACEHOLDER_HEADER + "\n")
        for i, r in enumerate(sorted(rows, key=lambda x: (x["run"], x["win"]))):
            fh.write(format_row(i + 1, r["run"], r["time_ms"], r["label"],
                                r["metric"], run_len[r["run"]]) + "\n")
    _log(f"wrote {len(rows)} alarms -> {path}")


# ---------------------------------------------------------------------------
# stage: files  (three bucket files: see Checkpoint B decision 2)
# ---------------------------------------------------------------------------

BG_H_TESTING_ASSUMED = 286.5   # ASSUMPTION: portal fpr = FA_count / ~286.5 bg-hours
                               # (leaderboard evidence: fpr values = integers/286.5)
BUCKET_TARGET_FA = {"low": 24.0, "mid": 49.0, "high": 200.0}   # FA counts
BUCKET_EDGE_FA = {"low": 35.0, "mid": 71.0, "high": 285.0}     # bucket upper edges


def cmd_files(args):
    """Choose per-file thresholds so EXPECTED testing false alarms hit the
    targets, calibrated on the TRAINING pool (b) and corrected by the
    like-for-like testing/validation rows-per-hour ratio at the same
    threshold.  Writes _scratch/ previews only; final CSVs stay gated."""
    from scipy.stats import chi2
    import local_scorer as LS

    cal = json.load(open(CALIB_JSON))
    floor = cal["floor"]["thr"]
    tc = pickle.load(open(TRAIN_SCORES, "rb"))
    cc = pickle.load(open(TEST_SCORES, "rb"))
    va = pickle.load(open(VAL_ALARM_PKL, "rb"))      # ALL validation peaks
    ta = pickle.load(open(TEST_ALARM_PKL, "rb"))      # ALL testing peaks

    # -- training pool (b): background peak values + background hours --------
    farm300 = L.farm_of(tc, range(N_TRAIN_RUNS))
    hrs_b = sum(L.hrs_of(farm300)[r] for r in range(N_TRAIN_RUNS))
    pv_list = []
    for r in range(N_TRAIN_RUNS):
        p = L.mx31_peaks(tc[r]["mx31"], sep=SEP)
        bg = p[tc[r]["dmin"][p] > L.R_EXCL]      # background-only peaks
        pv_list.append(tc[r]["mx31"][bg])
    pv_b = np.sort(np.concatenate(pv_list))[::-1]   # descending bg-peak values

    mv = np.sort([a["metric"] for a in va])[::-1]
    mt = np.sort([a["metric"] for a in ta])[::-1]
    h_val = sum(tc[r]["wtime"][-1] / 3600 for r in L.TEST_RUNS)
    h_test = sum(cc[r]["wtime"][-1] / 3600 for r in range(N_TEST_RUNS))

    def n_ge(arr, t):          # rows >= t (counts, exact for array values)
        return np.searchsorted(-arr, -t, side="right")

    def ratio(t):              # like-for-like testing/validation rows-per-hour
        nv, nt = n_ge(mv, t), n_ge(mt, t)
        return (nt / h_test) / max(nv / h_val, 1e-9)

    def rate_b(t, pvs=pv_b):   # training pool-b background peak rate
        return np.searchsorted(-pvs, -t, side="right") / hrs_b

    def fa_exp(t, pvs=pv_b):   # expected testing FA count at threshold t
        return rate_b(t, pvs) * ratio(t) * BG_H_TESTING_ASSUMED

    rng = np.random.default_rng(7)
    det, _tc, _cal, names, alarms = LS.load_state(SEP)
    enc = LS.gt_encounters(tc, names)
    R_LIKE_LOW, R_LIKE_HIGH = 1.19, 1.34   # user-cited like-for-like band (bg-dominant region)
    out = {"assumed_bg_hours_testing": BG_H_TESTING_ASSUMED,
           "drift_scenarios": [R_LIKE_LOW, R_LIKE_HIGH],
           "pool_b": {"bg_hours": float(hrs_b), "n_bg_peaks": int(len(pv_b))},
           "buckets": {}}
    _log(f"pool(b) {hrs_b:.1f} bg-h, {len(pv_b)} bg peaks; "
         f"val {len(mv)} peaks / {h_val:.1f} h, test {len(mt)} peaks / "
         f"{h_test:.1f} h | assumed testing bg hours {BG_H_TESTING_ASSUMED}")

    # evaluation grid: every distinct pool-b bg peak value >= floor
    grid = np.unique(pv_b[pv_b >= floor])
    for bname, target in BUCKET_TARGET_FA.items():
        fa_g = np.array([fa_exp(t) for t in grid])
        # fa_g decreasing in grid; invert at target
        idx = np.searchsorted(-fa_g, -target)          # first grid pt with fa<=target
        thr = float(grid[min(idx, len(grid) - 1)])
        # bootstrap threshold CI: resample pool-b bg peaks (Poisson-like count noise)
        boots = []
        n = len(pv_b)
        for _ in range(args.boot):
            s = pv_b[rng.integers(0, n, n)]
            k = int(np.ceil(target / (ratio(thr) * BG_H_TESTING_ASSUMED) * hrs_b))
            boots.append(np.sort(s)[::-1][max(min(k, n) - 1, 0)])
        boots = np.array(boots)
        t_ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
        # Poisson 95% CI on the FA count at the chosen threshold
        cnt = rate_b(thr) * hrs_b
        p_lo = chi2.ppf(0.025, 2 * cnt) / 2 / hrs_b if cnt > 0 else 0.0
        p_hi = chi2.ppf(0.975, 2 * cnt) / 2 / hrs_b
        fa_ci = (p_lo * ratio(thr) * BG_H_TESTING_ASSUMED,
                 p_hi * ratio(thr) * BG_H_TESTING_ASSUMED)
        # -- drift scenarios: measured ratio may be < band low at high thr
        #    because validation rows there are dominated by detection peaks.
        #    Worst case = Poisson-upper bg rate x upper drift. --------------
        r_meas = ratio(thr)
        r_lo_sc = min(r_meas, R_LIKE_LOW)
        r_hi_sc = max(r_meas, R_LIKE_HIGH)
        fa_worst = p_hi * r_hi_sc * BG_H_TESTING_ASSUMED
        # safe threshold: lowest grid point whose worst-case (Poisson upper x
        # 1.34 drift) FA stays under the bucket edge
        safe_thr = None
        for t in grid[grid >= thr]:
            n_t = np.searchsorted(-pv_b, -t, side="right")
            hi_t = chi2.ppf(0.975, 2 * n_t) / 2 / hrs_b if n_t else 0.0
            if hi_t * r_hi_sc * BG_H_TESTING_ASSUMED <= BUCKET_EDGE_FA[bname]:
                safe_thr = float(t)
                break
        rows = int(n_ge(mt, thr))
        # expected recall: validation sweep at this threshold (exact & center rule)
        s_ex = LS.sweep_all(alarms, enc, thr, "exact")
        s_ct = LS.sweep_all(alarms, enc, thr, "center")
        emax = np.concatenate([L.emax_mx(tc, r) for r in L.TEST_RUNS])
        rec_notebook = float((emax >= thr).mean())
        # MODEL 2 bound: testing FA ~ validation unmatched-alarm rate scaled to
        # testing hours (no label use; covers near-source false rows that the
        # pool-b background mask excludes).  Ratio from same rows/hour drift.
        fa_m2 = s_ct["fa_unmatched"] / h_val * h_test * ratio(thr)
        out["buckets"][bname] = {
            "thr": thr, "target_fa": target,
            "fa_expected": float(fa_exp(thr)),
            "fa_poisson_ci": [float(x) for x in fa_ci],
            "thr_boot_ci": list(t_ci),
            "drift_ratio": float(ratio(thr)),
            "pool_b_rate_h": float(rate_b(thr)),
            "testing_rows": rows, "rows_per_run": rows / N_TEST_RUNS,
            "recall_val_exact": s_ex["recall"], "recall_val_center": s_ct["recall"],
            "recall_val_emax": rec_notebook, "fa_model2": float(fa_m2),
            "drift_measured": float(r_meas),
            "drift_scenarios": [float(r_lo_sc), float(r_hi_sc)],
            "fa_worst_case": float(fa_worst),
            "safe_thr": safe_thr,
        }
        if safe_thr is not None and safe_thr != thr:
            s2 = LS.sweep_all(alarms, enc, safe_thr, "exact")
            rows2 = int(n_ge(mt, safe_thr))
            n2 = int(np.searchsorted(-pv_b, -safe_thr, side="right"))
            hi2 = chi2.ppf(0.975, 2 * n2) / 2 / hrs_b if n2 else 0.0
            # also the NEXT grid point above safe_thr (real margin, since
            # safe_thr is by construction at the worst-case edge)
            nxt = grid[grid > safe_thr]
            alt = None
            if len(nxt):
                t3 = float(nxt[0])
                n3 = int(np.searchsorted(-pv_b, -t3, side="right"))
                hi3 = chi2.ppf(0.975, 2 * n3) / 2 / hrs_b if n3 else 0.0
                s3 = LS.sweep_all(alarms, enc, t3, "exact")
                alt = {"thr": t3, "testing_rows": int(n_ge(mt, t3)),
                       "recall_val_exact": s3["recall"],
                       "fa_at_worst": float(hi3 * r_hi_sc * BG_H_TESTING_ASSUMED)}
            out["buckets"][bname]["safe_stats"] = {
                "testing_rows": rows2, "rows_per_run": rows2 / N_TEST_RUNS,
                "recall_val_exact": s2["recall"],
                "fa_at_worst": float(hi2 * r_hi_sc * BG_H_TESTING_ASSUMED),
                "margin_point": alt}
        b = out["buckets"][bname]
        _log(f"[{bname}] thr={thr:.3f}  drift ratio={b['drift_ratio']:.2f}  "
             f"poolB rate={b['pool_b_rate_h']:.3f}/h")
        _log(f"        expected FA={b['fa_expected']:.1f} "
             f"[Poisson {b['fa_poisson_ci'][0]:.1f}..{b['fa_poisson_ci'][1]:.1f}] "
             f"target {target:.0f} (edge {BUCKET_EDGE_FA[bname]:.0f}) | "
             f"thr boot CI {t_ci[0]:.2f}..{t_ci[1]:.2f}")
        _log(f"        testing rows={rows} ({b['rows_per_run']:.2f}/run) | "
             f"val recall exact={s_ex['recall'] * 100:.1f}% "
             f"center={s_ct['recall'] * 100:.1f}% emax={rec_notebook * 100:.1f}%")
        _log(f"        FA model-2 (unmatched-rows upper incl. near-source): "
             f"{fa_m2:.1f} -> {'OVER bucket edge' if fa_m2 > BUCKET_EDGE_FA[bname] else 'under edge'}")
        _log(f"        drift measured={r_meas:.2f}, scenarios=[{r_lo_sc:.2f},{r_hi_sc:.2f}] "
             f"-> FA worst-case={fa_worst:.1f} (edge {BUCKET_EDGE_FA[bname]:.0f})")
        if out["buckets"][bname].get("safe_thr") and \
                out["buckets"][bname]["safe_thr"] != thr:
            ss = out["buckets"][bname]["safe_stats"]
            _log(f"        SAFE thr={safe_thr:.3f}: rows={ss['testing_rows']} "
                 f"({ss['rows_per_run']:.2f}/run), recall(exact)="
                 f"{ss['recall_val_exact'] * 100:.1f}%, FA worst<=edge")
            if ss.get("margin_point"):
                mp = ss["margin_point"]
                _log(f"        SAFE+1 thr={mp['thr']:.3f}: rows={mp['testing_rows']}, "
                     f"recall={mp['recall_val_exact'] * 100:.1f}%, "
                     f"FA worst={mp['fa_at_worst']:.1f}")
        # preview file into _scratch (NOT the final submission; template pending)
        rows_b = [r for r in ta if r["metric"] >= thr]
        write_csv(rows_b, SCRATCH / f"preview_mx31_v1_fpr_{bname}.csv",
                  run_lengths_ms(cc))

    # optional wide file (fallback for a future portal sweep)
    rows_w = [r for r in ta if r["metric"] >= floor]
    out["wide"] = {"thr": floor, "testing_rows": len(rows_w),
                   "rows_per_run": len(rows_w) / N_TEST_RUNS,
                   "fa_expected_note": "wide file is NOT bucket-compliant; "
                                       "fallback only if portal sweeps metric_1"}
    write_csv(rows_w, SCRATCH / "preview_mx31_v1_fpr_wide.csv",
              run_lengths_ms(cc))

    json.dump(out, open(FILE_JSON, "w"), indent=1)
    _log(f"-> {FILE_JSON}")


def bucket_thr(b):
    """Threshold for a bucket name; '<base>_safe' = SAFE+1 margin point."""
    j = json.load(open(FILE_JSON))
    base = b[:-len("_safe")] if b.endswith("_safe") else b
    d = j["buckets"][base]
    if b.endswith("_safe"):
        return float(d["safe_stats"]["margin_point"]["thr"])
    return float(d["thr"])


def cmd_csv(args):
    cal = json.load(open(CALIB_JSON))
    floor = cal["floor"]["thr"]
    if args.split == "testing":
        if not args.i_confirm_template:
            sys.exit("REFUSED: CSV format not confirmed by user (placeholder "
                     "is UNCONFIRMED; the server validates after upload).")
        rows = pickle.load(open(TEST_ALARM_PKL, "rb"))
        rl = run_lengths_ms(pickle.load(open(TEST_SCORES, "rb")))
        if args.bucket:
            thr = bucket_thr(args.bucket)
            rows = [r for r in rows if r["metric"] >= thr]
            out = args.out or str(SUBDIR / f"submission_mx31_v1_fpr_{args.bucket}.csv")
            _log(f"bucket {args.bucket}: thr={thr:.4f}, {len(rows)} rows")
        else:
            rows = [r for r in rows if r["metric"] >= floor]
            out = args.out or str(SUBDIR / "submission_mx31_v1_wide.csv")
    else:
        rows = pickle.load(open(VAL_ALARM_PKL, "rb"))
        tc = pickle.load(open(TRAIN_SCORES, "rb"))
        rl = {k: v for k, v in run_lengths_ms(tc).items() if k in L.TEST_RUNS}
        if args.bucket:
            thr = bucket_thr(args.bucket)
            rows = [r for r in rows if r["metric"] >= thr]
        else:
            rows = [r for r in rows if r["metric"] >= floor]
        out = args.out or str(SUBDIR / "validation_dryrun_placeholder.csv")
    write_csv(rows, out, rl)


# ---------------------------------------------------------------------------
# stage: report (drift testing vs training, inference-only observation)
# ---------------------------------------------------------------------------

def cmd_report(args):
    tc = pickle.load(open(TRAIN_SCORES, "rb"))
    cc = pickle.load(open(TEST_SCORES, "rb"))
    def stats(cache, rids):
        rates = np.concatenate([cache[r]["bg_rate"] for r in rids])
        hrs = sum(cache[r]["wtime"][-1] / 3600 + 2 / 3600 for r in rids)
        qs = np.percentile(rates, [10, 25, 50, 75, 90])
        return hrs, rates.mean(), qs
    hA, mA, qA = stats(tc, range(N_TRAIN_RUNS))
    hB, mB, qB = stats(cc, range(N_TEST_RUNS))
    _log(f"{'':10}{'hours':>9}{'mean cps':>10}   p10/p25/p50/p75/p90 cps")
    _log(f"train300 {hA:9.1f}{mA:10.1f}   " +
         "/".join(f"{x:.0f}" for x in qA))
    _log(f"test300  {hB:9.1f}{mB:10.1f}   " +
         "/".join(f"{x:.0f}" for x in qB))
    _log(f"median-rate shift: {qB[2] / qA[2] - 1:+.1%}; "
         f"p90 shift: {qB[4] / qA[4] - 1:+.1%}")
    _log("(observation only; thresholds are NOT retuned on testing)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("score"); p.add_argument("--split", required=True,
                                               choices=["training", "testing"])
    p.add_argument("--force", action="store_true")
    p = sp.add_parser("calib"); p.add_argument("--floor-peak-h", type=float,
                                               default=25.0)
    p = sp.add_parser("emit"); p.add_argument("--split", required=True,
                                              choices=["validation", "testing"])
    p = sp.add_parser("csv"); p.add_argument("--split", required=True,
                                             choices=["validation", "testing"])
    p.add_argument("--out"); p.add_argument("--i-confirm-template",
                                            action="store_true")
    p.add_argument("--bucket",
                   choices=["low", "mid", "high", "low_safe", "mid_safe"],
                   default=None)
    p = sp.add_parser("files"); p.add_argument("--boot", type=int, default=400)
    sp.add_parser("report")
    args = ap.parse_args()
    {"score": cmd_score, "calib": cmd_calib, "emit": cmd_emit,
     "csv": cmd_csv, "files": cmd_files, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
