# -*- coding: utf-8 -*-
"""
official_scorer.py -- FAITHFUL local re-implementation of the organiser's
scoring (radai/radai/evaluation/), run on the VALIDATION split (training h5,
runs 25..124).  Replaces the old +/-30/60/120 s hypothesis scorer
(local_scorer.py) for portal-metric emulation.

Everything below mirrors this source code (file:line refs):

ANSWER KEY WINDOWS -- radai/evaluation/answerkey.py build_answerkey():
  * data opened via tools.open_radai_file(time_step_size=1000) -> ListmodeH5,
    label_manager.tier = 0  (raw source NAMES as labels, tools.py:117-128);
  * SNR series per name: data_set.py get_snr_for_run() (lines 1038-1152):
    events histogrammed per 1-s bin via np.histogram2d(time_ms, label_id,
    (edges=arange(0, get_end_time, 1000, uint32), label_edges=id -0.5/+0.5));
    B = column of id 0 ('BKG');  snr = S / sqrt(S + B) per id column (no
    energy cut); time listmode column = np.cumsum(dt, uint32)/1e3 ms
    (data_set.py:484); get_end_time = (end_ts - start_ts)*1e3 ms (run attrs);
  * per encounter (run['sources'] id+time, sorted by time):
    center_idx = round((CA_ms - 500)/1000)                       (l.92)
    greedy bounding window: expand left/right until >= 25 bins
    with snr < 0.01 inside, ties broken greedily by edge SNR      (l.95-134)
    snr_peak  = max over the bounding window                      (l.140)
    above     = window snr STRICTLY > 0.05 * snr_peak             (l.141)
    time_start = 1000 * (pos of FIRST above bin)                  (l.143)
    time_stop  = 1000 * (pos of LAST  above bin + 1)              (l.144)
    time_max   = 1000 * (pos of peak bin) + 500                   (l.145)
    assert window width <= 160 bins                               (l.301)
  (the max-SNR integral window l.149-193 is NOT used by the core metrics)

MATCHING -- radai/evaluation/visualization.py calculate_fp_info() (l.978-1039):
  alarm matches encounter iff same run and time_start <= t <= time_stop;
  >1 match -> RuntimeError (overlap forbidden, l.1006);
  matched alarm with metric_1 > current alg_metric OWNS the encounter
  (label_1 -> alg_label), losers are free (l.1018-1022);
  FP iff alarm time is inside NO window; penalty count =
  1 + int((time_stop - time_start)/1000 // max_alarm_time=160)  (l.1024-1037)

METRICS -- radai/evaluation/evaluation.py answers_to_metrics() (l.210-244):
  detection   = alg_label != 'BKG'            -> d_recall  = tp /encounters
  category    = true_cat == label2category(alg_label) -> c_recall = tc /enc
  id          = isotope  == alg_label         -> id_recall = tid /enc
  fpr[cat]    = #FP(label category == cat) / total_time_hr; Global = ALL FPs
  far[Global] = sum of fpr over alarm_categories (['NuclearMaterial']) only
  total_time_hr in answers_to_metrics = ak.run_id.max() - sum(window)/3600
  (l.211-213); compute_metrics instead uses sum(get_end_time) - sum(window)
  (l.303-305,374-376).  The portal output (per-category d/c/id/fpr rows +
  Global = SUM of category fprs) matches answers_to_metrics' layout; which
  denominator variant the portal really uses is unresolved -- we report both.

label2category / label2isotope: copied VERBATIM from tools.py (l.75-94).
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

import radai_lib as L

ROOT = Path(__file__).resolve().parent.parent
TRAIN_H5 = ROOT / "training_v4.3.h5"
SCRATCH = ROOT / "_scratch"
SUBDIR = Path(__file__).resolve().parent
AK_CSV = SCRATCH / "sub_val_answerkey.csv"
AK_STATS = SCRATCH / "sub_val_answerkey_stats.json"
VAL_ALARM_PKL = SCRATCH / "sub_validation_alarms.pkl"
TRAIN_SCORES = SCRATCH / "sub_training_scores.pkl"
FILE_JSON = SUBDIR / "file_thresholds.json"

STEP = 1000.0            # time_step_size, ms (open_radai_file call)
SNR_REL = 0.05           # snr_relative_thresh default
LOW_T, N_LOW = 0.01, 25  # bounding-window rule

# categories copied from radai/radai/evaluation/tools.py:18-52
CATEGORY_MAPPING = {
    "BKG": ["Background", "BKG"],
    "NORM": ["K-40", "Ra-226", "Th-232"],
    "Medical": ["Co-57", "F-18", "Tc-99m", "I-131", "Tl-201",
                "Cu-67", "Sr-90", "Lu-177", "Xe-133"],
    "Industrial": ["Co-60", "Cs-137", "Ba-133", "Ir-192", "Am-241"],
    "NuclearMaterial": ["DepletedU", "DU", "NatU", "RefinedU", "LEU",
                        "HEU", "FGPu", "WGPu"],
}


def label2category(label):          # verbatim tools.py l.75-80 (no raise here)
    for cat, sourcelist in CATEGORY_MAPPING.items():
        for source in sourcelist:
            if source in label:
                return cat
    raise ValueError(f"Could not find category for label {label}")


def label2isotope(label):           # verbatim tools.py l.83-94
    if label.startswith("Ir-192"):
        return "Ir-192"
    if "kg" in label:
        assert "-" in label
        tokens = label.split("-")
        return tokens[0]
    if "shielding" in label:
        tokens = label.split("_shielding")
        return tokens[0]
    return label


# ---------------------------------------------------------------------------
# stage: ak -- build the validation answer key exactly as build_answerkey()
# ---------------------------------------------------------------------------

def _snr_hist(run_id: int, num_labels: int):
    """Per-source 1-s count histogram + SNR series for one TRAINING run.

    Replicates get_snr_for_run(): time = cumsum(dt, uint32)/1e3 ms;
    np.histogram2d(time, label_id, (arange(0, end_ms, 1000, uint32),
    arange(num_labels+1)-0.5)); snr = S/sqrt(S+B) with B = column of id 0.
    Fast path (bincount) validated against histogram2d on first call.
    """
    with h5py.File(TRAIN_H5, "r") as f:
        g = f[f"runs/run{run_id}"]
        dt = g["listmode/dt"][:]
        eid = g["listmode/id"][:]
        end_ms = (g.attrs["end_timestamp"] - g.attrs["start_timestamp"]) * 1e3
    t_ms = np.cumsum(dt, dtype=np.uint32).astype(np.float64) / 1e3
    edges = np.arange(0, end_ms, STEP, dtype=np.uint32)
    nb = len(edges) - 1
    idx = np.floor(t_ms / STEP).astype(np.int64)
    keep = (idx >= 0) & (idx < nb)            # right edge inclusion is a
    idx2 = np.where(keep, idx, -1)            # measure-zero event
    edge_last = (~keep) & (idx == nb) & (t_ms == edges[-1])
    idx2[edge_last] = nb - 1
    ok = (idx2 >= 0)
    tvl = np.bincount(idx2[ok] * num_labels + eid[ok],
                      minlength=nb * num_labels).reshape(nb, num_labels)
    tvl = tvl.astype(np.float64)
    chk = AK_STATS.with_suffix(".histcheck")
    if not chk.exists():                      # one-time equivalence proof
        le = np.arange(num_labels + 1) - 0.5
        ref = np.histogram2d(t_ms, eid, bins=(edges.astype(np.float64), le))[0]
        np.save(chk, np.array([np.array_equal(tvl, ref)]))
        if not np.array_equal(tvl, ref):
            raise RuntimeError("bincount != histogram2d -- not faithful!")
    B = tvl[:, 0:1]
    with np.errstate(invalid="ignore"):
        snr = tvl / np.sqrt(tvl + B)
    snr[:, 0] = np.nan
    return snr, tvl, float(end_ms)


def _encounter_window(v: np.ndarray, center: int):
    """Greedy bounding window + 5%-of-peak span (answerkey.py l.95-145)."""
    n = len(v)
    n_below = 0
    n_below_left = 0
    n_below_right = 0
    wl = wr = 3
    win = None
    while n_below < N_LOW:
        win = v[center - wl: center + wr + 1]
        n_below = int((win < LOW_T).sum())
        if n_below == N_LOW:
            break
        snr_left = win[0]
        snr_right = win[-1]
        if (snr_left > snr_right and center - wl > 0) or \
           (snr_left <= snr_right and center + wr + 1 >= n):
            wl += 1
        elif (snr_left < snr_right and center + wr + 1 < n) or \
             (snr_left >= snr_right and center - wl <= 0):
            wr += 1
        elif center - wl == 0:
            wr += 1
            n_below_right += 1
        elif center + wr >= n - 1 or n_below_left < n_below_right:
            wl += 1
            n_below_left += 1
        elif n_below_right < n_below_left:
            wr += 1
            n_below_right += 1
        else:
            wl += 1
            n_below_left += 1
    lo = center - wl
    assert lo >= 0, "negative slice start (pandas-from-end bug) not handled"
    # pandas Series.max() skips NaN (bins with S=B=0 -> 0/0); mirror with
    # nanmax; all comparisons against NaN are False in both frameworks.
    peak_snr = float(np.nanmax(win))
    above = win > SNR_REL * peak_snr
    first = lo + int(np.argmax(above))                          # strict >
    last = lo + len(above) - 1 - int(np.argmax(above[::-1]))
    peak = lo + int(np.nanargmax(win))
    return first * STEP, (last + 1) * STEP, peak * STEP + STEP / 2, \
        peak_snr, last - first + 1


def build_ak(force=False):
    if AK_CSV.exists() and not force:
        return pd.read_csv(AK_CSV)
    with h5py.File(TRAIN_H5, "r") as f:
        source_ids = np.asarray(f.attrs["source_ids"])
        source_names = [str(s) for s in f.attrs["source_names"]]
    assert np.array_equal(source_ids, np.arange(len(source_ids))), \
        "label columns assumed to be ids 0..N-1"
    nl = len(source_ids)
    rows, tot_end = [], 0.0
    for rid in L.TEST_RUNS:
        snr, _, end_ms = _snr_hist(rid, nl)
        tot_end += end_ms
        with h5py.File(TRAIN_H5, "r") as f:
            g = f[f"runs/run{rid}"]
            sid = g["sources/id"][:]
            stime = g["sources/time"][:]
            snrp = g["sources/snr/peak"][:] if "sources/snr/peak" in g else None
        msk = sid != 0                            # drop_background
        sid, stime = sid[msk], np.asarray(stime)[msk]
        order = np.argsort(stime)
        for k in order:
            j = int(sid[k])
            nm = source_names[j]
            ts = float(stime[k])
            center = int(np.round((ts - STEP / 2) / STEP))
            t0, t1, tmax, peak_snr, wbin = _encounter_window(snr[:, j], center)
            rows.append(dict(run_id=rid, time=ts, name=nm,
                             category=label2category(nm),
                             isotope=label2isotope(nm),
                             time_start=t0, time_stop=t1, time_max=tmax,
                             snr_peak=peak_snr,
                             snr_peak_gt=float(snrp[k]) if snrp is not None else np.nan,
                             width_bins=wbin))
    ak = pd.DataFrame(rows).sort_values(["run_id", "time"]).reset_index(drop=True)
    # overlap audit (calculate_fp_info RuntimeError guard)
    ov = 0
    for _r, gg in ak.groupby("run_id"):
        s = gg.sort_values("time_start")
        ov += int((s.time_start.values[1:] <= s.time_stop.values[:-1]).sum())
    ak.to_csv(AK_CSV, index=False)
    stats = dict(
        n_runs=len(L.TEST_RUNS), n_encounters=len(ak),
        total_run_hours=tot_end / 3.6e6,
        sum_window_hours=float((ak.time_stop - ak.time_start).sum() / 3.6e6),
        width_s_mean=float(ak.width_bins.mean()), width_s_med=float(ak.width_bins.median()),
        width_s_p90=float(ak.width_bins.quantile(0.9)), width_s_max=float(ak.width_bins.max()),
        over160=int((ak.width_bins > 160).sum()),
        overlap_violations=ov,
        start_minus_ca_s_med=float(((ak.time_start - ak.time) / 1000).median()),
        stop_minus_ca_s_med=float(((ak.time_stop - ak.time) / 1000).median()),
        per_category={c: int((ak.category == c).sum()) for c in CATEGORY_MAPPING if c != "BKG"},
    )
    AK_STATS.write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))
    return ak


# ---------------------------------------------------------------------------
# stage: score -- calculate_fp_info + answers_to_metrics replication
# ---------------------------------------------------------------------------

def val_rows(thr: float):
    """Alarm rows (run, time_ms, label, metric) >= thr, in CSV write order."""
    alarms = pickle.load(open(VAL_ALARM_PKL, "rb"))
    rows = [(a["run"], a["win"], float(a["time_ms"]), a["label"], float(a["metric"]))
            for a in alarms if a["metric"] >= thr and a["label"] is not None]
    return sorted(rows, key=lambda x: (x[0], x[1]))


def _run_len_ms():
    """Same rule as make_submission.run_lengths_ms (drives the 4-s alarm span)."""
    tc = pickle.load(open(TRAIN_SCORES, "rb"))
    return {r: int(round((tc[r]["wtime"][-1] + L.DET_T / 2) * 1000)) for r in L.TEST_RUNS}


def score_rows(rows, ak, max_alarm_time=160.0):
    """Exact re-implementation of calculate_fp_info + answers_to_metrics."""
    ak = ak.copy().reset_index(drop=True)
    ak["alg_label"] = "BKG"
    ak["alg_metric"] = -np.inf
    fp = []                                   # dicts like fp_info rows
    n_matched = n_loser = n_multi = 0
    rl = _run_len_ms()
    by_run = {r: g for r, g in ak.groupby("run_id")}   # positional index == label
    for run, _win, t, lab, m in rows:
        g = by_run.get(run)
        hit = ((t >= g.time_start) & (t <= g.time_stop)).to_numpy() if g is not None \
            else np.zeros(0, bool)
        if hit.any():
            if hit.sum() > 1:
                n_multi += 1                   # official: RuntimeError
                continue
            gi = g.index[int(np.argmax(hit))]  # ak row label (ak was reset-indexed,
            n_matched += 1                     # so label == global position)
            if m > ak.at[gi, "alg_metric"]:
                ak.at[gi, "alg_label"] = lab
                ak.at[gi, "alg_metric"] = m
            else:
                n_loser += 1                   # matched but not the max: free
        else:
            span = min(t + 2000, rl[run]) - max(t - 2000, 0)
            for _ in range(int(1.0 + (span / 1000.0) // max_alarm_time)):
                fp.append(dict(run_id=run, time=t, label=lab, metric=m))
    fp_df = pd.DataFrame(fp)
    if len(fp_df):
        fp_df["category"] = fp_df.label.map(label2category)
    # ---- answers_to_metrics (l.210-244) -----------------------------------
    ak["detection"] = ak["alg_label"] != "BKG"
    ak["category_ok"] = ak["category"] == ak["alg_label"].apply(label2category)
    ak["identification"] = ak["isotope"] == ak["alg_label"]
    win_hr = float((ak.time_stop - ak.time_start).sum() / 3.6e6)
    with h5py.File(TRAIN_H5, "r") as f:
        total_meta_hr = sum(
            (f[f"runs/run{r}"].attrs["end_timestamp"]
             - f[f"runs/run{r}"].attrs["start_timestamp"]) / 3600
            for r in L.TEST_RUNS)
    den_compute = total_meta_hr - win_hr                 # l.374-376 variant
    den_answers = float(ak.run_id.max()) - win_hr        # l.211-213 variant
    cats = ["NORM", "Medical", "Industrial", "NuclearMaterial", "Global"]
    out = []
    for cat in cats:
        m = np.ones(len(ak), bool) if cat == "Global" else (ak.category == cat).to_numpy()
        mf = (np.ones(len(fp_df), bool) if cat == "Global"
              else (fp_df.category == cat).to_numpy()) if len(fp_df) else np.zeros(0, bool)
        enc = int(m.sum())
        tp = int(ak.detection.to_numpy()[m].sum())
        tcr = int(ak.category_ok.to_numpy()[m].sum())
        tid = int(ak.identification.to_numpy()[m].sum())
        nfp = int(mf.sum())
        out.append(dict(category=cat, encounters=enc, tp=tp, tc=tcr, tid=tid,
                        fp=nfp,
                        d_recall=tp / enc, c_recall=tcr / enc, id_recall=tid / enc,
                        fpr_compute=nfp / den_compute, fpr_answers=nfp / den_answers))
    res = pd.DataFrame(out).set_index("category")
    aux = dict(rows=len(rows), matched=n_matched, losers=n_loser,
               multi_match=n_multi,
               fps=len(fp_df), total_meta_hr=total_meta_hr, win_hr=win_hr,
               den_compute=den_compute, den_answers=den_answers)
    return res, ak, fp_df, aux


# ---------------------------------------------------------------------------
# stage: diag -- timing diagnostics (task 4)
# ---------------------------------------------------------------------------

def diag(rows, ak, fp_df):
    print("\n== timing diagnostics (alarms vs CA / windows) ==")
    print(f"alarm time definition: mx31(=31x2s trailing rolling max of bestA) "
          f"local max, rising-plateau FIRST index -> wtime centre of the bestA "
          f"argmax window (radai_lib.mx31_peaks)")
    w = ak[["run_id", "time", "time_start", "time_stop", "category"]]
    off_start = (w.time_start - w.time) / 1000
    off_stop = (w.time_stop - w.time) / 1000
    width = off_stop - off_start
    print(f"window rel CA (s): start q10/50/90 "
          f"{off_start.quantile([.1, .5, .9]).round(1).to_dict()}")
    print(f"                    stop  q10/50/90 "
          f"{off_stop.quantile([.1, .5, .9]).round(1).to_dict()}")
    print(f"window width (s):  mean {width.mean():.1f}  med {width.median():.1f}"
          f"  p90 {width.quantile(.9):.1f}  max {width.max():.1f}")

    runs = rows[:, 0].astype(np.int64)
    t_arr = rows[:, 2].astype(np.float64)
    in_win = np.zeros(len(rows), bool)
    near = np.zeros(len(rows), bool)          # within +-150 s of some CA
    for e in w.itertuples():
        sel = runs == e.run_id
        in_win |= sel & (t_arr >= e.time_start) & (t_arr <= e.time_stop)
        near |= sel & (np.abs(t_arr - e.time) <= 150_000)
    wing = near & ~in_win                     # near a source, outside every window
    far = ~near
    def offq(mask, ref="ca"):
        offs = []
        for e in w.itertuples():
            sel = (runs == e.run_id) & mask & (np.abs(t_arr - e.time) <= 150_000)
            offs.extend((t_arr[sel] - e.time) / 1000.0)
        offs = np.array(offs)
        return (f"n={len(offs)} q05/q25/q50/q75/q95 = "
                f"{np.round(np.percentile(offs, [5, 25, 50, 75, 95]), 1).tolist()}"
                if len(offs) else "n=0")
    print(f"\nalarms at thr: total={len(rows)}  in-window={in_win.sum()}  "
          f"wing(outside window, <=150s of CA)={wing.sum()}  far={far.sum()}")
    print(f"  FPs recorded = {len(fp_df)} (should equal wing+far = "
          f"{int(wing.sum() + far.sum())})")
    print(f"offset (alarm time - CA), s, IN-WINDOW alarms:   {offq(in_win)}")
    print(f"offset (alarm time - CA), s, WING alarms:        {offq(wing)}")
    # winner offsets: among the alarms inside each detected encounter's window
    winners = ak[ak.alg_metric > -np.inf] if "alg_metric" in ak.columns else ak.iloc[0:0]
    if len(winners):
        offs = []
        for e in winners.itertuples():
            sel = (runs == e.run_id) & (t_arr >= e.time_start) & (t_arr <= e.time_stop)
            if sel.any():
                offs.append(t_arr[sel][np.argmax(rows[sel, 4].astype(np.float64))]
                            - e.time)
        offs = np.array(offs) / 1000.0
        print(f"offset (WINNER alarm - CA), s: n={len(offs)} q05..q95 = "
              f"{np.round(np.percentile(offs, [5, 25, 50, 75, 95]), 1).tolist()}")
    print("\nd_recall by window width bucket (after this scoring):")
    ak2 = ak.copy()
    ak2["detection"] = ak2.get("alg_metric", pd.Series(-np.inf, index=ak2.index)) > -np.inf
    ak2["wd"] = pd.cut(width, [0, 15, 30, 1000], labels=["<=15s", "15-30s", ">30s"])
    print(ak2.groupby("wd", observed=True).agg(n=("detection", "size"),
                                               d=("detection", "mean")).round(3))
    print("\nd_recall by snr_peak bucket:")
    ak2["snr_b"] = pd.cut(ak2.snr_peak, [-1, 0.5, 1, 2, 1000],
                          labels=["<=0.5", "0.5-1", "1-2", ">2"])
    print(ak2.groupby("snr_b", observed=True).agg(n=("detection", "size"),
                                                  d=("detection", "mean")).round(3))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ak").add_argument("--force", action="store_true")
    sc = sub.add_parser("score")
    sc.add_argument("--thr", type=float, default=None)
    sc.add_argument("--curve", type=str, default=None,
                    help="comma list of thresholds -> compact d/fpr table")
    sub.add_parser("diag").add_argument("--thr", type=float, default=None)
    a = ap.parse_args()
    ak = build_ak(force=getattr(a, "force", False))
    if a.cmd == "ak":
        return
    thr = a.thr or float(json.load(open(FILE_JSON))["buckets"]["mid"]["thr"])
    rows = np.array([(r, w, t, lab, m) for r, w, t, lab, m in val_rows(thr)],
                    dtype=object)
    res, ak2, fp_df, aux = score_rows([tuple(x) for x in rows], ak)
    print(f"\n== validation scored at thr={thr:.4f} "
          f"({aux['rows']} rows; matched={aux['matched']}, FP={aux['fps']}) ==")
    print(f"denominators: compute-style {aux['den_compute']:.1f} h "
          f"| answers-style {aux['den_answers']:.1f} h "
          f"| windows sum {aux['win_hr']:.2f} h")
    print(res[["encounters", "tp", "tc", "tid", "fp", "d_recall", "c_recall",
               "id_recall", "fpr_compute"]].to_string(float_format=lambda v: f"{v:.4f}"))
    if a.cmd == "diag":
        diag(rows, ak2, fp_df)
    if a.cmd == "score" and a.curve:
        print("\n== curve ==")
        for t in [float(x) for x in a.curve.split(",")]:
            rr = val_rows(t)
            r2, _, f2, x2 = score_rows(rr, ak)
            g = r2.loc["Global"]
            print(f"thr={t:7.4f} rows={x2['rows']:5d} matched={x2['matched']:4d} "
                  f"fp={x2['fps']:4d} fpr={g.fpr_compute:.3f} "
                  f"d={g.d_recall:.3f} c={g.c_recall:.3f} id={g.id_recall:.3f}")


if __name__ == "__main__":
    main()
