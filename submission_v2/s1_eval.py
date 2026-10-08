# -*- coding: utf-8 -*-
"""
s1_eval.py -- Stage-1 evaluation: per-second OOF scores -> alarms -> official
metrics through the FROZEN v2_lib harness.

Alarm extraction (fixed BEFORE any result inspection, identical for all
configs): score s(t) = sigmoid p Gaussian-smoothed (sigma=1.5 s, +-4 s);
per run sort seconds by s desc, greedy NMS min separation SEP=15 s, drop
s < PMIN=0.02, cap CAP=300/run; time_ms=(sec+0.5)*1e3 span +-2 s;
metric_1 = s (threshold-ordered, subset property); label_1 = isotope-head
argmax at that second (official 24-label vocab).

Modes:
  parity : feed v1's cached alarms through THIS path and compare against
           reports/v1_baseline.json (check b; must match < 1e-9)
  table  : pooled dev OOF vs v1 at targets + bootstrap CIs + per-category +
           HIGH-BG slice + FP distribution + failure examples -> report files
  synth  : SYNTH-SHIFT fold (frozen shift_synth params), same fold models,
           re-scored OOF -> report-only d/fpr at dev thr + FP inflation

Note on synth rate_all: the frozen transform returns only in-grid totals, so
rate_all_synth = rate_in_synth * median(rate_all/rate_in) over train-cache runs
(training-file statistic; testing will have true rate_all).  Disclosed here.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "submission"))
import v2_lib as V                                   # noqa: E402
import s1_data as D                                  # noqa: E402

TARGETS = [0.085, 0.17, 0.35, 0.7]
SEP = 15.0
PMIN = 0.02
CAP = 300
SIGMA = 1.5


def gauss_smooth(p, sigma=SIGMA):
    r = int(np.ceil(4 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    return np.convolve(np.pad(p.astype(np.float64), r, mode="edge"),
                       k, mode="valid")


def alarms_from_run(run, p, iso_idx, iso):
    s = gauss_smooth(p)
    o = np.argsort(-s, kind="stable")
    out, taken = [], []
    for j in o:
        v = s[j]
        if v < PMIN or len(out) >= CAP:
            break
        if any(abs(j - t) < SEP for t in taken):
            continue
        taken.append(j)
        out.append(dict(run=int(run), time_ms=(j + 0.5) * 1e3,
                        time_start_ms=(j - 1.5) * 1e3,
                        time_stop_ms=(j + 2.5) * 1e3,
                        metric=float(v), label=str(iso[int(iso_idx[j])])))
    return out


def load_oof_alarms(cfg):
    iso = D.iso_vocab()
    od = V.CACHE / "oof" / cfg
    lock_runs, runs240, folds, hi = D.dev_pool()
    a = []
    for r in runs240:
        z = np.load(od / f"r{r}.npz")
        a += alarms_from_run(r, z["p"], z["iso"], iso)
    return a, runs240, folds, hi, lock_runs


def base_setup():
    ak = V.ak_table(V.load_encounters("train"))
    meta = V.run_meta("train")
    v1 = pickle.load(open(V.CACHE / "v1_alarms_train.pkl", "rb"))
    curve_v1 = V.AlarmCurve(ak, v1)
    return ak, meta, curve_v1


# ---------------------------------------------------------------------------
# (b) harness parity: v1 alarms through the new eval path == v1_baseline.json
# ---------------------------------------------------------------------------

def parity(runs240, ak, meta, curve_v1):
    base = json.load(open(V.REPORTS / "v1_baseline.json"))
    worst, where = 0.0, None
    for tg in TARGETS:
        ref = {r["category"]: r
               for r in base["dev_targets"][str(tg)]["table"]}
        row = curve_v1.row_for_thr(base["dev_targets"][str(tg)]["thr"])
        t = curve_v1.metrics(row, runs240, meta=meta)
        for cat, rr in ref.items():
            for f in ("d_recall", "c_recall", "id_recall", "fpr"):
                d = abs(float(t.loc[cat, f]) - float(rr[f]))
                if d > worst:
                    worst, where = d, (tg, cat, f)
    ok = worst < 1e-9
    print(f"PARITY v1-through-new-path vs reports/v1_baseline.json: "
          f"max|diff| = {worst:.3e} ({where}) -> {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# main comparison table
# ---------------------------------------------------------------------------

def fmt_tab(t):
    return t[["n_enc", "tp", "tc", "tid", "fp", "d_recall", "c_recall",
              "id_recall", "fpr"]].to_string(float_format=lambda v: f"{v:.4f}")


def table(cfg, runs240, ak, meta, folds, hi, curve_v1):
    alarms, _, _, _, _ = load_oof_alarms(cfg)
    curve = V.AlarmCurve(ak, alarms)
    den = V.denom_hours(meta, runs240)
    md = [f"# Stage-1 {cfg}: pooled dev OOF, {len(runs240)} runs, "
          f"den {den:.1f} bg-h; alarm NMS sep {SEP}s, sigma {SIGMA}s",
          "",
          "| target | thr | d_G | d CI95 | c_G | id_G | fpr | FP | "
          "v1 d_G | d - v1 | CI(diff) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    sel = {}
    for tg in TARGETS:
        thr, row, t = curve.thr_at_fpr(tg, runs240, meta=meta)
        thr1, row1, t1 = curve_v1.thr_at_fpr(tg, runs240, meta=meta)
        ci_d = V.boot_at_row(curve, row, runs240, meta, "d_recall", 1000)
        dd = V.boot_diff_at_rows(curve, row, curve_v1, row1, runs240, meta,
                                 "d_recall", 1000)
        sel[tg] = dict(thr=thr, row=row, table=t, ci_d=ci_d, dd=dd,
                       thr1=thr1, row1=row1, t1=t1)
        md.append(f"| {tg} | {thr:.4f} | {t.loc['Global','d_recall']:.4f} "
                  f"| [{ci_d[0]:.3f},{ci_d[1]:.3f}] | "
                  f"{t.loc['Global','c_recall']:.4f} | "
                  f"{t.loc['Global','id_recall']:.4f} | "
                  f"{t.loc['Global','fpr']:.4f} | "
                  f"{int(t.loc['Global','fp'])} | "
                  f"{t1.loc['Global','d_recall']:.4f} | "
                  f"{t.loc['Global','d_recall'] - t1.loc['Global','d_recall']:+.4f} | "
                  f"[{dd[0]:+.4f},{dd[1]:+.4f}] |")
    md.append("")
    for tg in TARGETS:
        md += [f"### {cfg} @ target fpr {tg} (thr {sel[tg]['thr']:.4f})",
               fmt_tab(sel[tg]["table"]),
               f"v1 comparator @ target fpr {tg} (thr {sel[tg]['thr1']:.4f})",
               fmt_tab(sel[tg]["t1"]), ""]
    # HIGH-BG slice (report only, model's own dev-selected thresholds)
    md += ["## HIGH-BG slice (OOF, thresholds from dev selection)",
           "| target | model d | model fpr | v1 d | v1 fpr |",
           "|---|---|---|---|---|"]
    hb = []
    for tg in TARGETS:
        t = curve.metrics(sel[tg]["row"], hi, meta=meta)
        t1 = curve_v1.metrics(sel[tg]["row1"], hi, meta=meta)
        hb.append(dict(target=tg, d=float(t.loc["Global", "d_recall"]),
                       fpr=float(t.loc["Global", "fpr"]),
                       v1_d=float(t1.loc["Global", "d_recall"]),
                       v1_fpr=float(t1.loc["Global", "fpr"])))
        md.append(f"| {tg} | {hb[-1]['d']:.4f} | {hb[-1]['fpr']:.4f} | "
                  f"{hb[-1]['v1_d']:.4f} | {hb[-1]['v1_fpr']:.4f} |")
    # FP distribution across runs @0.35
    enc_tab, fp_tab = curve.tables_at(sel[0.35]["row"])
    cnt = (fp_tab.groupby("run").size() if len(fp_tab)
           else pd.Series(dtype=int))
    n_run = len(runs240)
    top = cnt.sort_values(ascending=False).head(5)
    md += ["", f"## FP distribution @0.35: zero-FP runs "
           f"{n_run - len(cnt)}/{n_run}; max/run "
           f"{int(cnt.max()) if len(cnt) else 0}; top5 "
           f"{ {int(k): int(v) for k, v in top.items()} }; top3 share "
           f"{float(cnt.sort_values(ascending=False).head(3).sum()) / max(float(cnt.sum()), 1.0) if len(cnt) else 0:.2f}"]
    # missed encounters by isotope / SNR @0.35 (curve arrays align with ak)
    det = curve.det
    run_col = curve.enc_run
    in_pool = np.isin(run_col, runs240)
    alive = curve.enc_row <= sel[0.35]["row"]
    missed = in_pool & ~(det & alive)
    detd = in_pool & (det & alive)
    snr = V.load_encounters("train").drop_duplicates(
        subset=["run", "win_start_s", "win_stop_s"])
    akk = ak.reset_index(drop=True).copy()
    akk["snr_window"] = snr.set_index(
        ["run", "win_start_s", "win_stop_s"]).loc[
        list(zip(akk.run, akk.start_ms / 1e3, akk.stop_ms / 1e3)),
        "snr_window"].to_numpy()
    akk["detd"] = detd
    mm = akk[missed]
    g = mm.groupby("isotope").size().sort_values(ascending=False)
    tot_by_iso = akk.groupby("isotope").size()
    q_m = np.nanpercentile(mm.snr_window.to_numpy(float), [10, 50, 90])
    q_d = np.nanpercentile(akk.loc[akk.detd, "snr_window"].to_numpy(float),
                           [10, 50, 90])
    md += ["", f"## Encounters @0.35: detected {int(detd.sum())} / missed "
           f"{int(missed.sum())} of {int(in_pool.sum())}",
           "missed by isotope (missed/total, top): " + ", ".join(
               f"{k} {int(v)}/{int(tot_by_iso[k])}"
               for k, v in g.head(10).items()),
           f"snr_window quantiles missed p10/50/90: {np.round(q_m, 2)}",
           f"snr_window quantiles detected p10/50/90: {np.round(q_d, 2)}"]
    # typical FPs @0.35
    if len(fp_tab):
        fi = np.flatnonzero(curve.fp)          # full-array FP positions
        sel_fp = fi[curve.fp_row_f <= sel[0.35]["row"]][:8]
        A = curve.A
        md += ["", "sample FP alarms @0.35 (run, t_s, metric, label):"]
        rows = []
        for k in sel_fp:
            rows.append(f"  r{int(A['run'][k])} t={A['t'][k] / 1e3:.0f}s "
                        f"s={A['m'][k]:.3f} {A['lab'][k]}")
        md += rows
    js = dict(cfg=cfg, n_alarms=len(alarms), den_hr=den,
              alarm_params=dict(sep=SEP, pmin=PMIN, cap=CAP, sigma=SIGMA),
              targets={str(tg): dict(
                  thr=float(sel[tg]["thr"]), row=int(sel[tg]["row"]),
                  table=sel[tg]["table"].reset_index().to_dict("records"),
                  ci_d=[float(x) for x in sel[tg]["ci_d"]],
                  diff_vs_v1_d=[float(x) for x in sel[tg]["dd"]],
                  v1_thr=float(sel[tg]["thr1"]),
                  v1_table=sel[tg]["t1"].reset_index().to_dict("records"))
                  for tg in TARGETS},
              high_bg=hb)
    return md, js, sel, curve


# ---------------------------------------------------------------------------
# SYNTH fold (report only)
# ---------------------------------------------------------------------------

def synth(cfg, runs240, folds, ak, meta, sel, dev):
    import torch
    import h5py
    import s1_model as M
    import s1_train as S1
    import shift_synth as SH
    iso = D.iso_vocab()
    cf = S1.CFGS[cfg]
    fold_of = {r: k for k, fr in enumerate(folds) for r in fr}
    models = {}
    for k in range(5):
        s = torch.load(V.CACHE / "models" / cfg / f"fold{k}.pt",
                       map_location=dev, weights_only=False)
        net = M.Net(n_iso=24).to(dev)
        net.load_state_dict(s["best_state"])
        net.eval()
        models[k] = net
    # label-free-approx rate_all ratio from TRAIN cache (disclosed)
    with h5py.File(V.CACHE / "train.h5", "r") as f:
        ra = np.concatenate([f[f"r{r}"]["rate_all"][:600]
                             for r in runs240[:40]]).astype(np.float64)
        ri = np.concatenate([f[f"r{r}"]["rate_in"][:600]
                             for r in runs240[:40]]).astype(np.float64)
        ratio = float(np.median(np.clip(ra / np.maximum(ri, 1e-3), 1, 10)))
    print(f"synth rate_all/rate_in ratio = {ratio:.4f}")
    enc = V.load_encounters("train")
    cen = enc.groupby("run")["cen_bin"].apply(list).to_dict()
    alarms = []
    t0 = time.time()
    with h5py.File(V.CACHE / "train.h5", "r") as f, torch.no_grad():
        for i, r in enumerate(runs240):
            g = f[f"r{r}"]
            arrays = {"bg_comp": g["bg_comp"][:].astype(np.float64),
                      "src_rest": g["src_rest"][:].astype(np.float64)}
            q = 0
            while f"enc_{q}" in g:
                arrays[f"enc_{q}"] = g[f"enc_{q}"][:].astype(np.float64)
                q += 1
            sh = SH.synth_shift_run(arrays, cen.get(r, []), run=r)
            tot = sh["total"]
            ri_s = tot.sum(1).astype(np.float64)
            X = D.features(tot.astype(np.uint16), ri_s * ratio, ri_s)
            nb = X.shape[0]
            z = np.zeros((nb, 4), np.int8) - 1
            pre = {r: dict(X=X, y=np.zeros(nb, np.uint8),
                           ys=np.zeros(nb, np.float16),
                           tcat=z[:, 0], tiso=z[:, 0])}
            st = S1.Store([r], cf["ctx"], cf["target"], dev, pre=pre)
            j = st.valid_j
            P, I_ = [], []
            for k0 in range(0, len(j), 256):
                jj = j[k0:k0 + 256]
                with torch.autocast("cuda", dtype=torch.float16):
                    dl, cl, il = models[fold_of[r]](
                        S1.windows(st, jj), cf["ctx"])
                P.append(torch.sigmoid(dl).float().cpu().numpy())
                I_.append(il.argmax(1).cpu().numpy().astype(np.int8))
            alarms += alarms_from_run(r, np.concatenate(P),
                                      np.concatenate(I_), iso)
            del st, pre
            if (i + 1) % 40 == 0:
                print(f"  synth {i + 1}/{len(runs240)} "
                      f"({time.time() - t0:.0f}s)", flush=True)
    curve_sh = V.AlarmCurve(ak, alarms)
    out = []
    for tg in TARGETS:
        thr, row, t = curve_sh.thr_at_fpr(tg, runs240, meta=meta)
        row_d = curve_sh.row_for_thr(sel[tg]["thr"])
        tat = curve_sh.metrics(row_d, runs240, meta=meta)
        b = sel[tg]["table"].loc["Global"]
        out.append(dict(target=tg, thr_self=float(thr),
                        d_self=float(t.loc["Global", "d_recall"]),
                        d_at_dev_thr=float(tat.loc["Global", "d_recall"]),
                        fpr_at_dev_thr=float(tat.loc["Global", "fpr"]),
                        inflation=float(tat.loc["Global", "fpr"] /
                                       max(float(b["fpr"]), 1e-9))))
    return out, alarms


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["parity", "table", "synth"])
    ap.add_argument("--cfg", default="c64h")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    lock_runs, runs240, folds, hi = D.dev_pool()
    ak, meta, curve_v1 = base_setup()
    D.assert_no_lockbox(runs240, f"s1_eval {a.mode} pool",
                        V.CACHE / "models" / "leakguard.log")
    if a.mode == "parity":
        ok = parity(runs240, ak, meta, curve_v1)
        raise SystemExit(0 if ok else 1)
    if a.mode == "table":
        md, js, sel, curve = table(a.cfg, runs240, ak, meta, folds, hi,
                                   curve_v1)
        V.REPORTS.mkdir(exist_ok=True)
        (V.REPORTS / f"{a.cfg}_oof.md").write_text("\n".join(md),
                                                   encoding="utf-8")
        (V.REPORTS / f"{a.cfg}_oof.json").write_text(json.dumps(js, indent=1))
        print("\n".join(md[:20]))
        print(f"wrote reports/{a.cfg}_oof.md/json")
    elif a.mode == "synth":
        md, js, sel, curve = table(a.cfg, runs240, ak, meta, folds, hi,
                                   curve_v1)
        dev = __import__("torch").device(a.device)
        out, _ = synth(a.cfg, runs240, folds, ak, meta, sel, dev)
        rows = ["", "## SYNTH-SHIFT (frozen params, report only) OOF-scores",
                "| target | d self-cal | fpr self-cal | d @dev-thr | "
                "fpr @dev-thr | FP inflation |",
                "|---|---|---|---|---|---|"]
        for r in out:
            rows.append(f"| {r['target']} | {r['d_self']:.4f} | "
                        f"{r['thr_self']:.4f} thr | {r['d_at_dev_thr']:.4f} | "
                        f"{r['fpr_at_dev_thr']:.4f} | {r['inflation']:.2f} |")
        js["synth"] = out
        (V.REPORTS / f"{a.cfg}_oof.md").write_text(
            "\n".join(md + rows), encoding="utf-8")
        (V.REPORTS / f"{a.cfg}_oof.json").write_text(json.dumps(js, indent=1))
        print("\n".join(rows))


if __name__ == "__main__":
    main()
