# -*- coding: utf-8 -*-
"""
s2_metrics.py -- Stage-2 per-config metrics + paired comparisons (PLAN 2026-10-06).

mode cfg     --cfg <cfg>   (requires cache/oof/<cfg>/r*.npz from s2_train eval)
    Pooled dev OOF curve -> OWN thr per target {.085,.17,.35,.7} (thr_at_fpr),
    global d/c/id/fpr + boot CI, per-category rows (Medical watch),
    REAL speed-tercile d at own thrs (edges pinned 4.4/6.8 m/s per E039;
    fast = >=6.8 band = PRIMARY shift metric, per frozen PLAN 2026-10-06),
    calibrated-synth + out-of-range(kmax=2.0) retention/inflation at own thrs
    (s2_shift, SEED_CAL stream, ORIGINAL windows as labels; c64s reuses the
    Step-0 alarm caches), training cost from models/<cfg>/history.jsonl.
    -> cache/s2/metrics_<cfg>.json

mode compare             all trained configs:
    Paired run-cluster bootstrap (V.boot_diff_at_rows, B=1000, seed 11, field
    d_recall) of each s2 config vs c64s AND vs s2-0, per target, AT EACH
    CONFIG'S OWN thr; noise floor NF(tg)=|d(c64s)-d(s2-0)| point diff; claim =
    CI excludes 0 AND |diff| > NF.  -> cache/s2/compare.json

GPU path goes through s1_run.py launcher:
  .venv\\Scripts\\python.exe submission_v2\\s1_run.py submission_v2\\s2_metrics.py --cfg s2-bg
  .venv\\Scripts\\python.exe submission_v2\\s1_run.py submission_v2\\s2_metrics.py --mode compare
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "submission"))
import v2_lib as V          # noqa: E402
import s1_data as D         # noqa: E402
import s1_eval as E         # noqa: E402
import s2_cfg as C          # noqa: E402

TARGETS = E.TARGETS
SPEED_EDGES = [4.4, 6.8]    # frozen real-speed tercile edges (E039)


def plain_stats(cfg, runs240, ak, meta):
    alarms, *_ = E.load_oof_alarms(cfg)
    curve = V.AlarmCurve(ak, alarms)
    sel = {}
    for tg in TARGETS:
        thr, row, t = curve.thr_at_fpr(tg, runs240, meta=meta)
        ci = V.boot_at_row(curve, row, runs240, meta, "d_recall", 1000)
        g = t.loc["Global"]
        sel[str(tg)] = dict(
            thr=float(thr), d=float(g["d_recall"]), c=float(g["c_recall"]),
            id=float(g["id_recall"]), fpr=float(g["fpr"]),
            fp=float(g["fp"]), ci_d=[float(ci[0]), float(ci[1])],
            cats={str(i): dict(d=float(r["d_recall"]), c=float(r["c_recall"]),
                               id=float(r["id_recall"]), n_enc=int(r["n_enc"]))
                  for i, r in t.iterrows() if i != "Global"})
    return alarms, curve, sel


def encounter_speeds(runs240):
    """|velocity| at CA row per pool encounter (s1_diag.encounter_speeds
    convention: searchsorted on detector/position time_ms at ca_ms, clamp)."""
    import h5py
    enc = V.load_encounters("train")
    enc = enc[enc.run.isin(list(runs240))].reset_index(drop=True)
    sp = []
    with h5py.File(V.TRAIN_H5, "r") as f:
        pos = {r: f[f"runs/run{r}/detector/position"] for r in set(enc.run)}
        for _, e in enc.iterrows():
            g = pos[int(e.run)]
            t = g["time"][:]
            v = g["velocity"][:]
            i = min(max(int(np.searchsorted(t, e.ca_ms)), 0), len(t) - 1)
            sp.append(abs(float(v[i])))
    enc["speed_mps"] = sp
    return enc


def speed_terciles(cfg, runs240, ak, meta, sel):
    """PRIMARY shift metric per frozen PLAN 2026-10-06: REAL speed terciles,
    EDGES PINNED at 4.4 / 6.8 m/s (E039), NOT per-config quantiles:
    0=slow (<4.4), 1=mid, 2=FAST (>=6.8, the hard shift-sensitive band)."""
    enc = encounter_speeds(runs240)
    lab = np.digitize(enc["speed_mps"].to_numpy(), [4.4, 6.8])
    alarms, *_ = E.load_oof_alarms(cfg)
    out = {}
    for gi in range(3):
        sub = enc[lab == gi]
        ak_sub = ak.merge(sub[["run", "start_ms", "stop_ms"]],
                          on=["run", "start_ms", "stop_ms"])
        cur = V.AlarmCurve(ak_sub.reset_index(drop=True), alarms)
        d = {}
        for tg in TARGETS:
            row = cur.row_for_thr(sel[str(tg)]["thr"])
            m = cur.metrics(row, runs240, meta=meta)
            d[str(tg)] = float(m.loc["Global", "d_recall"])
        out[gi] = dict(n=int(len(sub)),
                       vlo=float(sub["speed_mps"].min()),
                       vhi=float(sub["speed_mps"].max()), d=d)
    ratio = {tg: out[2]["d"][tg] / max(out[0]["d"][tg], 1e-9)
             for tg in map(str, TARGETS)}
    return dict(col="speed_mps", edges=[4.4, 6.8], ter=out, fast_mean=float(
        np.mean([out[2]["d"][str(tg)] for tg in TARGETS])), ratio=ratio)


def synth_rows(cfg, curve, sel, runs240, ak, meta, folds, dev, params, tag):
    import s1_model as M
    import s2_step0 as ST
    import torch
    ST.torch = torch                        # score_runs uses module global
    ST.ENC = V.load_encounters("train")
    fold_of = {r: k for k, fr in enumerate(folds) for r in fr}
    models = {}
    for k in range(5):
        s = torch.load(V.CACHE / "models" / cfg / f"fold{k}.pt",
                       map_location=dev, weights_only=False)
        net = M.Net(n_iso=24).to(dev)
        net.load_state_dict(s["best_state"])
        net.eval()
        models[k] = net
    al = ST.score_runs("c64s", runs240, params, fold_of, models, dev,
                       D.iso_vocab(), tag)   # cf recipe == c64s by design
    del models
    torch.cuda.empty_cache()
    cs = V.AlarmCurve(ak, al)
    out = []
    for tg in TARGETS:
        thr_s, row_s, t_s = cs.thr_at_fpr(tg, runs240, meta=meta)
        row = cs.row_for_thr(sel[str(tg)]["thr"])
        m = cs.metrics(row, runs240, meta=meta)
        g, b = m.loc["Global"], sel[str(tg)]
        out.append(dict(target=tg, thr_self=float(thr_s),
                        d_self=float(t_s.loc["Global", "d_recall"]),
                        d_at_dev_thr=float(g["d_recall"]),
                        fpr_at_dev_thr=float(g["fpr"]),
                        retention=float(g["d_recall"]) / max(b["d"], 1e-9),
                        inflation=float(g["fpr"]) / max(b["fpr"], 1e-9)))
    return out


def cost(cfg):
    p = V.CACHE / "models" / cfg / "history.jsonl"
    if not p.exists():
        return None
    rs = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    per_fold = {}
    for r in rs:
        per_fold.setdefault(r["fold"], []).append(r)
    return dict(folds={str(f): dict(secs=round(sum(x["secs"] for x in v), 1),
                                    epochs=len(v),
                                    best_epoch=int(min(
                                        v, key=lambda x: x["val_focal"])["epoch"]),
                                    val_focal_min=float(min(
                                        x["val_focal"] for x in v)))
                       for f, v in per_fold.items()},
                total_secs=round(sum(r["secs"] for r in rs), 1),
                vram_peak_mib=float(max(r["vram_mib"] for r in rs)))


def mode_cfg(cfg, dev):
    lock, runs240, folds, hi = D.dev_pool()
    D.assert_no_lockbox(runs240, f"s2_metrics {cfg}",
                        C.S2 / "leakguard.log")
    ak = V.ak_table(V.load_encounters("train"))
    meta = V.run_meta("train")
    t0 = time.time()
    alarms, curve, sel = plain_stats(cfg, runs240, ak, meta)
    print(f"[{cfg}] plain done {time.time() - t0:.0f}s", flush=True)
    terc = speed_terciles(cfg, runs240, ak, meta, sel)
    print(f"[{cfg}] terciles: fast mean d {terc['fast_mean']:.4f}", flush=True)
    cal = C.calib()
    js = dict(cfg=cfg, sel=sel, terciles=terc, cost=cost(cfg))
    if cal:
        p1 = dict(kappa_max=cal["kappa_max"], theta=cal["theta"])
        p2 = dict(kappa_max=2.0, theta=cal["theta"])
        k1, t1 = p1["kappa_max"], p1["theta"]
        # E052: NORM semantics fixed to design reading -> ALL configs
        # (incl. c64s) recompute synth with the corrected generator; the
        # Step-0 pkl caches are NOT reused (they stacked f*1.2).
        tg1, tg2 = f"{cfg}_v2conf_{k1}_{t1}", f"{cfg}_v2out_2.0_{t1}"
        js["synth_cal"] = synth_rows(cfg, curve, sel, runs240, ak, meta,
                                     folds, dev, p1, tg1)
        print(f"[{cfg}] cal-synth done {time.time() - t0:.0f}s", flush=True)
        js["synth_out"] = synth_rows(cfg, curve, sel, runs240, ak, meta,
                                     folds, dev, p2, tg2)
        print(f"[{cfg}] out-synth done {time.time() - t0:.0f}s", flush=True)
    (C.S2 / f"metrics_{cfg}.json").write_text(json.dumps(js, indent=1))
    print(f"[{cfg}] METRICS OK ({time.time() - t0:.0f}s)", flush=True)


def mode_compare(trained):
    _lock, runs240, folds, hi = D.dev_pool()
    ak = V.ak_table(V.load_encounters("train"))
    meta = V.run_meta("train")
    curves, sels = {}, {}
    for cfg in trained:
        a, cur, sel = plain_stats(cfg, runs240, ak, meta)
        curves[cfg], sels[cfg] = cur, sel
        print(f"loaded {cfg}", flush=True)
    nf = {str(tg): abs(sels["c64s"][str(tg)]["d"] - sels["s2-0"][str(tg)]["d"])
          for tg in TARGETS}
    out = dict(nf=nf, pairs={})
    for cfg in [c for c in trained if c not in ("c64s", "s2-0")]:
        for cmp in ("c64s", "s2-0"):
            rows = {}
            for tg in TARGETS:
                r_a = curves[cfg].row_for_thr(sels[cfg][str(tg)]["thr"])
                r_b = curves[cmp].row_for_thr(sels[cmp][str(tg)]["thr"])
                dd = V.boot_diff_at_rows(curves[cfg], r_a, curves[cmp], r_b,
                                         runs240, meta, "d_recall", 1000)
                dv = sels[cfg][str(tg)]["d"] - sels[cmp][str(tg)]["d"]
                excl = not (dd[0] <= 0 <= dd[1])
                rows[str(tg)] = dict(diff=float(dv), ci=[float(dd[0]),
                                                         float(dd[1])],
                                     ci_excl_0=bool(excl),
                                     claim=bool(excl and abs(dv) > nf[str(tg)]))
            out["pairs"][f"{cfg}_vs_{cmp}"] = rows
            print(cfg, "vs", cmp, flush=True)
    (C.S2 / "compare.json").write_text(json.dumps(out, indent=1))
    print("COMPARE OK", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg")
    ap.add_argument("--mode", choices=["cfg", "compare"])
    a = ap.parse_args()
    _lock, runs240, folds, hi = D.dev_pool()
    if a.mode == "compare":
        trained = [c for c in ["c64s"] + C.ORDER
                   if (C.S2 / f"metrics_{c}.json").exists()]
        mode_compare(trained)
    else:
        import torch
        torch.zeros(1).cuda()
        mode_cfg(a.cfg, torch.device("cuda"))
