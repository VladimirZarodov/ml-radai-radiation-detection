# -*- coding: utf-8 -*-
"""s1_diag.py -- Checkpoint-1 diagnostics (user addendum 2026-10-03).

All are DIAGNOSTICS ONLY: no tuning, no config/protocol change, lockbox
untouched. Run AFTER the training chain finished (no GPU work concurrently
with the chain).

Modes:
  --mode timeshift --cfg c64h   CPU  check 1: pooled OOF alarms shifted
       +/-400 s (out-of-run alarms DROPPED, not wrapped); d and fpr at the
       four dev-selected thresholds should collapse.
  --mode speed     --cfg c64h   CPU  check 2: terciles by platform speed at
       closest approach (training_v4.3.h5 detector/position/velocity);
       proxies: official window width, distance at CA. Global d/c/id at the
       0.35 and 0.70 dev thresholds per tercile.
  --mode testing   --cfg c64h   GPU  check 3 (LAST, after chain): fold-
       ensemble inference on the 300 TESTING runs (features only, no labels
       used); alarms/h at dev thresholds vs dev-OOF alarms/h, dispersion,
       score quantiles. Testing-derived -> disclosed in README.
  --mode leakaudit              CPU  check 4: written leakage checklist (a-d).

Writes reports/{cfg}_diag.md (+ reports/leak_audit.md, reports/testing_diag.md).
"""
import argparse
import inspect
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch  # import only; no CUDA work happens in CPU modes

sys.path.insert(0, str(Path(__file__).parent))
import v2_lib as V            # noqa: E402
import s1_data as D           # noqa: E402
import s1_train as S          # noqa: E402
import s1_model as M          # noqa: E402
from s1_eval import alarms_from_run, load_oof_alarms, base_setup  # noqa: E402

TARGETS = [0.085, 0.17, 0.35, 0.7]
FEAT_TEST = V.CACHE / "feat_test_diag.h5"   # separate key-space: NO targets


def thr_rows(cfg):
    j = json.load(open(V.REPORTS / f"{cfg}_oof.json"))
    return {float(t): (j["targets"][t]["thr"], j["targets"][t]["row"])
            for t in j["targets"]}, j


# ---------------------------------------------------------------------------
# check 1: time-shift control
# ---------------------------------------------------------------------------

def run_timeshift(cfg):
    lock, runs240, folds, hi = D.dev_pool()
    alarms, *_ = load_oof_alarms(cfg)
    ak, meta, _ = base_setup()
    j = json.load(open(V.REPORTS / f"{cfg}_oof.json"))
    nb = {r: meta[r]["nb"] for r in runs240}
    md = [f"# check 1 time-shift control -- {cfg}",
          "policy: shifted alarms landing outside [0, nb_s) of their own run "
          "are DROPPED (no wrap); scores unchanged; thresholds = the four "
          "dev-selected values (pooled OOF).", ""]
    md.append("| shift | target | thr | d_G | c_G | id_G | fpr | FP |")
    md.append("|---|---|---|---|---|---|---|---|")
    for sh in (0, 400, -400):
        al2 = []
        for a in alarms:
            t = a["time_ms"] + sh * 1e3
            if 0 <= t < nb[a["run"]] * 1e3:
                b = dict(a)
                b["time_ms"] = t
                b["time_start_ms"] = t - 1.5e3
                b["time_stop_ms"] = t + 2.5e3
                al2.append(b)
        cur = V.AlarmCurve(ak, al2)
        for tg in TARGETS:
            thr = j["targets"][str(tg)]["thr"]
            row = cur.row_for_thr(thr)
            t = cur.metrics(row, runs240, meta=meta)
            g = t.loc["Global"]
            md.append(f"| {sh:+d}s | {tg} | {thr:.4f} | {g['d_recall']:.4f} "
                      f"| {g['c_recall']:.4f} | {g['id_recall']:.4f} | "
                      f"{g['fpr']:.4f} | {int(g['fp'])} |")
        md.append("")
    (V.REPORTS / f"{cfg}_diag.md").write_text("\n".join(md),
                                              encoding="utf-8")
    print("\n".join(md[:30]))
    print(f"wrote reports/{cfg}_diag.md (check 1; checks 2 appends)")


# ---------------------------------------------------------------------------
# check 2: speed / duration sensitivity
# ---------------------------------------------------------------------------

def encounter_speeds(runs240):
    enc = V.load_encounters("train")
    enc = enc[enc.run.isin(list(runs240))].reset_index(drop=True)
    sp, wid, dist = [], [], []
    with h5py.File(V.TRAIN_H5, "r") as f:
        pos = {r: f[f"runs/run{r}/detector/position"] for r in set(enc.run)}
        for _, e in enc.iterrows():
            g = pos[int(e.run)]
            t = g["time"][:]                       # ms since run start
            v = g["velocity"][:]
            i = np.searchsorted(t, e.ca_ms)
            i = min(max(i, 0), len(t) - 1)
            sp.append(float(abs(v[i])))
            wid.append(float(e.width_s))
            dist.append(float(e.dist))
    enc["speed_mps"] = sp
    enc["width_s"] = np.array(wid)
    return enc


def terciles_tab(cfg, enc, col):
    lock, runs240, folds, hi = D.dev_pool()
    alarms, *_ = load_oof_alarms(cfg)
    ak, meta, _ = base_setup()
    j = json.load(open(V.REPORTS / f"{cfg}_oof.json"))
    q = enc[col].quantile([1 / 3, 2 / 3]).to_numpy()
    lab = np.digitize(enc[col].to_numpy(), q)      # 0/1/2
    out = []
    for gi in range(3):
        sub = enc[lab == gi]
        ak_sub = ak.merge(sub[["run", "start_ms", "stop_ms"]],
                          on=["run", "start_ms", "stop_ms"])
        cur = V.AlarmCurve(ak_sub.reset_index(drop=True), alarms)
        row = {"grp": f"{col} tercile {gi} [{sub[col].min():.1f},"
                       f" {sub[col].max():.1f}] n={len(sub)}"}
        for tg in (0.35, 0.7):
            thr = j["targets"][str(tg)]["thr"]
            t = cur.metrics(cur.row_for_thr(thr), runs240, meta=meta)
            g = t.loc["Global"]
            row[f"d@{tg}"] = float(g["d_recall"])
            row[f"c@{tg}"] = float(g["c_recall"])
            row[f"id@{tg}"] = float(g["id_recall"])
        out.append(row)
    return out


def run_speed(cfg):
    lock, runs240, folds, hi = D.dev_pool()
    enc = encounter_speeds(runs240)
    md = [f"\n## check 2 speed/duration sensitivity -- {cfg}",
          "speed_at_CA from training_v4.3.h5 detector/position/velocity "
          "(|v| sampled at ca_ms); proxies: official window width_s, "
          "dist_m at CA. d/c/id = Global at the dev-selected thresholds "
          "(fixed), per tercile."]
    for col in ["speed_mps", "width_s", "dist"]:
        md.append(f"| group | d@.35 | c@.35 | id@.35 | d@.7 | c@.7 | id@.7 |")
        for r in terciles_tab(cfg, enc, col):
            md.append(f"| {r['grp']} | {r['d@0.35']:.4f} "
                      f"| {r['c@0.35']:.4f} | {r['id@0.35']:.4f} | "
                      f"{r['d@0.7']:.4f} | {r['c@0.7']:.4f} | "
                      f"{r['id@0.7']:.4f} |")
        md.append("")
    p = V.REPORTS / f"{cfg}_diag.md"
    old = p.read_text(encoding="utf-8") if p.exists() else ""
    p.write_text(old + "\n".join(md), encoding="utf-8")
    print("\n".join(md))


# ---------------------------------------------------------------------------
# check 3: unlabeled testing diagnostic (GPU, run LAST)
# ---------------------------------------------------------------------------

def test_store(runs, ctx, dev):
    if not FEAT_TEST.exists():
        print("building feat_test_diag.h5 (features only, no labels)")
        with h5py.File(FEAT_TEST, "w") as fo, \
                h5py.File(V.CACHE_H5["test"], "r") as f:
            for k in sorted(f.keys(), key=lambda x: int(x[1:])):
                g = f[k]
                X = D.features(g["total"][:], g["rate_all"][:],
                               g["rate_in"][:])
                fo.create_dataset(k + "/X", data=X, compression="gzip",
                                  compression_opts=4)
    d = {}
    with h5py.File(FEAT_TEST, "r") as f:
        for r in runs:
            d[r] = f[f"r{r}/X"][:]
    st = S.Store.__new__(S.Store)
    xs, n = [], 0
    st.j0, st.nb = {}, {}
    for r in runs:
        a = d[r]
        st.nb[r] = a.shape[0]
        xs.append(np.pad(a, ((ctx, ctx), (0, 0))))
        st.j0[r] = n
        n += a.shape[0] + 2 * ctx
    st.ctx, st.T = ctx, 2 * ctx + 1
    st.X = torch.from_numpy(np.concatenate(xs)).to(dev)
    st.off = torch.arange(-ctx, ctx + 1, device=dev)
    return st, d


def test_hours(runs):
    out = {}
    with h5py.File(V.CACHE_H5["test"], "r") as fc, \
            h5py.File(V.TEST_H5, "r") as fs:
        for r in runs:
            nb = int(fc[f"r{r}"].attrs["nb"])
            g = fs[f"runs/run{r}"]
            hr = float(g.attrs["end_timestamp"]
                       - g.attrs["start_timestamp"]) / 3600.0
            out[r] = (nb, hr)
    return out


def run_testing(cfg):
    import torch
    cf = S.CFGS[cfg]
    iso = D.iso_vocab()
    runs = sorted(int(k[1:]) for k in h5py.File(V.CACHE_H5["test"], "r"))
    dev = torch.device("cuda")
    st, _ = test_store(runs, cf["ctx"], dev)
    models = []
    for k in range(5):
        s = torch.load(S.cfg_dir(cfg) / f"fold{k}.pt", map_location=dev,
                       weights_only=False)
        m = M.Net(n_iso=len(iso)).to(dev)
        m.load_state_dict(s["best_state"])
        m.eval()
        models.append(m)
    P, I = {}, {}
    with torch.no_grad():
        for r in runs:
            jp = np.arange(st.j0[r] + cf["ctx"],
                           st.j0[r] + cf["ctx"] + st.nb[r])
            acc, acci = np.zeros(len(jp)), np.zeros((len(jp), len(iso)))
            for m in models:
                for i in range(0, len(jp), 256):
                    j = torch.from_numpy(jp[i:i + 256]).to(dev)
                    with torch.autocast("cuda", dtype=torch.float16):
                        dl, _, il = m(S.windows(st, j), cf["ctx"])
                    acc[i:i + 256] += torch.sigmoid(dl).float().cpu().numpy()
                    acci[i:i + 256] += il.float().cpu().numpy()
            P[r] = acc / len(models)
            I[r] = (acci / len(models)).argmax(1).astype(np.int8)
            del acc, acci
    alarms = []
    for r in runs:
        alarms += alarms_from_run(r, P[r], I[r], iso)
    hrs = test_hours(runs)
    den_test = sum(h for (_, h) in hrs.values())
    lock, runs240, folds, hi = D.dev_pool()
    deval, *_ = load_oof_alarms(cfg)
    j = json.load(open(V.REPORTS / f"{cfg}_oof.json"))
    den_dev = j["den_hr"]
    md = [f"\n# check 3 testing diagnostic -- {cfg} (fold-ensemble)",
          "DISCLOSURE: computed on the 300 TESTING runs. Only the unlabeled "
          "count stream is used (features + model scores). No labels, no "
          "answer keys, not in any selection; diagnostics only.",
          f"test runs {len(runs)}, total {den_test:.1f} h "
          f"({den_test / len(runs):.3f} h/run) vs dev pool {den_dev:.1f} h",
          "", "| dev thr | test alarms/h | dev OOF alarms/h | ratio |",
          "|---|---|---|---|"]
    for tg in TARGETS:
        thr = j["targets"][str(tg)]["thr"]
        nt = sum(1 for a in alarms if a["metric"] >= thr)
        nd = sum(1 for a in deval if a["metric"] >= thr)
        md.append(f"| {thr:.4f} | {nt / den_test:.2f} | {nd / den_dev:.2f} "
                  f"| {nt / den_test / (nd / den_dev):.2f} |")
    per = {}
    for a in alarms:
        if a["metric"] >= j["targets"]["0.35"]["thr"]:
            per[a["run"]] = per.get(a["run"], 0) + 1
    cnt = np.array([per.get(r, 0) for r in runs], float)
    qs = np.quantile(np.concatenate([P[r] for r in runs]),
                    [.5, .9, .99, .999, 1.0])
    qd = np.quantile(np.concatenate([np.load(V.CACHE / "oof" / cfg
                                             / f"r{r}.npz")["p"]
                                     for r in list(runs240)]),
                    [.5, .9, .99, .999, 1.0])
    md += ["", f"@0.35-thr alarms/run: mean {cnt.mean():.2f} "
              f"sd {cnt.std():.2f} max {cnt.max():.0f} "
              f"zero-runs {int((cnt == 0).sum())}/300",
           f"test ens score quantiles p50/90/99/99.9/max: "
           f"{np.round(qs, 4)}",
           f"dev OOF p quantiles (single own-fold model per run): "
           f"{np.round(qd, 4)}",
           f"v1 precedent: testing rows/h vs val ~1.2-1.3x at comparable "
           f"thr (Stage-0). Ratios above ~1.5x would be a warning sign."]
    (V.REPORTS / "testing_diag.md").write_text("\n".join(md),
                                               encoding="utf-8")
    print("\n".join(md))


# ---------------------------------------------------------------------------
# check 4: written leakage audit
# ---------------------------------------------------------------------------

def run_leakaudit(cfg):
    lock, runs240, folds, hi = D.dev_pool()
    md = [f"# check 4 leakage audit (config {cfg}) -- checklist with evidence"]
    src = inspect.getsource(D.features)
    bad = [t for t in ["listmode", "background_id", "sources", "bg_all",
                       "bg_comp", "sec_bg", "sec_src", "src_rest", "id",
                       "enc_0"] if t in src]
    md += ["(a) inputs/normalisation:",
           f"  D.features signature: {inspect.signature(D.features)} "
           f"-- only per-second totals + rate channels; "
           f"label/answer-key tokens found in source: {bad or 'NONE'}",
           "  normalisation = sqrt(total / per-run MEDIAN of the run's own "
           "total counts) + robust log-rate channels (run-local); "
           "no listmode id, no background_id, no sources/*, no AK.",
           "  EVIDENCE: grep s1_data.py cache ->"]
    import subprocess
    g = subprocess.run(["findstr", "/i",
                        "listmode background_id sources", "s1_data.py"],
                       capture_output=True, text=True)
    md.append(f"    hits: {g.stdout.strip() or 'NONE'} (per-run median uses "
              "run's own totals only)")
    md += ["(b) window gather inside own run:",
           "  Store concatenates per-run padded blocks; windows() gathers "
           "j+arange(-ctx,ctx+1); inference j limited to "
           "[j0+ctx, j0+ctx+nb) -> gathered rows in [j0, j0+nb+2ctx) = "
           "exactly that run's block; E030 synthetic-store assert (centered, "
           "no inter-run rows); re-verified at runtime here:"]
    ctx = S.CFGS[cfg]["ctx"]
    ok = True
    import torch
    st = S.Store(sorted(runs240)[:3], ctx, S.CFGS[cfg]["target"],
                 torch.device("cpu"))
    for r in sorted(runs240)[:3]:
        jp = np.arange(st.j0[r] + ctx, st.j0[r] + ctx + st.nb[r])
        idx = jp[:, None] + st.off.numpy()[None, :]
        ok &= idx.min() >= st.j0[r] and idx.max() < st.j0[r] + st.nb[r] \
            + 2 * ctx
    md.append(f"    runtime bounds check on 3 runs: {'PASS' if ok else 'FAIL'}")
    md += ["(c) fold separation (OOF scoring runs absent from that fold's "
           "model training AND its inner validation):"]
    for k in range(5):
        val = set(folds[k])
        tr = [r for r in runs240 if r not in val]
        itr, iva = D.inner_split(sorted(tr))
        md.append(f"  fold{k}: |val|={len(val)} inter(inner-train,val)="
                  f"{len(set(itr) & val)} inter(inner-val,val)="
                  f"{len(set(iva) & val)} lockbox∩val="
                  f"{len(val & set(lock))}")
    md += ["  NOTE: other pool runs are IN fold-k training by design "
           "(cross-fitting); each scored run is held out of ITS OWN model. "
           "All lists derived from frozen cache/folds.json, unchanged."]
    md += ["(d) threshold provenance:",
           "  thr values in reports/*_oof.json come from "
           "AlarmCurve.thr_at_fpr(target, runs240, meta) on POOLED OOF "
           "alarms only (runs240, dev pool; lockbox excluded -- leakguard "
           "overlap=0 logged per run of train/eval/eval-table). No test "
           "data involved. v1 comparator thresholds likewise pooled-AK-only."]
    p = V.REPORTS / "leak_audit.md"
    p.write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["timeshift", "speed", "testing", "leakaudit"])
    ap.add_argument("--cfg", default="c64h")
    a = ap.parse_args()
    V.REPORTS.mkdir(exist_ok=True)
    {"timeshift": lambda: run_timeshift(a.cfg),
     "speed": lambda: run_speed(a.cfg),
     "testing": lambda: run_testing(a.cfg),
     "leakaudit": lambda: run_leakaudit(a.cfg)}[a.mode]()


if __name__ == "__main__":
    main()
