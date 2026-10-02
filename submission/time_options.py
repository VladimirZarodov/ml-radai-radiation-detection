# -*- coding: utf-8 -*-
"""
time_options.py -- checkpoint D task 5: evaluate alarm-time / consolidation
options with the FAITHFUL official scorer (rules from official_scorer.py) on
VALIDATION only, compared at EQUAL official fpr, with run-cluster bootstrap.

Anchoring to portal measurement #1 (mid file, thr 5.7492, 413 rows):
    testing FP 153 (fpr 0.535/h over 286.5 bg-h)  vs  validation FP 21
    (fpr 0.2198/h over 95.5 bg-h)  at the same threshold
    => FP_test(thr) ~= SCALE * FP_val(thr),  SCALE = 153/21 = 7.29
    => recall transfer r = d_test/d_val = 0.0946/0.1453 = 0.651
These assume the val->test hardness ratio is roughly threshold-independent --
a strong assumption, stated as such; equal-fpr comparisons use FP_val = 21.

Options (detector scores & metric_1 semantics UNCHANGED; the peak family is
computed threshold-free so nestedness under a raised sweep threshold holds):
  base    : current alarm time (mx31 plateau-first = bestA-argmax window
            centre), sep=15 windows (30 s)
  smooth5 : argmax of Gaussian-smoothed bestA (sigma 5 s) in episode
            [p-30, p+15] windows
  smooth10: same, sigma 10 s
  centroid: score-weighted centroid of (bestA - run P5)+ over
            [p-30, plateau_end+15]
  sep30 / sep45 : keep only mx31 peaks surviving NMS with 60/90 s separation
            (consolidates the secondary episode peaks = WING false alarms)

Usage: .venv\\Scripts\\python.exe submission\\time_options.py
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import radai_lib as L
import official_scorer as OS

SCRATCH = Path(__file__).resolve().parent.parent / "_scratch"
TRAIN_SCORES = SCRATCH / "sub_training_scores.pkl"
VAL_ALARM_PKL = SCRATCH / "sub_validation_alarms.pkl"
TEST_ALARM_PKL = SCRATCH / "sub_testing_alarms.pkl"

FP_VAL_ANCHOR = 21.0            # val FP at portal-fpr 0.535 with current rows
SCALE = 153.0 / 21.0            # FP_test ~= SCALE * FP_val
R_DRIFT = 0.0946 / 0.1453       # d_test ~= R_DRIFT * d_val
BASELINE_THR = 5.7492
GRID = np.round(np.arange(5.0, 7.81, 0.05), 4)


# ---------------------------------------------------------------------------
# rows under each option
# ---------------------------------------------------------------------------

def _gauss(sigma_win):
    r = max(1, int(np.ceil(3 * sigma_win)))
    x = np.arange(-r, r + 1)
    k = np.exp(-0.5 * (x / sigma_win) ** 2)
    return k / k.sum()


def _smooth_argmax(bestA, p, sigma_win):
    lo, hi = max(0, p - 30), min(len(bestA), p + 16)
    seg = bestA[lo:hi].astype(np.float64)
    pad = np.concatenate([np.full(6, seg[0]), seg, np.full(6, seg[-1])])
    sm = np.convolve(pad, _gauss(sigma_win), mode="same")[6:-6]
    return lo + int(np.argmax(sm))


def peak_set(run_cache, sep):
    return set(L.mx31_peaks(run_cache["mx31"], sep=sep).tolist())


def option_rows(kind, thr, tc, alarms):
    """Rows (run, win, time_ms, label, metric) >= thr under an option."""
    sep_kind = {"sep30": 30, "sep45": 45}.get(kind, 15)
    if kind.startswith("sep"):
        kind = "base"
    if kind in ("sep",):
        kind = "base"
    peak_cache = {}
    out = []
    for a in alarms:
        if a["metric"] < thr or a["label"] is None:
            continue
        if sep_kind != 15:
            ps = peak_cache.setdefault(a["run"], peak_set(tc[a["run"]], sep_kind))
            if a["win"] not in ps:
                continue
        c = tc[a["run"]]
        if kind == "base":
            t_ms = float(a["time_ms"])
        elif kind in ("smooth5", "smooth10"):
            w2 = _smooth_argmax(c["bestA"], a["win"],
                                (5.0 if kind == "smooth5" else 10.0) / L.STRIDE)
            t_ms = float(c["wtime"][w2]) * 1e3
        elif kind == "centroid":
            base = float(np.percentile(c["bestA"], 5))
            lo, hi = max(0, a["win"] - 30), min(len(c["bestA"]), a["win_end"] + 15)
            seg = np.maximum(c["bestA"][lo:hi].astype(np.float64) - base, 0.0)
            w2 = int(np.round((np.arange(lo, hi) * seg).sum() / seg.sum())) \
                if seg.sum() > 0 else a["win"]
            t_ms = float(c["wtime"][w2]) * 1e3
        else:
            raise SystemExit(f"unknown option {kind}")
        out.append((a["run"], a["win"], t_ms, a["label"], float(a["metric"])))
    return sorted(out, key=lambda x: (x[0], x[1]))


# ---------------------------------------------------------------------------
# per-run independent scoring (enables exact run-cluster bootstrap)
# ---------------------------------------------------------------------------

def _windows_by_run(ak):
    W = {}
    for e in ak.itertuples():
        W.setdefault(int(e.run_id), []).append(
            (float(e.time_start), float(e.time_stop), str(e.category),
             str(e.isotope)))
    return W


def score_run(rows_r, wins_r):
    """(fp, tp, tc, tid, enc) for ONE run's rows under official rules."""
    owned = [False] * len(wins_r)
    fp = tp = tcc = tid = 0
    for _run, _w, tm, lab, m in sorted(rows_r, key=lambda x: -x[4]):
        hit = next((i for i, (a0, a1, _c, _i) in enumerate(wins_r)
                    if a0 <= tm <= a1), -1)
        if hit >= 0:
            if not owned[hit]:
                owned[hit] = True
                tp += 1
                _a0, _a1, cat, iso = wins_r[hit]
                if OS.label2category(lab) == cat:
                    tcc += 1
                if iso == lab:
                    tid += 1
        else:
            fp += 1
    return fp, tp, tcc, tid, len(wins_r)


def run_group(rows, W):
    g = {}
    for r in rows:
        g.setdefault(r[0], []).append(r)
    return g


def metrics_at(rows, ak, W=None):
    W = W or _windows_by_run(ak)
    fp = tp = tcc = tid = enc = 0
    by_run = run_group(rows, W)
    for r, wins in W.items():                    # ALL ak runs, incl. row-less
        f, t, c, i, n = score_run(by_run.get(r, []), wins)
        fp += f
        tp += t
        tcc += c
        tid += i
        enc += n
    return dict(fp=fp, tp=tp, tc=tcc, tid=tid, enc=enc,
                d=tp / enc, c=tcc / enc, id=tid / enc)


def curve_for_rows(rows, ak, grid, W=None):
    """Cumulative official metrics at each sweep threshold (exact, single pass
    per threshold via per-run metric-desc ownership)."""
    W = W or _windows_by_run(ak)
    by_run_all = run_group(rows, W)
    out = []
    for t in grid:
        rr = [x for r in by_run_all for x in by_run_all[r] if x[4] >= t]
        m = metrics_at(rr, ak, W)
        out.append(m | {"thr": t})
    return pd.DataFrame(out).set_index("thr")


def bootstrap(kind, thr_point, B=300, seed=7):
    """Run-cluster bootstrap at FIXED thr: CI for FP_val and d_val."""
    alarms = pickle.load(open(VAL_ALARM_PKL, "rb"))
    tc = pickle.load(open(TRAIN_SCORES, "rb"))
    ak = OS.build_ak()
    W = _windows_by_run(ak)
    base = option_rows(kind, thr_point, tc, alarms)
    per_run = run_group(base, W)
    enc_by_run = {r: len(w) for r, w in W.items()}
    stat = {r: score_run(per_run.get(r, []), wins)
            for r, wins in W.items()}
    runs = sorted(stat)
    rng = np.random.default_rng(seed)
    ds, fs = [], []
    for _ in range(B):
        pick = rng.choice(runs, len(runs), replace=True)
        fp = tp = enc = 0
        for r in pick:
            f, t, _c, _i, _n = stat[r]
            fp += f
            tp += t
            enc += enc_by_run[r]
        ds.append(tp / enc)
        fs.append(fp)
    return np.percentile(ds, [2.5, 97.5]), np.percentile(fs, [2.5, 97.5])


# ---------------------------------------------------------------------------

def main():
    alarms = pickle.load(open(VAL_ALARM_PKL, "rb"))
    tc = pickle.load(open(TRAIN_SCORES, "rb"))
    ak = OS.build_ak()
    W = _windows_by_run(ak)
    n_enc = len(ak)
    print(f"SCALE={SCALE:.2f} (FP_test per FP_val), "
          f"R_DRIFT={R_DRIFT:.3f} (d_test per d_val), val enc={n_enc}")

    # ---- wing analysis + blind duplicate proxy -----------------------------
    base = option_rows("base", BASELINE_THR, tc, alarms)
    by_run_rows = run_group(base, W)
    allrows = [(r, x[2], x[4]) for r, rr in by_run_rows.items() for x in rr]
    fp_rows = []
    for r, rr in by_run_rows.items():
        wins = W.get(r, [])
        for row in sorted(rr, key=lambda x: -x[4]):
            hit = next((i for i, (a0, a1, _c, _i) in enumerate(wins)
                        if a0 <= row[2] <= a1), -1)
            if hit < 0:
                fp_rows.append((r, row[2], row[4]))
    ca_by_run = {}
    for e in ak.itertuples():
        ca_by_run.setdefault(int(e.run_id), []).append(float(e.time))
    wing = supp = 0
    for r, tm, m in fp_rows:
        if any(abs(tm - ca) <= 150_000 for ca in ca_by_run.get(r, [])):
            wing += 1
            if any(r2 == r and m2 > m and abs(t2 - tm) <= 150_000
                   for r2, t2, m2 in allrows):
                supp += 1
    print(f"val FP={len(fp_rows)}: wing (<=150 s of a CA)={wing}, "
          f"of which suppressable by a higher-metric row within 150 s={supp}; "
          f"far={len(fp_rows) - wing}")
    for tag, pk in [("validation", VAL_ALARM_PKL), ("TESTING (blind)", TEST_ALARM_PKL)]:
        a = pickle.load(open(pk, "rb"))
        rr = [x for x in a if x["metric"] >= BASELINE_THR and x["label"]]
        by = {}
        for x in rr:
            by.setdefault(x["run"], []).append(x)
        dup = sum(1 for x in rr
                  if any(y["metric"] > x["metric"]
                         and abs(y["time_ms"] - x["time_ms"]) <= 150_000
                         for y in by[x["run"]]))
        print(f"{tag}: rows@5.7492={len(rr)}  with higher-metric row within "
              f"150 s: {dup} ({dup / len(rr):.2f})")

    # ---- option curves at equal official fpr (FP_val ~= 21) ----------------
    print("\n== options: curve then thr with FP_val <= 21 (max rows) ==")
    sel = {}
    for kind in ["base", "smooth5", "smooth10", "centroid", "sep30", "sep45"]:
        rows = option_rows(kind, GRID[0], tc, alarms)
        cur = curve_for_rows(rows, ak, GRID, W)
        ok = cur[cur.fp <= FP_VAL_ANCHOR]
        if not len(ok):
            sel[kind] = None
            print(f"{kind:9s} NO threshold in grid reaches FP_val<={FP_VAL_ANCHOR:.0f} "
                  f"(best FP at thr {GRID[-1]:.2f}: {int(cur.fp.iloc[-1])}, "
                  f"d={cur.d.iloc[-1]:.4f})")
            continue
        t_sel = float(ok.index[0])
        m = cur.loc[t_sel]
        sel[kind] = t_sel
        print(f"{kind:9s} thr={t_sel:5.2f} FPv={int(m.fp):2d} d_val={m.d:.4f} "
              f"c={m.c:.4f} id={m.id:.4f}  proj: FP_test={SCALE * m.fp:5.1f} "
              f"d_test={R_DRIFT * m.d:.4f}  rows>=thr:{int((cur.loc[t_sel].tp))}")
    # paired: same threshold as base for transparency
    tb = sel["base"]
    print(f"\n== same thr ({tb:.2f}) cross-option ==")
    for kind in sel:
        rows = option_rows(kind, tb, tc, alarms)
        m = metrics_at(rows, ak, W)
        print(f"{kind:9s} rows={len(rows)} FPv={m['fp']:2d} d_val={m['d']:.4f} "
              f"proj d_test={R_DRIFT * m['d']:.4f}")

    print(f"\n== run-cluster bootstrap @ thr={tb:.2f} (B=300) ==")
    for kind in ["base", "smooth5", "centroid", "sep30", "sep45"]:
        (dlo, dhi), (flo, fhi) = bootstrap(kind, tb)
        rows = option_rows(kind, tb, tc, alarms)
        m = metrics_at(rows, ak, W)
        print(f"{kind:9s} d_val={m['d']:.4f} CI[{dlo:.3f},{dhi:.3f}]  "
              f"FPv={m['fp']:2d} CI[{flo:.0f},{fhi:.0f}]  "
              f"proj d_test={R_DRIFT * m['d']:.4f}")


if __name__ == "__main__":
    main()
