# -*- coding: utf-8 -*-
"""
v2_lib.py -- shared harness library for RADAI v2 (Stage 0+).

Faithful official scoring semantics (== submission/official_scorer.py, which
is a verified replication of radai/radai/evaluation/):
  * alarm matches an encounter iff same run and time_start <= t <= time_stop;
  * among matched alarms the one with max metric owns the encounter label
    (harness tie rule: equal metrics -> earlier (run, time) wins; official
    iterates CSV row order with strict >, identical for distinct metrics);
    other matched alarms are FREE (neither TP nor FP);
  * alarm matching no window -> FP (our spans 4 s <= 160 s -> penalty 1);
  * fpr = FP / (Σ run-hours - Σ window-hours)  [compute_metrics variant,
    matches portal denominator];
  * Global d/c/id over all encounters; Global fpr = ALL FPs / same denom;
    per-category rows split encounters by TRUE category and FPs by ALARM
    label category.

Objects: AlarmCurve (full vs-threshold curve, O(A log A)), per-run bootstrap
with multiplicity, frozen split loaders, cache loaders.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import h5py

ROOT = Path(__file__).resolve().parent.parent
S2 = Path(__file__).resolve().parent
CACHE = S2 / "cache"
REPORTS = S2 / "reports"
sys.path.insert(0, str(ROOT / "submission"))
import official_scorer as OS    # noqa: E402  (faithful label helpers, verbatim)

TRAIN_H5 = ROOT / "training_v4.3.h5"
DEV_H5 = ROOT / "developer_v4.3.h5"
TEST_H5 = ROOT / "testing_v4.3.h5"
CACHE_H5 = {"train": CACHE / "train.h5", "dev": CACHE / "dev.h5",
            "test": CACHE / "test.h5"}

CATS = ["NORM", "Medical", "Industrial", "NuclearMaterial", "Global"]
TRUE_CATS = CATS[:4]


# ---------------------------------------------------------------------------
# answer key & run metadata
# ---------------------------------------------------------------------------

def load_encounters(split):
    df = pd.read_csv(CACHE / f"encounters_{split}.csv")
    df["start_ms"] = df.win_start_s * 1e3
    df["stop_ms"] = df.win_stop_s * 1e3
    return df


def ak_table(enc_df):
    return enc_df[["run", "start_ms", "stop_ms", "category", "isotope"]] \
        .reset_index(drop=True)


def run_meta(split):
    """run -> dict(nb, n_enc, hr, win_hr).  hr from the source h5 run attrs
    (end_timestamp - start_timestamp)/3600 EXACTLY (official denominator);
    nb is the cached 1-s bin count (floor)."""
    ak = load_encounters(split)
    src = {"train": TRAIN_H5, "dev": DEV_H5, "test": TEST_H5}[split]
    out = {}
    with h5py.File(CACHE_H5[split], "r") as f:
        for k in f:
            g = f[k]
            out[int(k[1:])] = dict(nb=int(g.attrs["nb"]),
                                   n_enc=int(g.attrs["n_enc"]))
    with h5py.File(src, "r") as f:
        for r in out:
            g = f[f"runs/run{r}"]
            out[r]["hr"] = float(g.attrs["end_timestamp"]
                                 - g.attrs["start_timestamp"]) / 3600.0
    win = ((ak.stop_ms - ak.start_ms) / 3.6e6).groupby(ak.run).sum()
    for r in out:
        out[r]["win_hr"] = float(win.get(r, 0.0))
    return out


def denom_hours(meta, runs):
    cnt = pd.Series(list(runs)).value_counts()
    return float(sum(c * (meta[r]["hr"] - meta[r]["win_hr"])
                     for r, c in cnt.items()))


# ---------------------------------------------------------------------------
# alarms
# ---------------------------------------------------------------------------

def clean_alarms(alarms):
    """drop unlabelled (never emitted); return arrays sorted (m desc, run, t)."""
    a = [al for al in alarms if al.get("label")]
    m = np.fromiter((float(al["metric"]) for al in a), np.float64)
    r = np.fromiter((int(al["run"]) for al in a), np.int64)
    t = np.fromiter((float(al["time_ms"]) for al in a), np.float64)
    lab = np.array([str(al["label"]) for al in a], object)
    order = np.lexsort((t, r, -m))
    return dict(m=m[order], run=r[order], t=t[order], lab=lab[order],
                cat=np.array([OS.label2category(x) for x in lab[order]],
                             object),
                iso=np.array([OS.label2isotope(x) for x in lab[order]], object))


# ---------------------------------------------------------------------------
# AlarmCurve
# ---------------------------------------------------------------------------

class AlarmCurve:
    """Incremental vs-threshold curve for one AK + alarm set.

    Rows of thr_desc (descending unique metrics) are threshold levels; as the
    row index grows the threshold falls and alarms are added.
    curve.metrics(row, runs[, meta]) -> official table; thr_at_fpr(target,
    runs, meta) -> (thr, row, table) lowest thr (most alarms) with Global
    fpr <= target.
    """

    def __init__(self, ak, alarms):
        self.ak = ak.reset_index(drop=True)
        A = clean_alarms(alarms)
        self.A = A
        n = len(A["m"])
        # per-run encounter interval lookup (windows are disjoint per run)
        pos = {}
        s_ = self.ak.start_ms.to_numpy()
        e_ = self.ak.stop_ms.to_numpy()
        for r, gg in self.ak.groupby("run"):
            idx = gg.index.to_numpy()
            o = idx[np.argsort(s_[idx])]
            pos[r] = (o, s_[o], e_[o])
        det = np.zeros(len(self.ak), bool)
        cok = np.zeros(len(self.ak), bool)
        iok = np.zeros(len(self.ak), bool)
        win_m = np.full(len(self.ak), -np.inf)      # -inf = never matched
        taken = np.zeros(len(self.ak), bool)
        fp = np.zeros(n, bool)
        for k in range(n):
            r, t = int(A["run"][k]), A["t"][k]
            p = pos.get(r)
            hit = -1
            if p is not None:
                o, ss, ee = p
                i = np.searchsorted(ee, t)          # first stop >= t
                if i < len(o) and ss[i] <= t <= ee[i]:
                    hit = int(o[i])
            if hit >= 0:
                if not taken[hit]:                  # A sorted m-desc: winner
                    taken[hit] = True
                    det[hit] = True
                    cok[hit] = self.ak.category.iloc[hit] == A["cat"][k]
                    iok[hit] = self.ak.isotope.iloc[hit] == A["iso"][k]
                    win_m[hit] = A["m"][k]
            else:
                fp[k] = True
        # overlapping-window sanity (official would RuntimeError):
        for r, (o, ss, ee) in pos.items():
            if len(o) > 1 and (ss[1:] <= ee[:-1]).any():
                raise RuntimeError(f"overlapping AK windows in run {r}")
        self.fp = fp
        # curve bookkeeping
        thr_desc = np.unique(A["m"])[::-1]
        self.thr_desc = thr_desc
        nt = len(thr_desc)
        neg = -thr_desc                              # ascending
        self.enc_row = np.clip(np.searchsorted(
            neg, -win_m, side="left"), 0, nt)        # -inf -> nt (never)
        self.det, self.cok, self.iok = det, cok, iok
        self.fp_row = np.searchsorted(neg, -A["m"], side="left")
        self.enc_run = self.ak.run.to_numpy()
        self.enc_cat = self.ak.category.to_numpy()
        self.fp_run = A["run"][fp]
        self.fp_cat = A["cat"][fp]
        self.fp_row_f = self.fp_row[fp]
        self.n_alarms = n

    def tables_at(self, row):
        alive_e = self.enc_row <= row
        enc_tab = pd.DataFrame(dict(
            run=self.enc_run, cat=self.enc_cat,
            det=alive_e & self.det, cok=alive_e & self.cok,
            iok=alive_e & self.iok))
        alive_f = self.fp_row_f <= row
        fp_tab = pd.DataFrame(dict(run=self.fp_run[alive_f],
                                   cat=self.fp_cat[alive_f]))
        return enc_tab, fp_tab

    # ---- vectorized cumulative curve --------------------------------------
    def _subset_arrays(self, runs):
        """Cumulative per-row counts (row = threshold level in thr_desc)."""
        nt = len(self.thr_desc)
        runs = np.unique(list(runs))
        ie = np.isin(self.enc_run, runs)
        cat_e = self.enc_cat[ie]
        rows_e = self.enc_row[ie]
        cats = TRUE_CATS
        n = np.array([int((cat_e == c).sum()) for c in cats] + [int(ie.sum())])
        out = {"n": n}
        for j, flags in enumerate((self.det[ie], self.cok[ie], self.iok[ie])):
            mat = [np.bincount(rows_e[flags & (cat_e == c)],
                               minlength=nt + 2).cumsum() for c in cats]
            mat.append(np.bincount(rows_e[flags], minlength=nt + 2).cumsum())
            out[("det", "cok", "iok")[j]] = np.stack(mat)     # (5, nt+2)
        arr = np.zeros((5, nt + 2), np.int64)
        if len(self.fp_run):
            inf = np.isin(self.fp_run, runs)
            r_f = self.fp_row_f[inf]
            c_f = self.fp_cat[inf]
            for j, c in enumerate(cats):
                arr[j] = np.bincount(r_f[c_f == c],
                                     minlength=nt + 2).cumsum()
            arr[-1] = np.bincount(r_f, minlength=nt + 2).cumsum()
        out["fp"] = arr
        return out

    def metrics(self, row, runs, meta=None, den_hr=None):
        if den_hr is None:
            den_hr = denom_hours(meta, runs)
        a = self._subset_arrays(runs)
        rows = []
        for j, cat in enumerate(CATS):
            n = int(a["n"][j])
            tp = int(a["det"][j, row]) if row >= 0 else 0
            tc = int(a["cok"][j, row]) if row >= 0 else 0
            tid = int(a["iok"][j, row]) if row >= 0 else 0
            fp = int(a["fp"][j, row]) if row >= 0 else 0
            rows.append(dict(category=cat, n_enc=n, tp=tp, tc=tc, tid=tid,
                             fp=fp, d_recall=tp / n if n else np.nan,
                             c_recall=tc / n if n else np.nan,
                             id_recall=tid / n if n else np.nan,
                             fpr=fp / den_hr))
        return pd.DataFrame(rows).set_index("category")

    def thr_at_fpr(self, target, runs, meta=None, den_hr=None):
        """Largest alarm set (lowest thr) whose Global fpr <= target.

        thr_desc is metric-DESCENDING (row 0 = highest thr, fewest alarms);
        fpr is non-decreasing in row; pick the highest row meeting the budget.
        row -1 = no alarms at all.  Returns (thr, row, table)."""
        if den_hr is None:
            den_hr = denom_hours(meta, runs)
        a = self._subset_arrays(runs)
        nt = len(self.thr_desc)
        fpr_g = a["fp"][-1, :nt] / den_hr
        ok = np.nonzero(fpr_g <= target + 1e-12)[0]
        row = int(ok[-1]) if len(ok) else -1
        t = self.metrics(row, runs, meta=meta, den_hr=den_hr)
        thr = float(self.thr_desc[row]) if row >= 0 else np.inf
        return thr, row, t

    def row_for_thr(self, x):
        """Curve row whose alarm set = {metric >= x} exactly."""
        if len(self.thr_desc) == 0:
            return -1
        ge = np.nonzero(self.thr_desc >= x)[0]
        return int(ge[-1]) if len(ge) else -1

    def d_fpr_arrays(self, runs, meta=None, den_hr=None):
        """(rows 0..nt-1) global d and fpr arrays for curve plotting/cuts."""
        if den_hr is None:
            den_hr = denom_hours(meta, runs)
        a = self._subset_arrays(runs)
        nt = len(self.thr_desc)
        n_g = max(int(a["n"][-1]), 1)
        return (a["det"][-1, :nt] / n_g, a["fp"][-1, :nt] / den_hr)


# ---------------------------------------------------------------------------
# bootstrap (run-cluster, with multiplicity)
# ---------------------------------------------------------------------------

def _pertable(enc_tab, fp_tab):
    return ({r: g for r, g in enc_tab.groupby("run")},
            {r: g for r, g in fp_tab.groupby("run")})


FIELD_COL = {"d_recall": "det", "c_recall": "cok", "id_recall": "iok"}


def boot_at_row(curve, row, runs, meta, field="d_recall", B=1000, seed=7):
    """CI percentile [2.5, 97.5] of Global field (d/c/id_recall or fpr)."""
    rng = np.random.default_rng(seed)
    runs = np.asarray(runs)
    enc_tab, fp_tab = curve.tables_at(row)
    et, ft = _pertable(enc_tab, fp_tab)
    col = FIELD_COL[field] if field in FIELD_COL else None
    vals = []
    for _ in range(B):
        samp = rng.choice(runs, len(runs), replace=True)
        cnt = pd.Series(samp).value_counts()
        den = denom_hours(meta, samp)
        if field == "fpr":
            fpn = sum(len(ft[r]) * c for r, c in cnt.items() if r in ft)
            vals.append(fpn / den)
        else:
            num = tot = 0
            for r, c in cnt.items():
                g = et.get(r)
                if g is not None and len(g):
                    tot += len(g) * c
                    num += int(g[col].sum()) * c
            vals.append(num / tot if tot else np.nan)
    return np.nanpercentile(vals, [2.5, 97.5])


def boot_diff_at_rows(c_ours, r_ours, c_cmp, r_cmp, runs, meta,
                      field="d_recall", B=1000, seed=11):
    """Paired bootstrap CI of (ours - comparator) Global metric at fixed rows."""
    rng = np.random.default_rng(seed)
    runs = np.asarray(runs)
    e1, f1 = _pertable(*c_ours.tables_at(r_ours))
    e2, f2 = _pertable(*c_cmp.tables_at(r_cmp))
    col = FIELD_COL[field] if field in FIELD_COL else None
    vals = []
    for _ in range(B):
        samp = rng.choice(runs, len(runs), replace=True)
        cnt = pd.Series(samp).value_counts()
        den = denom_hours(meta, samp)
        if field == "fpr":
            v = (sum(len(f1[r]) * c for r, c in cnt.items() if r in f1)
                 - sum(len(f2[r]) * c for r, c in cnt.items() if r in f2)) / den
        else:
            n1 = t1 = n2 = t2 = 0
            for r, c in cnt.items():
                for tab, kk, et in ((e1, 1, col), (e2, 2, col)):
                    g = tab.get(r)
                    if g is not None and len(g):
                        if kk == 1:
                            t1 += int(g[et].sum()) * c
                            n1 += len(g) * c
                        else:
                            t2 += int(g[et].sum()) * c
                            n2 += len(g) * c
            v = (t1 / n1 if n1 else 0.0) - (t2 / n2 if n2 else 0.0)
        vals.append(v)
    return np.nanpercentile(vals, [2.5, 97.5])


# ---------------------------------------------------------------------------
# frozen split metadata
# ---------------------------------------------------------------------------

def load_frozen():
    lock = json.load(open(CACHE / "lockbox.json"))
    folds = json.load(open(CACHE / "folds.json"))
    shift = json.load(open(CACHE / "shift_slice.json"))
    return lock, folds, shift


def load_run_cache(split="train", runs=None, keys=None):
    out = {}
    with h5py.File(CACHE_H5[split], "r") as f:
        ks = sorted(f.keys(), key=lambda k: int(k[1:])) if runs is None \
            else [f"r{r}" for r in runs]
        for k in ks:
            g = f[k]
            names = keys if keys is not None else list(g.keys())
            out[int(k[1:])] = {nm: g[nm][:] for nm in names}
    return out
