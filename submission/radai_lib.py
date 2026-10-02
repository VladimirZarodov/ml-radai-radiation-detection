# -*- coding: utf-8 -*-
"""
radai_lib.py -- generic RADAI matched-filter pipeline library (v1, frozen detector).

Extracted from Matched_Filter_Detection.ipynb (the notebook is the source of
truth).  This library parameterises the .h5 path and run-id list so the SAME
frozen detector can score both training_v4.3.h5 and testing_v4.3.h5.

Hard rules honoured here:
  * The detector (7 background components M, 61 templates U) is fit ONLY on
    training-file runs TRAIN_RUNS = [0, 3..19] (exactly as the notebook).
  * Inference on a run reads ONLY listmode/dt and listmode/energy (blind-safe;
    never sources/*, listmode/id, listmode/background_id).
  * Ground-truth fields are read ONLY from the training file, for fitting M/U
    (TRAIN_RUNS) and for calibration masks / evaluation.
  * Numeric formulas and dtype casts are copied VERBATIM from the notebook
    (including the cumsum-difference windowing) so cached scores reproduce
    bit-for-bit.  See regression_check.py.

New file, Oct 2026. Python 3.11 (.venv), numpy 1.24.2 / pandas 1.5.3 /
h5py 3.13.0 per requirements.txt.
"""
from __future__ import annotations

import os
import re
import time
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import h5py

# ---------------------------------------------------------------------------
# Frozen detector constants (notebook cells 17-19 verbatim)
# ---------------------------------------------------------------------------
EMIN, EMAX, NB, STRIDE = 15.0, 3000.0, 128, 2.0
SMIN, SMAX = np.sqrt(EMIN), np.sqrt(EMAX)
DET_T = 2.0                    # window length for the detector, s (notebook T)
MX_N = 31                      # trailing rolling-max length, windows (62 s)
R_EXCL = 150.0                 # background-mask radius dmin > 150 s
TRAIN_RUNS = [0] + list(range(3, 20))          # 18 runs: fit M and U
VAL_RUNS = [20, 21, 22, 23, 24]                # notebook "val" (unused by mx31 fit/calib)
TEST_RUNS = list(range(25, 125))               # notebook internal hold-out = "validation"
ALL_RUNS = TRAIN_RUNS + VAL_RUNS + TEST_RUNS   # 123 runs scored in nb_mf_cache.pkl

# ---------------------------------------------------------------------------
# Official RADAI label vocabulary + categories (organiser-provided, from the
# RADAI source-list page used by the online scorer; supersedes earlier maps).
# ---------------------------------------------------------------------------
OFFICIAL_LABELS = [
    "K-40", "Co-57", "Co-60", "Cs-137", "Ba-133", "Ir-192", "Ra-226", "Th-232",
    "Am-241", "F-18", "Tc-99m", "I-131", "Tl-201", "NatU", "RefinedU", "LEU",
    "HEU", "FGPu", "WGPu", "Cu-67", "Sr-90", "DU", "Lu-177", "Xe-133",
]
OFFICIAL_CATEGORIES = {
    "NORM": ["K-40", "Ra-226", "Th-232"],
    "Industrial": ["Co-60", "Cs-137", "Ba-133", "Ir-192"],
    "Medical": ["Co-57", "F-18", "Tc-99m", "I-131", "Tl-201", "Cu-67", "Sr-90",
                "Lu-177", "Xe-133"],
    "Nuclear Material": ["Am-241", "NatU", "RefinedU", "LEU", "HEU", "FGPu",
                         "WGPu", "DU"],
}
LABEL2CAT = {lab: cat for cat, labs in OFFICIAL_CATEGORIES.items() for lab in labs}


def strip_shielding(full_name: str) -> str:
    """'Cs-137_shielding_id=2' -> 'Cs-137' (notebook names[] transform)."""
    return full_name.split("_shielding")[0]


def source_names(h5path):
    """Full source_names attr (identical in both files); returned stripped."""
    with h5py.File(h5path, "r") as f:
        return [strip_shielding(str(n)) for n in f.attrs["source_names"]]


def base_of(name: str) -> str:
    """Strip the mass suffix '-<n>kg' (notebook baseof())."""
    m = re.match(r"^(.*)-(\d+(?:\.\d+)?kg)$", name)
    return m.group(1) if m else name


def official_label(name: str):
    """Map an internal (shielding/mass-free) template name to the official
    24-label vocabulary.

    Rules (organiser-stated): shielding id and mass are not part of the label;
    both Ir-192 variants map to 'Ir-192'.  Anything that does not land in the
    official vocabulary returns None -- never guessed.
    """
    b = base_of(strip_shielding(name))
    if b == "Ir-192_industrial":
        b = "Ir-192"
    return b if b in OFFICIAL_LABELS else None


# ---------------------------------------------------------------------------
# HDF5 reading (streaming per run; blind-safe core)
# ---------------------------------------------------------------------------

def read_listmode_blind(h5path, rid):
    """Load ONLY listmode/dt + listmode/energy for one run (works on testing).

    Returns (t, e, inr, bi) exactly as notebook ev_arrays() (minus GT fields).
    """
    with h5py.File(h5path, "r") as f:
        g = f[f"runs/run{rid}"]
        dt = g["listmode/dt"][:]
        e = g["listmode/energy"][:]
    t = np.cumsum(dt, dtype=np.uint64) / 1e6
    inr = (e >= EMIN) & (e < EMAX)
    bi = np.zeros(len(e), np.int64)
    bi[inr] = np.clip(np.floor((np.sqrt(e[inr]) - SMIN) / (SMAX - SMIN) * NB)
                      .astype(np.int64), 0, NB - 1)
    return t, e, inr, bi


def read_listmode_gt(h5path_train, rid):
    """Load listmode incl. ground-truth id/background_id. TRAINING FILE ONLY."""
    t, e, inr, bi = read_listmode_blind(h5path_train, rid)
    with h5py.File(h5path_train, "r") as f:
        g = f[f"runs/run{rid}/listmode"]
        eid = g["id"][:]
        bid = g["background_id"][:]
    return t, e, inr, bi, eid, bid


def windows_of_run(t, inr, bi, T=DET_T):
    """2 s windows via the notebook run_windows() cumsum-difference, verbatim.

    Returns spec (ns x 128, float64), wtime (float64, window centres, s).
    Reproducing this bit-exactly matters: cum[i+k]-cum[i] is NOT guaranteed
    equal to a directly-binned row in floating point.
    """
    dur = t[-1]
    nfull = int(np.floor(dur / STRIDE))
    wi = np.floor(t / STRIDE).astype(np.int64)
    ok = (wi < nfull) & inr
    spec_s = np.bincount(wi[ok] * NB + bi[ok],
                         minlength=nfull * NB).reshape(nfull, NB).astype(np.float64)
    k = int(round(T / STRIDE))
    ns = max(nfull - k + 1, 1)
    idx = np.arange(ns)
    cum = np.vstack([np.zeros((1, NB)), np.cumsum(spec_s, 0)])
    spec = cum[idx + k] - cum[idx]
    wtime = idx * STRIDE + T / 2.0
    return spec, wtime


# ---------------------------------------------------------------------------
# Model fitting on TRAINING runs (frozen objects: 7 components, 61 templates)
# ---------------------------------------------------------------------------

def fit_detector(h5path_train, train_runs=TRAIN_RUNS, cache_path=None):
    """Fit M and U from training GT, mirroring notebook cell 17 verbatim
    (same iteration order / arithmetic).  Returns dict(M, U, U2, ids)."""
    if cache_path and os.path.exists(cache_path):
        return pickle.load(open(cache_path, "rb"))
    comp_hist = np.zeros((8, NB))
    src_hist = {}
    for rid in train_runs:
        t, e, inr, bi, eid, bid = read_listmode_gt(h5path_train, rid)
        for k in range(1, 8):
            comp_hist[k] += np.bincount(bi[inr & (bid == k) & (eid == 0)],
                                        minlength=NB)
        sm = inr & (eid != 0)
        for sidx in np.unique(eid[sm]):
            sidx = int(sidx)
            src_hist[sidx] = src_hist.get(sidx, np.zeros(NB)) + \
                np.bincount(bi[sm & (eid == sidx)], minlength=NB)
    keep = comp_hist.sum(1) > 100
    M = comp_hist[keep] / comp_hist[keep].sum(1, keepdims=True)
    ids = np.array(sorted(src_hist))
    U = np.array([src_hist[i] for i in ids])
    U = U / U.sum(1, keepdims=True)
    det = dict(M=M, U=U, U2=U ** 2, ids=ids)
    if cache_path:
        pickle.dump(det, open(cache_path, "wb"))
    return det


# ---------------------------------------------------------------------------
# Scoring (formulas + casts identical to notebook det_scores + rollmax)
# ---------------------------------------------------------------------------

def poisson_fit(X, M, iters=60):
    """Poisson EM fit S = A @ M (notebook poisson_fit verbatim, M explicit)."""
    A = np.full((X.shape[0], M.shape[0]), X.sum(1, keepdims=True) / M.shape[0])
    norm = M.sum(1)[None, :]
    for _ in range(iters):
        B = A @ M + 1e-9
        A *= ((X / B) @ M.T) / norm
    return A @ M + 1e-9


def rollmax(v, n=MX_N):
    """Trailing rolling max as float32 (notebook rollmax verbatim)."""
    return pd.Series(v).rolling(n, min_periods=1).max().to_numpy(np.float32)


def score_run(h5path, rid, det, with_gt=False):
    """Per-run inference -> dict(bestA, mx31, wtime, am, bg_rate[, dmin, stime,
    snr, sid]).  Reads listmode/dt+energy only, plus sources/* if with_gt
    (training file only)."""
    t, e, inr, bi = read_listmode_blind(h5path, rid)
    spec, wtime = windows_of_run(t, inr, bi)
    M, U, U2 = det["M"], det["U"], det["U2"]
    S = np.maximum(poisson_fit(spec, M), 1.0)
    zA = ((spec - S) @ U.T) / np.sqrt(S @ U2.T + 1e-9)
    bestA = zA.max(1)
    am = np.argmax(zA, 1)
    del zA
    out = dict(
        bestA=bestA.astype(np.float32),
        mx31=rollmax(bestA.astype(np.float32), MX_N),
        wtime=wtime.astype(np.float32),
        am=am.astype(np.uint8),
        bg_rate=spec.sum(1).astype(np.float32),
    )
    if with_gt:
        with h5py.File(h5path, "r") as f:
            g = f[f"runs/run{rid}"]
            stime64 = g["sources/time"][:] / 1e3
        if len(stime64):
            dmin = np.min(np.abs(wtime[:, None] - stime64[None, :]), 1)
        else:                                # pure-background run (defensive)
            dmin = np.full(len(wtime), np.inf)
        out["dmin"] = dmin.astype(np.float32)
        out["stime"] = stime64.astype(np.float32)
        with h5py.File(h5path, "r") as f:
            out["snr"] = f[f"runs/run{rid}/sources/snr/peak"][:].astype(np.float32)
            out["sid"] = f[f"runs/run{rid}/sources/id"][:]
    return out


def score_runs(h5path, run_ids, det, cache_path=None, with_gt=False,
               progress_every=25, recompute=False):
    """Score many runs with progress log; pickle cache at cache_path.

    Entries stored as: score arrays plus 'dmin'/'stime'/... when with_gt.
    Set recompute=True to ignore the cache file (still overwrites it)."""
    cache = {}
    if cache_path and os.path.exists(cache_path) and not recompute:
        cache = pickle.load(open(cache_path, "rb"))
        print(f"cache loaded: {cache_path} ({len(cache)} runs)", flush=True)
    todo = [r for r in run_ids if r not in cache]
    if not todo:
        print(f"cache hit for all {len(run_ids)} runs", flush=True)
        return cache
    t0 = time.time()
    for i, rid in enumerate(todo):
        cache[rid] = score_run(h5path, rid, det, with_gt=with_gt)
        if (i + 1) % progress_every == 0:
            print(f"  scored {i + 1}/{len(todo)} ({time.time() - t0:.0f}s)",
                  flush=True)
    print(f"scoring done: {len(todo)} new runs in {time.time() - t0:.0f}s",
          flush=True)
    if cache_path:
        pickle.dump(cache, open(cache_path, "wb"))
    return cache


# ---------------------------------------------------------------------------
# Threshold calibration (notebook cells 19-20 protocol, verbatim formulas)
# ---------------------------------------------------------------------------

def farm_of(cache, run_ids, r_excl=R_EXCL):
    """Background masks dmin > r_excl (TRAINING runs only -- needs dmin)."""
    return {r: cache[r]["dmin"] > r_excl for r in run_ids}


def hrs_of(farm):
    return {r: int(m.sum()) * STRIDE / 3600 for r, m in farm.items()}


def build_cand(pool_values):
    """CAND grid from log-quantiles of the background pool (verbatim)."""
    dq = np.logspace(np.log10(2e-4), np.log10(30), 320)
    return np.array(sorted(set(np.quantile(pool_values, 1 - dq / 100)),
                           reverse=True))


def onset_at(s, thr, mask):
    """Episode onsets above thr within mask windows (notebook onset_at body)."""
    a = s >= thr
    return int(np.count_nonzero((a & ~np.r_[False, a[:-1]])[mask]))


def onset_matrix(cache, run_ids, farm, cand, stat="mx31"):
    """Omat() verbatim: episodes-per-run x threshold grid."""
    O = np.zeros((len(run_ids), len(cand)))
    for i, r in enumerate(run_ids):
        s = cache[r][stat]
        for j, t in enumerate(cand):
            O[i, j] = onset_at(s, t, farm[r])
    return O


def branch_end(fr, target):
    """Monotone-branch rule (notebook cell 19 verbatim)."""
    runmax, br = fr[0], 0
    for k in range(1, len(fr)):
        if fr[k] > runmax:
            runmax, br = fr[k], k
        elif fr[k] < runmax - max(0.5, 0.2 * runmax) and runmax > target:
            break
    return br


def calib_index(om_rows, hv_rows, target):
    """calib(): pick grid index with FAR <= target, nearest to target on the
    monotone branch. om_rows: (n_runs, n_thr); hv_rows: (n_runs,) hours."""
    fr = om_rows.sum(0) / hv_rows.sum()
    br = branch_end(fr, target)
    d = np.where(fr[:br + 1] <= target, target - fr[:br + 1], np.inf)
    return int(np.argmin(d)) if np.isfinite(d).any() else 0


def emax_mx(cache, rid, tol=60.0):
    """Max of mx31 within CA +- tol seconds, per encounter (cell 20 verbatim)."""
    c = cache[rid]
    return np.array([c["mx31"][np.abs(c["wtime"] - t) < tol].max()
                     for t in c["stime"] if (np.abs(c["wtime"] - t) < tol).any()])


# ---------------------------------------------------------------------------
# Alarm extraction (threshold-sweep-safe: raising the metric_1 cut yields a
# SUBSET of emitted alarms -- local maxima, NOT threshold episodes)
# ---------------------------------------------------------------------------

def mx31_peaks(mx31, sep=15):
    """Local maxima of mx31 with minimum separation `sep` windows (2 s each).

    Computed WITHOUT any threshold, so emitting peaks above a floor is closed
    under raising the metric threshold (subset property required by the
    portal's sweep).  A rising plateau contributes its first index (matches
    the notebook's episode-onset convention).  When two peaks lie closer than
    `sep`, the higher-value peak is kept (ties: earlier index).
    Returns peak indices sorted ascending.
    """
    s = mx31.astype(np.float64)
    n = len(s)
    cand_i, cand_v = [], []
    i = 0
    while i < n:
        j = i
        while j + 1 < n and s[j + 1] == s[i]:
            j += 1                      # plateau [i..j]
        left_ok = (i == 0) or (s[i - 1] < s[i])
        right_ok = (j == n - 1) or (s[j + 1] < s[i])
        if left_ok and right_ok:
            cand_i.append(i)
            cand_v.append(s[i])
        i = j + 1
    idx = np.array(cand_i, dtype=np.int64)
    val = np.array(cand_v, dtype=np.float64)
    if sep > 1 and len(idx) > 1:
        order = np.lexsort((idx, -val))         # value desc, then index asc
        keep = np.zeros(len(idx), bool)
        taken = []
        for o in order:
            if all(abs(int(idx[o]) - t) >= sep for t in taken):
                keep[o] = True
                taken.append(int(idx[o]))
        idx = idx[keep]
    return np.sort(idx)


def plateau_end(mx31, i):
    """Last index of the run of equal values starting at i (mx31 plateau)."""
    s = mx31.astype(np.float64)
    n = len(s)
    j = i
    while j + 1 < n and s[j + 1] == s[i]:
        j += 1
    return j


def peak_farm_count(cache, rid, farm, peaks):
    """Split peaks into background (mask True) vs encounter-region peaks."""
    m = farm[rid]
    p_bg = np.zeros(len(peaks), bool)
    for i, p in enumerate(peaks):
        lo = max(0, p - MX_N // 2)
        hi = min(len(m), p + MX_N // 2 + 1)
        p_bg[i] = bool(m[lo:hi].all())
    return p_bg
