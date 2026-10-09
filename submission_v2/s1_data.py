# -*- coding: utf-8 -*-
"""
s1_data.py -- Stage-1 dataset layer: label-free features + per-second targets.

Features per 1-s row (F = 132), ALL computable blind on testing (only totals
and rates, no decomposition, no answer key):
  ch 0..127  sqrt((total+0.5) / (per-bin run median+0.5))   (Poisson-style
             variance-stabilising ratio vs the run's own robust spectral
             baseline; label-free, gain-partially-invariant)
  ch 128     robust z of log1p(rate_all)   (z=(x-med)/(IQR/1.349), clip +-6)
  ch 129     robust z of log1p(rate_in)    (in-band 15-3000 keV cps)
  ch 130     rise: log1p(rate_all) - 31 s centered rolling median, clip +-6
  ch 131     rise: log1p(rate_in)  - 31 s centered rolling median, clip +-6
Edge context is zero-padded (spec->0 = no counts; z->0 = median level).

Targets (official AK geometry, win from cache/encounters_train.csv):
  hard[b] = 1 iff start_ms <= (b+0.5)*1e3 <= stop_ms   (TP rule verbatim)
  soft[b] = max(hard, per-window 0.9*exp(-((b+0.5-win_max_s)^2)/(2*SIG^2))
          restricted to [win_start_s-12, win_stop_s+12]), SIG=6 s  (CA-peaked)
  tcat/tiso = category / official-label index on hard-positive seconds,
          else MASKED (255); heads train on hard positives only.

Leak guard: assert_no_lockbox() runs at every train/eval split build and logs.
Feature cache: cache/feat.h5 (junction -> C:) built once, resumable.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "submission"))
import v2_lib as V                                    # noqa: E402

F_SPEC = 128
F_TOT = 132
SOFT_SIG = 6.0
SOFT_AMP = 0.9
SOFT_PAD = 12.0          # s beyond window ends where the ramp may reach
MASK = 255
CAT2IDX = {c: i for i, c in enumerate(V.TRUE_CATS)}

# ---------------------------------------------------------------------------
# vocab (fixed from TRAIN GT only -- allowed for targets/augmentation)
# ---------------------------------------------------------------------------

def iso_vocab():
    """Frozen official 24-label vocabulary (radai_lib.OFFICIAL_LABELS); train
    encounters use 22 of them, the head spans all 24."""
    import radai_lib as L
    return sorted(L.OFFICIAL_LABELS)


# ---------------------------------------------------------------------------
# feature transform (shared with synth / future test inference)
# ---------------------------------------------------------------------------

def _robust_z(lr):
    med = np.median(lr)
    iqr = np.percentile(lr, 75) - np.percentile(lr, 25)
    return np.clip((lr - med) / (iqr / 1.349 + 1e-6), -6.0, 6.0)


def _rolling_median(x, w=31):
    p = np.pad(x, w // 2, mode="edge")
    from numpy.lib.stride_tricks import sliding_window_view
    return np.median(sliding_window_view(p, w), axis=1)


def features(total, rate_all, rate_in):
    """(nb,132) float16 from cached uint16 total + rate arrays."""
    t = total.astype(np.float64)
    base = np.median(t, axis=0) + 0.5
    spec = np.sqrt((t + 0.5) / base)
    za, zi = _robust_z(np.log1p(rate_all.astype(np.float64))), \
        _robust_z(np.log1p(rate_in.astype(np.float64)))
    ra = np.clip(np.log1p(rate_all) - _rolling_median(np.log1p(rate_all)),
                 -6, 6)
    ri = np.clip(np.log1p(rate_in) - _rolling_median(np.log1p(rate_in)),
                 -6, 6)
    X = np.concatenate([spec, za[:, None], zi[:, None], ra[:, None],
                        ri[:, None]], axis=1)
    return X.astype(np.float16)


# ---------------------------------------------------------------------------
# targets
# ---------------------------------------------------------------------------

def targets(nb, rows, iso2idx):
    """rows: encounter rows (DataFrame slice) for one run."""
    y = np.zeros(nb, np.uint8)
    ys = np.zeros(nb, np.float32)
    tcat = np.full(nb, MASK, np.int8)
    tiso = np.full(nb, MASK, np.int8)
    for _, r in rows.iterrows():
        # TP rule: start_ms <= (b+0.5)*1e3 <= stop_ms
        lo = int(np.ceil(r.start_ms / 1e3 - 0.5 - 1e-9))
        hi = int(np.floor(r.stop_ms / 1e3 - 0.5 + 1e-9))
        a, b = max(lo, 0), min(hi, nb - 1)
        if b >= a:
            y[a:b + 1] = 1
            tcat[a:b + 1] = CAT2IDX[r.category]
            tiso[a:b + 1] = iso2idx[r.official_label]
            cs, ce = r.win_start_s - SOFT_PAD, r.win_stop_s + SOFT_PAD
            ii = np.arange(max(int(np.floor(cs)), 0),
                           min(int(np.ceil(ce)) + 1, nb))
            g = SOFT_AMP * np.exp(
                -((ii + 0.5 - r.win_max_s) ** 2) / (2 * SOFT_SIG ** 2))
            keep = (ii + 0.5 >= cs) & (ii + 0.5 <= ce)
            ys[ii[keep]] = np.maximum(ys[ii[keep]], g[keep].astype(np.float32))
    ys = np.maximum(ys, y.astype(np.float32))
    return y, ys, tcat, tiso


# ---------------------------------------------------------------------------
# build / load cache
# ---------------------------------------------------------------------------

FEAT_H5 = V.CACHE / "feat.h5"


def build(split="train", log=print):
    enc = V.load_encounters(split)
    iso2idx = {s: i for i, s in enumerate(iso_vocab())}
    n = len(iso2idx)
    log(f"iso vocab: {n} labels")
    assert n == 24, f"expected 24 official labels, got {n}"
    by_run = dict(tuple(enc.groupby("run")))
    with h5py.File(FEAT_H5, "a") as f:
        done = set(f.keys())
        with h5py.File(V.CACHE_H5[split], "r") as src:
            ks = sorted(src.keys(), key=lambda k: int(k[1:]))
            for i, k in enumerate(ks):
                if k in done:
                    continue
                g = src[k]
                rid = int(k[1:])
                total = g["total"][:]
                X = features(total, g["rate_all"][:], g["rate_in"][:])
                rows = by_run.get(rid, enc.iloc[0:0])
                y, ys, tcat, tiso = targets(total.shape[0], rows, iso2idx)
                out = f.require_group(k)
                out.create_dataset("X", data=X, compression="gzip",
                                   compression_opts=4)
                out.create_dataset("y", data=y)
                out.create_dataset("ys", data=ys, compression="gzip",
                                   compression_opts=4)
                out.create_dataset("tcat", data=tcat)
                out.create_dataset("tiso", data=tiso)
                if (i + 1) % 50 == 0:
                    log(f"  feat {i + 1}/{len(ks)}")
                    f.flush()
        log(f"built {len(set(f.keys()))} runs")


def load_runs(runs, device=None):
    """dict run -> {X f16, y u8, ys f16, tcat i8, tiso i8} (+ torch on device)."""
    out = {}
    with h5py.File(FEAT_H5, "r") as f:
        for r in runs:
            g = f[f"r{r}"]
            d = {nm: g[nm][:] for nm in ("X", "y", "ys", "tcat", "tiso")}
            out[r] = d
    if device is not None:
        import torch
        for r in out:
            d = out[r]
            d["X"] = torch.from_numpy(d["X"]).to(device)
            d["y"] = torch.from_numpy(d["y"]).to(device)
            d["ys"] = torch.from_numpy(d["ys"]).to(device)
            d["tcat"] = torch.from_numpy(d["tcat"]).to(device)
            d["tiso"] = torch.from_numpy(d["tiso"]).to(device)
    return out


# ---------------------------------------------------------------------------
# splits + leak guard
# ---------------------------------------------------------------------------

def dev_pool():
    lock, folds, shift = V.load_frozen()
    meta = V.run_meta("train")
    lock_runs = set(lock["runs"])
    runs240 = sorted(r for r in meta if r not in lock_runs)
    allf = [set(x) for x in folds["folds"]]
    assert set().union(*allf) == set(runs240), "folds must cover dev pool"
    assert not (set().union(*allf) & lock_runs), "fold overlap w/ lockbox"
    return lock_runs, runs240, folds["folds"], shift["high_bg_runs"]


def assert_no_lockbox(runs, purpose, logf):
    lock = set(json.load(open(V.CACHE / "lockbox.json"))["runs"])
    bad = set(runs) & lock
    msg = ("LEAKGUARD %s: n=%d lockbox-overlap=%d %s"
           % (purpose, len(runs), len(bad), sorted(bad) if bad else "OK"))
    Path(logf).parent.mkdir(parents=True, exist_ok=True)
    with open(logf, "a") as fh:
        fh.write(msg + "\n")
    if bad:
        raise RuntimeError("LOCKBOX LEAK: " + msg)
    return msg


def inner_split(train_runs):
    """frozen rule: sort by run id, last 15% (>=1) is inner validation."""
    s = sorted(train_runs)
    nv = max(1, int(round(0.15 * len(s))))
    return s[:len(s) - nv], s[len(s) - nv:]


if __name__ == "__main__":
    import time
    t0 = time.time()
    build()
    print(f"build done in {time.time() - t0:.0f}s")
    print(free := Path.stat(FEAT_H5).st_size / 2**20, "MB feat.h5")
