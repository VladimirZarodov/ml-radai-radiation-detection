# -*- coding: utf-8 -*-
"""
shift_synth.py -- FROZEN synthetic distribution-shift transform (harness).

Mimics the known training->testing shift at histogram level on cached 1-s
arrays (DESIGN.md §3 shift fold b).  Parameters are FIXED here at Stage 0
(pre-registration).  Stage 2 may add a *training* augmentation pipeline with
its own randomized parameters, but THIS shift-eval transform never changes.

Per run (rng = default_rng([SEED, split_id, run])):
  * global rate factor f ~ U(0.90, 1.30)            (portal: testing ~ +17 %)
  * background: NORM-like components (background_id 1,2,3 = K-40, U, Th)
    scaled f*1.20 (portal NORM +20 %); others f; Poisson re-draw keeps
    Poisson noise character: new_k ~ Poisson(bg_k * scale_k)
  * sources per encounter: time-compression c ~ U(1.00, 1.68) about CA
    (8.0 -> 13.4 m/s; per-run draw) and thinning t ~ U(0.45, 0.95)
    (portal d-ratio ~0.65), Poisson integerised.
Returns new 'total' (bg_all + hoods + src_rest) plus components; caller uses
total for v1 scoring and per-run spectra for learned models.
"""
from __future__ import annotations

import numpy as np

SEED = 20261003
W = 150                    # hood half-width, s (== cache_build)
NORM_LIKE = (1, 2, 3)      # background_id values K-40, U, Th chains
BG_RATE = (0.90, 1.30)
NORM_BOOST = 1.20
SPEED = (1.00, 1.68)
THIN = (0.45, 0.95)


def rng_for(run, split_id=0):
    return np.random.default_rng([SEED, int(split_id), int(run)])


def compress_hood(hood, c):
    """new_p(u) = p(u*c) by linear interpolation about CA row W (u in -W..W)."""
    nt = hood.shape[0]
    u = np.arange(nt, dtype=np.float64) - W
    src = u * c                                  # old offsets to sample
    ok = np.abs(src) <= W
    fl = np.floor(src).astype(np.int64)
    frac = src - fl
    lo = np.clip(fl, -W, W - 1) + W
    hi = np.clip(lo + 1, 0, nt - 1)
    prof = hood[lo] * (1.0 - frac[:, None]) + hood[hi] * frac[:, None]
    return np.where(ok[:, None], prof, 0.0)


def synth_shift_run(arrays, cens, rng=None, run=0, split_id=0):
    """arrays: dict with bg_comp (7,nb,128), src_rest (nb,128), enc_q
    (2W+1,128) for q in range(len(cens)); cens: CA bin per encounter.
    Returns new dict (int64 arrays + total/rate_in)."""
    if rng is None:
        rng = rng_for(run, split_id)
    nb, NB = arrays["src_rest"].shape
    f = rng.uniform(*BG_RATE)
    comps = []
    for k in range(1, 8):
        s = f * (NORM_BOOST if k in NORM_LIKE else 1.0)
        comps.append(rng.poisson(
            arrays["bg_comp"][k - 1].astype(np.float64) * s).astype(np.int64))
    bg_all = sum(comps)
    c = rng.uniform(*SPEED)
    tot = bg_all + arrays["src_rest"].astype(np.int64)
    out = dict(bg_comp=np.stack(comps), bg_all=bg_all,
               src_rest=arrays["src_rest"].astype(np.int64))
    for q, cen in enumerate(cens):
        key = f"enc_{q}"
        if key not in arrays:
            continue
        t = rng.uniform(*THIN)
        prof = compress_hood(arrays[key].astype(np.float64), c)
        new = rng.poisson(prof * t).astype(np.int64)
        out[key] = new
        lo, hi = int(cen) - W, int(cen) + W + 1
        r_lo, r_hi = max(lo, 0), min(hi, nb)
        tot[r_lo:r_hi] += new[r_lo - lo: r_hi - lo]
    out["total"] = tot
    out["rate_in"] = tot.sum(1).astype(np.float32)
    out["_f_bg"], out["_c_speed"] = float(f), float(c)
    return out
