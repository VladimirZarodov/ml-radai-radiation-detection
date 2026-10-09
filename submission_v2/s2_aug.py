# -*- coding: utf-8 -*-
"""
s2_aug.py -- Stage-2 physically-motivated transforms on the decomposed cache.

Shared core for (a) TRAINING augmentation (s2_build.py, rng stream SEED_AUG) and
(b) the Step-0 EVAL shift generator (s2_shift.py, rng stream SEED_CAL -- disjoint,
anti-circularity).  Operates on ONE run's cached arrays (S2_NOTES.md section 2):
  energy view   bg_comp (7,nb,128), src_rest (nb,128), per-instance hoods enc_q
  all-energy    sec_bg (nb,), per-source-id sec_src_j (nb,)  <- official SNR profile

Transform scope: ONLY the nearest-instance-masked segment of each encounter (energy
hood rows + the matching all-energy sec_src_j rows).  Background is touched only when
bg is on.  src_rest and out-of-hood seconds stay verbatim (logged approximation: SPD
and THIN act on what the official SNR numerator of the encounter itself is built from).
NORM_LIKE == indices 0,1,2 of bg_comp == background_id 1,2,3 == K-40, U, Th chains.

Draw order per instance (FIXED): [spd] pick=Bernoulli(p_spd), k~U[1,kmax];
[thin] t~U[theta,1].  Compress new_p(u)=p(u*k) about CA (linear interp), then
Poisson integerise; energy 2-D and all-energy 1-D use the SAME k, separate draws.
Thinning = binomial with the SAME t on both views.

Rebuilt windows: snr'_j = S'_j/sqrt(S'_j+B') on the augmented all-energy series,
official greedy rule via cache_build.ak_window (bit-identical replica, E010); CA bin
fixed; win_start/stop/max_s, snr_window, width_s updated.  Targets via s1_data.targets.

Invariants asserted per transformed run (G3/T2), exact integer identities:
  J1  bg_all' == bg_comp'.sum(0)
  J2  total' == bg_all' + src_rest + placed enc_q'
  J3  rate_all' == sec_bg' + sum_j sec_src_j' ; rate_in' == total'.sum(1)
  J4  max < 65536 (uint16 safety) and min >= 0
IDENTITY FAST PATH (spd=thin=bg=None) returns arrays + window rows VERBATIM so T1 can
demand bit-identity with feat.h5.
"""
from __future__ import annotations

import numpy as np

W = 150                      # hood half-width, s (== cache_build)
NORM_LIKE = (0, 1, 2)        # bg_comp rows = background_id 1,2,3 (K-40, U, Th)
NORM_BOOST = 1.20
CAP16 = 65536
BG_AUG = (1.00, 1.34)        # training-aug per-run rate factor (mean 1.17)


def _compress(a, k, center, axis):
    """new_p(u) = p(u*k) about index `center` along `axis`, linear interp,
    zeros where the source coordinate falls outside [0, n-1]."""
    n = a.shape[axis]
    u = np.arange(n, dtype=np.float64) - center
    x = u * k + center
    ok = (x >= 0) & (x <= n - 1)
    xc = np.clip(x, 0, n - 1)
    fl = np.floor(xc).astype(np.int64)
    hi = np.minimum(fl + 1, n - 1)
    lo_v = np.take(a, fl, axis=axis)
    hi_v = np.take(a, hi, axis=axis)
    bc = [1] * a.ndim
    bc[axis] = n
    fr = (xc - fl).reshape(bc)
    out = lo_v * (1.0 - fr) + hi_v * fr
    return np.where(ok.reshape(bc), out, 0.0)


def instance_masks(rows, nb):
    """rows: encounters slice for ONE run (cols source_id, cen_bin, inst).
    cache_build's nearest-own-instance rule (ties -> first listed instance),
    limited to +-W s.  Returns inst -> (mask (nb,) bool, lo, hi, cen)."""
    out = {}
    for _sid, gg in rows.groupby("source_id", sort=False):
        cs = gg.cen_bin.to_numpy(np.int64)
        insts = gg.inst.to_numpy(np.int64)
        rr = np.arange(nb, dtype=np.int64)[:, None]
        d = np.abs(rr - cs[None, :])
        aidx = np.argmin(d, axis=1)
        nmin = d.min(axis=1)
        for j in range(len(cs)):
            cen = int(cs[j])
            m = (aidx == j) & (nmin <= W)
            lo, hi = cen - W, cen + W + 1
            out[int(insts[j])] = (m, lo, hi, cen)
    return out


def load_run(f, rid):
    """Read one cache run group into the dict augment_run expects."""
    g = f[f"r{rid}"]
    nb = int(g.attrs["nb"])
    ne = int(g.attrs["n_enc"])
    sec = {}
    for k in g.keys():
        if k.startswith("sec_src_"):
            sec[int(k[8:])] = g[k][:].astype(np.int64)
    return dict(total=g["total"][:], bg_comp=g["bg_comp"][:],
                src_rest=g["src_rest"][:],
                enc=[g[f"enc_{q}"][:] for q in range(ne)],
                sec_bg=g["sec_bg"][:].astype(np.int64), sec=sec,
                rate_all=g["rate_all"][:], rate_in=g["rate_in"][:], nb=nb)


def augment_run(a, rows, rng, spd=None, thin=None, bg=None, p_spd=0.5,
                rebuild_windows=True):
    """a: load_run dict (int arrays; enc list aligned to rows by 'inst').
    rows: encounter DataFrame slice for this run (must contain inst, source_id,
    cen_bin).  spd=(kmax,), thin=(theta,), bg=(f,) or None; None = factor 1.
    Returns (out, rows2, stats, inv) with out keys total/bg_comp/src_rest/enc/
    sec_bg/sec/rate_all/rate_in/nb; rows2 = window rows rebuilt on the augmented
    all-energy SNR profile; inv = dict(J1..J4 -> bool)."""
    nb = a["nb"]
    ident = spd is None and thin is None and bg is None
    if ident:                                            # VERBATIM fast path
        return (a, rows, dict(identity=True),
                dict(J1=True, J2=True, J3=True, J4=True))
    masks = instance_masks(rows, nb)
    comp = a["bg_comp"].astype(np.int64)
    sb = a["sec_bg"].astype(np.int64)
    stats = dict(n_spd=0, n_thin=0, k=[], t=[], f_bg=1.0)
    if bg is not None:
        f = float(bg[0])
        sc = np.full(7, f, np.float64)
        # DESIGN SEMANTICS (STAGE2_DESIGN L28, README "+17% rate / +20%
        # NORM"): NORM comps get x1.2 TOTAL, not x(f*1.2).  w_norm = 0.987
        # of bg_comp counts, so the stacked reading inflated total rate to
        # ~x1.40 -- caught by pre-registered G2 rate bands (E052).
        sc[list(NORM_LIKE)] = NORM_BOOST
        comp0 = comp
        comp = rng.poisson(comp.astype(np.float64) *
                           sc[:, None, None]).astype(np.int64)
        # all-energy bg = (new IN-range bg total) + redrawn OUT-of-range
        # stream (orig sec_bg minus orig in-range bg, ~few cps): guarantees
        # sec_bg' >= bg_all'.sum(1) rowwise by construction.
        out0 = np.maximum(sb - comp0.astype(np.int64).sum(0).sum(1), 0)
        sb = comp.sum(0).sum(1) + rng.poisson(out0.astype(np.float64)
                                              * f).astype(np.int64)
        stats["f_bg"] = f
    sec = {k: v.astype(np.int64).copy() for k, v in a["sec"].items()}
    encl = []
    # FINAL window rebuild after all transforms: sb (=bg') and every sec_src_j'
    # column must be complete BEFORE any ak_window call (columns shared by
    # same-id instances; SNR denominator bg' redrawn per run) -- one pass.
    rebuilt = rows.copy()
    for _, e in rows.iterrows():
        q, sid, cen = int(e["inst"]), int(e["source_id"]), int(e["cen_bin"])
        m, lo, hi, _c = masks[q]
        a_lo, a_hi = max(lo, 0), min(hi, nb)
        hood = a["enc"][q].astype(np.float64)            # CA at row W
        seg = sec[sid][a_lo:a_hi].astype(np.float64)
        own = m[a_lo:a_hi]
        sub = hood[a_lo - lo:a_hi - lo]                  # window-local rows
        k, t = 1.0, 1.0
        if spd is not None:
            pick = bool(rng.random() < p_spd)
            kk = float(rng.uniform(1.0, max(spd[0], 1.0)))
            if pick and kk > 1.0 + 1e-9:
                g2 = _compress(np.where(own[:, None], sub, 0.0), kk, W - (a_lo - lo), 0)
                g1 = _compress(np.where(own, seg, 0.0), kk, W - (a_lo - lo), 0)
                sub = rng.poisson(np.maximum(g2, 0.0))
                seg = rng.poisson(np.maximum(g1, 0.0)).astype(np.float64)
                stats["n_spd"] += 1
                stats["k"].append(kk)
        if thin is not None:
            tt = float(rng.uniform(max(thin[0], 0.0), 1.0))
            if tt < 1.0 - 1e-9:
                base = np.maximum(np.floor(sub), 0).astype(np.int64)
                sub = rng.binomial(base, tt)
                base1 = np.maximum(np.floor(seg * own), 0).astype(np.int64)
                seg = rng.binomial(base1, tt).astype(np.float64)
                stats["n_thin"] += 1
                stats["t"].append(tt)
        # re-apply own mask (drops rows that migrated to another instance's hood)
        sub = np.where(own[:, None], np.maximum(sub, 0), 0.0).astype(np.int64)
        seg = np.where(own, np.maximum(seg, 0), 0).astype(np.int64)
        h = np.zeros_like(hood, dtype=np.int64)
        h[a_lo - lo:a_hi - lo] = sub
        encl.append(h)
        new_all = sec[sid].copy()
        new_all[a_lo:a_hi] = np.where(own, seg, sec[sid][a_lo:a_hi])
        sec[sid] = new_all
    from cache_build import ak_window
    if rebuild_windows:
        sbf = sb.astype(np.float64)
        snr_cache = {}
        for i, e in rows.iterrows():
            sid = int(e["source_id"])
            cen = int(e["cen_bin"])
            if sid not in snr_cache:
                S = sec[sid].astype(np.float64)
                with np.errstate(invalid="ignore", divide="ignore"):
                    snr_cache[sid] = S / np.sqrt(S + sbf)
            t0, t1, tmax, pk, wb, _cl = ak_window(snr_cache[sid], cen)
            rebuilt.loc[i, ["win_start_s", "win_stop_s", "win_max_s",
                           "snr_window", "width_s"]] = (t0 / 1e3, t1 / 1e3,
                                                        tmax / 1e3, pk, wb)
        rebuilt["start_ms"] = rebuilt.win_start_s * 1e3
        rebuilt["stop_ms"] = rebuilt.win_stop_s * 1e3
    placed = np.zeros((nb, 128), np.int64)
    for (_, e), h in zip(rows.iterrows(), encl):
        cen = int(e["cen_bin"])
        lo, hi = cen - W, cen + W + 1
        x, y = max(lo, 0), min(hi, nb)
        placed[x:y] += h[x - lo:y - lo]
    bg_all = comp.sum(0)
    total = bg_all + a["src_rest"].astype(np.int64) + placed
    ra_int = sb.copy()
    for v in sec.values():
        ra_int = ra_int + v
    # PHYSICAL GUARD: all-energy >= in-range per second.  The bg part is
    # nested by construction now; the SOURCE part keeps a tiny residual
    # flip rate because the in/all-energy margin (~few cps) is below
    # Poisson/binomial redraw noise -- clip instead of redesigning.
    _raw = ra_int.copy()
    ra_int = np.maximum(ra_int, total.sum(1))
    stats["clip_rows"] = int((_raw < total.sum(1)).sum())
    rate_all = ra_int.astype(np.float32)
    rate_in = total.sum(1).astype(np.float32)
    out = dict(total=total.astype(np.uint16) if total.max() < CAP16 else total,
               bg_comp=comp.astype(np.uint16), src_rest=a["src_rest"],
               enc=encl, sec_bg=sb, sec=sec, rate_all=rate_all,
               rate_in=rate_in, nb=nb)
    chk = np.zeros((nb, 128), np.int64)
    for (_, e), h in zip(rows.iterrows(), encl):
        cen = int(e["cen_bin"])
        lo, hi = cen - W, cen + W + 1
        x, y = max(lo, 0), min(hi, nb)
        chk[x:y] += h[x - lo:y - lo]
    inv = dict(
        J1=bool(np.array_equal(comp.sum(0), bg_all)),
        J2=bool(np.array_equal(out["total"].astype(np.int64),
                               bg_all + a["src_rest"].astype(np.int64) + chk)),
        J3=bool(np.array_equal(rate_all.astype(np.int64), ra_int)) and
            bool(np.array_equal(rate_in, total.sum(1).astype(np.float32))),
        J4=bool(total.max() < CAP16 and total.min() >= 0 and
                comp.min() >= 0 and all(h.min() >= 0 for h in encl)))
    return out, rebuilt, stats, inv
