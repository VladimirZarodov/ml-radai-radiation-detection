# -*- coding: utf-8 -*-
"""
cache_build.py -- Stage 0 decomposed-spectrogram cache builder (v2).

One streaming pass per h5 run; writes `submission_v2/cache/{train,dev,test}.h5`
plus `cache/encounters_{train,dev}.csv` and `cache/build_log_<split>.json`.
Resumable: existing r{rid} groups are skipped.

Binning (DESIGN.md §2):
  * time: official 1-s bins -- t_ms = cumsum(dt, uint32)/1e3, nb from
    edges arange(0, end_ms, 1000, uint32), guards identical to
    submission/official_scorer.py::_snr_hist (proven == organiser histogram2d);
  * energy: 128 bins uniform in sqrt(E) over [15, 3000) keV -- exact formula
    of radai_lib.read_listmode_blind.

Stored per TRAIN/DEV run `r{rid}` (uint16 unless noted):
  total    (nb,128) in-energy-range events
  bg_all   (nb,128) id==0
  bg_comp  (7,nb,128) id==0 by background_id 1..7
  enc_{q}  (301,128) per-encounter source counts, CA+-150 s hood, own id only
  src_rest (nb,128) source events outside every own-id hood
  rate_all (nb,) f32 cps all energies; rate_in (nb,) f32 cps in range
  sec_bg   (nb,) i32 all-energy id==0 counts/s   (official SNR denominator)
  sec_src_{id} (nb,) i32 all-energy per-id counts/s (official SNR numerator)
Answer-key windows rebuilt IN-PASS with official_scorer._encounter_window
(S/sqrt(S+B) on the all-energy sec x id matrix, greedy 25-bin bound, strict
>5%-of-peak) -> columns in encounters CSV.
TESTING runs (blind, only dt+energy exist): total, rate_all, rate_in.

Invariants asserted per run (fail => stop):
  V1 all stored bins < 65536 (uint16 safety),
  V2 sum(bg_comp) == bg_all; bg_all + Σ_src cols == total;
     Σ enc slices (in-hood rows) + src_rest == Σ_src cols.

Usage: .venv\\Scripts\\python.exe submission_v2\\cache_build.py --split train|dev|test
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "submission"))
import radai_lib as L           # frozen binning formula + official labels
import official_scorer as OS    # faithful window builder (verbatim reuse)

EMIN, EMAX, NB = L.EMIN, L.EMAX, L.NB
STEP = 1000.0                   # ms
W = 150                         # source hood half-width, s
CACHE = Path(__file__).resolve().parent / "cache"

SPLITS = {
    "train": dict(h5=ROOT / "training_v4.3.h5", out=CACHE / "train.h5", gt=True),
    "dev":   dict(h5=ROOT / "developer_v4.3.h5", out=CACHE / "dev.h5", gt=True),
    "test":  dict(h5=ROOT / "testing_v4.3.h5", out=CACHE / "test.h5", gt=False),
}

ENC_COLS = ["run", "inst", "source_id", "ca_ms", "cen_bin", "raw_name",
            "official_label", "category", "isotope", "win_start_s",
            "win_stop_s", "win_max_s", "snr_window", "width_s", "dist",
            "act", "snr_peak_gt"]


def energy_bins(e):
    """128 sqrt-E bins -- verbatim radai_lib.read_listmode_blind."""
    inr = (e >= EMIN) & (e < EMAX)
    bi = np.zeros(len(e), np.int64)
    smin, smax = np.sqrt(EMIN), np.sqrt(EMAX)
    bi[inr] = np.clip(np.floor((np.sqrt(e[inr]) - smin) / (smax - smin) * NB)
                      .astype(np.int64), 0, NB - 1)
    return inr, bi


def second_index(dt, end_ms):
    """Official 1-s bin index per event (-1 = dropped), same guards as
    official_scorer._snr_hist."""
    if not (10 < end_ms < 2e8):
        raise RuntimeError(f"implausible run length end_ms={end_ms}")
    t_ms = np.cumsum(dt, dtype=np.uint32).astype(np.float64) / 1e3
    edges = np.arange(0, end_ms, STEP, dtype=np.uint32)
    nb = len(edges) - 1
    if not (10 <= nb <= 8000):
        raise RuntimeError(f"implausible run length: nb={nb} "
                           f"end_ms={end_ms} dt_sum_ms={t_ms[-1]:.0f}")
    idx = np.floor(t_ms / STEP).astype(np.int64)
    ok = (idx >= 0) & (idx < nb)
    last = (~ok) & (idx == nb) & (t_ms == edges[-1])
    idx = np.where(last, nb - 1, idx)
    idx = np.where((idx >= 0) & (idx < nb), idx, -1)
    return idx, nb


def u16(a, what, log):
    m = int(a.max()) if a.size else 0
    if m > 65535:
        raise RuntimeError(f"V1 overflow {what}: max={m}")
    log.setdefault("hist_max", [])
    return a.astype(np.uint16)


def ak_window(v, center):
    """Guarded replica of official_scorer._encounter_window.

    Bit-identical behaviour on every run where the official loop
    terminates: the window grows by one bin per iteration, so n_below
    can only reach N_LOW by hitting it exactly (== vs >= is moot).
    Differs ONLY in the case where the official loop is non-terminating
    -- developer runs with a persistent source whose whole SNR column
    holds < N_LOW bins below LOW_T: there the official slice start goes
    negative and wraps forever.  Here the window is clamped to the full
    column and the 5%-of-peak span logic proceeds over it.  Returns
    (t0_ms, t1_ms, tmax_ms, peak_snr, wbin, clipped).
    """
    n = len(v)
    N_LOW, LOW_T, SNR_REL, STEP = OS.N_LOW, OS.LOW_T, OS.SNR_REL, OS.STEP
    n_below = 0
    n_below_left = 0
    n_below_right = 0
    wl = wr = 3
    win = None
    clipped = False
    for it in range(2 * n + 100):
        if wl > center or wr > n - 1 - center:      # overflow -> clamp, stop
            wl, wr = min(wl, center), min(wr, n - 1 - center)
            win = v[center - wl: center + wr + 1]
            clipped = True
            break
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
    else:
        raise RuntimeError("ak_window runaway")
    lo = center - wl
    assert lo >= 0, "negative slice start (pandas-from-end bug) not handled"
    peak_snr = float(np.nanmax(win))
    above = win > SNR_REL * peak_snr
    first = lo + int(np.argmax(above))
    last = lo + len(above) - 1 - int(np.argmax(above[::-1]))
    peak = lo + int(np.nanargmax(win))
    return (first * STEP, (last + 1) * STEP, peak * STEP + STEP / 2,
            peak_snr, last - first + 1, clipped)


def process_run(f, rid, nl, names, gt):
    g = f[f"runs/run{rid}"]
    lm = g["listmode"]
    dt = lm["dt"][:]
    e = lm["energy"][:]
    end_ms = (g.attrs["end_timestamp"] - g.attrs["start_timestamp"]) * 1e3
    idx, nb = second_index(dt, end_ms)
    inr, bi = energy_bins(e)
    keep = idx >= 0
    sel = keep & inr
    sec = idx[sel]
    log = dict(overflow=[], run=rid, nb=nb)
    out = {}

    out["total"] = u16(np.bincount(sec * NB + bi[sel], minlength=nb * NB)
                       .reshape(nb, NB), f"total r{rid}", log)
    out["rate_all"] = np.bincount(idx[keep], minlength=nb).astype(np.float32)
    out["rate_in"] = np.bincount(sec, minlength=nb).astype(np.float32)
    enc_meta = []

    if gt:
        eid = lm["id"][:]
        bid = lm["background_id"][:]
        # all-energy sec x id matrix -> official SNR columns
        tvl = np.bincount(idx[keep] * nl + eid[keep], minlength=nb * nl) \
            .reshape(nb, nl).astype(np.int64)
        B = tvl[:, 0:1].astype(np.float64)
        with np.errstate(invalid="ignore"):
            snr = tvl / np.sqrt(tvl + B)          # (nb, nl) float64, col0 nan-y
        snr[:, 0] = np.nan
        # background split (in energy range)
        comp = np.zeros((7, nb * NB), np.int64)
        m0 = sel & (eid == 0)
        for k in range(1, 8):
            m = m0 & (bid == k)
            comp[k - 1] = np.bincount(idx[m] * NB + bi[m], minlength=nb * NB)
        bg_all = comp.sum(0)
        out["bg_comp"] = u16(comp.reshape(7, nb, NB), f"bg_comp r{rid}", log)
        out["bg_all"] = u16(bg_all.reshape(nb, NB), f"bg_all r{rid}", log)
        assert np.array_equal(out["bg_comp"].sum(0), out["bg_all"]), f"V2a r{rid}"
        # source per-id energy matrices + hood assignment + AK windows
        sid = np.asarray(g["sources/id"][:])
        stime = np.asarray(g["sources/time"][:], dtype=np.float64)
        msk = sid != 0
        order = np.argsort(stime[msk], kind="stable")
        sids, stimes = sid[msk][order], stime[msk][order]
        mS = sel & (eid != 0)
        src_sum = np.bincount(idx[mS] * NB + bi[mS], minlength=nb * NB)
        hood_sum = np.zeros(nb * NB, np.int64)
        rows = np.arange(nb)
        for q, (j, c_ms) in enumerate(zip(sids, stimes)):
            j = int(j)
            m = sel & (eid == j)
            col = np.bincount(idx[m] * NB + bi[m], minlength=nb * NB) \
                .reshape(nb, NB)
            cen = int(np.round((c_ms - STEP / 2) / STEP))
            # own-id hood = +-150 s around CA, nearest-instance rule among
            # instances of the SAME id (rare multi-instance ids handled)
            same = np.where(sids == j)[0]
            cs = np.round((stimes[same] - STEP / 2) / STEP).astype(np.int64)
            d = np.abs(rows[:, None] - cs[None, :])
            aidx = np.argmin(d, 1)
            near = d.min(1) <= W
            seg = near & (aidx == q)
            lo, hi = cen - W, cen + W + 1
            sl = np.zeros((2 * W + 1, NB), np.uint16)
            r_lo, r_hi = max(lo, 0), min(hi, nb)
            sub = col[r_lo:r_hi]
            keep_r = seg[r_lo:r_hi]
            sl[r_lo - lo: r_hi - lo] = np.where(keep_r[:, None], sub, 0) \
                .astype(np.uint16)
            out[f"enc_{q}"] = sl
            hood_sum += np.where(seg[:, None], col, 0).ravel()
            t0, t1, tmax, pk, wb, clipped = ak_window(snr[:, j], cen)
            if clipped:
                print(f"  [r{rid}] enc{q}: CLIPPED ak window "
                      f"(persistent source, {len(snr)}-bin column)",
                      flush=True)
            enc_meta.append(dict(run=rid, inst=q, source_id=j, ca_ms=c_ms,
                                 cen_bin=cen,
                                 win_start_s=t0 / 1e3, win_stop_s=t1 / 1e3,
                                 win_max_s=tmax / 1e3, snr_window=pk,
                                 width_s=wb,
                                 dist=float(g["sources/distance"][:][order][q])
                                 if "sources/distance" in g else np.nan,
                                 act=float(g["sources/activity"][:][order][q])
                                 if "sources/activity" in g else np.nan,
                                 snr_peak_gt=float(
                                     g["sources/snr/peak"][:][order][q])
                                 if "sources/snr/peak" in g else np.nan))
        out["src_rest"] = u16((src_sum - hood_sum).reshape(nb, NB),
                              f"src_rest r{rid}", log)
        assert np.array_equal(
            bg_all + src_sum, out["total"].astype(np.int64).ravel()), \
            f"V2b r{rid}"
        assert int((src_sum - hood_sum).min()) >= 0, f"V2c r{rid}"
        out["sec_bg"] = tvl[:, 0].astype(np.int32)
        for j in np.unique(eid[eid != 0]):
            out[f"sec_src_{int(j)}"] = tvl[:, int(j)].astype(np.int32)
        # enrich labels
        for mrow in enc_meta:
            nm = names[mrow["source_id"]]
            mrow["raw_name"] = nm
            lab = L.official_label(nm)
            mrow["official_label"] = lab or ""
            eff = lab or nm
            try:
                mrow["category"] = OS.label2category(eff)
            except ValueError:
                mrow["category"] = ""
            mrow["isotope"] = OS.label2isotope(eff)
    return out, enc_meta, log


def build(split):
    cfg = SPLITS[split]
    CACHE.mkdir(parents=True, exist_ok=True)
    with h5py.File(cfg["h5"], "r") as f:
        runs = sorted(int(k[3:]) for k in f["runs"].keys())
        nl = int(f.attrs["source_ids"].max()) + 1
        assert np.array_equal(np.asarray(f.attrs["source_ids"]),
                              np.arange(nl)), "ids assumed 0..N-1"
        names = [str(x) for x in np.atleast_1d(f.attrs["source_names"])]
    done = []
    if cfg["out"].exists():
        with h5py.File(cfg["out"], "r") as f:
            done = [int(k[1:]) for k in f if k.startswith("r")]
    todo = [r for r in runs if r not in done]
    print(f"[{split}] runs={len(runs)} done={len(done)} todo={len(todo)}",
          flush=True)
    met = CACHE / f"encounters_{split}.csv"
    new_met = not (met.exists() and done)
    stats_log = dict(split=split, n_done=len(done), overflow=[])
    t0 = time.time()
    with h5py.File(cfg["out"], "a") as gout, \
            open(met, "w" if new_met else "a", newline="",
                 encoding="utf-8") as mfh:
        w = csv.writer(mfh)
        if new_met:
            w.writerow(ENC_COLS)
        for i, rid in enumerate(todo):
            with h5py.File(cfg["h5"], "r") as f:
                out, enc_meta, log = process_run(f, rid, nl, names, cfg["gt"])
            g = gout.create_group(f"r{rid}")
            for k, v in out.items():
                g.create_dataset(k, data=v)
            g.attrs["nb"] = out["total"].shape[0]
            g.attrs["n_enc"] = len(enc_meta)
            g.attrs["end_ms"] = out["total"].shape[0] * STEP
            for mrow in enc_meta:
                w.writerow([mrow.get(c, "") for c in ENC_COLS])
            stats_log["per_run"] = stats_log.get("per_run", [])
            stats_log["per_run"].append(dict(
                run=rid, nb=log["nb"], n_enc=len(enc_meta),
                cps_med=float(np.median(out["rate_all"])),
                cps_p99=float(np.percentile(out["rate_all"], 99))))
            gout.flush()
            if (i + 1) % 10 == 0 or i + 1 == len(todo):
                el = time.time() - t0
                print(f"[{split}] {i + 1}/{len(todo)} ({el:.0f}s, "
                      f"{el / (i + 1):.1f}s/run)", flush=True)
    stats_log["secs"] = time.time() - t0
    (CACHE / f"build_log_{split}.json").write_text(json.dumps(stats_log, indent=1))
    print(f"[{split}] DONE {len(todo)} runs in {stats_log['secs']:.0f}s",
          flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=list(SPLITS), required=True)
    a = ap.parse_args()
    build(a.split)
