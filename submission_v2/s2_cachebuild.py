# -*- coding: utf-8 -*-
"""
s2_cachebuild.py -- REBUILDS the train decomposition into cache/train_fix.h5
with the per-encounter hood-mask bug FIXED (original train.h5 left untouched).

BUG in cache_build.process_run (discovered 2026-10-06, Stage-2 T1b):
    seg = near & (aidx == q)
compares the argmin POSITION within the same-id instance group against the
GLOBAL instance index q.  Only q=0 ever receives rows -> enc_{q>0} empty in
296/300 train runs; their source events silently live in src_rest instead.
Everything else in the old cache is CORRECT (total, bg_comp, bg_all, sec_*,
rate_*, windows/CSV -- all mask-independent; Stage-1 features/targets never
read enc_q), so no Stage-1 number is affected.

Fix here: pos = np.searchsorted(same, q);  seg = near & (aidx == pos).
Consequence: src_rest gets SMALLER (per-id hood counts move into enc_q);
bg_all + src_rest + Σ placed enc == total still exact by construction.
Resumable; verifies against the old cache: total/bg_all/bg_comp/sec_* must
be BIT-IDENTICAL, windows CSV bit-identical, and each pool run must now have
non-empty hood for every instance (unless the instance truly has no in-range
counts within +-150 s... logged, not assumed).

Usage: .venv\\Scripts\\python.exe submission_v2\\s2_cachebuild.py [--redo]
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "submission"))
import cache_build as CB                      # noqa: E402  (reuse, do not modify)
import radai_lib as L                         # noqa: E402
import official_scorer as OS                  # noqa: E402

STEP, W, NB = CB.STEP, CB.W, CB.NB
RAW = HERE.parent / "training_v4.3.h5"
OUT = CB.CACHE / "train_fix.h5"
OUTCSV = CB.CACHE / "encounters_train_fix.csv"


def process_run_fix(f, rid, nl, names):
    g = f[f"runs/run{rid}"]
    lm = g["listmode"]
    dt = lm["dt"][:]
    e = lm["energy"][:]
    end_ms = (g.attrs["end_timestamp"] - g.attrs["start_timestamp"]) * 1e3
    idx, nb = CB.second_index(dt, end_ms)
    inr, bi = CB.energy_bins(e)
    keep = idx >= 0
    sel = keep & inr
    sec = idx[sel]
    eid = lm["id"][:]
    bid = lm["background_id"][:]
    out = {}
    out["total"] = CB.u16(np.bincount(sec * NB + bi[sel], minlength=nb * NB)
                          .reshape(nb, NB), f"total r{rid}", {})
    out["rate_all"] = np.bincount(idx[keep], minlength=nb).astype(np.float32)
    out["rate_in"] = np.bincount(sec, minlength=nb).astype(np.float32)
    tvl = np.bincount(idx[keep] * nl + eid[keep], minlength=nb * nl) \
        .reshape(nb, nl).astype(np.int64)
    B = tvl[:, 0:1].astype(np.float64)
    with np.errstate(invalid="ignore"):
        snr = tvl / np.sqrt(tvl + B)
    snr[:, 0] = np.nan
    comp = np.zeros((7, nb * NB), np.int64)
    m0 = sel & (eid == 0)
    for k in range(1, 8):
        m = m0 & (bid == k)
        comp[k - 1] = np.bincount(idx[m] * NB + bi[m], minlength=nb * NB)
    bg_all = comp.sum(0)
    out["bg_comp"] = CB.u16(comp.reshape(7, nb, NB), f"bg_comp r{rid}", {})
    out["bg_all"] = CB.u16(bg_all.reshape(nb, NB), f"bg_all r{rid}", {})
    assert np.array_equal(out["bg_comp"].sum(0), out["bg_all"])
    sid = np.asarray(g["sources/id"][:])
    stime = np.asarray(g["sources/time"][:], dtype=np.float64)
    msk = sid != 0
    order = np.argsort(stime[msk], kind="stable")
    sids, stimes = sid[msk][order], stime[msk][order]
    mS = sel & (eid != 0)
    src_sum = np.bincount(idx[mS] * NB + bi[mS], minlength=nb * NB)
    hood_sum = np.zeros(nb * NB, np.int64)
    rows = np.arange(nb)
    enc_meta = []
    for q, (j, c_ms) in enumerate(zip(sids, stimes)):
        j = int(j)
        m = sel & (eid == j)
        col = np.bincount(idx[m] * NB + bi[m], minlength=nb * NB) \
            .reshape(nb, NB)
        cen = int(np.round((c_ms - STEP / 2) / STEP))
        same = np.where(sids == j)[0]
        cs = np.round((stimes[same] - STEP / 2) / STEP).astype(np.int64)
        d = np.abs(rows[:, None] - cs[None, :])
        aidx = np.argmin(d, 1)
        near = d.min(1) <= W
        pos = int(np.searchsorted(same, q))          # <<< THE FIX
        seg = near & (aidx == pos)
        lo, hi = cen - W, cen + W + 1
        sl = np.zeros((2 * W + 1, NB), np.uint16)
        r_lo, r_hi = max(lo, 0), min(hi, nb)
        sl[r_lo - lo: r_hi - lo] = np.where(seg[r_lo:r_hi, None],
                                            col[r_lo:r_hi], 0).astype(np.uint16)
        out[f"enc_{q}"] = sl
        hood_sum += np.where(seg[:, None], col, 0).ravel()
        t0, t1, tmax, pk, wb, clipped = CB.ak_window(snr[:, j], cen)
        enc_meta.append(dict(run=rid, inst=q, source_id=j, ca_ms=c_ms,
                             cen_bin=cen,
                             win_start_s=t0 / 1e3, win_stop_s=t1 / 1e3,
                             win_max_s=tmax / 1e3, snr_window=pk, width_s=wb,
                             dist=float(g["sources/distance"][:][order][q]),
                             act=float(g["sources/activity"][:][order][q]),
                             snr_peak_gt=float(g["sources/snr/peak"]
                                               [:][order][q])))
    out["src_rest"] = CB.u16((src_sum - hood_sum).reshape(nb, NB),
                             f"src_rest r{rid}", {})
    assert int((src_sum - hood_sum).min()) >= 0, f"V2c r{rid}"
    assert np.array_equal(bg_all + src_sum,
                          out["total"].astype(np.int64).ravel())
    out["sec_bg"] = tvl[:, 0].astype(np.int32)
    for j in np.unique(eid[eid != 0]):
        out[f"sec_src_{int(j)}"] = tvl[:, int(j)].astype(np.int32)
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
    return out, enc_meta


def main(redo):
    with h5py.File(RAW, "r") as f:
        runs = sorted(int(k[3:]) for k in f["runs"].keys())
        nl = int(f.attrs["source_ids"].max()) + 1
        names = [str(x) for x in np.atleast_1d(f.attrs["source_names"])]
    done = []
    if OUT.exists() and not redo:
        with h5py.File(OUT, "r") as f:
            done = [int(k[1:]) for k in f]
    todo = [r for r in runs if r not in done]
    print(f"[fix] runs={len(runs)} done={len(done)} todo={len(todo)}",
          flush=True)
    new_csv = not (OUTCSV.exists() and done) or redo
    if redo:
        OUT.unlink(missing_ok=True)
    t0 = time.time()
    with h5py.File(OUT, "a") as gout, \
            open(OUTCSV, "w" if new_csv else "a", newline="",
                 encoding="utf-8") as mfh:
        w = csv.writer(mfh)
        if new_csv:
            w.writerow(CB.ENC_COLS)
        for i, rid in enumerate(todo):
            with h5py.File(RAW, "r") as f:
                out, enc_meta = process_run_fix(f, rid, nl, names)
            g = gout.create_group(f"r{rid}")
            for k, v in out.items():
                g.create_dataset(k, data=v)
            g.attrs["nb"] = out["total"].shape[0]
            g.attrs["n_enc"] = len(enc_meta)
            g.attrs["end_ms"] = out["total"].shape[0] * STEP
            for mrow in enc_meta:
                w.writerow([mrow.get(c, "") for c in CB.ENC_COLS])
            gout.flush()
            if (i + 1) % 25 == 0 or i + 1 == len(todo):
                el = time.time() - t0
                print(f"[fix] {i + 1}/{len(todo)} ({el:.0f}s, "
                      f"{el / (i + 1):.1f}s/run)", flush=True)
    (CB.CACHE / "build_fix.json").write_text(json.dumps(
        dict(runs=len(runs), secs=time.time() - t0)))
    print("[fix] DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--redo", action="store_true")
    main(ap.parse_args().redo)
