# -*- coding: utf-8 -*-
"""
s1_train.py -- Stage-1 launcher: timing / dev-parity / overfit / train / eval.

Configurations (<= 4 learned, pre-registered for the Stage-1 comparison):
  c64h  ctx +-64 s, hard window target     batch 192
  c128h ctx +-128 s, hard window target    batch 96
  c64s  ctx +-64 s, soft CA-peaked target  batch 192
  c128s ctx +-128 s, soft CA-peaked target batch 96
One architecture (s1_model.Net, ~0.92 M params), AMP fp16, AdamW wd 1e-5,
cosine LR from 1.5e-3, grad clip 1.0, focal(2, .75) + 0.5*CE_cat + 0.5*CE_iso,
negative sub-sampling NEG_RATE=0.25/epoch (rotating fixed seed), max --epochs
(default 15) with patience 3 on INNER-SPLIT val focal (frozen rule: fold-train
runs sorted by id, last 15% inner validation).  Lockbox asserted absent from
every split at start of every train/eval run (leak guard, logged).

Determinism: torch/np/random seeded per (SEED, cfg, fold); cudnn.deterministic
on, benchmark off.  Remaining non-determinism: fp16 atomics in some conv/
reduce backward kernels (cuDNN) and nondeterministic index-select backward --
run-to-run bitwise reproducibility is NOT guaranteed on GPU; distribution is.

Usage:
  python s1_train.py --mode build
  python s1_train.py --mode timing  --cfg c64h [--ctx-eval]
  python s1_train.py --mode devparity --cfg c64h
  python s1_train.py --mode overfit --cfg c64h [--ovf-epochs 80]
  python s1_train.py --mode train   --cfg c64h --fold 0|all
  python s1_train.py --mode eval    --cfg c64h --fold 0|all
Outputs: cache/models/{cfg}/fold{k}.pt (+.done, history.jsonl, log.txt),
cache/oof/{cfg}/r{run}.npz (p, cat argmax, iso argmax per second).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as TF

import s1_data as D
import s1_model as M
import v2_lib as V

SEED = 20261003
NEG_RATE = 0.12
CFGS = {
    "c64h":  dict(ctx=64,  target="hard", batch=192),
    "c128h": dict(ctx=128, target="hard", batch=96),
    "c64s":  dict(ctx=64,  target="soft", batch=192),
    "c128s": dict(ctx=128, target="soft", batch=96),
}
MODELS = V.CACHE / "models"
OOF = V.CACHE / "oof"


def cfg_dir(cfg):
    p = MODELS / cfg
    p.mkdir(parents=True, exist_ok=True)
    return p


def log_line(cfg, msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    (cfg_dir(cfg) / "log.txt").open("a").write(line + "\n")


def set_seeds(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ---------------------------------------------------------------------------
# fold data on GPU (single padded-coordinate system, see module notes)
# ---------------------------------------------------------------------------

class Store:
    def __init__(self, runs, ctx, target, dev, iso_n=24, pre=None):
        if pre is not None:
            d = {r: {k: np.asarray(v) for k, v in pre[r].items()}
                 for r in runs}
        else:
            d = D.load_runs(runs, device=None)      # numpy arrays
        Xs, ys, yss, tc, ti = [], [], [], [], []
        self.j0 = {}
        n = 0
        for r in runs:
            a = d[r]
            nb = a["X"].shape[0]
            Xp = np.pad(a["X"], ((ctx, ctx), (0, 0)))       # f16 zero pad
            yp = np.zeros(nb + 2 * ctx, np.uint8)
            yp[ctx:ctx + nb] = a["y"]
            ysp = np.zeros(nb + 2 * ctx, np.float16)
            ysp[ctx:ctx + nb] = a["ys"]
            tcp = np.full(nb + 2 * ctx, D.MASK, np.int8)
            tcp[ctx:ctx + nb] = a["tcat"]
            tip = np.full(nb + 2 * ctx, D.MASK, np.int8)
            tip[ctx:ctx + nb] = a["tiso"]
            self.j0[r] = n
            n += nb + 2 * ctx
            Xs.append(Xp); ys.append(yp); yss.append(ysp)
            tc.append(tcp); ti.append(tip)
        self.ctx = ctx
        self.T = 2 * ctx + 1
        self.X = torch.from_numpy(np.concatenate(Xs)).to(dev)
        self.off = torch.arange(-ctx, ctx + 1, device=dev)   # centered
        ycat = torch.from_numpy(np.concatenate(ys)).to(dev)
        tgt_np = (np.concatenate(ys).astype(np.float32)
                  if target == "hard"
                  else np.concatenate(yss).astype(np.float32))
        self.y = ycat
        self.ys = torch.from_numpy(np.concatenate(yss)).to(dev)
        self.tcat = torch.from_numpy(np.concatenate(tc)).to(dev)
        self.tiso = torch.from_numpy(np.concatenate(ti)).to(dev)
        self.tgt = torch.from_numpy(tgt_np).to(dev)
        valid = torch.zeros(n, dtype=torch.bool, device=dev)
        for r in runs:
            nb = d[r]["X"].shape[0]
            valid[self.j0[r] + ctx: self.j0[r] + ctx + nb] = True
        self.valid_j = torch.nonzero(valid).squeeze(1)
        self.nb = {r: d[r]["X"].shape[0] for r in runs}
        act = self.tgt > 0.5
        self.pos_j = torch.nonzero(valid & act).squeeze(1).cpu().numpy()
        self.neg_j = torch.nonzero(valid & ~act).squeeze(1).cpu().numpy()
        vj = self.valid_j.cpu().numpy()
        self.val_j = torch.from_numpy(np.unique(np.concatenate(
            [self.pos_j, vj[::4]]))).to(dev)


def windows(store, j):
    """(B,T,F) float16 centered at padded rows j (torch int64 on device)."""
    return store.X[j[:, None] + store.off[None, :]].unsqueeze(1)


# ---------------------------------------------------------------------------
# training one fold
# ---------------------------------------------------------------------------

def train_fold(cfgname, fold, epochs, dev, allow_resume=True):
    cf = CFGS[cfgname]
    cdir = cfg_dir(cfgname)
    ck = cdir / f"fold{fold}.pt"
    done = cdir / f"fold{fold}.done"
    if done.exists():
        log_line(cfgname, f"fold{fold}: already done, skip")
        return
    lock_runs, runs240, folds, hi_runs = D.dev_pool()
    val_runs = sorted(folds[fold])
    train_runs = sorted(r for r in runs240 if r not in set(val_runs))
    inner_tr, inner_va = D.inner_split(train_runs)
    log_line(cfgname, D.assert_no_lockbox(
        train_runs + val_runs + inner_va, f"train fold{fold}",
        cdir / "leakguard.log"))
    log_line(cfgname, D.assert_no_lockbox(
        inner_tr, f"inner-train fold{fold}", cdir / "leakguard.log"))

    seed = (SEED * 7919 + fold * 101 + len(cfgname) * 7907 +
            (1 if cf["target"] == "soft" else 0) * 31337) % 2**31
    set_seeds(seed)

    t0 = time.time()
    st_tr = Store(inner_tr, cf["ctx"], cf["target"], dev)
    st_va = Store(inner_va, cf["ctx"], cf["target"], dev)
    log_line(cfgname, f"fold{fold}: data load {time.time() - t0:.0f}s "
             f"train {len(inner_tr)} runs ({len(st_tr.pos_j)} pos, "
             f"{len(st_tr.neg_j)} neg), inner-val {len(inner_va)}")

    model = M.Net(n_iso=len(D.iso_vocab())).to(dev)
    log_line(cfgname, f"params: {M.n_params(model):,}")
    niso = len(D.iso_vocab())
    tc_np = st_tr.tcat.cpu().numpy()
    ti_np = st_tr.tiso.cpu().numpy()
    y_np = st_tr.y.cpu().numpy()
    cw_cat = M.class_weights(np.array(
        [int(((tc_np == c) & (y_np == 1)).sum()) for c in range(4)])).to(dev)
    cw_iso = M.class_weights(np.array(
        [int(((ti_np == i) & (y_np == 1)).sum()) for i in range(niso)])).to(dev)

    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-5)
    scaler = torch.amp.GradScaler("cuda")
    focal = M.Focal()

    first_epoch = 0
    best_val, best_state, bad = np.inf, None, 0
    if allow_resume and ck.exists():
        s = torch.load(ck, map_location=dev, weights_only=False)
        model.load_state_dict(s["model"])
        opt.load_state_dict(s["opt"])
        scaler.load_state_dict(s["scaler"])
        best_val, best_state = s["best_val"], s["best_state"]
        bad = s["bad"]
        first_epoch = s["epoch"] + 1
        log_line(cfgname, f"fold{fold}: resume from epoch {first_epoch} "
                 f"(best val {best_val:.4f})")

    hist = cdir / "history.jsonl"
    for ep in range(first_epoch, epochs):
        rng = np.random.default_rng([seed, fold, ep])
        neg = rng.choice(st_tr.neg_j,
                         size=int(NEG_RATE * len(st_tr.neg_j)),
                         replace=False)
        pool = np.concatenate([st_tr.pos_j, neg])
        perm = rng.permutation(len(pool))
        pool = torch.from_numpy(pool[perm]).to(dev)
        model.train()
        run_l, run_n, t1 = 0.0, 0, time.time()
        for i in range(0, len(pool), cf["batch"]):
            j = pool[i:i + cf["batch"]]
            if i == 0:
                set_seeds(seed + ep * 977 + 13)
            x = windows(st_tr, j)
            lr = 1.5e-3 * (0.1 + 0.9 * 0.5 *
                           (1 + math.cos(math.pi * min(1.0, (ep + i / len(pool)) /
                                                       max(epochs, 1)))) )
            for g in opt.param_groups:
                g["lr"] = lr
            with torch.autocast("cuda", dtype=torch.float16):
                dl, cl, il = model(x, cf["ctx"])
                l_det = focal(dl, st_tr.tgt[j])
                hm = st_tr.y[j] == 1
                if hm.any():
                    l_cat = TF.cross_entropy(
                        cl[hm], st_tr.tcat[j][hm].long(), weight=cw_cat)
                    l_iso = TF.cross_entropy(
                        il[hm], st_tr.tiso[j][hm].long(), weight=cw_iso)
                else:
                    l_cat = l_iso = torch.zeros((), device=dev)
                loss = l_det + 0.5 * l_cat + 0.5 * l_iso
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            run_l += float(loss) * len(j)
            run_n += len(j)
        tr_loss = run_l / run_n
        vl, auc, ap = evaluate(model, st_va, cf, dev, focal)
        rec = dict(cfg=cfgname, fold=fold, epoch=ep, seed=seed,
                   train_loss=tr_loss, val_focal=vl, val_auc=auc, val_ap=ap,
                   lr=lr, secs=time.time() - t1,
                   vram_mib=torch.cuda.max_memory_allocated() / 2**20
                   if dev.type == "cuda" else 0)
        with hist.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        log_line(cfgname, f"fold{fold} ep{ep}: train {tr_loss:.4f} "
                 f"val {vl:.4f} auc {auc:.4f} ap {ap:.4f} "
                 f"{rec['secs']:.0f}s vram {rec['vram_mib']:.0f} MiB")
        if vl < best_val - 1e-5:
            best_val, bad = vl, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad += 1
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(),
                         scaler=scaler.state_dict(), epoch=ep,
                         best_val=best_val, best_state=best_state, bad=bad,
                         cfg=cf, seed=seed), ck)
        if bad >= 3:
            log_line(cfgname, f"fold{fold}: early stop at ep{ep} "
                     f"(patience 3)")
            break
    (cdir / f"fold{fold}.done").touch()
    log_line(cfgname, f"fold{fold}: DONE best val focal {best_val:.4f} "
             f"in {time.time() - t0:.0f}s")


@torch.no_grad()
def evaluate(model, st, cf, dev, focal, batch=256):
    """Monitoring pass on the THINNED eval set (all positives + 1/4 of
    seconds, deterministic): early-stop metric val_focal and AUC/AP are
    computed on this subsample (monitor only, never for selection).
    batch 256 + empty_cache: avoids allocator fragmentation after AMP
    backward (fp32 GroupNorm intermediates need large contiguous blocks)."""
    model.eval()
    torch.cuda.empty_cache()
    js = st.val_j
    ls, n = 0.0, 0
    ps, ts, ys = [], [], []
    for i in range(0, len(js), batch):
        j = js[i:i + batch]
        x = windows(st, j)
        with torch.autocast("cuda", dtype=torch.float16):
            dl, _, _ = model(x, cf["ctx"])
        ls += float(focal(dl, st.tgt[j])) * len(j)
        n += len(j)
        ps.append(torch.sigmoid(dl).float().cpu().numpy())
        ts.append(st.tgt[j].cpu().numpy())
        ys.append(st.y[j].cpu().numpy())
    p = np.concatenate(ps); y = np.concatenate(ys)
    return (ls / n, M.auc_np(y, p), M.prap_np(y, p))


# ---------------------------------------------------------------------------
# per-fold out-of-fold inference
# ---------------------------------------------------------------------------

def eval_fold(cfgname, fold, dev):
    cf = CFGS[cfgname]
    cdir = cfg_dir(cfgname)
    ck = cdir / f"fold{fold}.pt"
    od = OOF / cfgname
    od.mkdir(parents=True, exist_ok=True)
    if (od / f"fold{fold}.done").exists():
        log_line(cfgname, f"eval fold{fold}: already done, skip")
        return
    lock_runs, runs240, folds, _ = D.dev_pool()
    val_runs = sorted(folds[fold])
    log_line(cfgname, D.assert_no_lockbox(
        val_runs, f"eval fold{fold} dev-runs", cdir / "leakguard.log"))
    s = torch.load(ck, map_location=dev, weights_only=False)
    model = M.Net(n_iso=len(D.iso_vocab())).to(dev)
    model.load_state_dict(s["best_state"])
    model.eval()
    st = Store(val_runs, cf["ctx"], cf["target"], dev)
    torch.cuda.empty_cache()
    iso_vocab = D.iso_vocab()
    t0 = time.time()
    with torch.no_grad():
        for r in val_runs:
            jp = np.arange(st.j0[r] + cf["ctx"],
                           st.j0[r] + cf["ctx"] + st.nb[r])
            P, C_, I_ = [], [], []
            for i in range(0, len(jp), 256):
                j = torch.from_numpy(jp[i:i + 256]).to(dev)
                with torch.autocast("cuda", dtype=torch.float16):
                    dl, cl, il = model(windows(st, j), cf["ctx"])
                P.append(torch.sigmoid(dl).float().cpu().numpy())
                C_.append(cl.argmax(1).cpu().numpy().astype(np.int8))
                I_.append(il.argmax(1).cpu().numpy().astype(np.int8))
            np.savez_compressed(od / f"r{r}.npz",
                                p=np.concatenate(P).astype(np.float16),
                                cat=np.concatenate(C_),
                                iso=np.concatenate(I_))
    (od / f"fold{fold}.done").touch()
    log_line(cfgname, f"eval fold{fold}: {len(val_runs)} runs in "
             f"{time.time() - t0:.0f}s (best val focal {s['best_val']:.4f})")


# ---------------------------------------------------------------------------
# timing calibration / dev parity / overfit
# ---------------------------------------------------------------------------

def timing(cfgname, dev, epochs=12):
    cf = CFGS[cfgname]
    cdir = cfg_dir(cfgname)
    lock_runs, runs240, folds, _ = D.dev_pool()
    train_runs = sorted(r for r in runs240 if r not in set(folds[0]))
    inner_tr, inner_va = D.inner_split(train_runs)
    t0 = time.time()
    st = Store(inner_tr, cf["ctx"], cf["target"], dev)
    load_s = time.time() - t0
    model = M.Net(n_iso=24).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda")
    focal = M.Focal()
    n_neg_e = int(NEG_RATE * len(st.neg_j))
    pool_sz = len(st.pos_j) + n_neg_e
    bs = cf["batch"]
    batch_idx = [torch.from_numpy(st.pos_j[i:i + bs]).to(dev)
                 for i in range(0, 6 * bs, bs)]
    torch.cuda.reset_peak_memory_stats()
    set_seeds(SEED)
    for k, j in enumerate(batch_idx):
        x = windows(st, j)
        with torch.autocast("cuda", dtype=torch.float16):
            dl, cl, il = model(x, cf["ctx"])
            loss = focal(dl, st.tgt[j]) + 0.5 * TF.cross_entropy(
                cl, st.tcat[j].clamp(min=0).long())
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
    torch.cuda.synchronize()
    t0 = time.time()
    for k in range(30):
        j = torch.from_numpy(
            st.pos_j[k * bs:(k + 1) * bs]).to(dev) if k * bs + bs <= len(
            st.pos_j) else batch_idx[0]
        x = windows(st, j)
        with torch.autocast("cuda", dtype=torch.float16):
            dl, _, _ = model(x, cf["ctx"])
            loss = focal(dl, st.tgt[j])
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        scaler.step(opt)
        scaler.update()
    torch.cuda.synchronize()
    s_bwd = (time.time() - t0) / 30
    t0 = time.time()
    model.eval()
    torch.cuda.empty_cache()
    for k in range(12):
        j = torch.from_numpy(st.pos_j[k * 256:(k + 1) * 256]).to(dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            model(windows(st, j), cf["ctx"])
    torch.cuda.synchronize()
    t_warm = time.time() - t0
    t0 = time.time()
    for rep in range(3):
        for k in range(12):
            j = torch.from_numpy(st.pos_j[k * 256:(k + 1) * 256]).to(dev)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                model(windows(st, j), cf["ctx"])
    torch.cuda.synchronize()
    s_fwd_eval = (time.time() - t0) / 36 * 2   # per-256 -> stored as /512
    model.train()
    mean_nb = int(round(np.mean([st.nb[r] for r in inner_tr])))
    ev_samples = mean_nb * len(inner_va) * (0.25 + 0.06)   # thinned val set
    train_batches = math.ceil(pool_sz / bs)
    epoch_train_s = s_bwd * train_batches
    epoch_val_s = s_fwd_eval * math.ceil(ev_samples / 512)
    n240 = 880000   # approx total dev-pool seconds for OOF inference pass
    res = dict(
        cfg=cfgname, params=M.n_params(model), batch=bs, epochs=epochs,
        load_s=round(load_s), train_pool=pool_sz,
        s_per_batch_bwd=round(s_bwd, 4),
        s_per_sample_fwd=round(s_fwd_eval / 512 * 1000, 3),
        epoch_train_s=round(epoch_train_s, 1),
        epoch_val_s=round(epoch_val_s, 1),
        fold_s=round((epoch_train_s + epoch_val_s) * epochs, 1),
        folds5_s=round((epoch_train_s + epoch_val_s) * epochs * 5, 1),
        oof_infer_s=round(n240 / 512 * s_fwd_eval, 1),
        peak_vram_mib=round(torch.cuda.max_memory_allocated() / 2**20),
        note="fold_s = worst-case (all epochs) train+thinned-val; early stop "
             "typically cuts 20-40%; oof_infer = full 240-run inference once",
    )
    log_line(cfgname, "TIMING " + json.dumps(res))
    (cfg_dir(cfgname) / "timing.json").write_text(json.dumps(res, indent=1))


def devparity(cfgname):
    """CPU vs CUDA implementation parity, fp32, TF32 OFF, dropout off.
    Real training intentionally keeps AMP fp16 + TF32 on (faster; run-to-run
    bitwise GPU reproducibility NOT guaranteed -- documented in PLAN)."""
    cf = CFGS[cfgname]
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    g1 = torch.Generator().manual_seed(SEED)
    x = torch.randn(64, 1, cf["ctx"] * 2 + 1, 132, generator=g1)
    y = (torch.arange(64) % 8 == 0).float()
    outs = {}
    for device in ("cpu", "cuda"):
        set_seeds(SEED)
        torch.backends.cudnn.deterministic = True
        model = M.Net(n_iso=24).to(device)
        model.eval()   # dropout RNG streams differ by design (CPU vs CUDA)
        focal = M.Focal()
        xd, yd = x.to(device), y.to(device)
        dl, cl, il = model(xd, cf["ctx"])
        with torch.no_grad():
            fwd = dl.detach().cpu().clone()
        loss = focal(dl, yd) + 0.1 * TF.cross_entropy(
            cl, torch.zeros(64, dtype=torch.long, device=device))
        model.zero_grad(set_to_none=True)
        loss.backward()
        grads = {n: p.grad.detach().cpu().clone()
                 for n, p in model.named_parameters() if p.grad is not None}
        # few bounded steps with tiny lr to check optimizer-path alignment
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        for _ in range(5):
            dl2, cl2, il2 = model(xd, cf["ctx"])
            l2 = focal(dl2, yd) + 0.1 * TF.cross_entropy(
                cl2, torch.zeros(64, dtype=torch.long, device=device))
            opt.zero_grad(set_to_none=True)
            l2.backward()
            opt.step()
        outs[device] = dict(fwd=fwd, grads=grads,
                            w={n: p.detach().cpu().clone()
                               for n, p in model.named_parameters()},
                            loss=float(l2.detach()))
    scale = max(t.abs().max() for t in outs["cpu"]["grads"].values())
    df = max(float((outs["cuda"]["fwd"] - outs["cpu"]["fwd"]).abs().max()),
             1e-12)
    dg = max(float((outs["cuda"]["grads"][k] - outs["cpu"]["grads"][k]).abs().max())
             for k in outs["cpu"]["grads"])
    dw = max(float((outs["cuda"]["w"][k] - outs["cpu"]["w"][k]).abs().max())
             for k in outs["cpu"]["w"])
    ok = df < 1e-4 and dg / scale < 2e-3
    print(f"devparity fp32 TF32-off: fwd max|d|={df:.2e} "
          f"grad rel max|d|={dg / scale:.2e} |dW|(5 Adam steps, info)={dw:.2e} "
          f"loss cpu {outs['cpu']['loss']:.5f} gpu {outs['cuda']['loss']:.5f} "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


def overfit(cfgname, dev, epochs=80):
    cf = CFGS[cfgname]
    cdir = cfg_dir(cfgname)
    lock_runs, runs240, folds, _ = D.dev_pool()
    D.assert_no_lockbox(folds[0][:4], "overfit", cdir / "leakguard.log")
    runs = sorted(folds[0])[:4]
    set_seeds(SEED)
    st = Store(runs, cf["ctx"], cf["target"], dev)
    model = M.Net(n_iso=len(D.iso_vocab())).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    scaler = torch.amp.GradScaler("cuda")
    focal = M.Focal()
    pool = st.valid_j.cpu().numpy()          # ALL seconds, no subsampling
    t0 = time.time()
    for ep in range(epochs):
        rng = np.random.default_rng([SEED, ep])
        p = torch.from_numpy(rng.permutation(pool)).to(dev)
        model.train()
        tot, n = 0.0, 0
        for i in range(0, len(p), cf["batch"]):
            j = p[i:i + cf["batch"]]
            with torch.autocast("cuda", dtype=torch.float16):
                dl, cl, il = model(windows(st, j), cf["ctx"])
                hm = st.y[j] == 1
                loss = focal(dl, st.tgt[j])
                if hm.any():
                    loss = loss + 0.5 * TF.cross_entropy(
                        cl[hm], st.tcat[j][hm].long()) + 0.5 * TF.cross_entropy(
                        il[hm], st.tiso[j][hm].long())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            scaler.step(opt)
            scaler.update()
            tot += float(loss) * len(j); n += len(j)
        if ep % 10 == 9 or ep == epochs - 1:
            vl, auc, ap = evaluate(model, st, cf, dev, focal)
            log_line(cfgname, f"overfit ep{ep + 1}: train {tot / n:.4f} "
                     f"eval auc {auc:.4f} ap {ap:.4f} "
                     f"({time.time() - t0:.0f}s)")
            if tot / n < 0.01:
                break
    ok = tot / n < 0.05
    log_line(cfgname, f"overfit verdict: {'PASS' if ok else 'FAIL'} "
             f"(final train loss {tot / n:.4f})")
    return ok


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["build", "timing", "devparity", "overfit",
                             "train", "eval"])
    ap.add_argument("--cfg", default="c64h")
    ap.add_argument("--fold", default="all")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--ovf-epochs", type=int, default=80)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if a.mode == "build":
        D.build()
        return
    dev = torch.device(a.device)
    if a.device == "cuda":
        torch.cuda.init()
    if a.mode == "timing":
        timing(a.cfg, dev, a.epochs)
    elif a.mode == "devparity":
        ok = devparity(a.cfg)
        raise SystemExit(0 if ok else 1)
    elif a.mode == "overfit":
        ok = overfit(a.cfg, dev, a.ovf_epochs)
        raise SystemExit(0 if ok else 1)
    elif a.mode == "train":
        folds = range(5) if a.fold == "all" else [int(a.fold)]
        for k in folds:
            train_fold(a.cfg, k, a.epochs, dev)
    elif a.mode == "eval":
        folds = range(5) if a.fold == "all" else [int(a.fold)]
        for k in folds:
            eval_fold(a.cfg, k, dev)


if __name__ == "__main__":
    main()
