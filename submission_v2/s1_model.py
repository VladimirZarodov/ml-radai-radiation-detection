# -*- coding: utf-8 -*-
"""
s1_model.py -- Stage-1 architecture: 2D-conv spectral front-end + TCN.

Input (B, 1, T, 132) window of s1_data features (T = 2*ctx+1).  Energy axis
pooled 4x, time axis preserved; per-frame projection -> 96-d; 4 dilated TCN
residual blocks (k=5, dilations 1/2/4/8, receptive field 1+4*2*(1+2+4+8)=121 s
-> ctx 64 nearly covered, ctx 128 partially -- zero-pad context remains).
Heads at the CENTER frame: detection logit (sigmoid), category (4),
isotope (24 official labels).  ~0.92 M params (< 5 M cap; AMP on).

Imbalance handling (reported): focal loss gamma=2 alpha=0.75 on the per-second
detection target (soft targets supported) + per-epoch random negative
sub-sampling (NEG_RATE of background seconds, rotating seed); category /
isotope CE use sqrt-inverse-frequency class weights computed from fold-train
ONLY, on hard-positive seconds.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, f_in=132, width=96, n_cat=4, n_iso=24):
        super().__init__()
        e = f_in
        self.stem = nn.Sequential(
            nn.Conv2d(1, 24, (3, 5), padding=(1, 2)), nn.GroupNorm(4, 24),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),                      # energy pooled EARLY
            nn.Conv2d(24, 32, 3, padding=1), nn.GroupNorm(4, 32), nn.ReLU(),
            nn.Conv2d(32, 48, 3, padding=1), nn.GroupNorm(4, 48), nn.ReLU(),
            nn.MaxPool2d((1, 2)))
        e4 = ((((e - 1) // 2 + 1) - 2) // 2 + 1)      # energy after pools
        self.proj = nn.Linear(48 * e4, width)
        self.dils = (1, 2, 4, 8)
        self.tcn = nn.ModuleList()
        self.gn1 = nn.ModuleList()
        self.gn2 = nn.ModuleList()
        for d in self.dils:
            self.tcn.append(nn.ModuleList([
                nn.Conv1d(width, width, 5, padding=2 * d, dilation=d),
                nn.Conv1d(width, width, 5, padding=2 * d, dilation=d)]))
            self.gn1.append(nn.GroupNorm(4, width))
            self.gn2.append(nn.GroupNorm(4, width))
        self.drop = nn.Dropout(0.05)
        self.det = nn.Conv1d(width, 1, 1)
        self.cat = nn.Linear(width, n_cat)
        self.iso = nn.Linear(width, n_iso)

    def forward(self, x, center):
        """x (B,1,T,F); center = int index of target frame."""
        h = self.stem(x)                               # (B,48,T,E/4)
        B, C, T, E = h.shape
        h = h.permute(0, 2, 1, 3).reshape(B, T, C * E)
        h = torch.relu(self.proj(h)).transpose(1, 2)   # (B,W,T)
        for blk, g1, g2 in zip(self.tcn, self.gn1, self.gn2):
            r = h
            a = self.drop(torch.relu(g1(blk[0](h))))
            a = torch.relu(g2(blk[1](a)))
            h = torch.relu(r + self.drop(a))
        d = self.det(h)[:, 0, center]                 # (B,)
        f = h[:, :, center]                           # (B,W)
        return d, self.cat(f), self.iso(f)


class Focal:
    def __init__(self, gamma=2.0, alpha=0.75):
        self.g, self.a = gamma, alpha

    def __call__(self, logits, target):
        ce = F.binary_cross_entropy_with_logits(
            logits, target.float(), reduction="none")
        pt = torch.exp(-ce)
        w = self.a * target + (1 - self.a) * (1 - target)
        return (w * (1 - pt).pow(self.g) * ce).mean()


def class_weights(counts):
    c = np.asarray(counts, np.float64)
    w = np.where(c > 0, 1.0 / np.sqrt(np.maximum(c, 1)), 0.0)
    return torch.tensor(w / w.sum() * len(w), dtype=torch.float32)


def auc_np(y, p):
    y = np.asarray(y).astype(bool)
    p = np.asarray(p, np.float64)
    npos, nneg = int(y.sum()), int((~y).sum())
    if npos == 0 or nneg == 0:
        return np.nan
    import pandas as pd
    ranks = pd.Series(np.asarray(p, np.float64)).rank(
        method="average").to_numpy()
    return float((ranks[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def prap_np(y, p):
    y = np.asarray(y).astype(bool)
    o = np.argsort(-p, kind="stable")
    ys = y[o]
    tp = np.cumsum(ys)
    fp = np.cumsum(~ys)
    prec = tp / (tp + fp)
    rec = tp / max(int(ys.sum()), 1)
    rec1 = np.concatenate([[0.0], rec])
    return float(np.sum(np.diff(rec1) * prec))


def n_params(model):
    return sum(p.numel() for p in model.parameters())
