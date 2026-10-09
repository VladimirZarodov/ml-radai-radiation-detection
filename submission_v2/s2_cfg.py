# -*- coding: utf-8 -*-
"""s2_cfg.py -- Stage-2 config table (single source of truth for chain/builder/
trainer/metrics).  Recipe fields mirror c64s exactly; only aug flags differ.
theta_aug resolution: Step-0 writes cache/s2/calib.json {kappa_max, theta};
  s2-thin/s2-all use theta_aug = theta if theta < 1 else 0.5 (pre-registered
  fallback, PLAN 2026-10-06) -- resolved AT BUILD TIME and echoed into the
  build stats json so the trained config's actual parameter is on record.
"""
from __future__ import annotations

import json
from pathlib import Path

import v2_lib as V

SEED2 = 20261006           # training seed base (c64s used 20261003)
SEED_AUG = 20261006        # aug rng base
SEED_CAL = 20261007        # eval-shift rng base (disjoint streams)

S2 = V.CACHE / "s2"
S2.mkdir(exist_ok=True)

# Rebuilt decomposition with per-instance hoods FIXED (E046); original
# train.h5 stays untouched but must NOT be read by Stage-2 code.
TRAIN_FIX = V.CACHE / "train_fix.h5"

BASE = dict(ctx=64, target="soft", batch=192, epochs=12)
KAPPA_MAX_AUG = 1.7        # design-fixed training-aug compression cap
P_SPD = 0.5                # Bernoulli prob to compress an encounter
BG_AUG = (1.00, 1.34)      # per-run bg factor range (mean 1.17)

CFGS = {
    "s2-0":    dict(slot=0, aug={}),
    "s2-spd":  dict(slot=1, aug={"spd": KAPPA_MAX_AUG}),
    "s2-thin": dict(slot=2, aug={"thin": "theta_aug"}),
    "s2-bg":   dict(slot=3, aug={"bg": True}),
    "s2-all":  dict(slot=4, aug={"spd": KAPPA_MAX_AUG, "thin": "theta_aug",
                                 "bg": True}),
}
ORDER = ["s2-0", "s2-spd", "s2-thin", "s2-bg", "s2-all"]


def theta_aug():
    """Augmentation thinning lower bound (see module docstring)."""
    cal = calib()
    th = float(cal.get("theta", 1.0)) if cal else 0.5
    return th if th < 1.0 else 0.5


def calib():
    p = S2 / "calib.json"
    return json.loads(p.read_text()) if p.exists() else None


def aug_spec(cfgname):
    """Resolve aug flags for a config AFTER Step-0: dict with optional keys
    spd=(kmax,), thin=(theta_aug,), bg=('draw',).  s2-0 -> {} (identity)."""
    a = CFGS[cfgname]["aug"]
    out = {}
    if "spd" in a:
        out["spd"] = (float(a["spd"]),)
    if "thin" in a:
        out["thin"] = (theta_aug(),)
    if "bg" in a:
        out["bg"] = ("draw",)
    return out


def seed_for(cfgname, fold):
    """Identical across all s2 configs (init hygiene); differs from c64s only
    by SEED base (len/soft terms equal to c64s: len 4, soft 1)."""
    return (SEED2 * 7919 + fold * 101 + 4 * 7907 + 31337) % 2 ** 31


def feat_path(cfgname):
    return V.CACHE / ("feat.h5" if cfgname == "s2-0"
                     else f"s2/feat_{cfgname}.h5")
