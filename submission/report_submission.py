# -*- coding: utf-8 -*-
"""
report_submission.py -- evidence report for the final submission CSVs
(Checkpoint C decision 4).  Read-only: parses each written file, prints row
counts, rows/run, runs with no alarms, time bounds vs run length (ms), label
histogram, metric range, md5, and a pandas dtype reload; then proves the
cross-file nestedness low subset-of mid subset-of high (and *_safe subset-of
base) on the exact tuple (run_id, time, label_1, metric_1).

Run from the repo root:
  .venv\\Scripts\\python.exe submission\\report_submission.py
"""
from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import pandas as pd

import make_submission as MS

ROOT = Path(__file__).resolve().parent.parent
SUB = ROOT / "submission"
cc = pickle.load(open(ROOT / "_scratch" / "sub_testing_scores.pkl", "rb"))
RL = MS.run_lengths_ms(cc)          # run_id -> detector-coverage end, ms

FILES = ["submission_mx31_v1_fpr_low.csv",
         "submission_mx31_v1_fpr_mid.csv",
         "submission_mx31_v1_fpr_high.csv",
         "submission_mx31_v1_fpr_low_safe.csv",
         "submission_mx31_v1_fpr_mid_safe.csv"]

keysets = {}
for f in FILES:
    p = SUB / f
    md5 = hashlib.md5(p.read_bytes()).hexdigest()
    df = pd.read_csv(p)
    rl = df["run_id"].map(RL)
    keysets[f] = {tuple(x) for x in
                  df[["run_id", "time", "label_1", "metric_1"]].values.tolist()}
    print(f"=== {f} ===")
    print(f"  rows: {len(df)} ({len(df) / 300:.2f}/run) | runs with >=1 alarm: "
          f"{df['run_id'].nunique()} | runs with NO alarms: {300 - df['run_id'].nunique()}")
    print(f"  time ms: min={df['time'].min()} max={df['time'].max()} | "
          f"run_length ms: min={min(RL.values())} max={max(RL.values())} | "
          f"smallest (run_length - time) = {int((rl - df['time']).min())} | "
          f"time_stop clamped rows: {int(((df['time'] + 2000) > rl).sum())} | "
          f"time_start clamped rows: {int((df['time'] < 2000).sum())}")
    print(f"  metric_1: min={df['metric_1'].min():.6f} "
          f"max={df['metric_1'].max():.6f}")
    print("  label histogram: " + ", ".join(
        f"{k}={v}" for k, v in df["label_1"].value_counts().items()))
    print(f"  md5: {md5}")
    print("  pandas dtypes: " + ", ".join(
        f"{k}:{v}" for k, v in df.dtypes.items()))
    print()

print("=== cross-file nestedness (tuple run_id/time/label_1/metric_1) ===")
PAIRS = [("submission_mx31_v1_fpr_low.csv", "submission_mx31_v1_fpr_mid.csv"),
         ("submission_mx31_v1_fpr_mid.csv", "submission_mx31_v1_fpr_high.csv"),
         ("submission_mx31_v1_fpr_low_safe.csv", "submission_mx31_v1_fpr_low.csv"),
         ("submission_mx31_v1_fpr_mid_safe.csv", "submission_mx31_v1_fpr_mid.csv")]
all_ok = True
for a, b in PAIRS:
    ka, kb = keysets[a], keysets[b]
    ins, extra = ka <= kb, len(kb - ka)
    strict = ins and extra > 0
    all_ok &= strict
    print(f"  {a:38s} subset of {b:38s}: {ins} | "
          f"extra rows in bigger file: {extra} | STRICT: {strict}")
print("NESTEDNESS:", "ALL STRICT SUBSETS OK" if all_ok else "PROBLEM")
