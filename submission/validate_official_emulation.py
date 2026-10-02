# -*- coding: utf-8 -*-
"""
validate_official_emulation.py -- emulate the ORGANISER'S validator.

The repo contains the official evaluation package (radai/radai/evaluation/):
  * evaluation.py   : validate_submission(), compute_metrics()
  * visualization.py: calculate_fp_info()   (match + FP rules)
  * tools.py        : label vocabulary, categories
(The package itself can't be imported in this venv -- numba missing -- so the
rules below are copied VERBATIM from that source.)

Official rules extracted (2026-10-02):
  * submission read with pd.read_csv(path, comment="#")  -> an UNQUOTED '#'
    first header cell makes pandas drop the WHOLE header row (required
    columns lost: FATAL, discovered by this emulation 2026-10-02); our
    HEADER therefore writes the cell quoted as '"#"', which survives both
    with and without comment="#" (verified empirically);
  * REQUIRED columns: run_id, time, label_1, metric_1; time_start/time_stop
    are used by compute_metrics (variation lookup, duration penalty), so we
    keep them numeric;
  * run_id: castable int, 0 <= run_id <= answer-key max (299 for testing);
  * time: castable float, 0 <= time <= ak.time.max() + 122000 ms
    (hard-coded for v4.3; answer key unknown -> we can only check the low end
    and note the risk);
  * label_1: exact membership in all_sources (note: Am-241 is INDUSTRIAL,
    DU/DepletedU/NatU/... are NuclearMaterial, Background/BKG are legal);
  * metric_1: castable float;
  * MATCHING (calculate_fp_info): an alarm is TRUE-POSITIVE if its `time`
    falls inside [time_start, time_stop] of an answer-key encounter (same
    run); highest metric_1 among matched alarms sets that encounter's
    label/detection; extra matched alarms are free;
  * FALSE POSITIVE if the time falls in NO encounter window; penalty count
    = 1 + floor((time_stop - time_start)/160 s)  -> our 4 s alarms count 1;
  * fpr (per label category and Global) = FP count / total_time_bkg_only_hr,
    bkg_only = sum(run lengths) - sum(ak time_stop - time_start);
    CONSISTENT with the empirically found ~286.5-h testing denominator;
  * metrics.loc['Global','far'] sums fpr only over alarm_categories =
    ['NuclearMaterial'] (NM-labelled false alarms only).
"""
from pathlib import Path

import pandas as pd

# --- verbatim from radai/radai/evaluation/tools.py -------------------------
category_mapping = {
    "BKG": ["Background", "BKG"],
    "NORM": ["K-40", "Ra-226", "Th-232"],
    "Medical": ["Co-57", "F-18", "Tc-99m", "I-131", "Tl-201",
                "Cu-67", "Sr-90", "Lu-177", "Xe-133"],
    "Industrial": ["Co-60", "Cs-137", "Ba-133", "Ir-192", "Am-241"],
    "NuclearMaterial": ["DepletedU", "DU", "NatU", "RefinedU", "LEU",
                        "HEU", "FGPu", "WGPu"],
}
ALL_SOURCES = [s for ll in category_mapping.values() for s in ll]
ALARM_CATEGORIES = ["NuclearMaterial"]
REQUIRED = ["run_id", "time", "label_1", "metric_1"]

SUB = Path(__file__).resolve().parent
FILES = ["submission_mx31_v1_fpr_low.csv", "submission_mx31_v1_fpr_mid.csv",
         "submission_mx31_v1_fpr_high.csv", "submission_mx31_v1_fpr_low_safe.csv",
         "submission_mx31_v1_fpr_mid_safe.csv"]

all_ok = True
for f in FILES:
    sub = pd.read_csv(SUB / f, comment="#")        # <- official reader
    problems = []
    problems += [f"missing column {c}" for c in REQUIRED if c not in sub.columns]
    if problems:
        status = "PROBLEM"
        all_ok = False
        print(f"{f:38s} {status}  cols read={list(sub.columns)}")
        for p in problems:
            print("   -", p)
        continue
    sub["run_id"] = sub["run_id"].astype(int)
    if not (sub["run_id"].min() >= 0 and sub["run_id"].max() <= 299):
        problems.append("run_id outside [0, 299]")
    sub["time"] = sub["time"].astype(float)
    sub["metric_1"] = sub["metric_1"].astype(float)
    if sub["time"].min() < 0:
        problems.append("negative time")
    bad = sorted({x for x in sub["label_1"] if x not in ALL_SOURCES})
    if bad:
        problems.append(f"labels not in official vocabulary: {bad}")
    if not ({"time_start", "time_stop"} <= set(sub.columns)):
        problems.append("time_start/time_stop missing")
    nm_fp_share = sum(1 for x in sub["label_1"]
                      if x in category_mapping[ALARM_CATEGORIES[0]]) / len(sub)
    status = "OK" if not problems else "PROBLEM"
    all_ok &= not problems
    print(f"{f:38s} {status}  rows={len(sub)}  "
          f"cols={len(sub.columns)}  NM-label share={nm_fp_share:.2f}")
    for p in problems:
        print("   -", p)

print("OFFICIAL-RULE EMULATION VALIDATION:", "ALL PASS" if all_ok else "see above")
print("REMINDER: portal's time upper bound = ak.time.max()+122000 ms cannot be")
print("checked locally (testing answer key unknown); our max time is 3,607,000 ms.")
