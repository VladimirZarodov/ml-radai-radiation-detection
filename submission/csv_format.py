# -*- coding: utf-8 -*-
"""
csv_format.py -- THE SINGLE CONFIG PLACE for the submission CSV layout.

STATUS: UNCONFIRMED PLACEHOLDER FORMAT (user-confirmed "use placeholder
format", Checkpoint C decision 1, based on the portal's example row).
The server validates the file after upload; keep verify_submission.py checks
in sync if this layout changes.

When a different template ever arrives, update ONLY this file:
  * HEADER / COLUMNS: exact header line and column names,
  * format_row(): how an alarm row maps onto those columns,
  * parse_row() / IDX: used by verify_submission.py for column checks.

Columns (Checkpoint C decision 1):
  #         : 1-based row number
  run_id    : testing run index (0..299)
  time      : alarm time, integer ms from run start (mx31 peak window centre)
  time_start: NUMERIC  max(0, time - WINDOW_MS)
  time_stop : NUMERIC  min(run_length_ms, time + WINDOW_MS)
              (portal's own example uses time +/- 2 s)
  label_1   : official label (one of 24); never blank in our files
  metric_1  : mx31 score, 6 decimals
  label_2 / label_3 : EMPTY  (only columns allowed to be empty)
  metric_2 / metric_3 : 0
"""

COLUMNS = ["#", "run_id", "time", "time_start", "time_stop",
           "label_1", "metric_1", "label_2", "metric_2",
           "label_3", "metric_3"]

#: Header line AS WRITTEN.  The '#' column name is QUOTED because the
#: organiser's reader is pd.read_csv(path, comment="#") (radai/evaluation/
#: evaluation.py) -- an unquoted leading '#' makes pandas treat the ENTIRE
#: header row as a comment, destroying the columns (verified empirically
#: 2026-10-02).  A quoted field keeps the column name '#' and round-trips
#: both with and without comment="#".
HEADER = '"#",run_id,time,time_start,time_stop,label_1,metric_1,' \
         "label_2,metric_2,label_3,metric_3"

#: index of each field when reading a written row back with csv.reader
IDX = {name: i for i, name in enumerate(COLUMNS)}

#: half-width of the alarm window written into time_start/time_stop (ms)
WINDOW_MS = 2000

#: the ONLY columns that may be empty in a written row
ALLOWED_EMPTY = {"label_2", "label_3"}


def format_row(i, run_id, time_ms, label, metric, run_len_ms):
    """One data row (1-based serial `i`).  Column mapping lives HERE only.
    run_len_ms: total detector length of that run in integer ms, used to
    clamp time_stop (time + 2000 would otherwise spill past the run end)."""
    t = int(round(time_ms))
    ts = max(0, t - WINDOW_MS)
    te = min(int(round(run_len_ms)), t + WINDOW_MS)
    return (f"{i},{run_id},{t},{ts},{te},{label or ''},"
            f"{metric:.6f},,0,,0")


def parse_row(fields):
    """Reverse of format_row for verification tooling."""
    return dict(run_id=int(fields[IDX["run_id"]]),
                time_ms=int(fields[IDX["time"]]),
                time_start=int(fields[IDX["time_start"]]),
                time_stop=int(fields[IDX["time_stop"]]),
                label_1=fields[IDX["label_1"]],
                metric_1=float(fields[IDX["metric_1"]]))
