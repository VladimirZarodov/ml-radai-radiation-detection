# LOCKBOX_LOG.md -- PRE-RUN FROZEN RECORD (lockbox touched ONCE per finalist AFTER this file was written)

Written: 2026-10-06T23:05:56+03:00 (before ANY lockbox computation in this session).
Disk at record time: D: 4.27 GB free / C: 16.80 GB free.

## Finalists (fixed by user brief, 2026-10-06; NO additions)
v1 (physics matched filter), c64s (Stage-1 selection), s2-spd (Stage-2 selection). No ensembles, no extra seeds, no re-selection ever.

## Lockbox pool (from frozen cache/lockbox.json via s1_data.dev_pool)
n = 60 runs:
```
[3, 5, 18, 21, 22, 23, 25, 29, 30, 31, 34, 35, 38, 40, 41, 47, 49, 66, 68, 69, 72, 81, 85, 88, 89, 90, 96, 102, 106, 113, 121, 125, 126, 127, 134, 135, 145, 156, 157, 158, 166, 172, 200, 207, 211, 213, 214, 230, 236, 237, 244, 250, 251, 253, 265, 276, 283, 285, 288, 298]
```
Background hours of these 60 runs (harness meta, at record time): **60.44 h** (PLAN power table assumed 57.45 h).

## Pre-registered rule (verbatim from PLAN.md + user brief; frozen here)
1. Thresholds: each model's OWN pooled-dev-OOF threshold at targets 0.085 / 0.17 / 0.35 / 0.70 -- never re-fit on lockbox. Sources: metrics_c64s.json / metrics_s2-spd.json `sel[t].thr`, v1_baseline.json `dev_targets[t].thr`. Exact values frozen by this record:
   - v1: 0.085: 6.2266287804, 0.17: 5.9116830826, 0.35: 5.6019983292, 0.7: 5.2640399933
   - c64s: 0.085: 0.8617231573, 0.17: 0.8448914571, 0.35: 0.8234114872, 0.7: 0.7961097376
   - s2-spd: 0.085: 0.8535758255, 0.17: 0.8313147875, 0.35: 0.8044557294, 0.7: 0.7797453714
2. Learned models score EACH lockbox run with fold model run_id % 5 (threshold-transfer rule, PLAN 2026-10-04). v1: frozen cache/v1_alarms_train.pkl subset.
3. Alarm extraction FROZEN (s1_eval: gauss sigma 1.5 s, SEP 15 s, PMIN .02, CAP 300/run).
4. CONCLUSIONS ONLY at target 0.70 (PRIMARY) and 0.35 (SECONDARY/indicative); 0.085/0.17 REPORTED but must not drive any decision (PLAN 2026-10-03 power amendment). Any comparison that would flip by using .085/.17 is invalid.
5. CIs: run-cluster bootstrap B=1000 (v2_lib, fixed seeds: 7 global, 11 paired, per-cat same as global); paired diffs v1-referenced inside each json + the 3 pairwise finalists in the report script.
6. Real speed terciles: |v| at CA (searchsorted on training_v4.3.h5 detector/position, frozen s1_diag/s2_metrics convention), EDGES PINNED 4.4 / 6.8 m/s. Lockbox has only 60 runs -> LOW POWER, disclosed.
7. One touch: each finalist exactly one lockbox run; rerun allowed ONLY if a crash produced NO output (must be logged); NEVER after results are seen. Raw per-run alarms/scores/counts under reports/lockbox/<cfg>/.

## Expected lockbox FP counts per target (PLAN 2026-10-03 power table)
| target fpr | expected (target x 57.45 h) | v1 observed @ dev thr |
|--|--|--|
| 0.085 | ~4.3 | 4 |
| 0.17 | ~9.8 | 12 |
| 0.35 | ~20.1 | 22 |
| 0.7 | ~40.2 | 45 |

## Dev dry-runs via THIS code path, completed BEFORE this record
v1 / c64s / s2-spd: `DRY-RUN <cfg>: max|d - reference| = 0.000000 -> PASS` (DEV-240 fold-true OOF; artifacts reports/lockbox/dryrun_dev/, threshold check against v1_baseline.json / metrics_<cfg>.json).

## SHA-256 of everything used (frozen from this moment)
| file | sha256 |
|--|--|
| submission_v2/s2_lockbox.py (script) | 5aa731f198b9920397767767883139603acecb84c925da8bd213ed2ee6945c64 |
| submission_v2/s2_lockbox_report.py (script) | 67c5dde443fb23790e6f3bada7aa5132ec51cffc7e23f8b2ee43660c52040219 |
| submission_v2/s1_run.py (script) | 6a188fe93b069f04985778b98e02cf1b8c488061f134c9338ae40d17579689ea |
| submission_v2/s2_cfg.py (script) | 0cc3a804aad94ed1d9b2e65195e9a183158a0872cd0c4fb760deb6893e9c3341 |
| submission_v2/v2_lib.py (script) | 0201600cba8fc92f93fac4c6d01f4ec64d3497bb3f1a8fa250e0f3f3ebdffe0a |
| submission_v2/s1_data.py (script) | 3f945911d92899f0b187807b439c788d0ccb7eb8649c7ec244376932a89f8425 |
| submission_v2/s1_eval.py (script) | 59e18f31732ea71a2357a1a7de98bf172b2cf89c4ad17b492da4b05af7ebfc14 |
| submission_v2/s1_model.py (script) | 4e6c4c0ae0091aa0bd7530739da5d789371b46b69762253b6af71f0519a2de4f |
| submission_v2/s1_train.py (script) | e69e32d094883bf0adf1d12e7532bda1d817f9a1426ad7189d8a4af619872305 |
| submission_v2/cache/models/c64s/fold0.pt (checkpoint) | 229377082255be63ba40be7f189bd485f2972d26d46754b293d4b32fd253b16d |
| submission_v2/cache/models/c64s/fold1.pt (checkpoint) | d0695051d0ba6a6ad54fa8010011acfdceb1b9237db1fcc65f65af5db427146e |
| submission_v2/cache/models/c64s/fold2.pt (checkpoint) | 9f15ace9e9013cf1e65e42bcedaf7aa178e7d9bdef50f78e16f39d1eae267653 |
| submission_v2/cache/models/c64s/fold3.pt (checkpoint) | e4f5320c3a1c2f6193d67fcde43d87f2337d16154c54ba4d8274a6e413bcd4b0 |
| submission_v2/cache/models/c64s/fold4.pt (checkpoint) | b27be07c9a1018719703ec6163869ae32296522e085dc3ff3896054568a4b1fe |
| submission_v2/cache/models/s2-spd/fold0.pt (checkpoint) | 561a661decce57359a7467af0172484756b11bd05f13c174d47a18730c29b8a2 |
| submission_v2/cache/models/s2-spd/fold1.pt (checkpoint) | a10877c0faf781309794393c858b1ed11e9e60e59327a3389345476ac057d3c4 |
| submission_v2/cache/models/s2-spd/fold2.pt (checkpoint) | b8edd562dcb616c639f094acaab30a4a740fa9ba1542f934aaa8f9bfcc692b68 |
| submission_v2/cache/models/s2-spd/fold3.pt (checkpoint) | daad01481275c81ae6fd039bb1ce5a4c3b6f0082f4d6cdaadd7d9f0ea0de3ee2 |
| submission_v2/cache/models/s2-spd/fold4.pt (checkpoint) | 6f16ae2c4b32000f191a99c22ff71bedff4bc13c3bf215f324069b26339ab484 |
| submission_v2/cache/s2/metrics_c64s.json (threshold json) | 32b2a3d57e4b6e65ee96fee9744fe7b186577fa9689269a65692fd3f7acfbd99 |
| submission_v2/cache/s2/metrics_s2-spd.json (threshold json) | e23d0e2a788b2c04164c2c552b946a6039b2d5098c3f9e5b167a0ec64e0fc75d |
| submission_v2/reports/v1_baseline.json (threshold json) | f68cc954976c81e417e4604abc21b88af4c5e156fcec580320ecb6e1a32ed92a |
| submission_v2/cache/lockbox.json (data) | 280c28c339a75b13f761e63e796d7eef98e080dbe235c4e0a66177d2fdaf610f |
| submission_v2/cache/feat.h5 (data) | 57aa0887548eb570c28a6cf3184c10e85063fa496898bea5cfbaf8335aca4112 |
| submission_v2/cache/v1_alarms_train.pkl (data) | be809471c169b1e546a7f02ccb2124799113d25d80e9a8be6d61cb58ba0fa878 |
| submission_v2/cache/encounters_train.csv (data) | a8ca02840790ba38fc698d0ada36e36155165964559d77544dabaf89905e039e |
| training_v4.3.h5 (data) | a39d5a4eacabea3d698dc39f04cd19c108d38a8ab7b0dc964ba508a03deb1ad5 |

Optional later input (data file, NOT code): reports/lockbox/third_try.json -- if the user confirms portal #3, it is written as a plain dict with keys tryname/model/thr/d/fpr/rows/md5/note; the report script reads it verbatim. Its absence leaves the portal row PENDING.

NO code, threshold or rule may change after this record.

## POST-RECORD EXCEPTION (appended 2026-10-06T23:23:11+03:00 -- full disclosure, frozen body untouched)
- FIRST real touch attempt (`--cfg v1 --lockbox --yes`, ~23:06:50) CRASHED
  with NO output: `TypeError: len() of unsized object` in V.boot_at_row --
  s1_data.dev_pool() returns the 60 lockbox ids as a SET; boot_at_row does
  np.asarray(runs) (0-d object array). Verified on disk before any rerun: no
  reports/lockbox real artifacts and no cache/s2/lockbox_v1.json existed
  (crash log submission_v2/logs/lb_v1.log). ZERO lockbox results were seen.
- Brief rule invoked: 'crashes before producing output -> may rerun only to
  complete it (log it, change nothing)'. Completion required one mechanical
  type coercion, ONE LINE in main():
  `lock = sorted(int(r) for r in lock)`  after `D.dev_pool()`.
  NO threshold, rule, data, model or protocol value changed: same 60 ids,
  deterministic order; downstream lookups are value-keyed, alarms sorted,
  bootstrap seeds unchanged.
- NEUTRALITY PROVED: re-ran all three DEV dry-runs with the patched code --
  v1/c64s/s2-spd all `max|d - reference| = 0.000000 PASS` and the
  regenerated reports/lockbox/dryrun_dev/lockbox_dryrun_*.json are
  BYTE-IDENTICAL to the pre-record artifacts (sha256 c64s
  7B4422EB551C99F3B682F59CD78F602BFB331E8BB787DE5B8B155AA816F75050, s2-spd
  6CC3584B0F29D96A33B17A13664F253B14E028AF52884FF920451BEBE5E7DDDF, v1
  AAA10DFEE66E718C757D430AD2292E5A55A6E602FD62D5418B77B7FEAF51637D --
  same values before and after).
- Patched-file hash (supersedes the table above for THIS ONE file only):
  s2_lockbox.py = 9F21EE8BB3EBC6E0E31E4571B196616861CF62EA9E5266E8D91C6BFE0F5DB738
- The v1 real run from now on is the logged COMPLETION rerun; c64s and
  s2-spd real runs remain first-and-only touches.
