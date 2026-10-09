# ML-RADAI: from a matched filter to a temporal CNN on RADAI v4.3

[English](README.md) · [Русский](README.ru.md)

This repository studies **detection and identification of gamma-ray sources** in the
[RADAI urban-search simulation dataset](https://bdc.lbl.gov/wiki/public/radai-interactive-datasets/)
(ORNL/LBNL, v4.3). A vehicle-mounted NaI(Tl) detector drives through a simulated city, and the task
is to raise an alarm when it passes a hidden source, to name the isotope, and to keep the false-alarm
rate within a practical budget. Everything is scored with the organisers' **official metrics**:
detection recall `d_recall` at a fixed number of false alarms per background hour (`fpr`).

The main deliverable is the notebook [`RADAI_pipeline.ipynb`](RADAI_pipeline.ipynb) (executed, outputs
embedded; the text is in Russian, code comments are in English). It contains a line-by-line replica of
the official scorer and answer-key builder; the v1 physics baseline (Poisson matched filter with 7
background components, 61 isotope templates and a 62 s rolling maximum); a frozen validation harness
(5-fold CV by run on 240 runs, a sealed lockbox of 60 runs, bootstrap by run); a temporal CNN (**c64s**,
547k parameters) and its variant trained with speed-compression augmentation (**s2-spd**); the generation
of the portal CSVs; and three measurements on the blind testing set through the official portal.

## Main results

All intervals are 95 % bootstrap intervals over runs. `d_recall` is the official detection recall.

**Development OOF** (240 runs, 229.4 background hours, out-of-fold predictions):

| fpr target, 1/h | v1 (mx31) | c64s | s2-spd |
|---|---|---|---|
| 0.085 | 0.1199 [0.1062–0.1324] | 0.4435 [0.4199–0.4665] | **0.4481 [0.4225–0.4723]** |
| 0.17 | 0.1287 [0.1143–0.1420] | 0.4926 [0.4682–0.5150] | **0.5014 [0.4779–0.5247]** |
| 0.35 | 0.1481 [0.1332–0.1625] | 0.5347 [0.5123–0.5553] | **0.5440 [0.5183–0.5667]** |
| 0.70 | 0.1657 [0.1499–0.1812] | 0.5736 [0.5525–0.5948] | **0.5806 [0.5566–0.6019]** |

**Sealed lockbox** (60 runs, 57.45 background hours; spent once on 2026-10-06; thresholds transferred from dev;
only 0.70 was used for conclusions, 0.35 is indicative, 0.085 and 0.17 are report-only because they hold 5–13 false alarms):

| fpr target, 1/h | v1 (mx31) | c64s | s2-spd |
|---|---|---|---|
| 0.085 | 0.0981 [0.0734–0.1238] (fpr 0.087, FP 5) | 0.4519 [0.4049–0.5000] (fpr 0.104, FP 6) | 0.4407 [0.3892–0.4948] (fpr 0.139, FP 8) |
| 0.17 | 0.1130 [0.0856–0.1407] (fpr 0.174, FP 10) | 0.4870 [0.4428–0.5332] (fpr 0.139, FP 8) | 0.5000 [0.4485–0.5509] (fpr 0.226, FP 13) |
| 0.35 | 0.1389 [0.1101–0.1686] (fpr 0.296, FP 17) | 0.5278 [0.4831–0.5738] (fpr 0.296, FP 17) | 0.5519 [0.5027–0.6015] (fpr 0.453, FP 26) |
| 0.70 | 0.1593 [0.1309–0.1903] (fpr 0.714, FP 41) | 0.5722 [0.5311–0.6131] (fpr 0.696, FP 40) | 0.5833 [0.5358–0.6296] (fpr 1.010, FP 58) |

**Blind testing set, official portal** (upload time as shown by the portal):

| # | date | model | file rows | `d_recall` | `fpr`, 1/h | retention (testing/dev) |
|---|---|---|---|---|---|---|
| 1 `first_try` | 2026-10-02 | v1, thr 5.7492 | 413 | 0.0946 | 0.535 | 0.65 |
| 2 `second_try` | 2026-10-04 | c64s, thr 0.8617 | 727 | 0.2530 | 0.0973 | 0.570 ± 0.045 |
| 3 `third_try` | 2026-10-06 | s2-spd, thr 0.8536 | 683 | 0.2402 | 0.0730 | 0.536 [0.491–0.585] |

What the numbers say:

* Both CNNs beat the v1 matched filter decisively: the lockbox difference in `d_recall` is +0.34…+0.42, with lower
  confidence bounds ≥ +0.29.
* **s2-spd is not separated from c64s**: dev +0.005…+0.011, lockbox +0.011 [−0.030, +0.050] at 0.70, portal −0.013 (≈1.1σ).
  At 0.70 the s2-spd lockbox run also fired at an actual fpr of 1.01 versus 0.70 for c64s, so its small
  `d_recall` advantage there was bought with more false alarms. Two seeds of the same training recipe differ by 0.061
  in dev `d_recall` at 0.085, so differences below that size are not significant.
* Going from dev to the blind testing set, the networks keep about 55 % of `d_recall` while their fpr barely
  inflates (≈1.2 and ≈0.9; v1 had ×2.4): the loss comes from missed weak and fast encounters, not from a flood of false alarms.
* The first internal estimate for v1 (25.2 / 40.5 / 70.0 % at 1 / 3 / 10 alarms per hour, archived notebook) used a
  more lenient criterion (±60 s tolerance, 150 s exclusion around encounters) and is **not comparable** with the official metric.
* Leaderboard context (snapshot of 2026-10-06; leaderboards change): the organisers' `baseline_nmf` row has Global `d_recall` 0.298 at
  fpr 0.122. In the table with fpr < 0.125 our rows #2 and #3 rank third and fourth by Global `d_recall`, behind one participant row and the NMF baseline.

### Where the models fail

Recall depends on how fast the platform passes the source. Real (not synthetic) speed terciles on dev, `d_recall` at fpr 0.085
(testing reaches 13.4 m/s while training stops at 8.0 m/s):

| speed tercile (v at CA) | n | c64s | s2-spd |
|---|---|---|---|
| ≤ 4.4 m/s | 715 | 0.562 | 0.564 |
| 4.4–6.8 m/s | 724 | 0.445 | 0.421 |
| > 6.8 m/s | 721 | 0.325 | 0.361 |

By category at the portal (`d_recall`):

| portal row | Global | Industrial | Medical | NORM | Nuclear Material |
|---|---|---|---|---|---|
| #2 c64s, `d_recall` | 0.253 | 0.337 | 0.156 | 0.048 | 0.287 |
| #3 s2-spd, `d_recall` | 0.240 | 0.271 | 0.153 | 0.009 | 0.290 |

NORM is detected poorly by every model (the spectra are close to the background). For Medical sources the category
accuracy collapses on testing (`c_recall`/`d_recall` ≈ 0.93 on dev, 0.43 and 0.55 at the portal): these encounters are detected but
labelled outside the Medical classes. Cu-67, Lu-177 and several shielded configurations are absent from the training data;
this is the leading hypothesis, not a measured cause.

![Summary of the results](figures/fig_summary.png)
<!-- VERIFY: open figures/fig_summary.png and fix the caption/alt text so it matches what the figure shows -->

![Sealed lockbox at the primary target](figures/fig_lockbox_primary.png)
<!-- VERIFY: same for figures/fig_lockbox_primary.png -->

## Notebook contents

| section | topic |
|---|---|
| §0 | Task, data structure, glossary, related work, overview (self-contained introduction) |
| §1–§2 | Data files; official scoring rules and a replica of the answer-key builder, with a parity check against the organisers' scorer |
| §3 | v1 matched-filter baseline and its development tables |
| §4–§5 | Decomposed 1-second caches; frozen validation harness (splits, lockbox, bootstrap) |
| §6–§7 | Temporal CNN c64s; out-of-fold tables, speed terciles, time-shift control, leakage audit |
| §8 | Speed-compression augmentation s2-spd and the ablation of augmentations |
| §9 | Generation of the portal CSVs; byte-for-byte reproduction of the uploaded files |
| §10–§11 | Sealed lockbox (results are only loaded); the three portal measurements and the summary |
| §12 | Attribution, what is original, limitations |

## Getting the data

The three HDF5 files are not included (about 75 GB in total; `*.h5` is git-ignored).

1. Register at [bdc.lbl.gov](https://bdc.lbl.gov/register) (academic requests are usually approved within about a day).
2. Download `training_v4.3.h5`, `developer_v4.3.h5` and `testing_v4.3.h5` from the
   [dataset page](https://bdc.lbl.gov/wiki/public/radai-interactive-datasets/).
3. Save them in the repository root, next to `RADAI_pipeline.ipynb`.

## Running the notebook

Python 3.11; package versions are pinned in `requirements.txt` because some findings are version-sensitive.

```
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # Linux / macOS
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace RADAI_pipeline.ipynb     # or run interactively
```

`requirements.txt` pins the CPU wheel of PyTorch; the results were produced with the CUDA wheel of the same version
(`pip install torch==2.14.0+cu126 --index-url https://download.pytorch.org/whl/cu126`). A GPU (RTX 3060 Laptop, 6 GB) is needed only in §9.
On Windows, `torch` must be imported before `numpy`/`pandas`/`h5py` (a DLL load-order problem); the first code cell does this.

Switches in the first code cell:

* `REPRODUCE = True` (default): numbers are recomputed from frozen caches, saved predictions, checkpoints and measurement records
  (≈16 min on the author's machine) and compared with saved results; six verifications are printed at the end of §11.
* `FULL = True`: additionally rebuilds the derived caches from the `.h5` files. <!-- VERIFY: state the measured time and exactly what a fresh clone needs (caches are not in git) -->
* `LOCKBOX_RECOMPUTE = False`: the lockbox is spent; §10 only loads the saved results.

**Model training is never executed by the notebook.** The training and augmentation code lives in `submission_v2/`
(`s1_train.py`, `s2_train.py`, launched through `s1_run.py`); the exact commands and every run, including failures,
are in `results/logs/EXPERIMENTS.md`. The checkpoints that produced the portal measurements are in `models/`.

## Repository layout

```
RADAI_pipeline.ipynb     the deliverable
README.md / README.ru.md
requirements.txt
submission/              scoring replica: official_scorer.py, radai_lib.py, csv_format.py
submission_v2/           library imported by the notebook; reports/ holds the frozen evidence it reads
models/                  fold checkpoints of the two finalists (c64s, s2-spd)
results/                 uploaded portal CSVs, pre-registration and experiment logs
figures/                 figures used in this README
archive/                 the earlier v1 study (original notebook and README), read-only
```

## References

* Jones et al., *Adaptive NMF* (LBNL), [arXiv:2507.10715](https://arxiv.org/abs/2507.10715) — adaptive background model with regularisation against NORM-like shapes.
* Bachleda et al., *waterfall CNN* (PNNL), [arXiv:2607.00270](https://arxiv.org/html/2607.00270v3) — convolutional network on raw spectral-temporal input.
* RADAI dataset paper, IEEE Trans. Nucl. Sci., 2026, [doi:10.1109/TNS.2026.3682654](https://doi.org/10.1109/tns.2026.3682654).

Numbers from these papers are not directly comparable with ours: the definitions of the false-alarm rate, the encounter
boundaries and the data splits differ.

## Dataset and attribution

RADAI is a synthetic urban-search gamma-ray dataset produced by ORNL and LBNL (funded by DOE NA-22) and released under CC-BY 4.0;
please cite the dataset paper above (the dataset page also lists ORNL/TM-2021/3265). The organisers' scoring package
(`radai`, BSD-3-Clause, © University of California / LBNL / ORNL) is not included; `submission/official_scorer.py`
mirrors its evaluation rules and keeps its copyright notice for the label tables copied from it.