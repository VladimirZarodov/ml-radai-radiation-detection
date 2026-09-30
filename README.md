# ML-RADAI: matched-filter detection of gamma-ray sources in urban search

[English](README.md) · [Русский](README.ru.md)

This repository studies **anomaly detection for gamma-ray radiation sources** in the
[RADAI urban-search simulation dataset](https://bdc.lbl.gov/wiki/public/radai-interactive-datasets/)
(`training_v4.3.h5`, ORNL/LBNL). A vehicle-mounted NaI(Tl) detector drives through a simulated
city, and the task is to raise an alarm when it passes a hidden source while keeping the
false-alarm rate (FAR) within a practical budget.

The main deliverable is the notebook [`Matched_Filter_Detection.ipynb`](Matched_Filter_Detection.ipynb).
It implements a Poisson matched-filter detector: a background model with 7 learned components,
61 isotope templates, and a 62 s rolling-maximum aggregation of the per-window score
(**"mx31"**). The alarm threshold is calibrated on background episodes using a
leave-one-run-out pool (**B_loo**), and all confidence intervals are obtained with a
run-cluster bootstrap. The text of the notebook is written in Russian.

## Main results

Test set: runs 25–124, 943 source encounters. A detection is counted if the alarm statistic
exceeds the threshold within ±60 s of the closest approach; an alarm is the onset of an episode
above the threshold. Values in brackets are 95 % bootstrap intervals.

| target FAR, 1/h | LogReg | XGBoost | MLP | single window (bestA) | **mx31 (B_loo)** |
|---|---|---|---|---|---|
| 1  | 6.9 % | 9.9 % | 4.7 % | 21.2 % | **25.2 %** [22.6–27.9] |
| 3  | 17.5 % | 18.6 % | 12.9 % | 28.1 % | **40.5 %** [37.5–44.2] |
| 10 | 38.7 % | 27.7 % | 31.7 % | 37.0 % | **70.0 %** [66.8–73.5] |

The first three columns are window-level classifiers trained on 2 s spectra (§2 of the notebook);
their ROC-AUC on individual windows is only 0.53–0.65.

Because an alarm is defined as an episode, mx31 cannot exceed a FAR of about 13 episodes per hour.
Above that level, a hybrid of mx31 and a per-window MLP is used (OR of the two branches; §3.5):

| target FAR, 1/h | 30 | 50 | 100 |
|---|---|---|---|
| mx31 + MLP, recall | 89.3 % [86.8–92.1] | 96.0 % [93.1–97.6] | 98.9 % [98.1–99.6] |

These intervals come from the full protocol in which both calibration and evaluation runs are
resampled, so they do not include the optimism of selecting thresholds on the test set.

### Where the method fails

Recall depends strongly on the type of source. Two groups are detected poorly at low FAR,
because their spectra have almost the same shape as the uranium and thorium background
components:

| source group | recall at FAR 1/h | recall at FAR 10/h |
|---|---|---|
| NORM (K-40, Th-232, Ra-226) | 9.7 % | 67.7 % |
| U-family (DU, LEU, NatU, RefinedU) | 8.3 % | 54.0 % |
| other (HEU, Pu, line sources, Cs-137, Sr-90, …) | 40.0 % | 81.6 % |

This limitation is discussed in §4.1 and §6 of the notebook. A small waterfall-CNN probe (§4.6),
inspired by Bachleda et al., reaches the level of mx31 on these groups but does not exceed it.
It indicates that the information is present in the raw spectral-temporal representation, and
that the limit is set by the model class rather than by the physics of the problem.

![Recall vs FAR: mx31 (B_loo) with confidence band and the mx31+ML hybrid](figures/01_headline.png)

![Recall by source group at FAR 1/3/10](figures/04_groups.png)

## Notebook contents

| section | topic |
|---|---|
| §0 | Task, dataset schema, glossary, related work, method overview (self-contained introduction) |
| §1 | Why window-level classifiers are limited: non-stationary background, weak source contribution per window |
| §2 | Baselines: LogReg, XGBoost, MLP on window spectra |
| §3 | Matched-filter detector, threshold calibration protocol, recall–FAR curves, hybrid for high FAR, sensitivity to the exclusion radius |
| §4 | Error analysis: recall by isotope and group, false-alarm tail and pulse pile-up, SNR at 50 % detection probability, Tikhonov regularisation, waterfall-CNN probe |
| §5 | Approaches that did not help (negative results) |
| §6 | Conclusions and limitations |

Other figures produced by the notebook are stored in [`figures/`](figures/).

## Getting the data

`training_v4.3.h5` (about 26.6 GB) is not included in the repository (it is too large and is
listed in `.gitignore`).

1. Register at [bdc.lbl.gov](https://bdc.lbl.gov/register) (academic requests are usually approved within about a day).
2. Download the RADAI **training** dataset v4.3 from the
   [dataset page](https://bdc.lbl.gov/wiki/public/radai-interactive-datasets/).
3. Save it in the repository root as `training_v4.3.h5`, next to `Matched_Filter_Detection.ipynb`.

## Running the notebook

Python 3.11 is used. Package versions are pinned in `requirements.txt`, because some of the
findings are sensitive to library versions.

```
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # Linux / macOS
pip install -r requirements.txt
python -m notebook Matched_Filter_Detection.ipynb      # interactive run
```

A headless run is also possible:

```
python _scratch/run_exec.py
# or
jupyter nbconvert --to notebook --execute --inplace Matched_Filter_Detection.ipynb
```

The notebook is self-contained: every intermediate cache (`_scratch/*.pkl`) is rebuilt from the
raw `.h5` file if it is missing, and the `figures/` and `_scratch/` folders are created
automatically. A full cold run (fresh kernel, empty cache directory, executed top to bottom with
`jupyter nbconvert --execute`) takes about 40 minutes and reproduces all published numbers
exactly; the headline values (25.2 / 40.5 / 70.0 %) are checked by assertions in the notebook.
The only expensive step is §4.6: if `_scratch/t5_wf.pkl` is missing, the notebook launches
`python _scratch/t5_waterfall.py` (PyTorch on CPU, fixed seed, about 10–20 of the 40 minutes)
before evaluating the results.

## Repository layout

- `Matched_Filter_Detection.ipynb` — the final notebook, stored with all outputs.
- `figures/` — PNG files saved by the notebook cells; some of them are used in this README.
- `_scratch/` — the notebook generator (`build_nb.py`), the headless runner (`run_exec.py`),
  standalone verification scripts cited in §5 of the notebook, and their reference `.out` logs.
  Regenerable caches (`*.pkl`, `*.log`) are ignored by git.
- `requirements.txt` — pinned dependency versions.

Earlier exploratory notebooks (dataset exploration, window-level LogReg / XGBoost / CNN,
adaptive-background ideas) are no longer kept in the repository. Their findings are reproduced
in §1–§2 of the notebook, and the dataset schema they documented has been checked against the
`.h5` file and moved to §0.2.

## References

- Jones et al., *Adaptive NMF* (LBNL), [arXiv:2507.10715](https://arxiv.org/abs/2507.10715) — adaptive background model with regularisation against NORM-like shapes.
- Bachleda et al., *waterfall CNN* (PNNL), [arXiv:2607.00270](https://arxiv.org/html/2607.00270v3) — convolutional network on raw spectral-temporal input.

The methodological differences from both papers are summarised in §0.4 of the notebook.
Numerical comparison with them is not possible, because the definitions of FAR, the encounter
boundaries and the data splits differ (see §4.4 and §5).

## Dataset and attribution

RADAI is a synthetic urban-search gamma-ray dataset produced by ORNL and LBNL (funded by DOE NA-22).
Please follow the citation and licensing instructions given on the
[dataset page](https://bdc.lbl.gov/wiki/public/radai-interactive-datasets/).
