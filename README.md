# Multiscale ordinal complexity features for machine-learning classification of volatility regimes in energy commodity markets

Code, input data and results for the manuscript of the same title (under review).

The study tests whether ordinal-pattern quantifiers (permutation entropy H,
statistical complexity C and Fisher information F, embedding dimension 3)
add out-of-sample information on future volatility regimes of four energy
spot markets (WTI and Brent crude oil, U.S. Gulf Coast gasoline and Henry Hub
natural gas) beyond realized-volatility (HAR) terms. Each feature block X is
evaluated as HAR+X against HAR on identical test dates, with moving-block
bootstrap intervals and Holm-adjusted paired tests. The direction of the
h-step price change serves as a negative control.

## Contents

| Path | Contents |
|---|---|
| `data/` | Daily FRED spot prices used in the study, frozen on 18 August 2026 (see `data/README.md`) |
| `features_gen.ipynb` | Builds the feature tables `Datasets/<market>_complexity_datasets/<ID>_MASTER.csv` from `data/` |
| `Datasets/` | Feature-generation settings per market (`<ID>_CONFIG.json`); the notebook writes the feature tables here |
| `EAAI_experiments/` | Experiments, analysis and figures (scripts listed below) |
| `Results_EAAI/` | Dated out-of-sample predictions, benchmark scores, model-selection records and the aggregated analysis tables (see `Results_EAAI/README.md`) |

| Script | Purpose |
|---|---|
| `eaai_common.py` | Markets, feature sets, causal lags, targets, benchmarks, mutual-information selector with a per-fold cache, group permutation importance |
| `eaai_experiments.py` | One run per market x target x horizon x feature set x learner: chronological split with embargo, randomized search on `TimeSeriesSplit`, refit, dated test predictions |
| `eaai_seq.py` | GRU learner, tuned on the same folds and criterion as the tabular learners (needs PyTorch) |
| `eaai_analysis.py` | AUC with block-bootstrap intervals, paired Holm-adjusted comparisons, AUC by year, calibration, summary tables |
| `eaai_figures.py` | Result figures and the split figure |
| `eaai_figures_static.py` | Dynamics of H, C, F and the complexity-entropy plane (needs the feature tables and `ordpy`) |
| `eaai_style.py` | Common figure style |
| `eaai_check.py` | Integrity check of a results folder after an interruption |
| `run_all.bat`, `run_gru.bat`, `resume.bat` | Windows batch files for the full run, the GRU arm and resuming after an interruption |

## Requirements

Python 3.14 and the packages in `requirements.txt`:

```
pip install -r requirements.txt
```

PyTorch (CPU build) is needed only for the GRU learner.

## Reproducing the results

Run all commands from the repository root.

### 1. Tables and figures from the included predictions

```
python EAAI_experiments/eaai_analysis.py
python EAAI_experiments/eaai_figures.py --only forest_vol forest_dir delta importance yearly calibration
```

`eaai_analysis.py` rewrites `Results_EAAI/_analysis/` from the dated
predictions in `Results_EAAI/` and reproduces the archived tables exactly
(about 3 min on two CPU cores); `eaai_figures.py` writes the figures to
`figures/`. No model is refitted. The split figure (`eaai_figures.py` without
`--only`) and the figures of `eaai_figures_static.py` also need the feature
tables from step 2.

### 2. Feature tables

Run `features_gen.ipynb` (Jupyter, working directory = repository root). With
`USE_LOCAL_DATA = True` (default) it reads the frozen prices in `data/`;
`USE_LOCAL_DATA = False` downloads the current FRED vintage instead (needs
`pandas-datareader`), which in general differs from the data of the study.
From `data/`, the notebook reproduces the feature tables of the study up to
floating-point rounding. The feature tables are written to `Datasets/` and are
also needed by `eaai_figures_static.py`.

### 3. Experiments

The results folder already contains every finished run, and
`eaai_experiments.py` skips finished runs. To refit the models, write to a new
folder:

```
python EAAI_experiments/eaai_experiments.py --smoke
python EAAI_experiments/eaai_experiments.py --models LR HGB --out Results_new
python EAAI_experiments/eaai_experiments.py --models RF --out Results_new
python EAAI_experiments/eaai_experiments.py --models GRU --targets vol --featuresets HAR HAR+HCF HAR+BASE+HCF HCF --out Results_new
python EAAI_experiments/eaai_experiments.py --models GRU --targets dir --featuresets BASE BASE+HCF --out Results_new
python EAAI_experiments/eaai_analysis.py --results Results_new
python EAAI_experiments/eaai_figures.py --results Results_new
```

`--smoke` is a short end-to-end check (one market and horizon, few
candidates) that writes to `Results_EAAI_smoke/`. The batch files use the
default folder `Results_EAAI/`.

The 468 learner runs took 35.1 h of summed fitting time on a six-core
AMD Ryzen 5 3600 under Windows 10 (logistic regression 2.3 h, random forest
11.0 h, gradient boosting 16.2 h, GRU 5.6 h).

## Licenses

- Code: MIT License (`LICENSE`).
- Results in `Results_EAAI/`: Creative Commons Attribution 4.0 International
  (CC BY 4.0), <https://creativecommons.org/licenses/by/4.0/>.
- Price data in `data/`: U.S. Energy Information Administration via FRED,
  public domain, citation requested (`data/README.md`).

## Citation

To be added upon publication.
