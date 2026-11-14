# Results

Outputs of `EAAI_experiments/eaai_experiments.py` (per run) and
`EAAI_experiments/eaai_analysis.py` (aggregated). Licensed under Creative
Commons Attribution 4.0 International (CC BY 4.0),
<https://creativecommons.org/licenses/by/4.0/>.

## Layout

```
<market>/<target>_h<h>/split.json                 split dates, sample sizes, class prevalence
<market>/<target>_h<h>/test_labels.csv            test dates, labels, prices and benchmark scores
<market>/<target>_h<h>/<feature set>/<learner>/
    predictions.csv                               date, y, score (predicted probability of class +1), pred
    best.json                                     selected hyperparameters, CV AUC, test AUC, fitting time
    cv_results.csv                                randomized-search table (all candidates, all folds)
    group_importance.csv                          group permutation importance on the test period (tabular learners, all sets except HAR)
    selected_features.csv                         mutual information (nats) of every candidate column on the final training period and whether the selector retained it (tabular learners)
_analysis/                                        aggregated tables (below)
config_<timestamp>.json                           settings and package versions of each invocation of eaai_experiments.py
```

- `<market>`: `WTI`, `Brent`, `Gasoline`, `HenryHub`.
- `<target>`: `vol` (volatility regime, classes -1 Contraction / +1 Expansion)
  or `dir` (direction of the h-step price change, -1 Down / +1 Up); `<h>`: 5, 10, 20.
- `<feature set>`: `+` in the set name is written as `_plus_` (for example
  `HAR_plus_HCF`).
- `<learner>`: `LR`, `RF`, `HGB`, `GRU`.

All runs of one market x target x horizon share the dates and labels of
`test_labels.csv`; `eaai_analysis.py` checks this. The full plan was run in
several invocations, resumed after interruptions, so there is one
`config_*.json` per invocation; in these files the two folder paths were
reduced to folder names.

## Aggregated tables (`_analysis/`)

| File | Contents |
|---|---|
| `metrics.csv` | Test AUC with 95 % block-bootstrap intervals, AUC on non-overlapping subsamples, CV AUC, accuracy, balanced accuracy, macro-F1, Brier score, for every run and benchmark |
| `cv_selected_models.csv` | The same metrics for the learner with the highest CV AUC per feature set |
| `comparisons.csv` | Paired AUC differences with bootstrap intervals, p-values and Holm-adjusted p-values within each family |
| `table_main_vol.csv`, `table_main_dir.csv`, `table_baselines.csv`, `tables_main.tex` | Summary tables of the manuscript |
| `auc_by_year.csv` | AUC by calendar year of the test period |
| `calibration.csv` | Reliability-diagram bins |
| `group_importance_all.csv` | Group permutation importance of all runs |
