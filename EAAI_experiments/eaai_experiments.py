#!/usr/bin/env python
"""
eaai_experiments.py
===================

Leakage-controlled experiments for the EAAI manuscript.

For every market x target x horizon x feature set x model the script
  1. builds the causal predictor matrix (transform first, lag second),
  2. aligns it with the target and makes a chronological split
     (first 80 % train, 250-observation embargo, remainder test),
  3. tunes [mutual-information selection -> StandardScaler -> classifier]
     by RandomizedSearchCV on TimeSeriesSplit(5, gap=250), ROC AUC,
  4. refits on the whole training period and scores the test period once,
  5. saves dated test predictions, the CV table, the selected features with
     their MI scores and a group permutation importance table.

Baselines that require no fitting and econometric baselines are written once per
market x target x horizon (test_labels.csv), on exactly the same test dates
as the models, so every comparison in eaai_analysis.py is paired.

All feature sets of one market x target x horizon share the same rows:
the union matrix is built first and every set is a column subset of it.

Usage (from the project folder):

    python EAAI_experiments/eaai_experiments.py --smoke
    python EAAI_experiments/eaai_experiments.py                # full plan
    python EAAI_experiments/eaai_experiments.py --markets Brent --horizons 5 \
        --targets vol --featuresets BASE HCF BASE+HCF --models LR HGB

Re-running skips finished cells (predictions.csv exists) unless --overwrite.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import eaai_common as ec  # noqa: E402

SK_VERSION = tuple(int(p) for p in sklearn.__version__.split(".")[:2])

# Default experimental plan --------------------------------------------------
DEFAULT_FEATURESETS = {
    # primary target: what each block adds to the HAR volatility benchmark,
    # plus the complexity block on its own
    "vol": ["HAR", "HAR+BASE", "HAR+H", "HAR+HCF", "HAR+BASE+HCF", "HCF"],
    # negative control: direction of the h-step price change
    "dir": ["PRICE", "BASE", "HCF", "BASE+HCF", "HCF+PRICE"],
}
SMOKE_FEATURESETS = {"vol": ["HAR", "HAR+HCF"], "dir": ["BASE", "HCF"]}
DEFAULT_MODELS = ["LR", "RF", "HGB"]
PERCENTILES = [5, 10, 20, 30, 50, 75, 100]


# ============================================================
# MODELS AND SEARCH SPACES
# ============================================================

def make_logistic():
    kwargs = dict(solver="saga", max_iter=3000, tol=1e-3)
    if SK_VERSION < (1, 8):
        kwargs["penalty"] = "elasticnet"  # l1_ratio is ignored otherwise
    return LogisticRegression(**kwargs)


def model_specs(names, cache_dir, seed):
    selector = ec.MISelector(cache_dir=cache_dir)
    specs = {}
    for name in names:
        if name == "LR":
            est = make_logistic()
            space = {
                "clf__C": loguniform(1e-3, 1e2),
                "clf__l1_ratio": uniform(0.0, 1.0),
                "clf__class_weight": [None, "balanced"],
            }
        elif name == "RF":
            est = RandomForestClassifier(n_jobs=1, random_state=seed)
            space = {
                "clf__n_estimators": randint(200, 801),
                "clf__max_depth": [None, 4, 8, 16, 32],
                "clf__max_features": ["sqrt", "log2", 0.2, 0.4],
                "clf__min_samples_leaf": [1, 5, 20, 50],
                "clf__max_samples": [None, 0.5, 0.7],
                "clf__class_weight": [None, "balanced_subsample"],
            }
        elif name == "HGB":
            # early_stopping=False: the internal validation split of
            # HistGradientBoosting is random and would break time order.
            est = HistGradientBoostingClassifier(early_stopping=False, random_state=seed)
            space = {
                "clf__learning_rate": loguniform(1e-2, 2e-1),
                "clf__max_iter": randint(100, 501),
                "clf__max_leaf_nodes": randint(7, 64),
                "clf__min_samples_leaf": randint(20, 301),
                "clf__l2_regularization": loguniform(1e-4, 1e1),
                "clf__class_weight": [None, "balanced"],
            }
            if SK_VERSION >= (1, 4):
                space["clf__max_features"] = uniform(0.2, 0.8)
        elif name == "KNN":
            est = KNeighborsClassifier()
            space = {
                "clf__n_neighbors": randint(5, 301),
                "clf__weights": ["uniform", "distance"],
                "clf__p": [1, 2],
            }
        else:
            raise KeyError(f"Unknown model '{name}'")
        space["sel__percentile"] = PERCENTILES
        pipe = Pipeline([("sel", selector), ("scale", StandardScaler()), ("clf", est)])
        specs[name] = (pipe, space)
    return specs


# ============================================================
# DATA PREPARATION
# ============================================================

def build_union_matrix(master, representation, n_lags):
    variables = ec.feature_set_variables("BASE+HCF", master.columns) + ["price"]
    X_all = ec.make_lagged_matrix(master, variables, representation, n_lags)
    return X_all.dropna()


def har_lagged(price, h, n_lags):
    """HAR control block (log RV at 1, 5, 22, h days) with lags 0..n_lags."""
    design = ec.har_design(price, h)
    blocks = [design]
    for lag in range(1, n_lags + 1):
        sh = design.shift(lag)
        sh.columns = [f"{c}_t-{lag}" for c in design.columns]
        blocks.append(sh)
    return pd.concat(blocks, axis=1), list(design.columns)


def prepare_experiment(master, X_all, target, h, test_size, gap, n_lags):
    price = master["price"]
    har_vars = []
    if target == "vol":
        har_block, har_vars = har_lagged(price, h, n_lags)
        X_all = X_all.join(har_block, how="inner").dropna()
    if target == "vol":
        tdf = ec.target_volatility_regime(price, h)
        # integrity: the label is reproducible from the stored columns
        chk = np.where(tdf["rv_fwd"] > tdf["rv_past"], 1, -1)
        assert np.array_equal(chk, tdf["label"].to_numpy()), "vol label check failed"
    elif target == "dir":
        tdf = ec.target_direction(price, h)
        chk = np.sign(tdf["fwd_change"]).astype(int).to_numpy()
        assert np.array_equal(chk, tdf["label"].to_numpy()), "dir label check failed"
    else:
        raise KeyError(target)

    common = X_all.index.intersection(tdf.index)
    X = X_all.loc[common]
    tdf = tdf.loc[common]
    y = tdf["label"].to_numpy()
    n = len(common)
    split_index, test_start = ec.chronological_split(n, test_size, gap)
    idx_train = np.arange(0, split_index)
    idx_test = np.arange(test_start, n)

    dates = common
    # embargo must exceed the label horizon
    assert test_start - split_index >= h, "embargo shorter than horizon"
    assert dates[split_index - 1] < dates[test_start]

    # ---------- baselines on the test dates ----------
    test_dates = dates[idx_test]
    out = pd.DataFrame(index=test_dates)
    out.index.name = "date"
    out["y"] = y[idx_test]
    out["price"] = price.reindex(test_dates).to_numpy()
    prev_train = float(np.mean(y[idx_train] == 1))
    out["base_prevalence"] = prev_train  # constant score -> AUC 0.5

    if target == "vol":
        out["rv_past"] = tdf["rv_past"].iloc[idx_test].to_numpy()
        out["rv_fwd"] = tdf["rv_fwd"].iloc[idx_test].to_numpy()
        # (i) volatility mean reversion, no fitting: high past RV -> contraction
        out["base_rv_reversion"] = -out["rv_past"]
        # (ii) HAR-type logistic model on log RV(1, 5, 22, h), fitted on train
        design = ec.har_design(price, h).reindex(dates)
        ok_train = design.iloc[idx_train].notna().all(axis=1).to_numpy()
        ok_test = design.iloc[idx_test].notna().all(axis=1).to_numpy()
        har = np.full(len(idx_test), np.nan)
        har[ok_test] = ec.fit_har_logit(
            design.iloc[idx_train][ok_train],
            y[idx_train][ok_train],
            design.iloc[idx_test][ok_test],
        )
        out["base_har_logit"] = har
    else:
        out["fwd_change"] = tdf["fwd_change"].iloc[idx_test].to_numpy()
        out["fwd_return"] = tdf["fwd_return"].iloc[idx_test].to_numpy()
        # time-series momentum: the past h-step change, observable at t
        out["base_momentum"] = ec.momentum_score(price, h).reindex(test_dates).to_numpy()

    info = {
        "target": target,
        "class_names": {"-1": ec.TARGET_NAMES[target][0], "+1": ec.TARGET_NAMES[target][1]},
        "horizon": h,
        "n_aligned": int(n),
        "n_train": int(len(idx_train)),
        "n_test": int(len(idx_test)),
        "embargo": int(gap),
        "test_size_nominal": test_size,
        "train_start": str(dates[0].date()),
        "train_end": str(dates[split_index - 1].date()),
        "test_start": str(dates[test_start].date()),
        "test_end": str(dates[-1].date()),
        "prevalence_train": prev_train,
        "prevalence_test": float(np.mean(y[idx_test] == 1)),
    }
    info["har_variables"] = har_vars
    return X, y, idx_train, idx_test, out, info, har_vars


# ============================================================
# ONE CELL: feature set x model
# ============================================================

def warm_mi_cache(X_tr, y_tr, cv, cache_dir, n_jobs):
    t0 = time.time()
    for tr, _ in cv.split(X_tr):
        ec.mi_scores(X_tr[tr], y_tr[tr], cache_dir=cache_dir, n_jobs=n_jobs)
    ec.mi_scores(X_tr, y_tr, cache_dir=cache_dir, n_jobs=n_jobs)
    return time.time() - t0


def run_tabular_cell(name, spec, X_tr, y_tr, X_te, y_te, test_dates, columns, cv, args, out_dir, log,
                     n_always=0):
    pipe, space = spec
    pipe = clone(pipe).set_params(sel__n_always=n_always)
    space = dict(space)
    if n_always >= len(columns):  # control block only: nothing to select
        space.pop("sel__percentile", None)
    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=space,
        n_iter=args.n_iter,
        cv=cv,
        scoring="roc_auc",
        n_jobs=args.n_jobs,
        refit=True,
        random_state=args.seed,
        error_score=np.nan,
        return_train_score=False,
    )
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        warnings.simplefilter("ignore", category=FutureWarning)
        search.fit(X_tr, y_tr)
    fit_seconds = time.time() - t0

    best = search.best_estimator_
    score = ec.positive_score(best, X_te)
    pred = best.predict(X_te)
    auc = roc_auc_score(y_te, score)

    out_dir.mkdir(parents=True, exist_ok=True)

    cv_table = pd.DataFrame(search.cv_results_)
    keep = [c for c in cv_table.columns if c.startswith("param_")] + [
        "mean_test_score", "std_test_score", "rank_test_score", "mean_fit_time"
    ] + [c for c in cv_table.columns if c.startswith("split") and c.endswith("_test_score")]
    cv_table[keep].sort_values("rank_test_score").to_csv(out_dir / "cv_results.csv", index=False)

    sel = best.named_steps["sel"]
    pd.DataFrame(
        {"feature": columns, "mi": sel.scores_, "selected": sel.get_support()}
    ).sort_values("mi", ascending=False).to_csv(out_dir / "selected_features.csv", index=False)

    groups = ec.feature_groups(columns)
    if args.importance_repeats > 0 and len(groups) > 1:
        imp = ec.group_permutation_importance(
            best, X_te, y_te, groups, n_repeats=args.importance_repeats, random_state=args.seed
        )
        imp.to_csv(out_dir / "group_importance.csv", index=False)

    best_params = {k: (v if isinstance(v, (int, float, str, type(None), bool)) else str(v))
                   for k, v in search.best_params_.items()}
    meta = {
        "model": name,
        "best_params": best_params,
        "cv_best_auc": float(search.best_score_),
        "test_auc": float(auc),
        "n_candidates": int(len(columns)),
        "n_selected": int(sel.get_support().sum()),
        "n_always_kept": int(n_always),
        "fit_seconds": round(fit_seconds, 1),
        "n_iter": args.n_iter,
        "finished": datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / "best.json").write_text(json.dumps(meta, indent=2))

    # written last: its presence marks the cell as finished
    pd.DataFrame(
        {"y": y_te, "score": score, "pred": pred}, index=pd.Index(test_dates, name="date")
    ).to_csv(out_dir / "predictions.csv")
    log(f"      {name:4s} test AUC {auc:.3f} | CV {search.best_score_:.3f} | "
        f"{meta['n_selected']}/{len(columns)} features | {fit_seconds:.0f}s")


def run_sequence_cell(X_df, y, idx_train, idx_test, variables, cv, args, out_dir, log):
    """GRU with the same CV folds and ROC AUC criterion as the tabular learners."""
    import eaai_seq  # imported lazily: needs PyTorch

    t0 = time.time()
    score, info, cv_table = eaai_seq.fit_predict_gru(
        X_df, y, idx_train, idx_test, variables, cv, n_lags=args.lags, seed=args.seed, log=log,
        hidden_grid=args.gru_hidden, epoch_grid=args.gru_epochs, n_seeds=args.gru_seeds,
    )
    y_te = y[idx_test]
    auc = roc_auc_score(y_te, score)
    out_dir.mkdir(parents=True, exist_ok=True)
    cv_table.to_csv(out_dir / "cv_results.csv", index=False)
    import torch
    info.update({"model": "GRU", "test_auc": float(auc), "torch": torch.__version__,
                 "fit_seconds": round(time.time() - t0, 1), "n_iter": info["n_cv_candidates"],
                 "finished": datetime.now().isoformat(timespec="seconds")})
    (out_dir / "best.json").write_text(json.dumps(info, indent=2))
    # written last: its presence marks the cell as finished
    pd.DataFrame(
        {"y": y_te, "score": score, "pred": np.where(score >= 0.5, 1, -1)},
        index=pd.Index(X_df.index[idx_test], name="date"),
    ).to_csv(out_dir / "predictions.csv")
    bp = info["best_params"]
    log(f"      GRU  test AUC {auc:.3f} | CV {info['cv_best_auc']:.3f} | "
        f"hidden {bp['hidden']}, {bp['epochs']} epochs | {info['fit_seconds']:.0f}s")


# ============================================================
# MI PERMUTATION NULL (optional)
# ============================================================

def mi_permutation_null(X_tr, y_tr, n_draws, min_shift, cache_dir, n_jobs, seed):
    """
    Circular-shift null: the target is rotated by a random offset of at
    least `min_shift` observations, preserving its serial dependence.
    Returns observed scores, the null maxima, and pooled null values.
    """
    rng = np.random.default_rng(seed)
    observed = ec.mi_scores(X_tr, y_tr, cache_dir=cache_dir, n_jobs=n_jobs)
    n = len(y_tr)
    maxima, pooled = [], []
    for _ in range(n_draws):
        shift = int(rng.integers(min_shift, n - min_shift))
        null = ec.mi_scores(X_tr, np.roll(y_tr, shift), cache_dir=None, n_jobs=n_jobs)
        maxima.append(null.max())
        pooled.append(null)
    return observed, np.asarray(maxima), np.concatenate(pooled)


# ============================================================
# MAIN
# ============================================================

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=str(HERE.parent / "Datasets"))
    p.add_argument("--out", default=str(HERE.parent / "Results_EAAI"))
    p.add_argument("--markets", nargs="+", default=list(ec.MARKETS))
    p.add_argument("--horizons", nargs="+", type=int, default=[5, 10, 20])
    p.add_argument("--targets", nargs="+", default=["vol", "dir"], choices=["vol", "dir"])
    p.add_argument("--featuresets", nargs="+", default=None,
                   help="default: the plan in DEFAULT_FEATURESETS for each target")
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                   help="LR RF HGB KNN GRU (GRU needs PyTorch)")
    p.add_argument("--gru-hidden", nargs="+", type=int, default=[16, 32, 64],
                   help="GRU hidden sizes (CV candidates)")
    p.add_argument("--gru-epochs", nargs="+", type=int, default=[5, 10, 15, 20, 30, 40, 50, 60],
                   help="GRU numbers of training epochs (CV candidates)")
    p.add_argument("--gru-seeds", type=int, default=3, help="GRU networks averaged in the final refit")
    p.add_argument("--representation", default="levels", choices=list(ec.REPRESENTATIONS))
    p.add_argument("--lags", type=int, default=20)
    p.add_argument("--test-size", type=float, default=0.20)
    p.add_argument("--gap", type=int, default=250, help="train/test embargo and CV gap")
    p.add_argument("--cv-splits", type=int, default=5)
    p.add_argument("--n-iter", type=int, default=30, help="random-search candidates per model")
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--importance-repeats", type=int, default=5)
    p.add_argument("--mi-null", type=int, default=0,
                   help="circular-shift MI null draws for HCF and BASE+HCF (0 = off)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--prepare-only", action="store_true",
                   help="write split.json and test_labels.csv (targets, baselines) without fitting models")
    p.add_argument("--smoke", action="store_true",
                   help="quick end-to-end check: Brent, h=5, both targets, 2 sets, LR+HGB, n_iter=3")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.smoke:
        args.markets, args.horizons = ["Brent"], [5]
        args.featuresets = None  # filled per target below
        args.models = [m for m in args.models if m in ("LR", "HGB", "GRU")] or ["LR", "HGB"]
        args.n_iter, args.importance_repeats = 3, 2
        args.gru_hidden, args.gru_epochs, args.gru_seeds = [16], [2, 4], 1
        args.out = str(Path(args.out).with_name(Path(args.out).name + "_smoke"))

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    cache_dir = out_root / "_mi_cache"
    log_path = out_root / f"run_{datetime.now():%Y%m%d_%H%M%S}.log"

    def log(msg):
        line = f"[{datetime.now():%H:%M:%S}] {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    run_config = vars(args).copy()
    run_config.update({
        "python": sys.version.split()[0], "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
        "percentile_grid": PERCENTILES,
    })
    (out_root / f"config_{datetime.now():%Y%m%d_%H%M%S}.json").write_text(json.dumps(run_config, indent=2))
    log(f"output: {out_root.resolve()}")
    log(f"scikit-learn {sklearn.__version__}, pandas {pd.__version__}, numpy {np.__version__}")

    cv = TimeSeriesSplit(n_splits=args.cv_splits, gap=args.gap)
    tabular = [m for m in args.models if m != "GRU"]
    specs = model_specs(tabular, str(cache_dir), args.seed)

    for market in args.markets:
        master, series_id = ec.load_master(args.data_dir, market)
        X_all = build_union_matrix(master, args.representation, args.lags)
        log(f"== {market} ({series_id}): union matrix {X_all.shape}, "
            f"{X_all.index.min().date()} -> {X_all.index.max().date()}")

        for target in args.targets:
            fsets = args.featuresets or (SMOKE_FEATURESETS if args.smoke else DEFAULT_FEATURESETS)[target]
            for h in args.horizons:
                exp_dir = out_root / market / f"{target}_h{h}"
                X_df, y, idx_tr, idx_te, labels_test, info, har_vars = prepare_experiment(
                    master, X_all, target, h, args.test_size, args.gap, args.lags
                )
                info.update({"market": market, "series_id": series_id,
                             "representation": args.representation, "lags": args.lags})
                exp_dir.mkdir(parents=True, exist_ok=True)
                (exp_dir / "split.json").write_text(json.dumps(info, indent=2))
                labels_test.to_csv(exp_dir / "test_labels.csv")
                log(f"  -- {target} h={h}: train {info['n_train']} ({info['train_start']}..{info['train_end']}), "
                    f"test {info['n_test']} ({info['test_start']}..{info['test_end']}), "
                    f"prevalence(+1) test {info['prevalence_test']:.3f}")

                if args.prepare_only:
                    continue
                y_tr, y_te = y[idx_tr], y[idx_te]
                test_dates = X_df.index[idx_te]

                for fs in fsets:
                    uses_har, core = ec.parse_feature_set(fs)
                    if uses_har and target != "vol":
                        log(f"    [{fs}] HAR control block is defined for the volatility target only; skipped")
                        continue
                    core_vars = ec.feature_set_variables(core, master.columns) if core else []
                    core_cols = ec.columns_for_set(
                        [c for c in X_df.columns if not c.startswith(ec.HAR_PREFIX)], core_vars)
                    control_cols = list(har_vars) if uses_har else []  # lag-0 HAR terms
                    columns = control_cols + core_cols
                    variables = (list(har_vars) if uses_har else []) + core_vars  # for the GRU
                    X_np = X_df[columns].to_numpy(dtype=np.float64)
                    X_tr = np.ascontiguousarray(X_np[idx_tr])
                    X_te = np.ascontiguousarray(X_np[idx_te])
                    fs_dir = exp_dir / fs.replace("+", "_plus_")
                    todo = [m for m in args.models
                            if args.overwrite or not (fs_dir / m / "predictions.csv").exists()]
                    if not todo:
                        log(f"    [{fs}] done, skipped")
                        continue
                    log(f"    [{fs}] {len(columns)} candidate predictors"
                        + (f" ({len(control_cols)} HAR terms always kept)" if control_cols else ""))

                    if any(m in specs for m in todo):
                        secs = warm_mi_cache(X_tr, y_tr, cv, cache_dir, args.n_jobs)
                        log(f"      MI cache ready ({secs:.0f}s)")

                    if args.mi_null > 0 and core in ("HCF", "BASE+HCF") and \
                            (args.overwrite or not (fs_dir / "mi_null.json").exists()):
                        obs, maxima, pooled = mi_permutation_null(
                            X_tr, y_tr, args.mi_null, args.gap, cache_dir, args.n_jobs, args.seed)
                        p_fwer = float((1 + np.sum(maxima >= obs.max())) / (1 + len(maxima)))
                        srt = np.sort(pooled)
                        p_marg = 1.0 - np.searchsorted(srt, obs, side="left") / len(srt)
                        fs_dir.mkdir(parents=True, exist_ok=True)
                        pd.DataFrame({"feature": columns, "mi": obs, "p_pooled": p_marg}).to_csv(
                            fs_dir / "mi_null_features.csv", index=False)
                        (fs_dir / "mi_null.json").write_text(json.dumps({
                            "draws": int(len(maxima)), "min_shift": args.gap,
                            "observed_max": float(obs.max()),
                            "null_max_mean": float(maxima.mean()),
                            "null_max_q95": float(np.quantile(maxima, 0.95)),
                            "null_pooled_q99": float(np.quantile(pooled, 0.99)),
                            "p_family_wise": p_fwer,
                            "null_maxima": [float(v) for v in maxima],
                        }, indent=2))
                        log(f"      MI null: observed max {obs.max():.4f}, null q95 "
                            f"{np.quantile(maxima, 0.95):.4f}, FWER p={p_fwer:.3f}")

                    for m in todo:
                        try:
                            if m == "GRU":
                                run_sequence_cell(X_df, y, idx_tr, idx_te, variables, cv, args,
                                                  fs_dir / m, log)
                            else:
                                run_tabular_cell(m, specs[m], X_tr, y_tr, X_te, y_te, test_dates,
                                                 columns, cv, args, fs_dir / m, log,
                                                 n_always=len(control_cols))
                        except Exception as exc:  # keep going; the log records it
                            log(f"      {m} FAILED: {exc!r}")
                            fs_dir.mkdir(parents=True, exist_ok=True)
                            with open(fs_dir / f"{m}_error.txt", "w", encoding="utf-8") as fh:
                                traceback.print_exc(file=fh)
    log("finished")


if __name__ == "__main__":
    main()
