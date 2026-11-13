#!/usr/bin/env python
"""
eaai_analysis.py
================

Out-of-sample inference for the runs produced by eaai_experiments.py.

For every market x target x horizon it reads the dated test predictions of
every feature set x model and the baselines written on the same dates, and
reports

* ROC AUC with a moving-block bootstrap 95 % interval (block = 2h),
* the AUC averaged over the h non-overlapping subsamples (every h-th
  observation), which removes the dependence created by overlapping
  h-step targets,
* accuracy, balanced accuracy, macro F1, Brier score (models),
* paired block-bootstrap tests of AUC differences on identical dates,
  Holm-adjusted within each family of comparisons,
* AUC by calendar year of the test period,
* reliability (calibration) tables,
* aggregated group permutation importance and MI-null summaries.

The model shown for each market x horizon x feature set in the main tables
is the one with the highest CROSS-VALIDATION AUC (selected on the training
period), never the one with the highest test AUC.

Usage:
    python EAAI_experiments/eaai_analysis.py --results Results_EAAI --boot 2000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, f1_score

HERE = Path(__file__).resolve().parent
BASELINE_LABELS = {
    "base_prevalence": "Majority class",
    "base_rv_reversion": "RV mean reversion",
    "base_har_logit": "HAR logistic",
    "base_momentum": "Momentum",
}


# ============================================================
# AUC AND BOOTSTRAP
# ============================================================

def auc_fast(y_pos, s):
    """Mann-Whitney AUC; y_pos boolean, ties in s get average ranks."""
    n_pos = int(y_pos.sum())
    n_neg = y_pos.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan
    r = rankdata(s)
    return (r[y_pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def block_indices(n, block, n_boot, rng):
    """Moving-block bootstrap index matrix (n_boot x n)."""
    block = int(max(1, min(block, n)))
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)
    return idx[:, :n]


def bootstrap_aucs(y_pos, scores: dict, idx):
    out = {}
    for name, s in scores.items():
        out[name] = np.array([auc_fast(y_pos[b], s[b]) for b in idx])
    return out


def nonoverlap_auc(y_pos, s, h):
    vals = [auc_fast(y_pos[o::h], s[o::h]) for o in range(h)]
    vals = np.asarray([v for v in vals if np.isfinite(v)])
    return float(vals.mean()), float(vals.std(ddof=1)) if len(vals) > 1 else np.nan


def paired_test(boot_a, boot_b, point_a, point_b):
    d = boot_a - boot_b
    d = d[np.isfinite(d)]
    lo, hi = np.quantile(d, [0.025, 0.975])
    p = 2.0 * min(np.mean(d <= 0), np.mean(d >= 0))
    p = float(min(1.0, max(p, 1.0 / len(d))))
    return point_a - point_b, lo, hi, p


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


# ============================================================
# LOADING
# ============================================================

def load_experiment(exp_dir):
    split = json.loads((exp_dir / "split.json").read_text())
    labels = pd.read_csv(exp_dir / "test_labels.csv", index_col=0, parse_dates=True)
    models = {}
    for pred_path in sorted(exp_dir.glob("*/*/predictions.csv")):
        fs_dir, model = pred_path.parent.parent, pred_path.parent.name
        fs = fs_dir.name.replace("_plus_", "+")
        pred = pd.read_csv(pred_path, index_col=0, parse_dates=True)
        if not pred.index.equals(labels.index):
            raise ValueError(f"{pred_path}: dates differ from test_labels.csv")
        if not np.array_equal(pred["y"].to_numpy(), labels["y"].to_numpy()):
            raise ValueError(f"{pred_path}: labels differ from test_labels.csv")
        best_path = pred_path.parent / "best.json"
        best = json.loads(best_path.read_text()) if best_path.exists() else {}
        models[(fs, model)] = (pred, best)
    return split, labels, models


# ============================================================
# MAIN
# ============================================================

def analyse(results, n_boot, seed, out_dir):
    rng = np.random.default_rng(seed)
    metric_rows, comp_rows, year_rows, calib_rows, imp_rows, null_rows = [], [], [], [], [], []

    exp_dirs = sorted(p.parent for p in results.glob("*/*/split.json"))
    if not exp_dirs:
        raise SystemExit(f"No experiments found under {results}")

    for exp_dir in exp_dirs:
        split, labels, models = load_experiment(exp_dir)
        market, target, h = split["market"], split["target"], int(split["horizon"])
        y = labels["y"].to_numpy()
        y_pos = y == 1
        n = len(y)
        idx = block_indices(n, 2 * h, n_boot, rng)

        sources = {}
        for c in labels.columns:
            if c.startswith("base_") and labels[c].notna().all():
                sources[("baseline", c)] = labels[c].to_numpy(dtype=float)
        for (fs, model), (pred, _) in models.items():
            sources[(fs, model)] = pred["score"].to_numpy(dtype=float)

        boots = bootstrap_aucs(y_pos, sources, idx)
        points = {k: auc_fast(y_pos, s) for k, s in sources.items()}

        # ---------------- per-source metrics ----------------
        for key, s in sources.items():
            fs, model = key
            b = boots[key]
            row = {
                "market": market, "target": target, "h": h,
                "feature_set": fs, "model": BASELINE_LABELS.get(model, model),
                "is_baseline": fs == "baseline",
                "n_test": n, "prevalence": float(y_pos.mean()),
                "auc": points[key],
                "auc_lo": float(np.nanquantile(b, 0.025)),
                "auc_hi": float(np.nanquantile(b, 0.975)),
                "p_auc_le_0.5": float(max(np.mean(b <= 0.5), 1.0 / n_boot)),
            }
            row["auc_nonoverlap"], row["auc_nonoverlap_sd"] = nonoverlap_auc(y_pos, s, h)
            if fs != "baseline":
                pred, best = models[key]
                yhat = pred["pred"].to_numpy()
                row.update({
                    "cv_auc": best.get("cv_best_auc", np.nan),
                    "accuracy": float(np.mean(yhat == y)),
                    "majority_floor": float(max(y_pos.mean(), 1 - y_pos.mean())),
                    "balanced_accuracy": float(balanced_accuracy_score(y, yhat)),
                    "macro_f1": float(f1_score(y, yhat, average="macro")),
                    "n_selected": best.get("n_selected", np.nan),
                    "n_candidates": best.get("n_candidates", np.nan),
                    "percentile": best.get("best_params", {}).get("sel__percentile", np.nan),
                })
                if np.all((s >= 0) & (s <= 1)):
                    row["brier"] = float(brier_score_loss(y_pos, s))
                    # reliability table, 10 quantile bins
                    bins = np.asarray(pd.qcut(s, 10, labels=False, duplicates="drop"), dtype=float)
                    for bval in np.unique(bins[np.isfinite(bins)]):
                        m = bins == bval
                        calib_rows.append({
                            "market": market, "target": target, "h": h, "feature_set": fs,
                            "model": model, "bin": int(bval), "n": int(m.sum()),
                            "mean_predicted": float(s[m].mean()), "observed": float(y_pos[m].mean()),
                        })
            metric_rows.append(row)

            # AUC by calendar year
            years = labels.index.year
            for yr in np.unique(years):
                m = years == yr
                if m.sum() >= 50 and 0 < y_pos[m].sum() < m.sum():
                    year_rows.append({
                        "market": market, "target": target, "h": h, "feature_set": fs,
                        "model": BASELINE_LABELS.get(model, model), "year": int(yr),
                        "n": int(m.sum()), "auc": auc_fast(y_pos[m], s[m]),
                    })

        # ---------------- paired comparisons ----------------
        def compare(family, a, b):
            if a not in sources or b not in sources:
                return
            d, lo, hi, p = paired_test(boots[a], boots[b], points[a], points[b])
            comp_rows.append({
                "family": family, "market": market, "target": target, "h": h,
                "a": f"{a[0]}|{BASELINE_LABELS.get(a[1], a[1])}",
                "b": f"{b[0]}|{BASELINE_LABELS.get(b[1], b[1])}",
                "auc_a": points[a], "auc_b": points[b],
                "delta": d, "delta_lo": lo, "delta_hi": hi, "p": p,
            })

        model_names = sorted({m for (_, m) in models})
        # The GRU forms its own Holm families (suffix " [GRU]"), so that adding it
        # leaves the pre-specified families of the tabular learners unchanged.
        if target == "vol":
            for m in model_names:
                sfx = " [GRU]" if m == "GRU" else ""
                for core in ("BASE", "H", "HC", "HCF", "BASE+HCF"):
                    compare(f"vol: HAR+{core} vs HAR{sfx}", (f"HAR+{core}", m), ("HAR", m))
                compare(f"vol: HAR+BASE+HCF vs HAR+BASE{sfx}", ("HAR+BASE+HCF", m), ("HAR+BASE", m))
                compare(f"vol: HAR (ML) vs HAR logistic{sfx}", ("HAR", m), ("baseline", "base_har_logit"))
                compare(f"vol: HCF vs RV mean reversion{sfx}", ("HCF", m), ("baseline", "base_rv_reversion"))
        else:
            for m in model_names:
                sfx = " [GRU]" if m == "GRU" else ""
                for fs in ("PRICE", "BASE", "HCF", "BASE+HCF", "HCF+PRICE"):
                    compare(f"dir: {fs} vs momentum{sfx}", (fs, m), ("baseline", "base_momentum"))
                compare(f"dir: BASE+HCF vs BASE{sfx}", ("BASE+HCF", m), ("BASE", m))

        # ---------------- importance and MI null ----------------
        for imp_path in exp_dir.glob("*/*/group_importance.csv"):
            imp = pd.read_csv(imp_path)
            imp["market"], imp["target"], imp["h"] = market, target, h
            imp["feature_set"] = imp_path.parent.parent.name.replace("_plus_", "+")
            imp["model"] = imp_path.parent.name
            imp_rows.append(imp)
        for null_path in exp_dir.glob("*/mi_null.json"):
            info = json.loads(null_path.read_text())
            null_rows.append({
                "market": market, "target": target, "h": h,
                "feature_set": null_path.parent.name.replace("_plus_", "+"),
                **{k: v for k, v in info.items() if k != "null_maxima"},
            })
        print(f"{market:9s} {target} h={h:2d}: {len(models)} model runs, n_test={n}")

    metrics = pd.DataFrame(metric_rows)
    comps = pd.DataFrame(comp_rows)
    if len(comps):
        comps["p_holm"] = np.nan
        for fam, g in comps.groupby("family"):
            comps.loc[g.index, "p_holm"] = holm(g["p"].to_numpy())

    out_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    comps.to_csv(out_dir / "comparisons.csv", index=False)
    pd.DataFrame(year_rows).to_csv(out_dir / "auc_by_year.csv", index=False)
    pd.DataFrame(calib_rows).to_csv(out_dir / "calibration.csv", index=False)
    if imp_rows:
        pd.concat(imp_rows).to_csv(out_dir / "group_importance_all.csv", index=False)
    pd.DataFrame(null_rows).to_csv(out_dir / "mi_null_summary.csv", index=False)

    write_main_tables(metrics, comps, out_dir)
    print(f"\nwritten to {out_dir}")
    return metrics, comps


# ============================================================
# MAIN TABLES (CV-selected model per feature set)
# ============================================================

def cv_selected(metrics):
    mod = metrics[~metrics["is_baseline"]].copy()
    if mod.empty:
        return mod
    mod = mod.sort_values("cv_auc", ascending=False)
    return mod.groupby(["market", "target", "h", "feature_set"], as_index=False).first()


def fmt_ci(r):
    return f"{r['auc']:.3f} [{r['auc_lo']:.3f}, {r['auc_hi']:.3f}]"


def write_main_tables(metrics, comps, out_dir):
    sel = cv_selected(metrics)
    sel.to_csv(out_dir / "cv_selected_models.csv", index=False)
    base = metrics[metrics["is_baseline"]]
    base_tab = base.assign(auc_ci=base.apply(fmt_ci, axis=1)).pivot_table(
        index=["target", "market", "h"], columns="model", values="auc_ci", aggfunc="first")
    base_tab.to_csv(out_dir / "table_baselines.csv")
    if sel.empty:
        return
    lines = []
    for target, g in sel.groupby("target"):
        cols = list(dict.fromkeys(g["feature_set"]))
        tab = []
        for (market, h), gg in g.groupby(["market", "h"]):
            row = {"Market": market, "h": h}
            for _, r in base[(base.market == market) & (base.h == h) & (base.target == target)].iterrows():
                if r["model"] != "Majority class":
                    row[r["model"]] = fmt_ci(r)
            for _, r in gg.iterrows():
                row[f"{r['feature_set']} ({r['model']})"] = fmt_ci(r)
            tab.append(row)
        df = pd.DataFrame(tab)
        df.to_csv(out_dir / f"table_main_{target}.csv", index=False)
        lines.append(df.to_latex(index=False, escape=True))
    (out_dir / "tables_main.tex").write_text("\n\n".join(lines))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default=str(HERE.parent / "Results_EAAI"))
    p.add_argument("--boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--out", default=None, help="default: <results>/_analysis")
    a = p.parse_args(argv)
    results = Path(a.results)
    analyse(results, a.boot, a.seed, Path(a.out) if a.out else results / "_analysis")


if __name__ == "__main__":
    main()
