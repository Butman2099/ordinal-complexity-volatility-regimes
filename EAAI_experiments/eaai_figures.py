#!/usr/bin/env python
"""
eaai_figures.py
===============

Result figures for the EAAI manuscript, drawn from the tables written by
eaai_analysis.py (<results>/_analysis) and from the split.json files.
Style: eaai_style.py (SciencePlots 'science' + 'no-latex', Computer Modern,
drawn at the cas-sc text width so that fonts print at 9 / 8 pt).
Vector PDF + 300 dpi PNG.

  fig_split.pdf        training / embargo / test partition of the four markets
  fig_forest_vol.pdf   test AUC with 95 % intervals by feature set, market, horizon
                       (CV-selected learner), benchmarks as vertical lines
  fig_forest_dir.pdf   same for the direction (negative-control) target
  fig_delta_vol.pdf    paired AUC differences HAR+X minus HAR, per learner
  fig_importance.pdf   group permutation importance (volatility target)
  fig_yearly_vol.pdf   AUC by calendar year: HAR, HAR+BASE+HCF, HAR logistic
  fig_calibration.pdf  reliability of the volatility-regime probabilities

Usage:
    python EAAI_experiments/eaai_figures.py --results Results_EAAI --out figures
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import eaai_style as st  # noqa: E402

st.apply()
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

MARKETS, MARKET_LABEL = st.MARKETS, st.MARKET_LABEL
VOL_ORDER = ["HAR", "HAR+BASE", "HAR+H", "HAR+HC", "HAR+HCF", "HAR+BASE+HCF", "HCF"]
DIR_ORDER = ["PRICE", "BASE", "HCF", "BASE+HCF", "HCF+PRICE"]
W = st.TEXT_WIDTH


def learner_handles(models):
    return [Line2D([], [], marker=m, color=c, ls="", ms=4.5, label=st.LEARNER_LABEL.get(l, l))
            for l, (m, c) in st.LEARNER_STYLE.items() if l in set(models)]


# ============================================================
# PARTITION
# ============================================================

def fig_split(results, data_dir, out):
    sys.path.insert(0, str(HERE))
    import eaai_common as ec
    units = {"WTI": "USD/bbl", "Brent": "USD/bbl", "Gasoline": "USD/gal", "HenryHub": "USD/MMBtu"}
    markets = [m for m in MARKETS if (results / m / "vol_h5" / "split.json").exists()]
    if not markets:
        print("skip fig_split: no split.json")
        return
    fig, axes = plt.subplots(len(markets), 1, figsize=(W, 1.3 * len(markets) + 0.4), sharex=True)
    axes = np.atleast_1d(axes)
    c_train, c_test, c_gap, c_burn = st.BLUE, st.ORANGE, "0.88", st.LIGHT
    for ax, mkt in zip(axes, markets):
        sp = json.loads((results / mkt / "vol_h5" / "split.json").read_text())
        price = ec.load_master(data_dir, mkt)[0]["price"].dropna()
        t0, t1 = pd.Timestamp(sp["train_start"]), pd.Timestamp(sp["train_end"])
        t2, t3 = pd.Timestamp(sp["test_start"]), pd.Timestamp(sp["test_end"])
        ax.axvspan(t1, t2, color=c_gap, lw=0, zorder=0)
        ax.plot(price[price.index < t0], lw=0.5, color=c_burn, rasterized=True)
        ax.plot(price[(price.index >= t0) & (price.index <= t1)], lw=0.5, color=c_train, rasterized=True)
        ax.plot(price[(price.index > t1) & (price.index < t2)], lw=0.5, color="0.55", rasterized=True)
        ax.plot(price[(price.index >= t2) & (price.index <= t3)], lw=0.5, color=c_test, rasterized=True)
        ax.set_ylabel(f"{MARKET_LABEL[mkt]}\n({units[mkt]})")
        ax.margins(x=0.005)
        ax.xaxis.set_major_locator(mdates.YearLocator(5))
        ax.xaxis.set_minor_locator(mdates.YearLocator(1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    handles = [Line2D([], [], color=c_burn, lw=1.5, label="feature burn-in"),
               Line2D([], [], color=c_train, lw=1.5, label="training period"),
               Patch(facecolor=c_gap, label="embargo (250 obs.)"),
               Line2D([], [], color=c_test, lw=1.5, label="test period")]
    fig.legend(handles=handles, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(h_pad=0.4, rect=(0, 0.05, 1, 1))
    st.save(fig, out, "fig_split")


# ============================================================
# FOREST PLOTS
# ============================================================

def forest(metrics, target, order, out, name):
    mod = metrics[(metrics.target == target) & (~metrics.is_baseline)]
    if mod.empty:
        print(f"skip {name}: no model results for target '{target}'")
        return
    mod = mod.sort_values("cv_auc", ascending=False).groupby(
        ["market", "h", "feature_set"], as_index=False).first()
    base = metrics[(metrics.target == target) & metrics.is_baseline & (metrics.model != "Majority class")]
    order = [f for f in order if f in set(mod.feature_set)]
    horizons = sorted(mod.h.unique())
    markets = [m for m in MARKETS if m in set(mod.market)]
    lo = min(mod.auc_lo.min(), base.auc.min(), 0.5) - 0.02
    hi = max(mod.auc_hi.max(), base.auc.max()) + 0.02
    row_h = 0.24 * len(order) + 0.25
    fig, axes = plt.subplots(len(markets), len(horizons), figsize=(W, row_h * len(markets) + 0.7),
                             sharex=True, sharey=True, squeeze=False)
    for i, mkt in enumerate(markets):
        for j, h in enumerate(horizons):
            ax = axes[i, j]
            g = mod[(mod.market == mkt) & (mod.h == h)].set_index("feature_set")
            ax.axvline(0.5, color="0.75", lw=0.6, zorder=0)
            for _, b in base[(base.market == mkt) & (base.h == h)].iterrows():
                c, ls = st.BASE_STYLE.get(b["model"], ("0.4", "--"))
                ax.axvline(b["auc"], color=c, ls=ls, lw=0.9, zorder=1)
            for k, fs in enumerate(order):
                if fs not in g.index:
                    continue
                r = g.loc[fs]
                mk, col = st.LEARNER_STYLE.get(r["model"], ("o", "k"))
                ax.errorbar(r["auc"], k, xerr=[[r["auc"] - r["auc_lo"]], [r["auc_hi"] - r["auc"]]],
                            fmt=mk, color=col, ms=3.8, lw=0.9, capsize=1.6, zorder=3)
            st.categorical_y(ax)
            if i == 0:
                ax.set_title(f"$h = {h}$")
            if j == len(horizons) - 1:
                ax.text(1.03, 0.5, MARKET_LABEL[mkt], transform=ax.transAxes, rotation=270,
                        va="center", ha="left")
            if i == len(markets) - 1:
                ax.set_xlabel("test AUC")
    axes[0, 0].set_yticks(range(len(order)))
    axes[0, 0].set_yticklabels(order)
    axes[0, 0].set_ylim(len(order) - 0.5, -0.5)
    axes[0, 0].set_xlim(lo, hi)
    handles = [Line2D([], [], color=c, ls=ls, lw=1.0, label=lab) for lab, (c, ls) in st.BASE_STYLE.items()
               if lab in set(base.model)]
    handles += learner_handles(mod.model)
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(h_pad=0.5, w_pad=0.6, rect=(0, 0.035, 1, 1))
    st.save(fig, out, name)


# ============================================================
# PAIRED DIFFERENCES
# ============================================================

def delta_plot(comps, out):
    pat = re.compile(r"^vol: HAR\+(.+) vs HAR( \[GRU\])?$")
    c = comps[comps.family.str.match(pat)].copy()
    if c.empty:
        print("skip fig_delta_vol: no comparisons")
        return
    c["block"] = c.family.str.extract(pat)[0]
    c["learner"] = c.a.str.split("|").str[1]
    blocks = [b for b in ["BASE", "H", "HC", "HCF", "BASE+HCF"] if b in set(c.block)]
    learners = [l for l in st.LEARNER_STYLE if l in set(c.learner)]
    horizons = sorted(c.h.unique())
    n_rows = len(blocks) * len(MARKETS)
    fig, axes = plt.subplots(1, len(horizons), figsize=(W, 0.34 * n_rows + 1.0), sharey=True, squeeze=False)
    labels = [f"{MARKET_LABEL[m]}: +{b}" for m in MARKETS for b in blocks]
    step = 0.18
    for j, h in enumerate(horizons):
        ax = axes[0, j]
        ax.axvline(0, color="0.4", lw=0.7, zorder=0)
        for k in range(1, len(MARKETS)):
            ax.axhline(k * len(blocks) - 0.5, color="0.85", lw=0.5, zorder=0)
        for li, learner in enumerate(learners):
            mk, col = st.LEARNER_STYLE[learner]
            off = (li - (len(learners) - 1) / 2) * step
            for _, r in c[(c.h == h) & (c.learner == learner)].iterrows():
                if r.market not in MARKETS:
                    continue
                y = MARKETS.index(r.market) * len(blocks) + blocks.index(r.block) + off
                filled = r.p_holm < 0.05
                ax.errorbar(r.delta, y, xerr=[[r.delta - r.delta_lo], [r.delta_hi - r.delta]], fmt=mk,
                            color=col, mfc=col if filled else "white", ms=3.4, lw=0.8, capsize=1.3)
        st.categorical_y(ax)
        ax.set_title(f"$h = {h}$")
        ax.set_xlabel(r"$\Delta$AUC vs. HAR")
    axes[0, 0].set_yticks(range(len(labels)))
    axes[0, 0].set_yticklabels(labels)
    axes[0, 0].set_ylim(len(labels) - 0.5, -0.5)
    handles = learner_handles(learners)
    handles.append(Line2D([], [], marker="o", color="k", mfc="white", ls="", ms=4.5,
                          label=r"open: Holm-adjusted $p \geq 0.05$"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(w_pad=0.8, rect=(0, 0.035, 1, 1))
    st.save(fig, out, "fig_delta_vol")


# ============================================================
# PERMUTATION IMPORTANCE
# ============================================================

def importance_plot(imp, out, feature_set="HAR+BASE+HCF"):
    if imp is None or imp.empty:
        print("skip fig_importance: no importance tables")
        return
    d = imp[(imp.target == "vol") & (imp.feature_set == feature_set)]
    if d.empty:
        print("skip fig_importance: no volatility importance")
        return
    d = d.assign(kind=d.group.str.split(":").str[0], level=d.group.str.split(":").str[1])
    kinds = [k for k in ["family", "channel", "window", "lag"] if k in set(d.kind)]
    natural = {"channel": ["price", "return", "abs_return"], "window": ["50", "75", "100", "250"],
               "lag": ["0", "1-5", "6-10", "11-20"]}
    label = {"price": r"$P_t$", "return": r"$r_t$", "abs_return": r"$|r_t|$"}
    title = {"family": "block / measure", "channel": "ordinal channel",
             "window": "window $W$", "lag": "lag (days)"}
    fig, axes = plt.subplots(1, len(kinds), figsize=(W, 2.3), squeeze=False)
    for j, k in enumerate(kinds):
        g = d[d.kind == k].groupby("level")["auc_drop_mean"].agg(["mean", "std", "count"])
        g = g.sort_values("mean", ascending=False) if k == "family" else \
            g.reindex([v for v in natural[k] if v in g.index])
        ax = axes[0, j]
        se = g["std"] / np.sqrt(g["count"].clip(lower=1))
        ax.barh(range(len(g)), g["mean"], xerr=se, color=st.BLUE, height=0.6,
                error_kw=dict(lw=0.7, capsize=1.5))
        ax.set_yticks(range(len(g)))
        ax.set_yticklabels([label.get(v, v) for v in g.index])
        st.categorical_y(ax)
        ax.invert_yaxis()
        ax.axvline(0, color="0.4", lw=0.6)
        ax.set_title(title.get(k, k))
        ax.set_xlabel("mean AUC drop")
    fig.tight_layout(w_pad=1.0)
    st.save(fig, out, "fig_importance")


# ============================================================
# AUC BY YEAR
# ============================================================

def yearly_plot(yearly, metrics, out):
    if yearly.empty:
        print("skip fig_yearly_vol")
        return
    y = yearly[yearly.target == "vol"]
    mod = metrics[(metrics.target == "vol") & (~metrics.is_baseline)]
    if mod.empty:
        return
    best = mod.sort_values("cv_auc", ascending=False).groupby(["market", "h", "feature_set"], as_index=False).first()
    horizons = sorted(y.h.unique())[:3]
    fig, axes = plt.subplots(len(MARKETS), len(horizons), figsize=(W, 6.6), sharex=True, sharey=True,
                             squeeze=False)
    seen = {}
    for i, mkt in enumerate(MARKETS):
        for j, h in enumerate(horizons):
            ax = axes[i, j]
            ax.axhline(0.5, color="0.8", lw=0.6)
            for fs, col, mk in (("HAR", st.BLUE, "o"), ("HAR+BASE+HCF", st.ORANGE, "s")):
                b = best[(best.market == mkt) & (best.h == h) & (best.feature_set == fs)]
                if b.empty:
                    continue
                g = y[(y.market == mkt) & (y.h == h) & (y.feature_set == fs) & (y.model == b.model.iloc[0])]
                ln, = ax.plot(g.year, g.auc, marker=mk, ms=2.8, lw=0.9, color=col)
                seen.setdefault(fs, ln)
            g = y[(y.market == mkt) & (y.h == h) & (y.model == "HAR logistic")]
            ln, = ax.plot(g.year, g.auc, ls=":", color=st.GREY, lw=1.0)
            seen.setdefault("HAR logistic", ln)
            ax.xaxis.set_major_locator(plt.MultipleLocator(2))
            ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
            if i == 0:
                ax.set_title(f"$h = {h}$")
            if j == 0:
                ax.set_ylabel(f"{MARKET_LABEL[mkt]}\ntest AUC")
    labels = {"HAR": "HAR (CV-selected learner)", "HAR+BASE+HCF": "HAR+BASE+HCF (CV-selected learner)",
              "HAR logistic": "HAR logistic benchmark"}
    fig.legend([seen[k] for k in seen], [labels[k] for k in seen], loc="lower center", ncol=len(seen),
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(h_pad=0.5, rect=(0, 0.035, 1, 1))
    st.save(fig, out, "fig_yearly_vol")


# ============================================================
# CALIBRATION
# ============================================================

def calibration_plot(cal, out):
    if cal.empty:
        print("skip fig_calibration")
        return
    c = cal[(cal.target == "vol") & (cal.feature_set.isin(["HAR", "HAR+BASE+HCF"]))]
    if c.empty:
        return
    horizons = sorted(c.h.unique())[:3]
    fig, axes = plt.subplots(1, len(horizons), figsize=(W, 2.75), sharey=True, squeeze=False)
    for j, h in enumerate(horizons):
        ax = axes[0, j]
        ax.plot([0, 1], [0, 1], color="0.6", lw=0.6, ls="--")
        for (fs, model), g in c[c.h == h].groupby(["feature_set", "model"]):
            gg = g.groupby("bin")[["mean_predicted", "observed"]].mean()
            mk, col = st.LEARNER_STYLE.get(model, ("o", "k"))
            ls = "-" if fs == "HAR" else "--"
            ax.plot(gg.mean_predicted, gg.observed, marker=mk, ms=2.6, lw=0.8, color=col, ls=ls,
                    mfc=col if fs == "HAR" else "white", label=f"{fs} ({model})")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_title(f"$h = {h}$")
        ax.set_xlabel("mean predicted probability")
    axes[0, 0].set_ylabel("observed frequency")
    # compact legend: colour/marker = learner, line style = feature set
    learners = [l for l in st.LEARNER_STYLE if l in set(c.model)]
    handles = [Line2D([], [], marker=st.LEARNER_STYLE[l][0], color=st.LEARNER_STYLE[l][1], ms=3.5,
                      lw=0.9, label=st.LEARNER_LABEL.get(l, l)) for l in learners]
    handles += [Line2D([], [], color="0.2", lw=0.9, ls="-", marker="o", ms=3.5, label="HAR (filled)"),
                Line2D([], [], color="0.2", lw=0.9, ls="--", marker="o", mfc="white", ms=3.5,
                       label="HAR+BASE+HCF (open)")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, 0.0),
               columnspacing=1.0, handlelength=1.8)
    fig.tight_layout(w_pad=0.8, rect=(0, 0.1, 1, 1))
    st.save(fig, out, "fig_calibration")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(HERE.parent / "Results_EAAI"))
    ap.add_argument("--data-dir", default=str(HERE.parent / "Datasets"))
    ap.add_argument("--out", default=str(HERE.parent / "figures"))
    ap.add_argument("--only", nargs="+", default=None,
                    help="subset: split forest_vol forest_dir delta importance yearly calibration")
    a = ap.parse_args(argv)
    results = Path(a.results)
    an = results / "_analysis"
    out = Path(a.out)

    def opt(name):
        p = an / name
        return pd.read_csv(p) if p.exists() and p.stat().st_size > 2 else pd.DataFrame()

    metrics = pd.read_csv(an / "metrics.csv")
    todo = set(a.only or ["split", "forest_vol", "forest_dir", "delta", "importance", "yearly", "calibration"])
    if "split" in todo:
        fig_split(results, a.data_dir, out)
    if "forest_vol" in todo:
        forest(metrics, "vol", VOL_ORDER, out, "fig_forest_vol")
    if "forest_dir" in todo:
        forest(metrics, "dir", DIR_ORDER, out, "fig_forest_dir")
    if "delta" in todo:
        comps = opt("comparisons.csv")
        if not comps.empty:
            delta_plot(comps, out)
    if "importance" in todo:
        importance_plot(opt("group_importance_all.csv"), out)
    if "yearly" in todo:
        yearly_plot(opt("auc_by_year.csv"), metrics, out)
    if "calibration" in todo:
        calibration_plot(opt("calibration.csv"), out)


if __name__ == "__main__":
    main()
