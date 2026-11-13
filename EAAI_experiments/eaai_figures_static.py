#!/usr/bin/env python
"""
eaai_figures_static.py
======================

Figures that depend only on the feature datasets (not on model runs).
Style: eaai_style.py (SciencePlots, Computer Modern, cas-sc text width).

  fig_dynamics.pdf  H, C and F at W = 250 through time, four markets; the test
                    period (from Results_EAAI/<market>/vol_h5/split.json, if
                    present) is shaded.
  fig_cecp.pdf      complexity-entropy plane, D = 3, with bounds computed on a
                    fine grid and the estimates shown as a density.

Requires ordpy (bounds of the complexity-entropy plane).

Usage:
    python EAAI_experiments/eaai_figures_static.py --data-dir Datasets --out figures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import eaai_common as ec  # noqa: E402
import eaai_style as st  # noqa: E402

st.apply()
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

W = st.TEXT_WIDTH
CMAPS = {"price": "Blues", "return": "Oranges", "abs_return": "Greens"}
CH_TITLE = {"price": r"price channel $P_t$", "return": r"return channel $r_t$",
            "abs_return": r"absolute-return channel $|r_t|$"}


def bounds(dx=3):
    import ordpy
    mx = np.asarray(ordpy.maximum_complexity_entropy(dx, m=2000))
    mn = np.asarray(ordpy.minimum_complexity_entropy(dx, size=2000))
    mx = mx[np.argsort(mx[:, 0])]
    mn = mn[np.argsort(mn[:, 0])]
    return mx, mn


def test_periods(results):
    out = {}
    for m in st.MARKETS:
        p = Path(results) / m / "vol_h5" / "split.json"
        if p.exists():
            sp = json.loads(p.read_text())
            out[m] = (pd.Timestamp(sp["test_start"]), pd.Timestamp(sp["test_end"]))
    return out


def fig_dynamics(data_dir, results, out, window=250):
    frames = {m: ec.load_master(data_dir, m)[0] for m in st.MARKETS}
    tests = test_periods(results)
    rows = [("H", "price"), ("H", "ret"), ("C", "price"), ("C", "ret"), ("F", "price"), ("F", "ret")]
    fig, axes = plt.subplots(len(rows), len(st.MARKETS), figsize=(W, 7.8), sharex="col", sharey="row")
    for i, (q, kind) in enumerate(rows):
        chans = ["price"] if kind == "price" else ["return", "abs_return"]
        for j, mkt in enumerate(st.MARKETS):
            ax = axes[i, j]
            if mkt in tests:
                ax.axvspan(*tests[mkt], color=st.ORANGE, alpha=0.12, lw=0, zorder=0)
            for ch in chans:
                s = frames[mkt][f"{q}_{ch}_w{window}"].dropna()
                ax.plot(s.index, s.to_numpy(), lw=0.5, color=st.CHANNEL_COLOR[ch], rasterized=True,
                        zorder=2 if ch != "abs_return" else 1)
            ax.xaxis.set_major_locator(mdates.YearLocator(10))
            ax.xaxis.set_minor_locator(mdates.YearLocator(5))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax.yaxis.set_major_locator(plt.MaxNLocator(4))
            if i == 0:
                ax.set_title(st.MARKET_LABEL[mkt])
            if j == 0:
                ax.set_ylabel(f"${q}$, " + ("$P_t$" if kind == "price" else "$r_t$, $|r_t|$"))
    handles = [Line2D([], [], color=st.CHANNEL_COLOR[c], lw=1.5, label=st.CHANNEL_LABEL[c])
               for c in ("price", "return", "abs_return")]
    if tests:
        handles.append(Patch(facecolor=st.ORANGE, alpha=0.12, label="test period"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(h_pad=0.3, w_pad=0.4, rect=(0, 0.03, 1, 1))
    st.save(fig, out, "fig_dynamics")


def fig_cecp(data_dir, out):
    frames = [ec.load_master(data_dir, m)[0] for m in ec.MARKETS]
    mx, mn = bounds(3)
    fig, axes = plt.subplots(2, 3, figsize=(W, 4.4), sharex=True, sharey=True)
    for r, w in enumerate((50, 250)):
        for c, ch in enumerate(("price", "return", "abs_return")):
            ax = axes[r, c]
            H = np.concatenate([f[f"H_{ch}_w{w}"].dropna().to_numpy() for f in frames])
            C = np.concatenate([f[f"C_{ch}_w{w}"].dropna().to_numpy() for f in frames])
            ax.plot(mx[:, 0], mx[:, 1], color="0.5", lw=0.8)
            ax.plot(mn[:, 0], mn[:, 1], color="0.5", lw=0.8)
            ax.hexbin(H, C, gridsize=(70, 45), extent=(0.80, 1.0, 0.0, 0.24),
                      cmap=CMAPS[ch], norm=LogNorm(), mincnt=1, linewidths=0)
            ax.set_xlim(0.80, 1.0)
            ax.set_ylim(0.0, 0.24)
            if r == 0:
                ax.set_title(CH_TITLE[ch])
            if c == 0:
                ax.set_ylabel(f"$C$ ($W = {w}$)")
            if r == 1:
                ax.set_xlabel("permutation entropy $H$")
            if r == 1 and c == 2:
                h0 = 0.995
                cmax = np.interp(h0, mx[:, 0], mx[:, 1])
                cmin = np.interp(h0, mn[:, 0], mn[:, 1])
                ax.annotate(f"band width {cmax - cmin:.4f}\nat $H = {h0}$", xy=(h0, cmax),
                            xytext=(0.812, 0.03), fontsize=st.TICK,
                            arrowprops=dict(arrowstyle="-", color="0.3", lw=0.6))
    fig.tight_layout(h_pad=0.6, w_pad=0.6)
    st.save(fig, out, "fig_cecp")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(HERE.parent / "Datasets"))
    ap.add_argument("--results", default=str(HERE.parent / "Results_EAAI"))
    ap.add_argument("--out", default=str(HERE.parent / "figures"))
    ap.add_argument("--only", nargs="+", default=["dynamics", "cecp"])
    a = ap.parse_args()
    if "dynamics" in a.only:
        fig_dynamics(a.data_dir, a.results, Path(a.out))
    if "cecp" in a.only:
        fig_cecp(a.data_dir, Path(a.out))
