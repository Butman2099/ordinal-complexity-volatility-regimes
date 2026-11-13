"""
eaai_style.py
=============

Common style of all manuscript figures.

* SciencePlots styles 'science' + 'no-latex' (pip install SciencePlots);
  'no-latex' keeps the figures independent of a LaTeX installation.
* Text in Computer Modern (matplotlib's bundled cmr10 + mathtext 'cm'), the
  body font of the cas-sc manuscript.
* Figures are drawn at the text width of cas-sc (468.33 pt = 6.48 in) and
  included with width=\\textwidth, i.e. at 100 % scale, so the font sizes
  below are the printed sizes: 9 pt for labels, titles and legends (the
  caption size; body text is 10 pt) and 8 pt for tick labels.

Labels must avoid characters that cmr10 lacks (|, <, >, _, unicode minus):
write them in mathtext, e.g. r"$|r_t|$", r"$p \\geq 0.05$".
"""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TEXT_WIDTH = 468.33 / 72.27          # inches, \textwidth of cas-sc
FONT = 9                             # labels, titles, legends (pt, printed)
TICK = 8                             # tick labels (pt, printed)

# SciencePlots 'science' colour cycle
BLUE, GREEN, ORANGE, RED, PURPLE, GREY, LIGHT = (
    "#0C5DA5", "#00B945", "#FF9500", "#FF2C00", "#845B97", "#474747", "#9E9E9E")

# learner identity: colour + marker (marker keeps them apart in grey scale / CVD)
LEARNER_STYLE = {"LR": ("o", BLUE), "RF": ("s", GREEN), "HGB": ("D", ORANGE),
                 "GRU": ("^", PURPLE), "KNN": ("v", GREY)}
LEARNER_LABEL = {"LR": "LR", "RF": "RF", "HGB": "HGB", "GRU": "GRU", "KNN": "KNN"}
# benchmarks: colour + line style
BASE_STYLE = {"RV mean reversion": (RED, "--"), "HAR logistic": (GREY, ":"),
              "Momentum": (RED, "--")}
# signal channels of the ordinal block
CHANNEL_COLOR = {"price": BLUE, "return": ORANGE, "abs_return": GREEN}
CHANNEL_LABEL = {"price": r"price $P_t$", "return": r"return $r_t$", "abs_return": r"$|r_t|$"}
MARKETS = ["WTI", "Brent", "Gasoline", "HenryHub"]
MARKET_LABEL = {"WTI": "WTI", "Brent": "Brent", "Gasoline": "Gasoline", "HenryHub": "Henry Hub"}

_applied = False


def apply():
    """Activate the style once per process."""
    global _applied
    if _applied:
        return
    try:
        import scienceplots  # noqa: F401  (registers the styles)
        plt.style.use(["science", "no-latex"])
    except Exception:  # pragma: no cover - fallback keeps the scripts usable
        warnings.warn("SciencePlots is not installed (pip install SciencePlots); "
                      "using an equivalent built-in configuration.")
        plt.rcParams.update({
            "axes.prop_cycle": plt.cycler(color=[BLUE, GREEN, ORANGE, RED, PURPLE, GREY, LIGHT]),
            "xtick.direction": "in", "ytick.direction": "in",
            "xtick.top": True, "ytick.right": True,
            "xtick.minor.visible": True, "ytick.minor.visible": True,
            "xtick.major.size": 3, "ytick.major.size": 3,
            "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
            "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
            "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
            "lines.linewidth": 1.0, "legend.frameon": False,
        })
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["cmr10", "Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "mathtext.rm": "serif",
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": True,
        "font.size": FONT, "axes.labelsize": FONT, "axes.titlesize": FONT,
        "legend.fontsize": FONT, "figure.titlesize": FONT,
        "xtick.labelsize": TICK, "ytick.labelsize": TICK,
        "legend.handlelength": 1.6, "legend.columnspacing": 1.4,
        "axes.titlepad": 4,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "figure.dpi": 100,
    })
    _applied = True


def categorical_y(ax):
    """No minor ticks on a categorical (label) axis."""
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.tick_params(axis="y", which="major", length=0)


def save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=300)
    plt.close(fig)
    print("written", out / f"{name}.pdf")
