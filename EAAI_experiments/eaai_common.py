"""
eaai_common.py
==============

Shared building blocks for the EAAI experiments:

* market registry and column groups of the *_MASTER.csv feature files
  produced by features_gen.ipynb;
* causal predictor representations and lag construction
  (transform first, lag second);
* the two targets
    - "vol": future volatility regime  (RV_forward > RV_past),
    - "dir": direction of the h-step price change (negative control);
* baselines that require no fitting, and econometric baselines;
* a mutual-information selector with a disk cache, so that the MI
  ranking of a training fold is computed once and shared by every
  hyperparameter candidate and every model;
* group permutation importance.

Every quantity indexed by t uses observations up to and including t only.
The label at t uses observations t+1 .. t+h only.
"""

from __future__ import annotations

import hashlib
import inspect
import os
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:  # scikit-learn >= 1.6
    from sklearn.utils.validation import validate_data as _validate_data
except ImportError:  # pragma: no cover - older scikit-learn
    _validate_data = None


# ============================================================
# MARKETS AND COLUMN GROUPS
# ============================================================

MARKETS = {
    "WTI": ("WTI_complexity_datasets", "DCOILWTICO"),
    "Brent": ("Brent_complexity_datasets", "DCOILBRENTEU"),
    "Gasoline": ("Gasoline_complexity_datasets", "DGASUSGULF"),
    "HenryHub": ("Henry_Hub_complexity_datasets", "DHHNGSP"),
}

MEASURES = ("H", "C", "F")
CHANNELS = ("price", "return", "abs_return")
WINDOWS = (50, 75, 100, 250)

BASE_PATTERN = re.compile(
    r"^(return_(mean|median|std|skew|kurt)"
    r"|abs_return_(mean|std)"
    r"|realized_vol"
    r"|cumulative_return)_w\d+$"
)
LAG_PATTERN = re.compile(r"^(?P<root>.+?)_t-(?P<lag>\d+)$")


def hcf_columns(measures="HCF"):
    return [
        f"{m}_{ch}_w{w}"
        for m in measures
        for ch in CHANNELS
        for w in WINDOWS
    ]


def base_columns(columns):
    return [c for c in columns if BASE_PATTERN.match(str(c))]


CORE_SETS = ("PRICE", "H", "HC", "HCF", "BASE", "BASE+HCF", "HCF+PRICE")
HAR_PREFIX = "log_rv_"


def parse_feature_set(name):
    """
    'HAR+<core>' -> (True, '<core>'); 'HAR' -> (True, None);
    '<core>' -> (False, '<core>').

    HAR denotes the control block of log realized volatilities at 1, 5, 22
    days and at the horizon h (volatility target only). In the tabular
    pipelines its lag-0 columns are always retained by the selector, so a
    'HAR+X' arm measures what X adds to the econometric benchmark.
    """
    if name == "HAR":
        return True, None
    if name.startswith("HAR+"):
        core = name[4:]
        if core not in CORE_SETS:
            raise KeyError(f"Unknown core set '{core}' in '{name}'")
        return True, core
    if name not in CORE_SETS:
        raise KeyError(f"Unknown feature set '{name}'. Known: {CORE_SETS} and HAR+<core>")
    return False, name


def feature_set_variables(name, master_columns):
    """Base (un-lagged) master variables of a core feature set."""
    base = base_columns(master_columns)
    sets = {
        "PRICE": ["price"],
        "H": hcf_columns("H"),
        "HC": hcf_columns("HC"),
        "HCF": hcf_columns("HCF"),
        "BASE": base,
        "BASE+HCF": base + hcf_columns("HCF"),
        "HCF+PRICE": hcf_columns("HCF") + ["price"],
    }
    if name not in sets:
        raise KeyError(f"Unknown feature set '{name}'. Known: {list(sets)}")
    return sets[name]


def load_master(data_dir, market):
    folder, series_id = MARKETS[market]
    path = Path(data_dir) / folder / f"{series_id}_MASTER.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    missing = [c for c in hcf_columns("HCF") + ["price"] if c not in df.columns]
    if missing:
        raise KeyError(f"{path.name}: missing columns {missing[:5]} ...")
    return df, series_id


# ============================================================
# PREDICTOR REPRESENTATION (causal) AND LAGS
# ============================================================

def rolling_zscore(x, window, min_periods=None, eps=1e-8):
    """Causal rolling z-score; the window ends at t inclusive (ddof=0)."""
    x = pd.to_numeric(x, errors="coerce").astype(float)
    roll = x.rolling(window, min_periods=min_periods or window)
    mean = roll.mean()
    scale = roll.std(ddof=0)
    scale = scale.mask(~np.isfinite(scale)).mask(scale.abs() <= eps, 1.0)
    return ((x - mean) / scale).replace([np.inf, -np.inf], np.nan)


def expanding_zscore(x, min_periods=250, eps=1e-8):
    """Causal expanding z-score; statistics use observations up to t."""
    x = pd.to_numeric(x, errors="coerce").astype(float)
    exp = x.expanding(min_periods=min_periods)
    mean = exp.mean()
    scale = exp.std(ddof=0)
    scale = scale.mask(~np.isfinite(scale)).mask(scale.abs() <= eps, 1.0)
    return ((x - mean) / scale).replace([np.inf, -np.inf], np.nan)


# group -> (kind, parameter); kind in {"none", "rolling", "expanding"}
REPRESENTATIONS = {
    # Complexity and conventional levels left intact (a StandardScaler
    # fitted per training fold sits inside every pipeline); the price
    # level, which is non-stationary, becomes a 250-day causal z-score.
    "levels": {
        "complexity": ("none", None),
        "base": ("none", None),
        "price": ("rolling", 250),
    },
    "expanding": {
        "complexity": ("expanding", 250),
        "base": ("expanding", 250),
        "price": ("expanding", 250),
    },
    # Short-window rolling z-score; kept for comparison only.
    "legacy_z20": {
        "complexity": ("rolling", 20),
        "base": ("rolling", 20),
        "price": ("rolling", 20),
    },
}


def variable_group(name):
    if re.match(r"^[HCF]_", name):
        return "complexity"
    if name == "price":
        return "price"
    return "base"


def transform_variable(series, representation):
    kind, param = REPRESENTATIONS[representation][variable_group(series.name)]
    if kind == "none":
        return pd.to_numeric(series, errors="coerce").astype(float)
    if kind == "rolling":
        return rolling_zscore(series, window=param)
    if kind == "expanding":
        return expanding_zscore(series, min_periods=param)
    raise ValueError(kind)


def make_lagged_matrix(master, variables, representation="levels", n_lags=20):
    """Transform each variable causally, then append lags 1..n_lags."""
    transformed = pd.DataFrame(
        {v: transform_variable(master[v].rename(v), representation) for v in variables},
        index=master.index,
    )
    blocks = [transformed]
    for lag in range(1, n_lags + 1):
        shifted = transformed.shift(lag)
        shifted.columns = [f"{c}_t-{lag}" for c in transformed.columns]
        blocks.append(shifted)
    out = pd.concat(blocks, axis=1)
    return out.replace([np.inf, -np.inf], np.nan)


def column_root_and_lag(column):
    m = LAG_PATTERN.match(column)
    if m:
        return m.group("root"), int(m.group("lag"))
    return column, 0


def columns_for_set(all_columns, variables):
    keep = set(variables)
    return [c for c in all_columns if column_root_and_lag(c)[0] in keep]


# ============================================================
# TARGETS
# ============================================================

def simple_returns(price):
    p = pd.to_numeric(price, errors="coerce").astype(float)
    p = p.mask(p == 0.0)
    return p.pct_change(fill_method=None)


def realized_vol(returns, k):
    """sqrt(sum_{i=0..k-1} r_{t-i}^2): known at t."""
    return returns.pow(2).rolling(k, min_periods=k).sum().pow(0.5)


def target_volatility_regime(price, h):
    """
    y_t = +1 if RV[t+1..t+h] > RV[t-h+1..t]  ("expansion"), -1 otherwise.
    """
    r = simple_returns(price)
    sq = r.pow(2)
    rv_past = sq.rolling(h, min_periods=h).sum().pow(0.5)
    rv_fwd = sq.shift(-h).rolling(h, min_periods=h).sum().pow(0.5)
    ref = rv_past.mask(~np.isfinite(rv_past) | (rv_past <= 0))
    out = pd.DataFrame(index=price.index)
    out["rv_past"] = rv_past
    out["rv_fwd"] = rv_fwd
    out["label"] = np.nan
    out.loc[rv_fwd > ref, "label"] = 1
    out.loc[rv_fwd < ref, "label"] = -1
    out = out.dropna(subset=["rv_past", "rv_fwd", "label"])
    out["label"] = out["label"].astype(int)
    return out


def target_direction(price, h):
    """
    y_t = sign(P_{t+h} - P_t), zero changes dropped.

    The sign of the price CHANGE is used instead of the sign of the simple
    return, because the two differ when P_t < 0 (WTI, 20 April 2020).
    The reference is exactly zero: no quantity observable at t enters the
    label, so a score that requires no fitting has no information about it by
    construction.
    """
    p = pd.to_numeric(price, errors="coerce").astype(float)
    change = p.shift(-h) - p
    out = pd.DataFrame(index=p.index)
    out["fwd_change"] = change
    out["fwd_return"] = change / p.abs()
    out["label"] = np.sign(change).replace(0.0, np.nan)
    out = out.dropna(subset=["label"])
    out["label"] = out["label"].astype(int)
    return out


TARGET_NAMES = {
    "vol": ("Contraction", "Expansion"),
    "dir": ("Down", "Up"),
}


# ============================================================
# BASELINES
# ============================================================

def har_design(price, h, eps=1e-6):
    """log realized volatility at 1, 5, 22 days and at the horizon h."""
    r = simple_returns(price)
    cols = {}
    for k in sorted({1, 5, 22, h}):
        cols[f"log_rv_{k}"] = np.log(realized_vol(r, k) + eps)
    return pd.DataFrame(cols, index=price.index)


def momentum_score(price, h):
    p = pd.to_numeric(price, errors="coerce").astype(float)
    return (p - p.shift(h)) / p.shift(h).abs()


def fit_har_logit(design_train, y_train, design_test):
    model = Pipeline(
        [("scale", StandardScaler()), ("clf", LogisticRegression(C=1.0, max_iter=2000))]
    )
    model.fit(design_train.to_numpy(), np.asarray(y_train))
    return model.predict_proba(design_test.to_numpy())[:, 1]


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

def chronological_split(n, test_size=0.2, gap=250):
    split_index = int(n * (1.0 - test_size))
    test_start = split_index + gap
    if split_index <= 0 or test_start >= n:
        raise ValueError("test_size/gap leave no train or test observations")
    return split_index, test_start


# ============================================================
# MUTUAL-INFORMATION SELECTOR WITH DISK CACHE
# ============================================================

_MI_HAS_NJOBS = "n_jobs" in inspect.signature(mutual_info_classif).parameters
_MI_CACHE_VERSION = "mi-v1"


def _mi_key(X, y, n_neighbors, random_state):
    hsh = hashlib.blake2b(digest_size=20)
    hsh.update(_MI_CACHE_VERSION.encode())
    hsh.update(str(X.shape).encode())
    hsh.update(np.ascontiguousarray(X).view(np.uint8).ravel())
    hsh.update(np.ascontiguousarray(y.astype(np.int64)).view(np.uint8).ravel())
    hsh.update(f"{n_neighbors}|{random_state}".encode())
    return hsh.hexdigest()


def mi_scores(X, y, n_neighbors=3, random_state=42, cache_dir=None, n_jobs=1):
    """
    Mutual information between every column of X and the discrete y
    (Kraskov / Ross nearest-neighbour estimator, scikit-learn).

    Results are cached on disk under a hash of (X, y, settings), so a
    training fold is scored once however many candidates and models use it.
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    y = np.asarray(y)
    path = None
    if cache_dir is not None:
        key = _mi_key(X, y, n_neighbors, random_state)
        path = Path(cache_dir) / f"{key}.npy"
        if path.exists():
            try:
                return np.load(path)
            except Exception:
                pass  # corrupted or partially written file: recompute
    kwargs = dict(discrete_features=False, n_neighbors=n_neighbors, random_state=random_state)
    if _MI_HAS_NJOBS:
        kwargs["n_jobs"] = n_jobs
    scores = mutual_info_classif(X, y, **kwargs)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        os.close(fd)
        try:
            with open(tmp, "wb") as fh:
                np.save(fh, scores)
            try:
                os.replace(tmp, path)
            except PermissionError:
                # Windows: another worker is writing or reading the same key;
                # its copy is identical, so this one can be discarded.
                pass
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
    return scores


class MISelector(SelectorMixin, BaseEstimator):
    """
    Keep the top `percentile` % of columns ranked by mutual information
    with the target, estimated on the data passed to fit() only.

    Equivalent to SelectPercentile(mutual_info_classif) except that the
    scores are cached, and ties are broken by column order.
    """

    def __init__(self, percentile=10, n_always=0, n_neighbors=3, random_state=42, cache_dir=None):
        self.percentile = percentile
        self.n_always = n_always
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        self.cache_dir = cache_dir

    def fit(self, X, y):
        """
        The first `n_always` columns (a control block such as the HAR
        volatility terms) are always kept; the percentile applies to the
        remaining columns, ranked by their mutual information with y.
        """
        if _validate_data is not None:
            X, y = _validate_data(self, X, y, dtype=np.float64, reset=True)
        X = np.ascontiguousarray(X, dtype=np.float64)
        n_cols = X.shape[1]
        self.n_features_in_ = n_cols
        self.scores_ = mi_scores(
            X, np.asarray(y), self.n_neighbors, self.random_state, self.cache_dir
        )
        a = int(min(max(self.n_always, 0), n_cols))
        mask = np.zeros(n_cols, dtype=bool)
        mask[:a] = True
        rest = n_cols - a
        if rest > 0:
            k = max(1, int(np.ceil(rest * float(self.percentile) / 100.0)))
            order = a + np.argsort(-self.scores_[a:], kind="stable")
            mask[order[:k]] = True
        self.mask_ = mask
        return self

    def _get_support_mask(self):
        return self.mask_


# ============================================================
# SCORING HELPERS AND GROUP PERMUTATION IMPORTANCE
# ============================================================

def positive_score(model, X):
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)[:, 1]
        except (AttributeError, NotImplementedError):
            pass
    return model.decision_function(X)


def feature_groups(columns):
    """
    Column-index groups used for permutation importance:
    family (H, C, F, BASE, PRICE), channel (complexity columns only),
    window and lag bucket.
    """
    groups = {}

    def add(name, i):
        groups.setdefault(name, []).append(i)

    for i, col in enumerate(columns):
        root, lag = column_root_and_lag(col)
        if root.startswith(HAR_PREFIX):
            add("family:HAR", i)
            continue
        if root == "price":
            add("family:PRICE", i)
        elif re.match(r"^[HCF]_", root):
            measure, rest = root.split("_", 1)
            channel = rest.rsplit("_w", 1)[0]
            add(f"family:{measure}", i)
            add(f"channel:{channel}", i)
        else:
            add("family:BASE", i)
        m = re.search(r"_w(\d+)$", root)
        if m:
            add(f"window:{m.group(1)}", i)
        if lag == 0:
            add("lag:0", i)
        elif lag <= 5:
            add("lag:1-5", i)
        elif lag <= 10:
            add("lag:6-10", i)
        else:
            add("lag:11-20", i)
    return {k: np.asarray(v) for k, v in groups.items()}


def group_permutation_importance(model, X, y, groups, n_repeats=5, random_state=0):
    """Drop in test ROC AUC when all columns of a group are permuted jointly."""
    rng = np.random.default_rng(random_state)
    X = np.asarray(X, dtype=np.float64)
    base = roc_auc_score(y, positive_score(model, X))
    rows = []
    for name, idx in groups.items():
        drops = []
        for _ in range(n_repeats):
            perm = rng.permutation(X.shape[0])
            Xp = X.copy()
            Xp[:, idx] = X[perm][:, idx]
            drops.append(base - roc_auc_score(y, positive_score(model, Xp)))
        rows.append(
            {
                "group": name,
                "n_columns": int(len(idx)),
                "auc_full": base,
                "auc_drop_mean": float(np.mean(drops)),
                "auc_drop_sd": float(np.std(drops, ddof=1)) if n_repeats > 1 else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("auc_drop_mean", ascending=False)
