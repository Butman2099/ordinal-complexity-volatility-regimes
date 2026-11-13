"""
eaai_seq.py
===========

Recurrent (GRU) comparator for the tabular pipelines.

The GRU receives the information set of the tabular models: the lag block
x_{t-L}, ..., x_t of the same causally transformed variables, reshaped from
columns into a sequence of length L+1 (oldest first). No mutual-information
selection is applied; the network consumes every variable of the set.

Protocol (mirrors RandomizedSearchCV + refit of the tabular learners)
--------------------------------------------------------------------
* Candidates: hidden size x number of training epochs
  (default 3 x 8 = 24 candidates; the tabular learners draw 30).
* Cross-validation on the SAME TimeSeriesSplit(n_splits, gap) folds of the
  training period as the tabular learners. In every fold the variables are
  standardized with statistics of that fold's training part only; for each
  hidden size one network is trained for max(epoch_grid) epochs and its
  validation ROC AUC is recorded after every epoch count in epoch_grid. The
  validation fold is never used for gradient steps or for early stopping,
  so the number of epochs is an ordinary hyperparameter.
* CV AUC of a candidate = mean validation AUC over the folds. The candidate
  with the highest CV AUC is selected; its CV AUC is written as
  `cv_best_auc`, the same quantity as `search.best_score_` of the tabular
  learners, so eaai_analysis.py can select between all learners by CV.
* Refit: standardization on the whole training period; `n_seeds` networks
  with the selected hidden size are trained for the selected number of
  epochs on the whole training period, and the test score is their mean
  predicted probability.

Requires PyTorch (CPU build is sufficient):
    pip install torch --index-url https://download.pytorch.org/whl/cpu
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HIDDEN_GRID = (16, 32, 64)
EPOCH_GRID = (5, 10, 15, 20, 30, 40, 50, 60)


def _sequence_tensor(X_df, variables, n_lags):
    blocks = []
    for lag in range(n_lags, -1, -1):  # oldest first
        names = [v if lag == 0 else f"{v}_t-{lag}" for v in variables]
        blocks.append(X_df[names].to_numpy(dtype=np.float32))
    return np.stack(blocks, axis=1)  # (n, L+1, n_vars)


def _scaler(seq, idx_fit):
    """Per-variable mean and SD from the lag-0 values of the fitting rows."""
    lag0 = seq[idx_fit, -1, :].astype(np.float64)
    mu = lag0.mean(axis=0)
    sd = lag0.std(axis=0)
    sd[sd <= 1e-8] = 1.0
    return mu.astype(np.float32), sd.astype(np.float32)


def fit_predict_gru(
    X_df, y, idx_train, idx_test, variables, cv, n_lags=20, seed=0, log=print,
    hidden_grid=HIDDEN_GRID, epoch_grid=EPOCH_GRID, n_seeds=3, dropout=0.2,
    lr=1e-3, weight_decay=1e-4, batch_size=256,
):
    """
    Returns (test_score, info, cv_table). `cv` is the splitter of the tabular
    learners; it is applied to positions 0..len(idx_train)-1 of the training
    period, exactly as RandomizedSearchCV applies it to X_train.
    """
    import torch
    from torch import nn

    torch.set_num_threads(max(1, (os.cpu_count() or 2)))

    seq = _sequence_tensor(X_df, list(variables), n_lags)
    target = (np.asarray(y) == 1).astype(np.float32)
    idx_train = np.asarray(idx_train)
    idx_test = np.asarray(idx_test)
    epoch_grid = sorted({int(e) for e in epoch_grid})
    max_epochs = max(epoch_grid)

    class GRUNet(nn.Module):
        def __init__(self, n_in, hidden):
            super().__init__()
            self.gru = nn.GRU(n_in, hidden, batch_first=True)
            self.drop = nn.Dropout(dropout)
            self.head = nn.Linear(hidden, 1)

        def forward(self, x):
            _, h_last = self.gru(x)
            return self.head(self.drop(h_last[-1])).squeeze(-1)

    def batch(idx, mu, sd):
        return torch.from_numpy((seq[idx] - mu) / sd)

    def predict(net, idx, mu, sd):
        net.eval()
        out = []
        with torch.no_grad():
            for start in range(0, len(idx), 2048):
                out.append(torch.sigmoid(net(batch(idx[start:start + 2048], mu, sd))).numpy())
        return np.concatenate(out)

    def train(fit_idx, mu, sd, hidden, n_epochs, s, eval_idx=None):
        """Fixed number of epochs; optional validation AUC after the epochs in epoch_grid."""
        torch.manual_seed(s)
        rng = np.random.default_rng(s)
        net = GRUNet(seq.shape[2], hidden)
        opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.BCEWithLogitsLoss()
        aucs = {}
        for epoch in range(1, n_epochs + 1):
            net.train()
            order = rng.permutation(fit_idx)
            for start in range(0, len(order), batch_size):
                b = order[start:start + batch_size]
                opt.zero_grad()
                loss = loss_fn(net(batch(b, mu, sd)), torch.from_numpy(target[b]))
                loss.backward()
                opt.step()
            if eval_idx is not None and epoch in epoch_grid:
                aucs[epoch] = float(roc_auc_score(target[eval_idx], predict(net, eval_idx, mu, sd)))
        return net, aucs

    # ---------------- cross-validation ----------------
    folds = list(cv.split(np.arange(len(idx_train))))
    fold_scores = {(hd, ep): [] for hd in hidden_grid for ep in epoch_grid}
    for k, (tr_pos, va_pos) in enumerate(folds):
        tr, va = idx_train[tr_pos], idx_train[va_pos]
        mu, sd = _scaler(seq, tr)
        best_in_fold = []
        for hd in hidden_grid:
            _, aucs = train(tr, mu, sd, hd, max_epochs, seed + k, eval_idx=va)
            for ep in epoch_grid:
                fold_scores[(hd, ep)].append(aucs[ep])
            best_in_fold.append(f"h{hd} {max(aucs.values()):.3f}")
        log(f"        GRU fold {k + 1}/{len(folds)} (train {len(tr)}, val {len(va)}): "
            + ", ".join(best_in_fold))

    rows = []
    for (hd, ep), vals in fold_scores.items():
        row = {"param_hidden": hd, "param_epochs": ep,
               "mean_test_score": float(np.mean(vals)), "std_test_score": float(np.std(vals))}
        row.update({f"split{i}_test_score": v for i, v in enumerate(vals)})
        rows.append(row)
    cv_table = pd.DataFrame(rows)
    # ties: prefer the smaller network, then fewer epochs
    cv_table = cv_table.sort_values(["mean_test_score", "param_hidden", "param_epochs"],
                                    ascending=[False, True, True]).reset_index(drop=True)
    cv_table["rank_test_score"] = np.arange(1, len(cv_table) + 1)
    best = cv_table.iloc[0]
    hidden, epochs = int(best["param_hidden"]), int(best["param_epochs"])

    # ---------------- refit on the whole training period ----------------
    mu, sd = _scaler(seq, idx_train)
    scores = []
    for s in range(n_seeds):
        net, _ = train(idx_train, mu, sd, hidden, epochs, seed + 100 + s)
        scores.append(predict(net, idx_test, mu, sd))

    n_cols = int(seq.shape[1] * seq.shape[2])
    info = {
        "best_params": {"hidden": hidden, "epochs": epochs},
        "cv_best_auc": float(best["mean_test_score"]),
        "cv_folds": len(folds),
        "n_cv_candidates": int(len(cv_table)),
        "hidden_grid": list(hidden_grid),
        "epoch_grid": list(epoch_grid),
        "n_seeds": n_seeds,
        "n_variables": len(variables),
        "sequence_length": n_lags + 1,
        "n_candidates": n_cols,
        "n_selected": n_cols,
        "fixed": {"layers": 1, "dropout": dropout, "optimizer": "Adam", "lr": lr,
                  "weight_decay": weight_decay, "batch_size": batch_size},
    }
    return np.mean(scores, axis=0), info, cv_table
