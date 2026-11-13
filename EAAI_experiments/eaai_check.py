#!/usr/bin/env python
"""
eaai_check.py
=============

Integrity check of Results_EAAI, e.g. after an unexpected shutdown.

A cell (feature set x model) counts as finished when its predictions.csv
exists; eaai_experiments.py writes that file last. If the computer lost
power exactly while that file was being written, it could be truncated.
This script finds such files and, with --fix, renames them to
predictions.csv.broken so that the next run recomputes the cell.

It also reports progress and any <model>_error.txt files.

Usage (from the project folder):
    python EAAI_experiments\\eaai_check.py
    python EAAI_experiments\\eaai_check.py --fix
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(HERE.parent / "Results_EAAI"))
    ap.add_argument("--fix", action="store_true", help="rename broken predictions.csv files")
    a = ap.parse_args(argv)
    root = Path(a.results)
    if not root.exists():
        raise SystemExit(f"{root} does not exist")

    ok, broken, by_model = 0, [], {}
    for exp_dir in sorted(p.parent for p in root.glob("*/*/split.json")):
        try:
            labels = pd.read_csv(exp_dir / "test_labels.csv", index_col=0)
        except Exception:
            print(f"[info] {exp_dir.relative_to(root)}: test_labels.csv unreadable "
                  f"(it is rewritten automatically on the next run)")
            labels = None
        for pred_path in sorted(exp_dir.glob("*/*/predictions.csv")):
            model = pred_path.parent.name
            reason = None
            try:
                pred = pd.read_csv(pred_path, index_col=0)
                if not {"y", "score", "pred"}.issubset(pred.columns):
                    reason = "missing columns"
                elif not np.isfinite(pred["score"].to_numpy(dtype=float)).all():
                    reason = "non-finite scores"
                elif labels is not None and len(pred) != len(labels):
                    reason = f"{len(pred)} rows instead of {len(labels)}"
                elif labels is not None and not np.array_equal(pred["y"].to_numpy(), labels["y"].to_numpy()):
                    reason = "labels differ from test_labels.csv"
            except Exception as exc:
                reason = f"unreadable ({type(exc).__name__})"
            if reason:
                broken.append((pred_path, reason))
            else:
                ok += 1
                by_model[model] = by_model.get(model, 0) + 1

    print(f"finished cells: {ok}  " + ", ".join(f"{m}: {n}" for m, n in sorted(by_model.items())))
    print("expected for the full plan: 132 per model (LR, HGB, RF) -> 264 after stage 1, 396 after stage 2")
    print("optional GRU arm (run_gru.bat): +72 cells -> 468 in total")
    for path, reason in broken:
        print(f"[BROKEN] {path.relative_to(root)}: {reason}")
        if a.fix:
            path.rename(path.with_name("predictions.csv.broken"))
            print("         renamed to predictions.csv.broken; it will be recomputed")
    if broken and not a.fix:
        print("run again with --fix to rename the broken files")
    errors = sorted(root.glob("*/*/*/*_error.txt"))
    for e in errors:
        print(f"[ERROR FILE] {e.relative_to(root)}")
    if not broken and not errors:
        print("no problems found")


if __name__ == "__main__":
    main()
