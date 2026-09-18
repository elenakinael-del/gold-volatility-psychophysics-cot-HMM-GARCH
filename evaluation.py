"""Leakage-aware walk-forward evaluation helpers for the research pipeline."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit


def walk_forward_scores(model, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> pd.DataFrame:
    """Evaluate expanding-window folds; never shuffle future observations into training."""
    rows = []
    for fold, (train_idx, test_idx) in enumerate(TimeSeriesSplit(n_splits=n_splits).split(X), 1):
        fitted = clone(model).fit(X.iloc[train_idx], y.iloc[train_idx])
        prediction = fitted.predict(X.iloc[test_idx])
        actual = y.iloc[test_idx]
        rows.append({
            "fold": fold,
            "train_end": str(X.index[train_idx[-1]].date()),
            "test_start": str(X.index[test_idx[0]].date()),
            "test_end": str(X.index[test_idx[-1]].date()),
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "mae": mean_absolute_error(actual, prediction),
            "rmse": float(np.sqrt(mean_squared_error(actual, prediction))),
            "r2": r2_score(actual, prediction),
        })
    return pd.DataFrame(rows)
