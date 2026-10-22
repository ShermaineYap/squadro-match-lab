from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold


def make_expanding_date_splits(
    date_series: pd.Series,
    n_splits: int = 3,
    minimum_training_fraction: float = 0.25,
) -> list[tuple[np.ndarray, np.ndarray]]:
    dates = pd.Series(pd.to_datetime(date_series)).reset_index(drop=True)
    unique_dates = np.array(sorted(dates.dropna().unique()))

    if len(unique_dates) < n_splits + 2:
        raise ValueError("Not enough unique dates for expanding-window validation.")

    start = max(1, int(len(unique_dates) * minimum_training_fraction))
    remaining = len(unique_dates) - start
    validation_width = max(1, remaining // n_splits)
    splits = []

    for fold in range(n_splits):
        train_end = start + fold * validation_width
        validation_end = (
            len(unique_dates)
            if fold == n_splits - 1
            else min(len(unique_dates), train_end + validation_width)
        )
        train_dates = unique_dates[:train_end]
        validation_dates = unique_dates[train_end:validation_end]

        train_indices = np.flatnonzero(dates.isin(train_dates).to_numpy())
        validation_indices = np.flatnonzero(dates.isin(validation_dates).to_numpy())
        if len(train_indices) == 0 or len(validation_indices) == 0:
            continue
        if dates.iloc[train_indices].max() >= dates.iloc[validation_indices].min():
            raise AssertionError("Chronological order was violated.")
        splits.append((train_indices, validation_indices))

    if len(splits) < 2:
        raise RuntimeError("At least two chronological folds are required.")
    return splits


def date_split_summary(
    date_series: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    dates = pd.Series(pd.to_datetime(date_series)).reset_index(drop=True)
    rows = []
    for fold, (train_indices, validation_indices) in enumerate(splits, start=1):
        rows.append(
            {
                "fold": fold,
                "training_rows": len(train_indices),
                "validation_rows": len(validation_indices),
                "training_start": dates.iloc[train_indices].min(),
                "training_end": dates.iloc[train_indices].max(),
                "validation_start": dates.iloc[validation_indices].min(),
                "validation_end": dates.iloc[validation_indices].max(),
            }
        )
    return pd.DataFrame(rows)


def make_stratified_group_splits(
    y: pd.Series,
    groups: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    placeholder = np.zeros((len(y), 1))
    splits = list(splitter.split(placeholder, y, groups))
    for train_indices, validation_indices in splits:
        train_groups = set(groups.iloc[train_indices])
        validation_groups = set(groups.iloc[validation_indices])
        if not train_groups.isdisjoint(validation_groups):
            raise AssertionError("Group leakage detected.")
    return splits


def group_split_summary(
    y: pd.Series,
    groups: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    rows = []
    for fold, (train_indices, validation_indices) in enumerate(splits, start=1):
        rows.append(
            {
                "fold": fold,
                "training_rows": len(train_indices),
                "validation_rows": len(validation_indices),
                "training_groups": groups.iloc[train_indices].nunique(),
                "validation_groups": groups.iloc[validation_indices].nunique(),
                "training_classes": y.iloc[train_indices].nunique(),
                "validation_classes": y.iloc[validation_indices].nunique(),
            }
        )
    return pd.DataFrame(rows)
