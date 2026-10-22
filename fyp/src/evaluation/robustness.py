from __future__ import annotations

from itertools import product
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score


def per_athlete_quality(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    rows = []
    for athlete_id, group in frame.groupby("athlete_id"):
        counts = group["injury_occurred"].value_counts(normalize=True)
        rows.append({
            "athlete_id": athlete_id,
            "records": len(group),
            "missing_cells": int(group[feature_columns].isna().sum().sum()),
            "missing_rate": float(group[feature_columns].isna().mean().mean()),
            "healthy_rate": float(counts.get(0, 0.0)),
            "low_risk_rate": float(counts.get(1, 0.0)),
            "high_risk_rate": float(counts.get(2, 0.0)),
        })
    return pd.DataFrame(rows)


def distribution_shift_table(
    development: pd.DataFrame,
    audit: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
) -> pd.DataFrame:
    rows = []
    for feature in numeric_features:
        left, right = development[feature], audit[feature]
        pooled = np.sqrt((left.var() + right.var()) / 2.0)
        smd = (right.mean() - left.mean()) / pooled if pooled and not np.isnan(pooled) else np.nan
        rows.append({
            "feature": feature, "type": "numeric", "shift_metric": "standardized_mean_difference",
            "shift_value": float(smd), "development_missing": float(left.isna().mean()),
            "audit_missing": float(right.isna().mean()),
        })
    for feature in categorical_features:
        left = development[feature].astype("string").fillna("Missing").value_counts(normalize=True)
        right = audit[feature].astype("string").fillna("Missing").value_counts(normalize=True)
        levels = left.index.union(right.index)
        distance = 0.5 * (left.reindex(levels, fill_value=0) - right.reindex(levels, fill_value=0)).abs().sum()
        rows.append({
            "feature": feature, "type": "categorical", "shift_metric": "total_variation_distance",
            "shift_value": float(distance), "development_missing": float(development[feature].isna().mean()),
            "audit_missing": float(audit[feature].isna().mean()),
        })
    return pd.DataFrame(rows).sort_values("shift_value", key=lambda x: x.abs(), ascending=False)


def select_class_multipliers(
    y_true, probabilities, grid=(0.6, 0.8, 1.0, 1.2, 1.5), minimum_low_risk_recall: float | None = None,
) -> dict:
    """Select class probability multipliers using grouped OOF predictions only."""
    y_true, probabilities = np.asarray(y_true), np.asarray(probabilities)
    candidates = []
    for values in product(grid, repeat=3):
        prediction = (probabilities * np.asarray(values)).argmax(axis=1)
        recalls = recall_score(y_true, prediction, labels=[0, 1, 2], average=None, zero_division=0)
        candidates.append({
            "multipliers": values,
            "macro_f1": float(f1_score(y_true, prediction, average="macro")),
            "balanced_accuracy": float(recalls.mean()),
            "healthy_recall": float(recalls[0]),
            "low_risk_recall": float(recalls[1]),
            "high_risk_recall": float(recalls[2]),
        })
    eligible = (
        [row for row in candidates if row["low_risk_recall"] >= minimum_low_risk_recall]
        if minimum_low_risk_recall is not None else candidates
    )
    pool = eligible or candidates
    return max(pool, key=lambda row: (row["macro_f1"], row["high_risk_recall"], row["balanced_accuracy"]))


def apply_class_multipliers(probabilities, multipliers):
    weighted = np.asarray(probabilities) * np.asarray(multipliers)
    return weighted.argmax(axis=1)


def out_of_distribution_flags(payload: dict, reference_quantiles: dict, categorical_options: dict | None = None):
    flags = []
    for feature, bounds in reference_quantiles.items():
        value = payload.get(feature)
        if value is None or pd.isna(value):
            flags.append({"feature": feature, "severity": "warning", "reason": "missing input"})
            continue
        lower, upper = bounds.get("q01"), bounds.get("q99")
        if lower is not None and upper is not None and not (float(lower) <= float(value) <= float(upper)):
            flags.append({"feature": feature, "severity": "warning", "reason": "outside development 1st–99th percentile"})
    for feature, allowed in (categorical_options or {}).items():
        value = payload.get(feature)
        # A blank category is an absent answer, not a category the model has
        # never seen. Reporting it as "unseen category" sent the reader looking
        # for an unusual value when what happened is that nothing was entered.
        if value is None or (not isinstance(value, str) and pd.isna(value)) or value == "":
            flags.append({"feature": feature, "severity": "warning", "reason": "missing input"})
            continue
        if not _category_is_known(value, allowed):
            flags.append({"feature": feature, "severity": "warning", "reason": "unseen category"})
    return flags


def _category_is_known(value, allowed) -> bool:
    """Whether a submitted category is one the model was fitted on.

    Compared as text first, then numerically, because the same category can
    reach here in either form: the injury pipeline encodes ``playing_surface``
    as the integer codes 0-4 while the schema endpoint offers them as the
    strings "0"-"4", and JSON carries a whole number as either. A text-only
    comparison reports 2.0 as an unseen surface and puts a false
    out-of-distribution warning on an ordinary request.
    """
    allowed_values = list(allowed)
    if str(value) in {str(option) for option in allowed_values}:
        return True
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    for option in allowed_values:
        try:
            if float(option) == numeric:
                return True
        except (TypeError, ValueError):
            continue
    return False
