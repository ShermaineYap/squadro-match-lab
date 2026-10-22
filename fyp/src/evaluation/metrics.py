from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any
import warnings

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def expected_calibration_error(
    y_true: np.ndarray,
    probability: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Binary expected calibration error."""
    y_true = np.asarray(y_true)
    probability = np.asarray(probability)
    boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    indices = np.digitize(probability, boundaries[1:-1], right=True)

    error = 0.0
    for bin_index in range(n_bins):
        mask = indices == bin_index
        if not mask.any():
            continue
        observed = y_true[mask].mean()
        predicted = probability[mask].mean()
        error += mask.mean() * abs(observed - predicted)
    return float(error)


def multiclass_expected_calibration_error(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Top-label multiclass expected calibration error."""
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = (predicted == y_true).astype(int)
    return expected_calibration_error(correct, confidence, n_bins=n_bins)


def multiclass_brier_score(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    classes: Sequence[int],
) -> float:
    classes = np.asarray(classes)
    encoded = label_binarize(y_true, classes=classes)
    if encoded.shape[1] == 1 and len(classes) == 2:
        encoded = np.column_stack([1 - encoded[:, 0], encoded[:, 0]])
    return float(np.mean(np.sum((encoded - probabilities) ** 2, axis=1)))


def binary_metrics(
    y_true: Sequence[int],
    prediction: Sequence[int],
    probability: Sequence[float],
) -> dict[str, float]:
    y_true = np.asarray(y_true)
    prediction = np.asarray(prediction)
    probability = np.asarray(probability)

    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "pr_auc": float(average_precision_score(y_true, probability)),
        # The label set is stated rather than inferred. A subgroup or bootstrap
        # resample can contain a single outcome, and an inferred label set would
        # then either raise or silently score a two-column probability matrix
        # against one class.
        "log_loss": float(
            log_loss(
                y_true,
                np.column_stack([1 - probability, probability]),
                labels=[0, 1],
            )
        ),
        "brier": float(brier_score_loss(y_true, probability, pos_label=1)),
        "ece": expected_calibration_error(y_true, probability),
    }


def multiclass_metrics(
    y_true: Sequence[int],
    prediction: Sequence[int],
    probabilities: np.ndarray,
    classes: Sequence[int] = (0, 1, 2),
) -> dict[str, float]:
    y_true = np.asarray(y_true)
    prediction = np.asarray(prediction)
    probabilities = np.asarray(probabilities)
    classes = np.asarray(classes)

    per_class_precision, per_class_recall, per_class_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            prediction,
            labels=classes,
            zero_division=0,
        )
    )

    encoded = label_binarize(y_true, classes=classes)
    try:
        macro_roc_auc = roc_auc_score(
            encoded,
            probabilities,
            average="macro",
            multi_class="ovr",
        )
    except ValueError:
        macro_roc_auc = np.nan

    try:
        macro_pr_auc = average_precision_score(
            encoded,
            probabilities,
            average="macro",
        )
    except ValueError:
        macro_pr_auc = np.nan

    result = {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        # `labels` keeps a class that is never predicted, or never present in a
        # resample, in the average instead of dropping it and inflating the mean.
        "macro_f1": float(
            f1_score(y_true, prediction, labels=classes, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, prediction, labels=classes, average="weighted", zero_division=0)
        ),
        "macro_roc_auc_ovr": float(macro_roc_auc),
        "macro_pr_auc_ovr": float(macro_pr_auc),
        "log_loss": float(log_loss(y_true, probabilities, labels=classes)),
        "multiclass_brier": multiclass_brier_score(y_true, probabilities, classes),
        "ece": multiclass_expected_calibration_error(y_true, probabilities),
    }

    for index, class_value in enumerate(classes):
        result[f"class_{class_value}_precision"] = float(per_class_precision[index])
        result[f"class_{class_value}_recall"] = float(per_class_recall[index])
        result[f"class_{class_value}_f1"] = float(per_class_f1[index])
    return result


def aggregate_fold_metrics(
    fold_metrics: pd.DataFrame,
    model_column: str = "model",
) -> pd.DataFrame:
    metric_columns = [
        column
        for column in fold_metrics.columns
        if column not in {model_column, "fold"}
        and pd.api.types.is_numeric_dtype(fold_metrics[column])
    ]
    rows: list[dict[str, Any]] = []
    for model_name, group in fold_metrics.groupby(model_column):
        row: dict[str, Any] = {"model": model_name, "folds": len(group)}
        for metric in metric_columns:
            row[f"mean_{metric}"] = group[metric].mean()
            row[f"std_{metric}"] = group[metric].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def cluster_bootstrap_confidence_intervals(
    frame: pd.DataFrame,
    cluster_column: str,
    metric_function: Callable[[pd.DataFrame], dict[str, float]],
    n_bootstrap: int = 500,
    random_state: int = 42,
    lower: float = 0.025,
    upper: float = 0.975,
) -> pd.DataFrame:
    """Bootstrap whole clusters, preserving repeated observations within a cluster.

    A replicate can still fail: ROC-AUC and average precision are undefined when
    the resampled clusters happen to contain a single outcome class. Those
    failures are not random - they are the most extreme replicates, exactly the
    tail the interval is meant to describe - so quietly discarding them narrows
    every interval reported. They are counted, carried on the returned frame as
    `failed_replicates` and warned about, and a run in which every replicate
    fails raises rather than returning an empty table.
    """
    rng = np.random.default_rng(random_state)
    clusters = np.asarray(frame[cluster_column].dropna().unique())
    if len(clusters) < 2:
        raise ValueError("At least two clusters are required for clustered bootstrap.")

    bootstrap_rows: list[dict[str, float]] = []
    failures: list[str] = []
    grouped = {cluster: group for cluster, group in frame.groupby(cluster_column)}

    for replicate in range(n_bootstrap):
        sampled_clusters = rng.choice(clusters, size=len(clusters), replace=True)
        sampled_parts = []
        for draw_number, cluster in enumerate(sampled_clusters):
            part = grouped[cluster].copy()
            # Preserve duplicate cluster draws as separate blocks.
            part["_bootstrap_draw"] = draw_number
            sampled_parts.append(part)
        sample = pd.concat(sampled_parts, ignore_index=True)
        try:
            metrics = metric_function(sample)
        except (ValueError, ZeroDivisionError) as error:
            failures.append(f"{type(error).__name__}: {error}")
            continue
        metrics["replicate"] = replicate
        bootstrap_rows.append(metrics)

    if not bootstrap_rows:
        raise ValueError(
            f"All {n_bootstrap} bootstrap replicates failed, so no interval can be "
            f"reported. First failure: {failures[0]}"
        )
    if failures:
        warnings.warn(
            f"{len(failures)} of {n_bootstrap} clustered bootstrap replicates could "
            "not be scored and are excluded from the intervals below, which are "
            "therefore narrower than the data supports. These are typically "
            f"single-class resamples. First failure: {failures[0]}",
            RuntimeWarning,
            stacklevel=2,
        )

    bootstrap_frame = pd.DataFrame(bootstrap_rows)
    rows = []
    for metric in bootstrap_frame.columns:
        if metric == "replicate":
            continue
        values = bootstrap_frame[metric].dropna()
        rows.append(
            {
                "metric": metric,
                "bootstrap_mean": values.mean(),
                "ci_lower": values.quantile(lower),
                "ci_upper": values.quantile(upper),
                "successful_replicates": len(values),
                "attempted_replicates": n_bootstrap,
                # Replicates the metric function refused outright, as opposed to
                # replicates it scored as NaN, which the count above already
                # excludes.
                "failed_replicates": len(failures),
            }
        )

    frame_out = pd.DataFrame(rows).set_index("metric")
    undefined = frame_out.index[
        frame_out["successful_replicates"] < len(bootstrap_frame)
    ].tolist()
    if undefined:
        warnings.warn(
            "These metrics were undefined in at least one replicate and their "
            "intervals are computed from the remainder, which is a non-random "
            f"subset and therefore narrower than the data supports: {undefined}. "
            "The per-metric `successful_replicates` column records how many.",
            RuntimeWarning,
            stacklevel=2,
        )
    return frame_out


def binary_subgroup_metrics(
    frame: pd.DataFrame,
    subgroup_columns: Sequence[str],
    target_column: str,
    prediction_column: str,
    probability_column: str,
    minimum_rows: int = 50,
) -> pd.DataFrame:
    rows = []
    for subgroup_column in subgroup_columns:
        if subgroup_column not in frame.columns:
            continue
        for subgroup_value, group in frame.groupby(subgroup_column, dropna=False):
            if len(group) < minimum_rows or group[target_column].nunique() < 2:
                continue
            metrics = binary_metrics(
                group[target_column],
                group[prediction_column],
                group[probability_column],
            )
            rows.append(
                {
                    "subgroup_variable": subgroup_column,
                    "subgroup_value": str(subgroup_value),
                    "rows": len(group),
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def multiclass_subgroup_metrics(
    frame: pd.DataFrame,
    subgroup_columns: Sequence[str],
    target_column: str,
    prediction_column: str,
    probability_columns: Sequence[str],
    minimum_rows: int = 50,
    classes: Sequence[int] = (0, 1, 2),
) -> pd.DataFrame:
    rows = []
    for subgroup_column in subgroup_columns:
        if subgroup_column not in frame.columns:
            continue
        for subgroup_value, group in frame.groupby(subgroup_column, dropna=False):
            if len(group) < minimum_rows or group[target_column].nunique() < 2:
                continue
            metrics = multiclass_metrics(
                group[target_column],
                group[prediction_column],
                group[list(probability_columns)].to_numpy(),
                classes=classes,
            )
            rows.append(
                {
                    "subgroup_variable": subgroup_column,
                    "subgroup_value": str(subgroup_value),
                    "rows": len(group),
                    **metrics,
                }
            )
    return pd.DataFrame(rows)
