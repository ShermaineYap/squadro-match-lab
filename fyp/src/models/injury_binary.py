"""Binary injury-risk modelling: combined Low Injury Risk versus High Injury Risk."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV

from src.evaluation.metrics import binary_metrics
from src.models.common import Candidate, RANDOM_STATE
from src.models.injury_modeling import make_standard_pipeline, make_smotenc_pipeline
from src.models.splitting import make_stratified_group_splits

try:  # pragma: no cover - import guard mirrors src.models.injury_modeling
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:  # pragma: no cover
    XGBOOST_AVAILABLE = False


BINARY_TARGET_MAPPING: dict[int, int] = {0: 0, 1: 0, 2: 1}

BINARY_TARGET_PROVENANCE: dict[str, Any] = {
    "source_target": "injury_occurred",
    "source_labels": {"0": "Healthy", "1": "Low Risk", "2": "High Risk/Injured"},
    "binary_labels": {"0": "Low Injury Risk", "1": "High Injury Risk"},
    "mapping": {
        "Healthy": "Low Injury Risk",
        "Low Risk": "Low Injury Risk",
        "High Risk/Injured": "High Injury Risk",
    },
    "positive_class": "High Injury Risk",
    "positive_class_index": 1,
}

BINARY_CLASS_NAMES: tuple[str, str] = ("Low Injury Risk", "High Injury Risk")

BINARY_RESULT_EXPLANATIONS: dict[str, str] = {
    "Low Injury Risk": (
        "The available measurements do not show a strong High Injury Risk "
        "pattern. This result combines the dataset's Healthy and Low Risk "
        "categories and is not medical clearance."
    ),
    "High Injury Risk": (
        "The available measurements resemble the dataset's High Risk/Injured "
        "category. This is a screening result, not an injury diagnosis."
    ),
}

BINARY_TASK_DESCRIPTION = (
    "Binary injury-risk classification that distinguishes a combined Low "
    "Injury Risk group from a High Injury Risk group."
)

PROHIBITED_RISK_TERMS: tuple[str, ...] = (
    "safe",
    "no injury risk",
    "injury-free",
    "injury free",
    "cleared to play",
    "will be injured",
    "will not be injured",
)


def to_binary_target(target: pd.Series | np.ndarray | list) -> pd.Series:
    """Map the provider's three-class codes onto the binary target.

    Raises on any code outside {0, 1, 2} rather than silently producing a
    negative label, because a quietly mis-mapped row would move an athlete out
    of the positive class without leaving a trace.
    """
    series = pd.Series(np.asarray(target).ravel())
    unknown = sorted(set(series.dropna().unique()) - set(BINARY_TARGET_MAPPING))
    if unknown:
        raise ValueError(
            "Cannot map to the binary injury target: unexpected source codes "
            f"{unknown}. Expected only {sorted(BINARY_TARGET_MAPPING)} "
            f"({BINARY_TARGET_PROVENANCE['source_labels']})."
        )
    if series.isna().any():
        raise ValueError("The source injury target contains missing values.")
    return series.map(BINARY_TARGET_MAPPING).astype(int)


if XGBOOST_AVAILABLE:  # pragma: no branch

    class FoldBalancedXGBClassifier(XGBClassifier):
        """XGBoost whose imbalance weight is derived from the fold it is fitted on.

        ``scale_pos_weight`` is a function of the class balance, so hard-coding
        it from the full development set would leak the validation fold's
        prevalence into every training fold. Deriving it inside ``fit`` keeps it
        fold-contained in exactly the way the imputers and scalers already are.

        The declared parameter is restored afterwards so ``get_params`` and
        ``clone`` still describe the configured estimator rather than the last
        fold it happened to see.
        """

        def fit(self, X, y, **kwargs):  # noqa: N803 - sklearn signature
            declared = self.scale_pos_weight
            labels = np.asarray(y).ravel()
            positives = int((labels == 1).sum())
            negatives = int((labels == 0).sum())
            self.scale_pos_weight = (
                float(negatives) / float(positives) if positives else 1.0
            )
            self.scale_pos_weight_used_ = self.scale_pos_weight
            try:
                return super().fit(X, y, **kwargs)
            finally:
                self.scale_pos_weight = declared


def build_binary_injury_candidates(
    manifest: dict,
    fast_mode: bool = True,
    random_state: int = RANDOM_STATE,
    include_smotenc: bool = False,
) -> dict[str, Candidate]:
    """The pre-registered binary candidate set.

    Oversampling is *not* included by default. The declared protocol is to
    establish first whether class weighting is sufficient for the binary task;
    ``include_smotenc=True`` adds the fold-contained SMOTENC pipeline as a
    separately reported comparison, never as an automatic addition.
    """
    numeric = list(manifest["numeric_features"])
    categorical = list(manifest["categorical_features"])
    all_features = numeric + categorical
    forest_estimators = 200 if fast_mode else 500
    xgb_estimators = 200 if fast_mode else 500

    candidates: dict[str, Candidate] = {
        "Prevalence Baseline": Candidate(
            name="Prevalence Baseline",
            estimator=make_standard_pipeline(
                DummyClassifier(strategy="prior", random_state=random_state),
                numeric,
                categorical,
            ),
            complexity_rank=0,
            feature_columns=all_features,
            family="baseline",
            imbalance_strategy="none",
        ),
        "Binary Logistic | class_weight": Candidate(
            name="Binary Logistic | class_weight",
            estimator=make_standard_pipeline(
                LogisticRegression(
                    max_iter=4000,
                    class_weight="balanced",
                    random_state=random_state,
                ),
                numeric,
                categorical,
            ),
            complexity_rank=1,
            feature_columns=all_features,
            family="logistic_regression",
            imbalance_strategy="class_weight",
        ),
        "Random Forest | class_weight": Candidate(
            name="Random Forest | class_weight",
            estimator=make_standard_pipeline(
                RandomForestClassifier(
                    n_estimators=forest_estimators,
                    min_samples_leaf=3,
                    max_features="sqrt",
                    class_weight="balanced_subsample",
                    random_state=random_state,
                    n_jobs=-1,
                ),
                numeric,
                categorical,
            ),
            complexity_rank=2,
            feature_columns=all_features,
            family="random_forest",
            imbalance_strategy="class_weight",
        ),
    }

    if XGBOOST_AVAILABLE:
        candidates["XGBoost | scale_pos_weight"] = Candidate(
            name="XGBoost | scale_pos_weight",
            estimator=make_standard_pipeline(
                FoldBalancedXGBClassifier(
                    n_estimators=xgb_estimators,
                    max_depth=3,
                    learning_rate=0.05,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=random_state,
                    n_jobs=-1,
                ),
                numeric,
                categorical,
            ),
            complexity_rank=3,
            feature_columns=all_features,
            family="xgboost",
            imbalance_strategy="fold_derived_scale_pos_weight",
        )

    if include_smotenc:
        candidates["Random Forest | SMOTENC"] = Candidate(
            name="Random Forest | SMOTENC",
            estimator=make_smotenc_pipeline(
                RandomForestClassifier(
                    n_estimators=forest_estimators,
                    min_samples_leaf=3,
                    max_features="sqrt",
                    random_state=random_state,
                    n_jobs=-1,
                ),
                numeric,
                categorical,
                random_state,
            ),
            complexity_rank=4,
            feature_columns=all_features,
            family="random_forest",
            imbalance_strategy="SMOTENC",
        )

    return candidates


BINARY_SEARCH_SPACES: dict[str, dict[str, Any]] = {
    "Binary Logistic | class_weight": {
        # ``penalty`` is deliberately not searched: scikit-learn 1.8 deprecates
        # it in favour of ``l1_ratio``, and the L2 default is what this study
        # declares. Only the regularisation strength is tuned.
        "model__C": np.logspace(-3, 2, 30),
    },
    "Random Forest | class_weight": {
        "model__n_estimators": [200, 350, 500],
        "model__max_depth": [None, 8, 12, 18],
        "model__min_samples_leaf": [1, 2, 4, 8],
        "model__max_features": ["sqrt", "log2", 0.5],
    },
    "XGBoost | scale_pos_weight": {
        "model__n_estimators": [200, 350, 500],
        "model__max_depth": [2, 3, 4, 5],
        "model__learning_rate": [0.02, 0.04, 0.06, 0.1],
        "model__subsample": [0.7, 0.85, 1.0],
        "model__colsample_bytree": [0.7, 0.85, 1.0],
        "model__min_child_weight": [1, 3, 5],
    },
    "Random Forest | SMOTENC": {
        "model__n_estimators": [200, 350, 500],
        "model__max_depth": [None, 8, 12, 18],
        "model__min_samples_leaf": [1, 2, 4, 8],
        "smotenc__k_neighbors": [3, 5, 7],
    },
}

INNER_SCORING = {
    "average_precision": "average_precision",
    "roc_auc": "roc_auc",
    "neg_log_loss": "neg_log_loss",
}
INNER_REFIT_METRIC = "average_precision"


def binary_fold_metrics(
    y_true: np.ndarray | pd.Series,
    probability: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Every metric the protocol requires, with counts alongside the rates."""
    y_true = np.asarray(y_true).astype(int)
    probability = np.asarray(probability, dtype=float)
    prediction = (probability >= threshold).astype(int)
    metrics = binary_metrics(y_true, prediction, probability)
    matrix = confusion_matrix(y_true, prediction, labels=[0, 1])
    (true_negative, false_positive), (false_negative, true_positive) = matrix
    specificity = (
        float(true_negative) / float(true_negative + false_positive)
        if (true_negative + false_positive)
        else float("nan")
    )
    return {
        **metrics,
        "threshold": float(threshold),
        "specificity": specificity,
        "true_negative": int(true_negative),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
        "true_positive": int(true_positive),
        "rows": int(len(y_true)),
        "positive_rows": int(y_true.sum()),
        "negative_rows": int(len(y_true) - y_true.sum()),
        "positive_prevalence": float(y_true.mean()),
    }


def nested_binary_grouped_cv(
    candidates: dict[str, Candidate],
    development: pd.DataFrame,
    manifest: dict,
    outer_splits: list[tuple[np.ndarray, np.ndarray]],
    iterations: int = 8,
    random_state: int = RANDOM_STATE,
    inner_folds: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Nested, athlete-grouped development validation for the binary target.

    Every candidate sees the *same* outer folds. Inside each outer fold the
    hyperparameter search runs on athlete-grouped sub-folds of that fold's
    training athletes only, so the held-out athletes score a model chosen
    without them. All preprocessing, weighting and any resampling live inside
    the fitted pipeline and are therefore fitted on training rows alone.
    """
    group_column = manifest["grouping_variable"]
    source_target = manifest["target"]
    binary_target = to_binary_target(development[source_target])

    fold_rows: list[dict[str, Any]] = []
    tuning_rows: list[dict[str, Any]] = []
    oof_parts: list[pd.DataFrame] = []

    for fold, (train_index, validation_index) in enumerate(outer_splits, start=1):
        train = development.iloc[train_index].reset_index(drop=True)
        validation = development.iloc[validation_index].reset_index(drop=True)
        y_train = binary_target.iloc[train_index].reset_index(drop=True)
        y_validation = binary_target.iloc[validation_index].reset_index(drop=True)

        if set(train[group_column]) & set(validation[group_column]):
            raise AssertionError(
                f"Outer fold {fold} shares athletes between training and validation."
            )

        inner_splits = make_stratified_group_splits(
            y_train,
            train[group_column],
            n_splits=inner_folds,
            random_state=random_state + 100 + fold,
        )

        for name, candidate in candidates.items():
            space = BINARY_SEARCH_SPACES.get(name)
            selected_parameters: dict[str, Any] = {}
            inner_score = float("nan")
            total_fits = 0

            if space:
                search = RandomizedSearchCV(
                    estimator=clone(candidate.estimator),
                    param_distributions=space,
                    n_iter=iterations,
                    scoring=INNER_SCORING,
                    refit=INNER_REFIT_METRIC,
                    cv=inner_splits,
                    random_state=random_state + 200 + fold,
                    n_jobs=-1,
                    return_train_score=False,
                )
                search.fit(train[candidate.feature_columns], y_train)
                estimator = search.best_estimator_
                selected_parameters = dict(search.best_params_)
                inner_score = float(search.best_score_)
                total_fits = int(len(search.cv_results_["params"]) * len(inner_splits) + 1)
            else:
                estimator = clone(candidate.estimator)
                estimator.fit(train[candidate.feature_columns], y_train)
                total_fits = 1

            probability = estimator.predict_proba(
                validation[candidate.feature_columns]
            )[:, 1]

            fold_rows.append(
                {
                    "outer_fold": fold,
                    "model": name,
                    "family": candidate.family,
                    "imbalance_strategy": candidate.imbalance_strategy,
                    "training_athletes": int(train[group_column].nunique()),
                    "validation_athletes": int(validation[group_column].nunique()),
                    **binary_fold_metrics(y_validation, probability, threshold=0.5),
                }
            )
            tuning_rows.append(
                {
                    "outer_fold": fold,
                    "model": name,
                    "tuned": bool(space),
                    "search_space": {
                        key: [str(value) for value in values]
                        for key, values in (space or {}).items()
                    },
                    "randomized_configurations": iterations if space else 0,
                    "inner_folds": len(inner_splits),
                    "inner_scoring": INNER_REFIT_METRIC if space else None,
                    "inner_best_score": inner_score,
                    "selected_parameters": {
                        key: _plain(value) for key, value in selected_parameters.items()
                    },
                    "total_fits": total_fits,
                    "random_seed": random_state + 200 + fold,
                }
            )

            part = validation[[group_column, "session_id", source_target]].copy()
            part["outer_fold"] = fold
            part["model"] = name
            part["actual_high_injury_risk"] = y_validation.to_numpy()
            part["probability_high_injury_risk"] = probability
            oof_parts.append(part)

    return (
        pd.DataFrame(fold_rows),
        pd.concat(oof_parts, ignore_index=True),
        pd.DataFrame(tuning_rows),
    )


def _plain(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def summarise_binary_folds(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation across outer folds, per candidate."""
    numeric = [
        column
        for column in fold_metrics.columns
        if pd.api.types.is_numeric_dtype(fold_metrics[column]) and column != "outer_fold"
    ]
    summary = fold_metrics.groupby("model")[numeric].agg(["mean", "std"])
    summary.columns = [f"{stat}_{metric}" for metric, stat in summary.columns]
    return summary.reset_index()


def selection_frequency(tuning_records: pd.DataFrame) -> pd.DataFrame:
    """How often each hyperparameter value was chosen across the outer folds."""
    rows = []
    for _, record in tuning_records.iterrows():
        if not record["tuned"]:
            continue
        for parameter, value in (record["selected_parameters"] or {}).items():
            rows.append(
                {
                    "model": record["model"],
                    "parameter": parameter,
                    "value": str(value),
                    "outer_fold": record["outer_fold"],
                }
            )
    if not rows:
        return pd.DataFrame(columns=["model", "parameter", "value", "folds_selected"])
    frame = pd.DataFrame(rows)
    return (
        frame.groupby(["model", "parameter", "value"])
        .size()
        .reset_index(name="folds_selected")
        .sort_values(["model", "parameter", "folds_selected"], ascending=[True, True, False])
        .reset_index(drop=True)
    )


def modal_parameters(tuning_records: pd.DataFrame, model: str) -> dict[str, Any]:
    """The most frequently selected value per parameter, across outer folds.

    These are the hyperparameters the deployed model is fitted with. They are
    the best found under the controlled search budget on grouped development
    folds; they are not claimed to be globally optimal.
    """
    frequency = selection_frequency(tuning_records)
    frequency = frequency[frequency["model"] == model]
    records = tuning_records[tuning_records["model"] == model]
    raw_values: dict[str, list[Any]] = {}
    for _, record in records.iterrows():
        for parameter, value in (record["selected_parameters"] or {}).items():
            raw_values.setdefault(parameter, []).append(value)

    chosen: dict[str, Any] = {}
    for parameter, group in frequency.groupby("parameter"):
        best = group.sort_values("folds_selected", ascending=False).iloc[0]["value"]
        for value in raw_values.get(parameter, []):
            if str(value) == best:
                chosen[parameter] = _plain(value)
                break
    return chosen


def threshold_sweep(
    y_true: np.ndarray,
    probability: np.ndarray,
    grid: np.ndarray | None = None,
) -> pd.DataFrame:
    """The complete precision / recall / specificity trade-off."""
    y_true = np.asarray(y_true).astype(int)
    probability = np.asarray(probability, dtype=float)
    if grid is None:
        grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    rows = []
    for threshold in grid:
        metrics = binary_fold_metrics(y_true, probability, threshold=float(threshold))
        rows.append(metrics)
    return pd.DataFrame(rows)


THRESHOLD_RULE = "maximise_balanced_accuracy"
THRESHOLD_RULE_DESCRIPTION = (
    "Maximise balanced accuracy on athlete-grouped out-of-fold development "
    "probabilities. Declared before candidate results were compared. The "
    "already-inspected 32-athlete audit cohort is not used."
)

BALANCED_SCREENING_RULE = "max_precision_with_both_recalls_at_least"
BALANCED_SCREENING_RULE_DESCRIPTION = (
    "Among thresholds where High Injury Risk recall and Low Injury Risk recall "
    "are both at least 0.80, select the threshold with the highest High Injury "
    "Risk precision. Ties are broken by the lower threshold. Selected on "
    "athlete-grouped out-of-fold development probabilities from the deployed "
    "calibrated model; the already-inspected 32-athlete audit cohort is not "
    "used. If no threshold satisfies both floors, the rule is infeasible and "
    "the pre-existing maximise-balanced-accuracy rule stands."
)

RECALL_CONSTRAINT_RULE_DESCRIPTION = (
    "Highest balanced accuracy among thresholds reaching at least 0.80 High "
    "Injury Risk recall while retaining at least 0.50 specificity. Reported "
    "for transparency; not the deployed rule."
)


def select_threshold(
    sweep: pd.DataFrame,
    rule: str = THRESHOLD_RULE,
    minimum_recall: float = 0.80,
    minimum_specificity: float = 0.50,
    minimum_negative_recall: float = 0.80,
) -> dict[str, Any]:
    if rule == BALANCED_SCREENING_RULE:
        # Low Injury Risk recall is specificity: the share of the combined
        # negative group the screen leaves unflagged.
        eligible = sweep[
            (sweep["recall"] >= minimum_recall)
            & (sweep["specificity"] >= minimum_negative_recall)
        ]
        if eligible.empty:
            return {
                "rule": rule,
                "description": BALANCED_SCREENING_RULE_DESCRIPTION,
                "feasible": False,
                "minimum_high_injury_risk_recall": minimum_recall,
                "minimum_low_injury_risk_recall": minimum_negative_recall,
            }
        best = eligible.sort_values(
            ["precision", "threshold"], ascending=[False, True]
        ).iloc[0]
        return {
            "rule": rule,
            "description": BALANCED_SCREENING_RULE_DESCRIPTION,
            "feasible": True,
            "minimum_high_injury_risk_recall": minimum_recall,
            "minimum_low_injury_risk_recall": minimum_negative_recall,
            "low_injury_risk_recall": float(best["specificity"]),
            **best.to_dict(),
        }
    if rule == "maximise_balanced_accuracy":
        best = sweep.sort_values(
            ["balanced_accuracy", "threshold"], ascending=[False, True]
        ).iloc[0]
        return {"rule": rule, "description": THRESHOLD_RULE_DESCRIPTION, **best.to_dict()}
    if rule == "minimum_recall_with_specificity_floor":
        eligible = sweep[
            (sweep["recall"] >= minimum_recall)
            & (sweep["specificity"] >= minimum_specificity)
        ]
        if eligible.empty:
            return {
                "rule": rule,
                "description": RECALL_CONSTRAINT_RULE_DESCRIPTION,
                "feasible": False,
                "minimum_recall": minimum_recall,
                "minimum_specificity": minimum_specificity,
            }
        best = eligible.sort_values(
            ["balanced_accuracy", "threshold"], ascending=[False, True]
        ).iloc[0]
        return {
            "rule": rule,
            "description": RECALL_CONSTRAINT_RULE_DESCRIPTION,
            "feasible": True,
            "minimum_recall": minimum_recall,
            "minimum_specificity": minimum_specificity,
            **best.to_dict(),
        }
    raise ValueError(f"Unknown threshold rule: {rule!r}")


def binary_calibration_audit(
    candidate: Candidate,
    development: pd.DataFrame,
    manifest: dict,
    outer_splits: list[tuple[np.ndarray, np.ndarray]],
    random_state: int = RANDOM_STATE,
    methods: tuple[str, ...] = ("sigmoid", "isotonic"),
    inner_folds: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare uncalibrated against calibrated probabilities on grouped folds.

    The calibrator is fitted on athlete-grouped sub-folds of each outer fold's
    training athletes, so the held-out athletes never contribute to the
    calibration map that scores them.
    """
    group_column = manifest["grouping_variable"]
    binary_target = to_binary_target(development[manifest["target"]])
    rows: list[dict[str, Any]] = []
    prediction_parts: list[pd.DataFrame] = []

    for fold, (train_index, validation_index) in enumerate(outer_splits, start=1):
        train = development.iloc[train_index].reset_index(drop=True)
        validation = development.iloc[validation_index].reset_index(drop=True)
        y_train = binary_target.iloc[train_index].reset_index(drop=True)
        y_validation = binary_target.iloc[validation_index].reset_index(drop=True)

        inner_splits = make_stratified_group_splits(
            y_train, train[group_column], n_splits=inner_folds,
            random_state=random_state + 300 + fold,
        )

        part = validation[[group_column, "session_id"]].copy()
        part["outer_fold"] = fold
        part["actual_high_injury_risk"] = y_validation.to_numpy()

        base = clone(candidate.estimator)
        base.fit(train[candidate.feature_columns], y_train)
        probabilities = {
            "uncalibrated": base.predict_proba(
                validation[candidate.feature_columns]
            )[:, 1]
        }

        for method in methods:
            calibrated = CalibratedClassifierCV(
                estimator=clone(candidate.estimator),
                method=method,
                cv=inner_splits,
                ensemble=True,
            )
            calibrated.fit(train[candidate.feature_columns], y_train)
            probabilities[method] = calibrated.predict_proba(
                validation[candidate.feature_columns]
            )[:, 1]

        for mode, probability in probabilities.items():
            rows.append(
                {
                    "mode": mode,
                    "outer_fold": fold,
                    **binary_fold_metrics(y_validation, probability, threshold=0.5),
                }
            )
            part[f"{mode}_probability_high_injury_risk"] = probability

        prediction_parts.append(part)

    metrics = pd.DataFrame(rows)
    numeric = [
        column
        for column in metrics.columns
        if pd.api.types.is_numeric_dtype(metrics[column]) and column != "outer_fold"
    ]
    summary = metrics.groupby("mode")[numeric].agg(["mean", "std"])
    summary.columns = [f"{stat}_{metric}" for metric, stat in summary.columns]
    return summary.reset_index(), pd.concat(prediction_parts, ignore_index=True)


def per_athlete_binary_performance(
    oof: pd.DataFrame,
    group_column: str = "athlete_id",
    threshold: float = 0.5,
    minimum_rows: int = 20,
) -> pd.DataFrame:
    """Per-athlete counts, prevalence and — where estimable — ranking quality."""
    rows = []
    for athlete, group in oof.groupby(group_column):
        y_true = group["actual_high_injury_risk"].to_numpy().astype(int)
        probability = group["probability_high_injury_risk"].to_numpy()
        prediction = (probability >= threshold).astype(int)
        record = {
            group_column: athlete,
            "sessions": int(len(group)),
            "high_injury_risk_sessions": int(y_true.sum()),
            "high_injury_risk_prevalence": float(y_true.mean()),
            "predicted_high_injury_risk_sessions": int(prediction.sum()),
            "mean_probability_high_injury_risk": float(probability.mean()),
        }
        if len(group) >= minimum_rows and 0 < y_true.sum() < len(y_true):
            record["roc_auc"] = float(roc_auc_score(y_true, probability))
            record["recall"] = float(
                (prediction[y_true == 1] == 1).mean()
            )
            record["specificity"] = float(
                (prediction[y_true == 0] == 0).mean()
            )
        rows.append(record)
    return pd.DataFrame(rows).sort_values(group_column).reset_index(drop=True)


def reliability_table(
    y_true: np.ndarray,
    probability: np.ndarray,
    bins: int = 10,
) -> pd.DataFrame:
    """Binned observed-versus-predicted frequencies behind a reliability curve."""
    y_true = np.asarray(y_true).astype(int)
    probability = np.asarray(probability, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(probability, edges[1:-1], right=False), 0, bins - 1)
    rows = []
    for bin_number in range(bins):
        mask = index == bin_number
        if not mask.any():
            continue
        rows.append(
            {
                "bin": bin_number + 1,
                "bin_lower": float(edges[bin_number]),
                "bin_upper": float(edges[bin_number + 1]),
                "rows": int(mask.sum()),
                "mean_predicted_probability": float(probability[mask].mean()),
                "observed_high_injury_risk_rate": float(y_true[mask].mean()),
                "observed_high_injury_risk_count": int(y_true[mask].sum()),
            }
        )
    return pd.DataFrame(rows)
