"""Injury-risk modelling: candidate families, athlete-grouped selection and audit."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.calibration import CalibrationDisplay
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline

from src.evaluation.metrics import (
    aggregate_fold_metrics,
    cluster_bootstrap_confidence_intervals,
    multiclass_metrics,
    multiclass_subgroup_metrics,
)
from src.models.common import (
    Candidate,
    RANDOM_STATE,
    environment_metadata,
    load_json,
    make_one_hot_encoder,
    save_json,
    save_model_bundle,
    set_reproducibility,
)
from src.models.splitting import (
    group_split_summary,
    make_stratified_group_splits,
)
from src.models.structured_multiclass import OrdinalRiskClassifier, TwoStageRiskClassifier

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False


CLASS_NAMES = {
    0: "Healthy",
    1: "Low Risk",
    2: "High Risk/Injured",
}


def make_standard_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", make_one_hot_encoder()),
                    ]
                ),
                categorical_features,
            ),
        ]
    )


def make_standard_pipeline(
    model: Any,
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", make_standard_preprocessor(numeric_features, categorical_features)),
            ("model", model),
        ]
    )


def make_smotenc_pipeline(
    model: Any,
    numeric_features: list[str],
    categorical_features: list[str],
    random_state: int,
) -> ImbPipeline:
    """SMOTENC after fold-contained imputation and ordinal encoding."""
    numeric_pre = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pre = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ordinal",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                ),
            ),
        ]
    )
    pre_smote = ColumnTransformer(
        [
            ("numeric", numeric_pre, numeric_features),
            ("categorical", categorical_pre, categorical_features),
        ],
        sparse_threshold=0,
    )

    numeric_indices = list(range(len(numeric_features)))
    categorical_indices = list(
        range(len(numeric_features), len(numeric_features) + len(categorical_features))
    )
    post_smote = ColumnTransformer(
        [
            ("numeric", StandardScaler(), numeric_indices),
            ("categorical", make_one_hot_encoder(), categorical_indices),
        ]
    )

    return ImbPipeline(
        [
            ("pre_smote", pre_smote),
            (
                "smotenc",
                SMOTENC(
                    categorical_features=categorical_indices,
                    random_state=random_state,
                    k_neighbors=5,
                ),
            ),
            ("post_smote", post_smote),
            ("model", model),
        ]
    )


def build_injury_candidates(
    manifest: dict,
    fast_mode: bool = True,
    random_state: int = RANDOM_STATE,
) -> dict[str, Candidate]:
    numeric = list(manifest["numeric_features"])
    categorical = list(manifest["categorical_features"])
    all_features = numeric + categorical
    forest_estimators = 80 if fast_mode else 500
    xgb_estimators = 100 if fast_mode else 500

    candidates: dict[str, Candidate] = {
        "Majority Baseline": Candidate(
            name="Majority Baseline",
            estimator=make_standard_pipeline(
                DummyClassifier(strategy="prior"),
                numeric,
                categorical,
            ),
            complexity_rank=0,
            feature_columns=all_features,
            family="baseline",
            imbalance_strategy="none",
        ),
        "Multinomial Logistic | none": Candidate(
            name="Multinomial Logistic | none",
            estimator=make_standard_pipeline(
                LogisticRegression(
                    max_iter=4000,
                    random_state=random_state,
                ),
                numeric,
                categorical,
            ),
            complexity_rank=1,
            feature_columns=all_features,
            family="logistic_regression",
            imbalance_strategy="none",
        ),
        "Multinomial Logistic | class_weight": Candidate(
            name="Multinomial Logistic | class_weight",
            estimator=make_standard_pipeline(
                LogisticRegression(
                    max_iter=4000,
                    class_weight="balanced",
                    random_state=random_state,
                ),
                numeric,
                categorical,
            ),
            complexity_rank=2,
            feature_columns=all_features,
            family="logistic_regression",
            imbalance_strategy="class_weight",
        ),
        "Decision Tree | class_weight": Candidate(
            name="Decision Tree | class_weight",
            estimator=make_standard_pipeline(
                DecisionTreeClassifier(
                    max_depth=8,
                    min_samples_leaf=20,
                    class_weight="balanced",
                    random_state=random_state,
                ),
                numeric,
                categorical,
            ),
            complexity_rank=3,
            feature_columns=all_features,
            family="decision_tree",
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
            complexity_rank=4,
            feature_columns=all_features,
            family="random_forest",
            imbalance_strategy="class_weight",
        ),
        "Random Forest | SMOTENC": Candidate(
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
            complexity_rank=5,
            feature_columns=all_features,
            family="random_forest",
            imbalance_strategy="SMOTENC",
        ),
        "Ordinal Logistic | class_weight": Candidate(
            name="Ordinal Logistic | class_weight",
            estimator=make_standard_pipeline(
                OrdinalRiskClassifier(random_state=random_state), numeric, categorical,
            ),
            complexity_rank=3,
            feature_columns=all_features,
            family="ordinal_logistic",
            imbalance_strategy="class_weight",
        ),
        "Two-stage Logistic | class_weight": Candidate(
            name="Two-stage Logistic | class_weight",
            estimator=make_standard_pipeline(
                TwoStageRiskClassifier(random_state=random_state), numeric, categorical,
            ),
            complexity_rank=3,
            feature_columns=all_features,
            family="two_stage_logistic",
            imbalance_strategy="class_weight",
        ),
    }

    if XGBOOST_AVAILABLE:
        candidates["XGBoost | none"] = Candidate(
            name="XGBoost | none",
            estimator=make_standard_pipeline(
                XGBClassifier(
                    n_estimators=xgb_estimators,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    objective="multi:softprob",
                    num_class=3,
                    eval_metric="mlogloss",
                    random_state=random_state,
                    n_jobs=-1,
                ),
                numeric,
                categorical,
            ),
            complexity_rank=6,
            feature_columns=all_features,
            family="xgboost",
            imbalance_strategy="none",
        )

    return candidates


INJURY_SEARCH_SPACES: dict[str, dict[str, Any]] = {
    "Multinomial Logistic | class_weight": {
        "model__C": np.logspace(-3, 2, 30),
    },
    "Ordinal Logistic | class_weight": {
        "model__C": np.logspace(-3, 2, 30),
    },
    "Two-stage Logistic | class_weight": {
        "model__C": np.logspace(-3, 2, 30),
    },
    "Decision Tree | class_weight": {
        "model__max_depth": [3, 5, 8, 12, None],
        "model__min_samples_leaf": [5, 10, 20, 40],
        "model__criterion": ["gini", "entropy", "log_loss"],
    },
    "Random Forest | class_weight": {
        "model__n_estimators": [250, 400, 600],
        "model__max_depth": [None, 8, 12, 18],
        "model__min_samples_leaf": [1, 2, 4, 8],
        "model__max_features": ["sqrt", "log2", 0.5],
    },
    "Random Forest | SMOTENC": {
        "model__n_estimators": [250, 400, 600],
        "model__max_depth": [None, 8, 12, 18],
        "model__min_samples_leaf": [1, 2, 4, 8],
        "model__max_features": ["sqrt", "log2", 0.5],
        "smotenc__k_neighbors": [3, 5, 7],
    },
    "XGBoost | none": {
        "model__n_estimators": [200, 350, 500],
        "model__max_depth": [2, 3, 4, 5],
        "model__learning_rate": [0.02, 0.04, 0.06, 0.1],
        "model__subsample": [0.7, 0.85, 1.0],
        "model__colsample_bytree": [0.7, 0.85, 1.0],
        "model__min_child_weight": [1, 3, 5],
    },
}


def tune_injury_candidates(
    candidates: dict[str, Candidate],
    frame: pd.DataFrame,
    target: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
    iterations: int = 12,
    random_state: int = RANDOM_STATE,
) -> tuple[dict[str, Candidate], pd.DataFrame]:
    """Fix the deployed hyperparameters by searching over ``splits``.

    This is a *fitting* step, not a comparison step. Scoring families against
    each other on the same folds the search optimised would flatter every tuned
    family against the untuned baselines, so champion selection uses
    :func:`evaluate_injury_candidates_nested_cv` instead and this function is
    run afterwards, once the family is already decided.
    """
    spaces = INJURY_SEARCH_SPACES

    updated = dict(candidates)
    rows = []

    for name, candidate in candidates.items():
        if name not in spaces:
            continue
        search = RandomizedSearchCV(
            estimator=clone(candidate.estimator),
            param_distributions=spaces[name],
            n_iter=iterations,
            scoring={
                "macro_f1": "f1_macro",
                "balanced_accuracy": "balanced_accuracy",
                "neg_log_loss": "neg_log_loss",
            },
            refit="macro_f1",
            cv=splits,
            random_state=random_state,
            n_jobs=-1,
            verbose=1,
            return_train_score=False,
        )
        search.fit(frame[candidate.feature_columns], target)
        updated[name] = Candidate(
            name=name,
            estimator=search.best_estimator_,
            complexity_rank=candidate.complexity_rank,
            feature_columns=candidate.feature_columns,
            family=candidate.family,
            imbalance_strategy=candidate.imbalance_strategy,
        )
        rows.append(
            {
                "model": name,
                "best_mean_cv_macro_f1": search.best_score_,
                "best_parameters": search.best_params_,
            }
        )

    return updated, pd.DataFrame(rows)


def apply_saved_injury_tuning(
    candidates: dict[str, Candidate],
    tuning_path: Path,
) -> tuple[dict[str, Candidate], pd.DataFrame]:
    records = load_json(tuning_path)
    updated = dict(candidates)
    for record in records:
        name = record["model"]
        if name not in candidates:
            continue
        candidate = candidates[name]
        estimator = clone(candidate.estimator)
        estimator.set_params(**record["best_parameters"])
        updated[name] = Candidate(
            name=name,
            estimator=estimator,
            complexity_rank=candidate.complexity_rank,
            feature_columns=candidate.feature_columns,
            family=candidate.family,
            imbalance_strategy=candidate.imbalance_strategy,
        )
    return updated, pd.DataFrame(records)

def evaluate_injury_candidates_cv(
    candidates: dict[str, Candidate],
    frame: pd.DataFrame,
    target: pd.Series,
    groups: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Score every candidate as given, one fit per athlete-grouped fold.

    Valid only when the candidates carry hyperparameters that were not chosen on
    these folds - untuned defaults, or parameters frozen by an earlier run. When
    a run tunes, use :func:`evaluate_injury_candidates_nested_cv`.
    """
    fold_rows = []
    prediction_rows = []

    for candidate_name, candidate in candidates.items():
        for fold_number, (train_indices, validation_indices) in enumerate(
            splits, start=1
        ):
            estimator = clone(candidate.estimator)
            X_train = frame.iloc[train_indices][candidate.feature_columns]
            X_validation = frame.iloc[validation_indices][candidate.feature_columns]
            y_train = target.iloc[train_indices]
            y_validation = target.iloc[validation_indices]

            train_groups = set(groups.iloc[train_indices])
            validation_groups = set(groups.iloc[validation_indices])
            if not train_groups.isdisjoint(validation_groups):
                raise AssertionError("Athlete leakage detected.")

            estimator.fit(X_train, y_train)
            prediction = estimator.predict(X_validation)
            probabilities = estimator.predict_proba(X_validation)
            metrics = multiclass_metrics(
                y_validation,
                prediction,
                probabilities,
                classes=(0, 1, 2),
            )

            fold_rows.append(
                {
                    "model": candidate_name,
                    "fold": fold_number,
                    **metrics,
                }
            )

            fold_predictions = frame.iloc[validation_indices][
                ["athlete_id", "session_id", "sport_type", "gender"]
            ].copy()
            fold_predictions["model"] = candidate_name
            fold_predictions["fold"] = fold_number
            fold_predictions["actual"] = y_validation.to_numpy()
            fold_predictions["prediction"] = prediction
            for class_value in (0, 1, 2):
                fold_predictions[f"probability_class_{class_value}"] = probabilities[
                    :, class_value
                ]
            prediction_rows.append(fold_predictions)

    fold_frame = pd.DataFrame(fold_rows)
    summary = aggregate_fold_metrics(fold_frame)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    return summary, fold_frame, predictions


def evaluate_injury_candidates_nested_cv(
    candidates: dict[str, Candidate],
    frame: pd.DataFrame,
    target: pd.Series,
    groups: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
    iterations: int = 12,
    inner_splits: int = 3,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare families with each family's tuning contained inside the fold.

    Tuning on the same grouped folds that later rank the candidates gives every
    tunable family several attempts at each held-out athlete set while the
    untuned baselines get one, so the comparison rewards the search rather than
    the family. Here the search for a fold runs on athlete-grouped sub-folds of
    that fold's *training* athletes only, and the held-out athletes are scored
    once, by a model whose hyperparameters were chosen without them.

    Returns the usual comparison triple plus the parameters each fold chose, so
    an unstable family - one whose fold winners disagree - is visible instead of
    hidden behind an averaged score.
    """
    fold_rows = []
    prediction_rows = []
    tuning_rows = []

    for candidate_name, candidate in candidates.items():
        space = INJURY_SEARCH_SPACES.get(candidate_name)
        for fold_number, (train_indices, validation_indices) in enumerate(
            splits, start=1
        ):
            train_frame = frame.iloc[train_indices]
            X_train = train_frame[candidate.feature_columns]
            X_validation = frame.iloc[validation_indices][candidate.feature_columns]
            y_train = target.iloc[train_indices]
            y_validation = target.iloc[validation_indices]

            if not set(groups.iloc[train_indices]).isdisjoint(
                set(groups.iloc[validation_indices])
            ):
                raise AssertionError("Athlete leakage detected.")

            if space and iterations > 0:
                search = RandomizedSearchCV(
                    estimator=clone(candidate.estimator),
                    param_distributions=space,
                    n_iter=iterations,
                    scoring={
                        "macro_f1": "f1_macro",
                        "balanced_accuracy": "balanced_accuracy",
                        "neg_log_loss": "neg_log_loss",
                    },
                    refit="macro_f1",
                    cv=make_stratified_group_splits(
                        y_train,
                        groups.iloc[train_indices],
                        n_splits=inner_splits,
                        random_state=random_state + fold_number,
                    ),
                    random_state=random_state + fold_number,
                    n_jobs=-1,
                    verbose=1,
                    return_train_score=False,
                )
                search.fit(X_train, y_train)
                estimator = search.best_estimator_
                tuning_rows.append(
                    {
                        "model": candidate_name,
                        "outer_fold": fold_number,
                        "inner_cv_macro_f1": search.best_score_,
                        "best_parameters": search.best_params_,
                    }
                )
            else:
                estimator = clone(candidate.estimator)
                estimator.fit(X_train, y_train)

            prediction = estimator.predict(X_validation)
            probabilities = estimator.predict_proba(X_validation)
            metrics = multiclass_metrics(
                y_validation,
                prediction,
                probabilities,
                classes=(0, 1, 2),
            )

            fold_rows.append(
                {
                    "model": candidate_name,
                    "fold": fold_number,
                    **metrics,
                }
            )

            fold_predictions = frame.iloc[validation_indices][
                ["athlete_id", "session_id", "sport_type", "gender"]
            ].copy()
            fold_predictions["model"] = candidate_name
            fold_predictions["fold"] = fold_number
            fold_predictions["actual"] = y_validation.to_numpy()
            fold_predictions["prediction"] = prediction
            for class_value in (0, 1, 2):
                fold_predictions[f"probability_class_{class_value}"] = probabilities[
                    :, class_value
                ]
            prediction_rows.append(fold_predictions)

    fold_frame = pd.DataFrame(fold_rows)
    summary = aggregate_fold_metrics(fold_frame)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    return summary, fold_frame, predictions, pd.DataFrame(tuning_rows)


def select_injury_champion(
    summary: pd.DataFrame,
    candidates: dict[str, Candidate],
    macro_f1_tolerance: float = 0.01,
) -> tuple[str, pd.DataFrame]:
    eligible = summary.copy()
    best_macro_f1 = eligible["mean_macro_f1"].max()
    eligible = eligible[
        eligible["mean_macro_f1"] >= best_macro_f1 - macro_f1_tolerance
    ].copy()
    eligible["complexity_rank"] = [
        candidates[name].complexity_rank for name in eligible.index
    ]
    eligible = eligible.sort_values(
        [
            "mean_class_2_recall",
            "mean_log_loss",
            "mean_multiclass_brier",
            "complexity_rank",
            "std_macro_f1",
        ],
        ascending=[False, True, True, True, True],
    )
    champion = str(eligible.index[0])
    return champion, eligible


def make_calibrated_classifier(
    estimator: Any,
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
    method: str = "sigmoid",
) -> CalibratedClassifierCV:
    try:
        return CalibratedClassifierCV(
            estimator=estimator,
            method=method,
            cv=cv_splits,
            ensemble=True,
        )
    except TypeError:
        return CalibratedClassifierCV(
            base_estimator=estimator,
            method=method,
            cv=cv_splits,
        )


def grouped_calibration_audit(
    candidate: Candidate,
    development: pd.DataFrame,
    target_column: str,
    group_column: str,
    outer_splits: list[tuple[np.ndarray, np.ndarray]],
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    all_predictions: list[pd.DataFrame] = []
    for fold, (train_indices, validation_indices) in enumerate(outer_splits, start=1):
        outer_train = development.iloc[train_indices].reset_index(drop=True)
        outer_validation = development.iloc[validation_indices].reset_index(drop=True)

        base = clone(candidate.estimator)
        base.fit(
            outer_train[candidate.feature_columns],
            outer_train[target_column],
        )
        uncalibrated_prediction = base.predict(
            outer_validation[candidate.feature_columns]
        )
        uncalibrated_probability = base.predict_proba(
            outer_validation[candidate.feature_columns]
        )

        inner_splits = make_stratified_group_splits(
            outer_train[target_column],
            outer_train[group_column],
            n_splits=3,
            random_state=random_state + fold,
        )
        calibrated = make_calibrated_classifier(
            clone(candidate.estimator),
            inner_splits,
            method="sigmoid",
        )
        calibrated.fit(
            outer_train[candidate.feature_columns],
            outer_train[target_column],
        )
        calibrated_prediction = calibrated.predict(
            outer_validation[candidate.feature_columns]
        )
        calibrated_probability = calibrated.predict_proba(
            outer_validation[candidate.feature_columns]
        )

        for mode, prediction, probability in [
            ("uncalibrated", uncalibrated_prediction, uncalibrated_probability),
            ("calibrated", calibrated_prediction, calibrated_probability),
        ]:
            metrics = multiclass_metrics(
                outer_validation[target_column],
                prediction,
                probability,
                classes=(0, 1, 2),
            )
            rows.append({"mode": mode, "fold": fold, **metrics})

        prediction_frame = outer_validation[[
            group_column, "session_id", target_column
        ]].copy()
        prediction_frame["fold"] = fold
        for class_value in (0, 1, 2):
            prediction_frame[f"uncalibrated_probability_class_{class_value}"] = (
                uncalibrated_probability[:, class_value]
            )
            prediction_frame[f"calibrated_probability_class_{class_value}"] = (
                calibrated_probability[:, class_value]
            )
        all_predictions.append(prediction_frame)

    metrics_frame = pd.DataFrame(rows)
    summary_rows = []
    for mode, group in metrics_frame.groupby("mode"):
        summary = {"mode": mode, "folds": len(group)}
        for column in group.columns:
            if column in {"mode", "fold"}:
                continue
            if pd.api.types.is_numeric_dtype(group[column]):
                summary[f"mean_{column}"] = group[column].mean()
                summary[f"std_{column}"] = group[column].std(ddof=1)
        summary_rows.append(summary)
    return pd.DataFrame(summary_rows).set_index("mode"), pd.concat(all_predictions, ignore_index=True)


def save_multiclass_calibration_comparison(
    predictions: pd.DataFrame,
    target_column: str,
    output_dir: Path,
) -> None:
    for class_value, class_name in CLASS_NAMES.items():
        figure, axis = plt.subplots(figsize=(6, 5))
        binary_target = (predictions[target_column] == class_value).astype(int)
        for mode in ["uncalibrated", "calibrated"]:
            probability = predictions[f"{mode}_probability_class_{class_value}"]
            observed, predicted = calibration_curve(
                binary_target,
                probability,
                n_bins=10,
                strategy="quantile",
            )
            axis.plot(predicted, observed, marker="o", label=mode.title())
        axis.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
        axis.set_xlabel("Mean predicted probability")
        axis.set_ylabel("Observed frequency")
        axis.set_title(f"Injury Calibration Audit — {class_name}")
        axis.legend()
        figure.tight_layout()
        figure.savefig(
            output_dir / f"injury_calibration_audit_class_{class_value}.png",
            dpi=200,
        )
        plt.close(figure)

def save_injury_figures(
    predictions: pd.DataFrame,
    output_dir: Path,
    title_suffix: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    y_true = predictions["injury_occurred"]
    y_pred = predictions["prediction"]

    figure, axis = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        # The three classes are stated rather than inferred: a fold or audit set
        # in which one class is never predicted would otherwise produce a matrix
        # of the wrong size and mismatch `display_labels`.
        labels=[0, 1, 2],
        display_labels=["Healthy", "Low Risk", "High Risk/Injured"],
        ax=axis,
    )
    axis.set_title(f"Injury Confusion Matrix — {title_suffix}")
    figure.tight_layout()
    figure.savefig(output_dir / "injury_confusion_matrix.png", dpi=200)
    plt.close(figure)

    for class_value, class_name in CLASS_NAMES.items():
        figure, axis = plt.subplots(figsize=(6, 5))
        CalibrationDisplay.from_predictions(
            (y_true == class_value).astype(int),
            predictions[f"probability_class_{class_value}"],
            n_bins=10,
            strategy="quantile",
            ax=axis,
        )
        axis.set_title(f"Injury Calibration — {class_name}")
        figure.tight_layout()
        figure.savefig(
            output_dir / f"injury_calibration_class_{class_value}.png",
            dpi=200,
        )
        plt.close(figure)


def run_injury_experiment(
    project_root: Path,
    fast_mode: bool = True,
    tune: bool = False,
    reuse_saved_tuning: bool = False,
    unlock_final_test: bool = False,
    bootstrap_replicates: int = 300,
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    set_reproducibility(random_state)
    project_root = Path(project_root).resolve()
    prepared_dir = project_root / "outputs" / "injury"
    output_dir = project_root / "outputs" / "modeling" / "injury"
    model_dir = project_root / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_json(prepared_dir / "injury_feature_manifest.json")
    development = pd.read_csv(prepared_dir / "injury_development.csv")
    final_test = pd.read_csv(prepared_dir / "injury_final_test.csv")
    target_column = manifest["target"]
    group_column = manifest["grouping_variable"]

    if set(development[group_column]).intersection(set(final_test[group_column])):
        raise AssertionError("Athlete overlap detected between development and test.")

    candidates = build_injury_candidates(manifest, fast_mode=fast_mode)
    group_splits = make_stratified_group_splits(
        development[target_column],
        development[group_column],
        n_splits=5,
        random_state=random_state,
    )
    split_summary = group_split_summary(
        development[target_column],
        development[group_column],
        group_splits,
    )
    split_summary.to_csv(output_dir / "injury_grouped_cv_folds.csv", index=False)

    tuning_results = pd.DataFrame()
    tuning_path = output_dir / "injury_tuning_results.json"
    search_iterations = 8 if fast_mode else 20

    if reuse_saved_tuning:
        if not tuning_path.exists():
            raise FileNotFoundError(
                "Saved injury tuning results are missing. Run tuning first or provide the frozen JSON."
            )
        candidates, tuning_results = apply_saved_injury_tuning(candidates, tuning_path)

    if tune and not reuse_saved_tuning:
        # Champion selection uses nested cross-validation: each family is tuned
        # on athlete-grouped sub-folds of each outer fold's training athletes, so
        # no family is ranked on athletes its own search optimised.
        (
            summary,
            fold_metrics,
            out_of_fold_predictions,
            nested_tuning_records,
        ) = evaluate_injury_candidates_nested_cv(
            candidates,
            development,
            development[target_column],
            development[group_column],
            group_splits,
            iterations=search_iterations,
            random_state=random_state,
        )
        nested_tuning_records.to_csv(
            output_dir / "injury_nested_cv_selected_parameters.csv", index=False
        )
    else:
        summary, fold_metrics, out_of_fold_predictions = evaluate_injury_candidates_cv(
            candidates,
            development,
            development[target_column],
            development[group_column],
            group_splits,
        )

    champion_name, tie_table = select_injury_champion(summary, candidates)

    if tune and not reuse_saved_tuning:
        # The family is now fixed. Searching once more over the full grouped
        # folds fixes the hyperparameters the deployed model is fitted with. This
        # search chooses parameters, not families, so it cannot bias a comparison
        # that has already happened.
        candidates, tuning_results = tune_injury_candidates(
            candidates,
            development,
            development[target_column],
            group_splits,
            iterations=search_iterations,
            random_state=random_state,
        )
        tuning_results.to_json(
            tuning_path,
            orient="records",
            indent=2,
        )

    champion_candidate = candidates[champion_name]

    summary.to_csv(output_dir / "injury_cv_model_comparison.csv")
    fold_metrics.to_csv(output_dir / "injury_cv_fold_metrics.csv", index=False)
    out_of_fold_predictions.to_csv(
        output_dir / "injury_cv_out_of_fold_predictions.csv", index=False
    )
    tie_table.to_csv(output_dir / "injury_champion_selection_tie_table.csv")

    calibration_audit, calibration_predictions = grouped_calibration_audit(
        champion_candidate,
        development,
        target_column,
        group_column,
        group_splits,
        random_state,
    )
    calibration_audit.to_csv(
        output_dir / "injury_calibration_audit_metrics.csv"
    )
    calibration_predictions.to_csv(
        output_dir / "injury_calibration_audit_predictions.csv",
        index=False,
    )
    save_multiclass_calibration_comparison(
        calibration_predictions,
        target_column,
        output_dir,
    )

    # Fit an uncalibrated copy for SHAP and a grouped-CV calibrated champion for
    # probability-based decision support.
    base_model = clone(champion_candidate.estimator)
    base_model.fit(
        development[champion_candidate.feature_columns],
        development[target_column],
    )

    calibrated = make_calibrated_classifier(
        clone(champion_candidate.estimator),
        group_splits,
        method="sigmoid",
    )
    calibrated.fit(
        development[champion_candidate.feature_columns],
        development[target_column],
    )

    reference_quantiles = {}
    for feature in manifest["numeric_features"]:
        series = development[feature]
        reference_quantiles[feature] = {
            "q01": float(series.quantile(0.01)) if series.notna().any() else None,
            "q10": float(series.quantile(0.10)) if series.notna().any() else None,
            "q25": float(series.quantile(0.25)) if series.notna().any() else None,
            "q50": float(series.quantile(0.50)) if series.notna().any() else None,
            "q75": float(series.quantile(0.75)) if series.notna().any() else None,
            "q90": float(series.quantile(0.90)) if series.notna().any() else None,
            "q99": float(series.quantile(0.99)) if series.notna().any() else None,
        }

    mutable_features = [
        feature
        for feature in [
            "training_load",
            "training_intensity",
            "training_duration",
            "recovery_score",
            "sleep_quality",
            "hydration_level",
            "fatigue_index",
            "stress_level",
        ]
        if feature in champion_candidate.feature_columns
    ]

    bundle = {
        "stream": "injury",
        "model_version": "2.0-candidate",
        "champion_name": champion_name,
        "champion_family": champion_candidate.family,
        "imbalance_strategy": champion_candidate.imbalance_strategy,
        "calibrated_model": calibrated,
        "base_model_for_explanation": base_model,
        "feature_columns": champion_candidate.feature_columns,
        "numeric_features": manifest["numeric_features"],
        "categorical_features": manifest["categorical_features"],
        "target": target_column,
        "class_names": [CLASS_NAMES[index] for index in (0, 1, 2)],
        "selection_rule": (
            "Highest mean athlete-grouped CV Macro-F1; models within 0.01 are "
            "compared by High Risk/Injured recall, log loss, multiclass Brier "
            "score, parsimony and variability."
        ),
        "grouping_variable": group_column,
        "final_test_evaluated": bool(unlock_final_test),
        "reference_quantiles": reference_quantiles,
        "mutable_what_if_features": mutable_features,
        "manifest": manifest,
        "environment": environment_metadata(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    final_test_metrics = None
    subgroup_metrics = pd.DataFrame()
    confidence_intervals = pd.DataFrame()

    if unlock_final_test:
        (output_dir / "injury_final_test_lock.json").unlink(missing_ok=True)
        prediction = calibrated.predict(final_test[champion_candidate.feature_columns])
        probabilities = calibrated.predict_proba(
            final_test[champion_candidate.feature_columns]
        )
        final_predictions = final_test.copy()
        final_predictions["prediction"] = prediction
        for class_value in (0, 1, 2):
            final_predictions[f"probability_class_{class_value}"] = probabilities[
                :, class_value
            ]
        final_predictions.to_csv(
            output_dir / "injury_final_test_predictions.csv", index=False
        )

        final_test_metrics = multiclass_metrics(
            final_test[target_column],
            prediction,
            probabilities,
            classes=(0, 1, 2),
        )
        pd.DataFrame([final_test_metrics], index=[champion_name]).to_csv(
            output_dir / "injury_final_test_metrics.csv"
        )

        subgroup_metrics = multiclass_subgroup_metrics(
            final_predictions,
            subgroup_columns=["sport_type", "gender", "playing_surface"],
            target_column=target_column,
            prediction_column="prediction",
            probability_columns=[
                "probability_class_0",
                "probability_class_1",
                "probability_class_2",
            ],
            minimum_rows=80,
            classes=(0, 1, 2),
        )
        subgroup_metrics.to_csv(
            output_dir / "injury_final_test_subgroup_metrics.csv", index=False
        )

        def bootstrap_metric(sample: pd.DataFrame) -> dict[str, float]:
            return multiclass_metrics(
                sample[target_column],
                sample["prediction"],
                sample[
                    [
                        "probability_class_0",
                        "probability_class_1",
                        "probability_class_2",
                    ]
                ].to_numpy(),
                classes=(0, 1, 2),
            )

        confidence_intervals = cluster_bootstrap_confidence_intervals(
            final_predictions,
            cluster_column=group_column,
            metric_function=bootstrap_metric,
            n_bootstrap=bootstrap_replicates,
            random_state=random_state,
        )
        confidence_intervals.to_csv(
            output_dir / "injury_athlete_bootstrap_confidence_intervals.csv"
        )
        save_injury_figures(final_predictions, output_dir, champion_name)
    else:
        for stale_name in [
            "injury_final_test_predictions.csv",
            "injury_final_test_metrics.csv",
            "injury_final_test_subgroup_metrics.csv",
            "injury_athlete_bootstrap_confidence_intervals.csv",
            "injury_confusion_matrix.png",
            "injury_calibration_class_0.png",
            "injury_calibration_class_1.png",
            "injury_calibration_class_2.png",
        ]:
            (output_dir / stale_name).unlink(missing_ok=True)
        save_json(
            {
                "status": "locked",
                "message": (
                    "The unseen-athlete final test has not been evaluated. Use "
                    "--unlock-final-test only after accepting the grouped-CV "
                    "champion and selection rule."
                ),
            },
            output_dir / "injury_final_test_lock.json",
        )

    model_path = model_dir / "injury_champion_bundle.joblib"
    save_model_bundle(bundle, model_path)

    metadata = {
        "champion_name": champion_name,
        "imbalance_strategy": champion_candidate.imbalance_strategy,
        "candidate_names": list(candidates),
        "development_rows": len(development),
        "development_athletes": development[group_column].nunique(),
        "final_test_rows": len(final_test),
        "final_test_athletes": final_test[group_column].nunique(),
        "final_test_evaluated": bool(unlock_final_test),
        "final_test_metrics": final_test_metrics,
        "bundle_path": str(model_path.relative_to(project_root)),
    }
    save_json(metadata, output_dir / "injury_run_metadata.json")

    return {
        "manifest": manifest,
        "development": development,
        "final_test": final_test,
        "candidates": candidates,
        "cv_summary": summary,
        "fold_metrics": fold_metrics,
        "champion_name": champion_name,
        "tie_table": tie_table,
        "calibration_audit": calibration_audit,
        "calibrated_model": calibrated,
        "base_model": base_model,
        "model_bundle": bundle,
        "model_path": model_path,
        "final_test_metrics": final_test_metrics,
        "subgroup_metrics": subgroup_metrics,
        "confidence_intervals": confidence_intervals,
    }
