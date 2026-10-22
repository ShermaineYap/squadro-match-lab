"""BWF match-outcome modelling: candidate families, selection and the audit run."""

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
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)
from sklearn.calibration import CalibrationDisplay
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

import matplotlib.pyplot as plt

from src.evaluation.metrics import (
    aggregate_fold_metrics,
    binary_metrics,
    binary_subgroup_metrics,
    cluster_bootstrap_confidence_intervals,
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
    date_split_summary,
    make_expanding_date_splits,
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False


def make_preprocessor(
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


def make_pipeline(
    model: Any,
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", make_preprocessor(numeric_features, categorical_features)),
            ("model", model),
        ]
    )


def build_bwf_candidates(
    manifest: dict,
    fast_mode: bool = True,
    random_state: int = RANDOM_STATE,
) -> dict[str, Candidate]:
    numeric = list(manifest["numeric_features"])
    categorical = list(manifest["categorical_features"])
    all_features = numeric + categorical

    forest_estimators = 180 if fast_mode else 500
    xgb_estimators = 200 if fast_mode else 500

    candidates: dict[str, Candidate] = {
        "Majority Baseline": Candidate(
            name="Majority Baseline",
            estimator=make_pipeline(
                DummyClassifier(strategy="prior"),
                numeric,
                categorical,
            ),
            complexity_rank=0,
            feature_columns=all_features,
            family="baseline",
        ),
        "Elo-only Logistic Regression": Candidate(
            name="Elo-only Logistic Regression",
            estimator=Pipeline(
                [
                    (
                        "preprocess",
                        ColumnTransformer(
                            [
                                (
                                    "elo",
                                    Pipeline(
                                        [
                                            ("imputer", SimpleImputer(strategy="median")),
                                            ("scaler", StandardScaler()),
                                        ]
                                    ),
                                    ["individual_elo_diff"],
                                )
                            ]
                        ),
                    ),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=3000,
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
            complexity_rank=1,
            feature_columns=["individual_elo_diff"],
            family="logistic_regression",
        ),
        "Full Logistic Regression": Candidate(
            name="Full Logistic Regression",
            estimator=make_pipeline(
                LogisticRegression(
                    max_iter=3000,
                    random_state=random_state,
                ),
                numeric,
                categorical,
            ),
            complexity_rank=2,
            feature_columns=all_features,
            family="logistic_regression",
        ),
        "Decision Tree": Candidate(
            name="Decision Tree",
            estimator=make_pipeline(
                DecisionTreeClassifier(
                    max_depth=8,
                    min_samples_leaf=20,
                    random_state=random_state,
                ),
                numeric,
                categorical,
            ),
            complexity_rank=3,
            feature_columns=all_features,
            family="decision_tree",
        ),
        "Random Forest": Candidate(
            name="Random Forest",
            estimator=make_pipeline(
                RandomForestClassifier(
                    n_estimators=forest_estimators,
                    max_depth=None,
                    min_samples_leaf=3,
                    max_features="sqrt",
                    random_state=random_state,
                    n_jobs=-1,
                ),
                numeric,
                categorical,
            ),
            complexity_rank=4,
            feature_columns=all_features,
            family="random_forest",
        ),
    }

    if XGBOOST_AVAILABLE:
        candidates["XGBoost"] = Candidate(
            name="XGBoost",
            estimator=make_pipeline(
                XGBClassifier(
                    n_estimators=xgb_estimators,
                    max_depth=4,
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
            complexity_rank=5,
            feature_columns=all_features,
            family="xgboost",
        )

    return candidates


BWF_SEARCH_SPACES: dict[str, dict[str, Any]] = {
    "Full Logistic Regression": {
        "model__C": np.logspace(-3, 2, 30),
    },
    "Decision Tree": {
        "model__max_depth": [3, 5, 8, 12, 16, None],
        "model__min_samples_leaf": [5, 10, 20, 40],
        "model__criterion": ["gini", "entropy", "log_loss"],
    },
    "Random Forest": {
        "model__n_estimators": [250, 400, 600],
        "model__max_depth": [None, 8, 12, 18],
        "model__min_samples_leaf": [1, 2, 4, 8],
        "model__max_features": ["sqrt", "log2", 0.5],
    },
    "XGBoost": {
        "model__n_estimators": [200, 350, 500],
        "model__max_depth": [2, 3, 4, 5],
        "model__learning_rate": [0.02, 0.04, 0.06, 0.1],
        "model__subsample": [0.7, 0.85, 1.0],
        "model__colsample_bytree": [0.7, 0.85, 1.0],
        "model__min_child_weight": [1, 3, 5],
        "model__reg_lambda": [0.5, 1.0, 2.0, 5.0],
    },
}


def tune_bwf_candidates(
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
    :func:`evaluate_bwf_candidates_nested_cv` instead and this function is run
    afterwards, once the family is already decided.
    """
    spaces = BWF_SEARCH_SPACES

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
                "roc_auc": "roc_auc",
                "neg_log_loss": "neg_log_loss",
            },
            refit="roc_auc",
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
                "best_mean_cv_roc_auc": search.best_score_,
                "best_parameters": search.best_params_,
            }
        )
    return updated, pd.DataFrame(rows)


def apply_saved_bwf_tuning(
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

def evaluate_bwf_candidates_cv(
    candidates: dict[str, Candidate],
    frame: pd.DataFrame,
    target: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Score every candidate as given, one fit per fold.

    Valid only when the candidates carry hyperparameters that were not chosen on
    these folds - untuned defaults, or parameters frozen by an earlier run. When
    a run tunes, use :func:`evaluate_bwf_candidates_nested_cv`.
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

            estimator.fit(X_train, y_train)
            prediction = estimator.predict(X_validation)
            probability = estimator.predict_proba(X_validation)[:, 1]
            metrics = binary_metrics(y_validation, prediction, probability)

            fold_rows.append(
                {
                    "model": candidate_name,
                    "fold": fold_number,
                    **metrics,
                }
            )

            fold_predictions = frame.iloc[validation_indices][
                ["match_id", "date", "tournament_id", "discipline"]
            ].copy()
            fold_predictions["model"] = candidate_name
            fold_predictions["fold"] = fold_number
            fold_predictions["actual"] = y_validation.to_numpy()
            fold_predictions["prediction"] = prediction
            fold_predictions["probability_team1_win"] = probability
            prediction_rows.append(fold_predictions)

    fold_frame = pd.DataFrame(fold_rows)
    summary = aggregate_fold_metrics(fold_frame)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    return summary, fold_frame, predictions


def evaluate_bwf_candidates_nested_cv(
    candidates: dict[str, Candidate],
    frame: pd.DataFrame,
    target: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
    iterations: int = 12,
    inner_splits: int = 3,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare families with each family's tuning contained inside the fold.

    Tuning on the same folds that later rank the candidates gives every tunable
    family several attempts at each validation window while the untuned
    baselines get one, so the comparison rewards the search rather than the
    family. Here the search for a fold runs on chronological sub-folds of that
    fold's training window only, and the fold's validation matches are scored
    once, by a model whose hyperparameters were chosen without them.

    The cost is that different folds may select different hyperparameters. That
    is the point: the reported score is the performance of the *procedure*
    "tune this family, then fit it", which is what a future season would get,
    rather than the performance of a single lucky parameter draw.

    Returns the usual comparison triple plus the parameters each fold chose, so
    an unstable family - one whose fold winners disagree - is visible instead of
    hidden behind an averaged score.
    """
    fold_rows = []
    prediction_rows = []
    tuning_rows = []

    for candidate_name, candidate in candidates.items():
        space = BWF_SEARCH_SPACES.get(candidate_name)
        for fold_number, (train_indices, validation_indices) in enumerate(
            splits, start=1
        ):
            train_frame = frame.iloc[train_indices]
            X_train = train_frame[candidate.feature_columns]
            X_validation = frame.iloc[validation_indices][candidate.feature_columns]
            y_train = target.iloc[train_indices]
            y_validation = target.iloc[validation_indices]

            if space and iterations > 0:
                search = RandomizedSearchCV(
                    estimator=clone(candidate.estimator),
                    param_distributions=space,
                    n_iter=iterations,
                    scoring={
                        "roc_auc": "roc_auc",
                        "neg_log_loss": "neg_log_loss",
                    },
                    refit="roc_auc",
                    cv=make_expanding_date_splits(
                        train_frame["date"], n_splits=inner_splits
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
                        "inner_cv_roc_auc": search.best_score_,
                        "best_parameters": search.best_params_,
                    }
                )
            else:
                estimator = clone(candidate.estimator)
                estimator.fit(X_train, y_train)

            prediction = estimator.predict(X_validation)
            probability = estimator.predict_proba(X_validation)[:, 1]
            metrics = binary_metrics(y_validation, prediction, probability)

            fold_rows.append(
                {
                    "model": candidate_name,
                    "fold": fold_number,
                    **metrics,
                }
            )

            fold_predictions = frame.iloc[validation_indices][
                ["match_id", "date", "tournament_id", "discipline"]
            ].copy()
            fold_predictions["model"] = candidate_name
            fold_predictions["fold"] = fold_number
            fold_predictions["actual"] = y_validation.to_numpy()
            fold_predictions["prediction"] = prediction
            fold_predictions["probability_team1_win"] = probability
            prediction_rows.append(fold_predictions)

    fold_frame = pd.DataFrame(fold_rows)
    summary = aggregate_fold_metrics(fold_frame)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    return summary, fold_frame, predictions, pd.DataFrame(tuning_rows)


def select_bwf_champion(
    summary: pd.DataFrame,
    candidates: dict[str, Candidate],
    auc_tolerance: float = 0.01,
) -> tuple[str, pd.DataFrame]:
    eligible = summary.copy()
    best_auc = eligible["mean_roc_auc"].max()
    eligible = eligible[
        eligible["mean_roc_auc"] >= best_auc - auc_tolerance
    ].copy()
    eligible["complexity_rank"] = [
        candidates[name].complexity_rank for name in eligible.index
    ]
    eligible = eligible.sort_values(
        [
            "mean_log_loss",
            "mean_brier",
            "complexity_rank",
            "std_roc_auc",
        ],
        ascending=[True, True, True, True],
    )
    champion = str(eligible.index[0])
    return champion, eligible


def evaluate_on_frame(
    candidate: Candidate,
    fit_frame: pd.DataFrame,
    evaluation_frame: pd.DataFrame,
    target_column: str,
) -> tuple[Any, dict[str, float], pd.DataFrame]:
    estimator = clone(candidate.estimator)
    estimator.fit(
        fit_frame[candidate.feature_columns],
        fit_frame[target_column],
    )
    prediction = estimator.predict(evaluation_frame[candidate.feature_columns])
    probability = estimator.predict_proba(
        evaluation_frame[candidate.feature_columns]
    )[:, 1]
    metrics = binary_metrics(
        evaluation_frame[target_column],
        prediction,
        probability,
    )
    predictions = evaluation_frame.copy()
    predictions["prediction"] = prediction
    predictions["probability_team1_win"] = probability
    return estimator, metrics, predictions


def make_calibrated_classifier(
    estimator: Any,
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
    method: str = "sigmoid",
) -> CalibratedClassifierCV:
    # `estimator` is the current argument name. `base_estimator` supports older
    # scikit-learn versions.
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


def save_bwf_figures(
    predictions: pd.DataFrame,
    output_dir: Path,
    title_suffix: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    y_true = predictions["team1_win"]
    y_pred = predictions["prediction"]
    probability = predictions["probability_team1_win"]

    figure, axis = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=["Team 2", "Team 1"],
        ax=axis,
    )
    axis.set_title(f"BWF Confusion Matrix — {title_suffix}")
    figure.tight_layout()
    figure.savefig(output_dir / "bwf_confusion_matrix.png", dpi=200)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_true, probability, ax=axis)
    axis.set_title(f"BWF ROC Curve — {title_suffix}")
    figure.tight_layout()
    figure.savefig(output_dir / "bwf_roc_curve.png", dpi=200)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, 5))
    CalibrationDisplay.from_predictions(
        y_true,
        probability,
        n_bins=10,
        strategy="quantile",
        ax=axis,
    )
    axis.set_title(f"BWF Calibration — {title_suffix}")
    figure.tight_layout()
    figure.savefig(output_dir / "bwf_calibration.png", dpi=200)
    plt.close(figure)


def save_binary_calibration_comparison(
    y_true: pd.Series,
    uncalibrated_probability: np.ndarray,
    calibrated_probability: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    figure, axis = plt.subplots(figsize=(6, 5))
    for label, probability in [
        ("Uncalibrated", uncalibrated_probability),
        ("Calibrated", calibrated_probability),
    ]:
        observed, predicted = calibration_curve(
            y_true,
            probability,
            n_bins=10,
            strategy="quantile",
        )
        axis.plot(predicted, observed, marker="o", label=label)
    axis.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    axis.set_xlabel("Mean predicted probability")
    axis.set_ylabel("Observed frequency")
    axis.set_title(title)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)

def run_bwf_experiment(
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
    prepared_dir = project_root / "outputs" / "bwf"
    output_dir = project_root / "outputs" / "modeling" / "bwf"
    model_dir = project_root / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_json(prepared_dir / "bwf_feature_manifest.json")
    data = pd.read_csv(
        prepared_dir / "bwf_model_rows_2022_2025.csv",
        parse_dates=["date"],
    )

    train = data[data["split"] == "train"].copy().reset_index(drop=True)
    validation = data[data["split"] == "validation"].copy().reset_index(drop=True)
    final_test = data[data["split"] == "test"].copy().reset_index(drop=True)
    target_column = manifest["target"]

    if train["date"].max() >= validation["date"].min():
        raise AssertionError("BWF training and validation dates overlap.")
    if validation["date"].max() >= final_test["date"].min():
        raise AssertionError("BWF validation and final-test dates overlap.")

    candidates = build_bwf_candidates(manifest, fast_mode=fast_mode)
    train_splits = make_expanding_date_splits(train["date"], n_splits=3)
    fold_summary = date_split_summary(train["date"], train_splits)
    fold_summary.to_csv(output_dir / "bwf_chronological_cv_folds.csv", index=False)

    tuning_results = pd.DataFrame()
    tuning_path = output_dir / "bwf_tuning_results.json"
    search_iterations = 8 if fast_mode else 20

    if reuse_saved_tuning:
        if not tuning_path.exists():
            raise FileNotFoundError(
                "Saved BWF tuning results are missing. Run tuning first or provide the frozen JSON."
            )
        candidates, tuning_results = apply_saved_bwf_tuning(candidates, tuning_path)

    if tune and not reuse_saved_tuning:
        # Champion selection uses nested cross-validation: each family is tuned
        # inside each fold's training window, so no family is ranked on folds its
        # own search optimised.
        (
            summary,
            fold_metrics,
            out_of_fold_predictions,
            nested_tuning_records,
        ) = evaluate_bwf_candidates_nested_cv(
            candidates,
            train,
            train[target_column],
            train_splits,
            iterations=search_iterations,
            random_state=random_state,
        )
        nested_tuning_records.to_csv(
            output_dir / "bwf_nested_cv_selected_parameters.csv", index=False
        )
    else:
        summary, fold_metrics, out_of_fold_predictions = evaluate_bwf_candidates_cv(
            candidates,
            train,
            train[target_column],
            train_splits,
        )

    champion_name, tie_table = select_bwf_champion(summary, candidates)

    if tune and not reuse_saved_tuning:
        # The family is now fixed. Searching once more over the full training
        # folds fixes the hyperparameters the deployed model is fitted with. This
        # search chooses parameters, not families, so it cannot bias a comparison
        # that has already happened.
        candidates, tuning_results = tune_bwf_candidates(
            candidates,
            train,
            train[target_column],
            train_splits,
            iterations=search_iterations,
            random_state=random_state,
        )
        tuning_results.to_json(
            tuning_path,
            orient="records",
            indent=2,
        )

    champion_candidate = candidates[champion_name]

    summary.to_csv(output_dir / "bwf_cv_model_comparison.csv")
    fold_metrics.to_csv(output_dir / "bwf_cv_fold_metrics.csv", index=False)
    out_of_fold_predictions.to_csv(
        output_dir / "bwf_cv_out_of_fold_predictions.csv", index=False
    )
    tie_table.to_csv(output_dir / "bwf_champion_selection_tie_table.csv")

    # The champion is selected from 2022–2023 chronological CV. 2024 is an audit
    # and calibration-development period, not a reason to switch families.
    fitted_on_train, validation_metrics, validation_predictions = evaluate_on_frame(
        champion_candidate,
        train,
        validation,
        target_column,
    )
    pd.DataFrame([validation_metrics], index=[champion_name]).to_csv(
        output_dir / "bwf_2024_validation_audit_metrics.csv"
    )
    validation_predictions.to_csv(
        output_dir / "bwf_2024_validation_predictions.csv", index=False
    )

    # Calibration audit on the separate 2024 period. Both models are fitted
    # using 2022–2023 only, so the comparison does not use 2024 labels in fit.
    calibrated_on_train = make_calibrated_classifier(
        clone(champion_candidate.estimator),
        train_splits,
        method="sigmoid",
    )
    calibrated_on_train.fit(
        train[champion_candidate.feature_columns],
        train[target_column],
    )
    calibrated_validation_prediction = calibrated_on_train.predict(
        validation[champion_candidate.feature_columns]
    )
    calibrated_validation_probability = calibrated_on_train.predict_proba(
        validation[champion_candidate.feature_columns]
    )[:, 1]
    calibrated_validation_metrics = binary_metrics(
        validation[target_column],
        calibrated_validation_prediction,
        calibrated_validation_probability,
    )
    calibration_audit = pd.DataFrame(
        [validation_metrics, calibrated_validation_metrics],
        index=["uncalibrated", "calibrated"],
    )
    calibration_audit.to_csv(
        output_dir / "bwf_2024_calibration_audit_metrics.csv"
    )
    calibration_predictions = validation[[
        "match_id", "date", "tournament_id", target_column
    ]].copy()
    calibration_predictions["uncalibrated_probability"] = (
        validation_predictions["probability_team1_win"].to_numpy()
    )
    calibration_predictions["calibrated_probability"] = (
        calibrated_validation_probability
    )
    calibration_predictions.to_csv(
        output_dir / "bwf_2024_calibration_audit_predictions.csv",
        index=False,
    )
    save_binary_calibration_comparison(
        validation[target_column],
        validation_predictions["probability_team1_win"].to_numpy(),
        calibrated_validation_probability,
        output_dir / "bwf_2024_calibration_comparison.png",
        "BWF 2024 Calibration Audit",
    )

    # Build a calibrated final development model using 2022–2024 only.
    development = pd.concat([train, validation], ignore_index=True)
    calibration_splits = make_expanding_date_splits(
        development["date"], n_splits=3, minimum_training_fraction=0.35
    )
    calibrated = make_calibrated_classifier(
        clone(champion_candidate.estimator),
        calibration_splits,
        method="sigmoid",
    )
    calibrated.fit(
        development[champion_candidate.feature_columns],
        development[target_column],
    )

    # SHAP explains an uncalibrated copy trained on the same 2022–2024
    # development period as the deployed calibrated bundle.
    explanation_model = clone(champion_candidate.estimator)
    explanation_model.fit(
        development[champion_candidate.feature_columns],
        development[target_column],
    )

    reference_quantiles = {}
    for feature in manifest["numeric_features"]:
        series = development[feature]
        reference_quantiles[feature] = {
            "q01": float(series.quantile(0.01)) if series.notna().any() else None,
            "q25": float(series.quantile(0.25)) if series.notna().any() else None,
            "q50": float(series.quantile(0.50)) if series.notna().any() else None,
            "q75": float(series.quantile(0.75)) if series.notna().any() else None,
            "q99": float(series.quantile(0.99)) if series.notna().any() else None,
        }

    bundle = {
        "stream": "bwf",
        "model_version": "2.0-candidate",
        "champion_name": champion_name,
        "champion_family": champion_candidate.family,
        "calibrated_model": calibrated,
        "base_model_for_explanation": explanation_model,
        "feature_columns": champion_candidate.feature_columns,
        "numeric_features": manifest["numeric_features"],
        "categorical_features": manifest["categorical_features"],
        "target": target_column,
        "class_names": ["Team 2 wins", "Team 1 wins"],
        "selection_rule": (
            "Highest mean chronological-CV ROC-AUC; models within 0.01 are "
            "compared by log loss, Brier score, parsimony and variability."
        ),
        "development_period": "2022–2024",
        "final_test_period": "2025",
        "final_test_evaluated": bool(unlock_final_test),
        "reference_quantiles": reference_quantiles,
        "manifest": manifest,
        "environment": environment_metadata(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    final_test_metrics = None
    subgroup_metrics = pd.DataFrame()
    confidence_intervals = pd.DataFrame()

    if unlock_final_test:
        (output_dir / "bwf_final_test_lock.json").unlink(missing_ok=True)
        final_prediction = calibrated.predict(
            final_test[champion_candidate.feature_columns]
        )
        final_probability = calibrated.predict_proba(
            final_test[champion_candidate.feature_columns]
        )[:, 1]
        final_predictions = final_test.copy()
        final_predictions["prediction"] = final_prediction
        final_predictions["probability_team1_win"] = final_probability
        final_predictions.to_csv(
            output_dir / "bwf_2025_final_test_predictions.csv", index=False
        )

        final_test_metrics = binary_metrics(
            final_test[target_column],
            final_prediction,
            final_probability,
        )
        pd.DataFrame([final_test_metrics], index=[champion_name]).to_csv(
            output_dir / "bwf_2025_final_test_metrics.csv"
        )

        subgroup_metrics = binary_subgroup_metrics(
            final_predictions,
            subgroup_columns=[
                "discipline",
                "tier_category",
                "is_qualification",
                "is_doubles",
                "either_team_cold_start",
            ],
            target_column=target_column,
            prediction_column="prediction",
            probability_column="probability_team1_win",
            minimum_rows=80,
        )
        subgroup_metrics.to_csv(
            output_dir / "bwf_2025_subgroup_metrics.csv", index=False
        )

        def bootstrap_metric(sample: pd.DataFrame) -> dict[str, float]:
            return binary_metrics(
                sample[target_column],
                sample["prediction"],
                sample["probability_team1_win"],
            )

        confidence_intervals = cluster_bootstrap_confidence_intervals(
            final_predictions,
            cluster_column="tournament_id",
            metric_function=bootstrap_metric,
            n_bootstrap=bootstrap_replicates,
            random_state=random_state,
        )
        confidence_intervals.to_csv(
            output_dir / "bwf_2025_cluster_bootstrap_confidence_intervals.csv"
        )
        save_bwf_figures(final_predictions, output_dir, champion_name)
    else:
        for stale_name in [
            "bwf_2025_final_test_predictions.csv",
            "bwf_2025_final_test_metrics.csv",
            "bwf_2025_subgroup_metrics.csv",
            "bwf_2025_cluster_bootstrap_confidence_intervals.csv",
            "bwf_confusion_matrix.png",
            "bwf_roc_curve.png",
            "bwf_calibration.png",
        ]:
            (output_dir / stale_name).unlink(missing_ok=True)
        save_json(
            {
                "status": "locked",
                "message": (
                    "The 2025 final test has not been evaluated. Use "
                    "--unlock-final-test only after the champion family and "
                    "selection rule are accepted."
                ),
            },
            output_dir / "bwf_final_test_lock.json",
        )

    model_path = model_dir / "bwf_champion_bundle.joblib"
    save_model_bundle(bundle, model_path)

    metadata = {
        "champion_name": champion_name,
        "candidate_names": list(candidates),
        "training_rows": len(train),
        "validation_rows": len(validation),
        "final_test_rows": len(final_test),
        "final_test_evaluated": bool(unlock_final_test),
        "validation_metrics": validation_metrics,
        "calibrated_validation_metrics": calibrated_validation_metrics,
        "final_test_metrics": final_test_metrics,
        "bundle_path": str(model_path.relative_to(project_root)),
    }
    save_json(metadata, output_dir / "bwf_run_metadata.json")

    return {
        "manifest": manifest,
        "data": data,
        "train": train,
        "validation": validation,
        "final_test": final_test,
        "candidates": candidates,
        "cv_summary": summary,
        "fold_metrics": fold_metrics,
        "champion_name": champion_name,
        "tie_table": tie_table,
        "validation_metrics": validation_metrics,
        "calibrated_validation_metrics": calibrated_validation_metrics,
        "validation_predictions": validation_predictions,
        "calibrated_model": calibrated,
        "model_bundle": bundle,
        "model_path": model_path,
        "final_test_metrics": final_test_metrics,
        "subgroup_metrics": subgroup_metrics,
        "confidence_intervals": confidence_intervals,
    }
