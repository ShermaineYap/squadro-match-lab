from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_MUTABLE_FEATURES = [
    "training_load",
    "training_intensity",
    "training_duration",
    "recovery_score",
    "sleep_quality",
    "hydration_level",
    "fatigue_index",
    "stress_level",
]


def _single_row_frame(
    row: pd.Series | dict,
    feature_columns: list[str],
) -> pd.DataFrame:
    if isinstance(row, pd.Series):
        frame = row.to_frame().T
    else:
        frame = pd.DataFrame([row])
    missing = [feature for feature in feature_columns if feature not in frame.columns]
    if missing:
        raise ValueError(f"Scenario row is missing required features: {missing}")
    return frame[feature_columns].copy()


def _numeric_scenario_value(feature: str, value: Any) -> float:
    """Coerce one requested what-if value, refusing anything that is not a number.

    The clipping below only guards features the bundle recorded quantiles for.
    A feature without them - a newly added lever, or one the export missed -
    took whatever the caller sent straight into a numeric column, so a stray
    ``"high"`` reached the fitted pipeline and came back as a pandas dtype error
    about converting a string to float. That is a real rejection wearing the
    costume of an internal failure, so it is caught here and named.
    """
    if isinstance(value, bool) or value is None:
        raise ValueError(
            f"What-if value for {feature!r} must be a number, received {value!r}."
        )
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"What-if value for {feature!r} must be a number, received {value!r}."
        ) from error


def _bounded_value(
    feature: str,
    value: float,
    reference_quantiles: dict,
) -> tuple[float, bool]:
    numeric = _numeric_scenario_value(feature, value)
    bounds = reference_quantiles.get(feature, {})
    lower = bounds.get("q01")
    upper = bounds.get("q99")
    if lower is None or upper is None or pd.isna(numeric):
        return numeric, False
    clipped = float(np.clip(numeric, float(lower), float(upper)))
    return clipped, clipped != numeric


def injury_what_if(
    model_bundle: dict,
    row: pd.Series | dict,
    changes: dict[str, Any],
) -> dict:
    feature_columns = list(model_bundle["feature_columns"])
    allowed = set(model_bundle.get("mutable_what_if_features", DEFAULT_MUTABLE_FEATURES))
    invalid = sorted(set(changes) - allowed)
    if invalid:
        raise ValueError(
            "The following fields are not approved for injury what-if analysis: "
            + ", ".join(invalid)
        )

    current = _single_row_frame(row, feature_columns)
    scenario = current.copy()
    clipping = []

    for feature, requested_value in changes.items():
        bounded, was_clipped = _bounded_value(
            feature,
            requested_value,
            model_bundle.get("reference_quantiles", {}),
        )
        scenario.loc[scenario.index[0], feature] = bounded
        clipping.append(
            {
                "feature": feature,
                "requested_value": requested_value,
                "used_value": bounded,
                "clipped_to_training_range": was_clipped,
            }
        )

    model = model_bundle.get("serving_model", model_bundle.get("calibrated_model"))
    current_probability = model.predict_proba(current)[0]
    scenario_probability = model.predict_proba(scenario)[0]
    class_names = list(model_bundle["class_names"])

    return {
        "current_probabilities": {
            class_names[index]: float(value)
            for index, value in enumerate(current_probability)
        },
        "scenario_probabilities": {
            class_names[index]: float(value)
            for index, value in enumerate(scenario_probability)
        },
        "probability_changes": {
            class_names[index]: float(scenario_probability[index] - current_probability[index])
            for index in range(len(class_names))
        },
        "changes": clipping,
        "scenario_row": scenario.iloc[0].to_dict(),
        "disclaimer": (
            "This is a model scenario. It shows how the fitted model responds "
            "to different inputs; it does not prove that making these changes "
            "will cause injury risk to change."
        ),
    }


def ranked_injury_scenarios(
    model_bundle: dict,
    row: pd.Series | dict,
    candidate_features: list[str] | None = None,
    maximum_scenarios: int = 12,
    target_class_index: int = 2,
) -> pd.DataFrame:
    """
    Generate simple, bounded one-feature scenarios using development-data
    quantiles. These are not causal recommendations.
    """
    allowed = list(model_bundle.get("mutable_what_if_features", DEFAULT_MUTABLE_FEATURES))
    if candidate_features is not None:
        allowed = [feature for feature in allowed if feature in candidate_features]

    quantiles = model_bundle.get("reference_quantiles", {})
    scenarios = []
    for feature in allowed:
        if feature not in quantiles:
            continue
        values = [
            quantiles[feature].get("q25"),
            quantiles[feature].get("q50"),
            quantiles[feature].get("q75"),
        ]
        for value in values:
            if value is None:
                continue
            result = injury_what_if(model_bundle, row, {feature: value})
            class_name = model_bundle["class_names"][target_class_index]
            scenarios.append(
                {
                    "feature": feature,
                    "scenario_value": value,
                    "target_class": class_name,
                    "current_probability": result["current_probabilities"][class_name],
                    "scenario_probability": result["scenario_probabilities"][class_name],
                    "probability_change": result["probability_changes"][class_name],
                }
            )

    return (
        pd.DataFrame(scenarios)
        .sort_values("scenario_probability")
        .head(maximum_scenarios)
        .reset_index(drop=True)
    )


def bwf_context_what_if(
    model_bundle: dict,
    row: pd.Series | dict,
    changes: dict[str, Any],
) -> dict:
    """
    Limited BWF scenario analysis. Only contextual inputs are editable.
    Historical strength features must be generated by the system and are locked.
    """
    allowed = {
        "tier_category",
        "round",
        "is_qualification",
        "host_country",
    }
    invalid = sorted(set(changes) - allowed)
    if invalid:
        raise ValueError(
            "BWF what-if only allows contextual fields. Historical features "
            f"cannot be edited directly: {invalid}"
        )

    feature_columns = list(model_bundle["feature_columns"])
    current = _single_row_frame(row, feature_columns)
    scenario = current.copy()
    for feature, value in changes.items():
        scenario.loc[scenario.index[0], feature] = value

    model = model_bundle.get("serving_model", model_bundle.get("calibrated_model"))
    current_probability = float(model.predict_proba(current)[0, 1])
    scenario_probability = float(model.predict_proba(scenario)[0, 1])

    return {
        "current_team1_win_probability": current_probability,
        "scenario_team1_win_probability": scenario_probability,
        "probability_change": scenario_probability - current_probability,
        "changes": changes,
        "disclaimer": (
            "This is a contextual model scenario, not proof that changing the "
            "tournament context will cause the match outcome to change."
        ),
    }
