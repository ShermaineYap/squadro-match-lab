"""Serving for the binary injury-risk screen: Low Injury Risk versus High Injury Risk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.decision_support.rule_engine import (
    MINIMUM_MATERIAL_CONTRIBUTION,
    generate_review_suggestions,
    load_rules,
)
from src.decision_support.what_if import injury_what_if
from src.evaluation.robustness import out_of_distribution_flags
from src.models.injury_binary import (
    BINARY_CLASS_NAMES,
    BINARY_RESULT_EXPLANATIONS,
    BINARY_TASK_DESCRIPTION,
)
from src.serving.categorical_vocabulary import trained_categorical_vocabulary
from src.serving.errors import predict_probabilities
from src.serving.explanation_service import build_attribution, explain_row
from src.serving.input_validation import build_feature_frame
from src.serving.json_safe import to_json_safe

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BINARY_RULES = PROJECT_ROOT / "rules" / "injury_binary_rules.json"

BACKGROUND_ROWS = 60

DISCLAIMER = (
    "Experimental screening support only. This output classifies patterns in "
    "the provider's dataset. It does not diagnose an injury, does not predict "
    "with certainty whether a future injury will occur, and does not replace a "
    "coach, sports scientist or healthcare professional."
)

WHAT_IF_DISCLAIMER = (
    "This is a model sensitivity check. It shows how the fitted model responds "
    "to different inputs; it does not show that making these changes would "
    "change an athlete's outcome."
)


def _positive_index(bundle: dict) -> int:
    return int(bundle.get("positive_class_index", 1))


def build_injury_binary_frame(bundle: dict, payload: dict[str, Any]) -> pd.DataFrame:
    return build_feature_frame(
        payload=payload,
        feature_columns=list(bundle["feature_columns"]),
        numeric_features=list(bundle["numeric_features"]),
        categorical_features=list(bundle["categorical_features"]),
    )


def _label_for(bundle: dict, probability: float) -> str:
    """Apply the bundle's locked operational threshold.

    There is no ``Not determined`` branch. The protocol permits one only when a
    genuine abstention rule has been validated, and none has been: adding a
    third label on an unvalidated cut-point would be exactly the arbitrary
    threshold the governance rules out.
    """
    threshold = float(bundle["threshold"])
    return BINARY_CLASS_NAMES[1] if probability >= threshold else BINARY_CLASS_NAMES[0]


def _target_definition(bundle: dict) -> dict[str, Any]:
    provenance = bundle["binary_target_mapping"]
    return {
        "task": bundle.get("task", BINARY_TASK_DESCRIPTION),
        "source_labels": provenance["source_labels"],
        "mapping": provenance["mapping"],
        "positive_class": provenance["positive_class"],
    }


def predict_injury_binary(bundle: dict, payload: dict[str, Any]) -> dict[str, Any]:
    frame = build_injury_binary_frame(bundle, payload)
    model = bundle["serving_model"]
    probabilities = predict_probabilities(model, frame, "injury_binary")[0]
    positive = float(probabilities[_positive_index(bundle)])
    label = _label_for(bundle, positive)

    ood_flags = out_of_distribution_flags(
        payload,
        bundle.get("reference_quantiles", {}),
        categorical_options=trained_categorical_vocabulary(bundle),
    )

    return to_json_safe(
        {
            "stream": "injury_binary",
            "prediction": label,
            "predicted_class_index": int(label == BINARY_CLASS_NAMES[1]),
            "explanation": BINARY_RESULT_EXPLANATIONS[label],
            "probabilities": {
                BINARY_CLASS_NAMES[0]: float(probabilities[1 - _positive_index(bundle)]),
                BINARY_CLASS_NAMES[1]: positive,
            },
            "probability_high_injury_risk": positive,
            "threshold": float(bundle["threshold"]),
            "threshold_rule": bundle.get("threshold_rule"),
            "model_version": bundle.get("model_version", "binary-1.0-research"),
            "model_family": bundle.get("champion_family"),
            "calibration": bundle.get("calibration_mode", "uncalibrated"),
            "target_definition": _target_definition(bundle),
            "input_quality_warnings": [
                flag for flag in ood_flags if flag.get("reason") == "missing input"
            ],
            "out_of_distribution_flags": [
                flag for flag in ood_flags if flag.get("reason") != "missing input"
            ],
            "evidence_status": bundle["grouped_validation_evidence"]["evidence_status"],
            "independent_evaluation_available": bool(
                bundle["grouped_validation_evidence"]["independent_evaluation_available"]
            ),
            "human_review_required": True,
            "limitations": list(bundle["limitations"]),
            "deployment_status": bundle.get("deployment_status"),
            "disclaimer": DISCLAIMER,
        }
    )


def explain_injury_binary(
    bundle: dict,
    background: pd.DataFrame,
    payload: dict[str, Any],
    maximum_factors: int = 3,
) -> dict[str, Any]:
    """SHAP for the exact estimator that produced the displayed probability.

    The bundle stores the calibrated serving pipeline as
    ``base_model_for_explanation`` as well, so the permutation explainer runs on
    the served model itself and the contributions are additive against the
    probability on screen rather than against an uncalibrated score nobody sees.
    That additivity is checked here on every request rather than assumed.
    """
    frame = build_injury_binary_frame(bundle, payload)
    model = bundle["serving_model"]
    probabilities = predict_probabilities(model, frame, "injury_binary")[0]
    positive_index = _positive_index(bundle)
    positive = float(probabilities[positive_index])
    label = _label_for(bundle, positive)

    explanation = explain_row(
        bundle=bundle,
        background=background,
        row=frame,
        target_class_index=positive_index,
        maximum_background_rows=BACKGROUND_ROWS,
    )

    contributions = pd.DataFrame(explanation["contributions"])
    reconstructed = float(explanation["base_value"]) + float(
        contributions["shap_value"].sum()
    )
    additivity_gap = abs(reconstructed - positive)

    rules = load_rules(BINARY_RULES) if BINARY_RULES.exists() else []
    # Same two rules as the three-class stream, for the same reasons: a review
    # area needs a factor that argued for the risk class *and* moved the
    # estimate materially, while the named factors are the largest movers in
    # either direction so they cannot contradict the figure they sit under.
    suggestions = generate_review_suggestions(
        contributions,
        rules,
        maximum_suggestions=maximum_factors,
        minimum_absolute_shap=MINIMUM_MATERIAL_CONTRIBUTION,
    )

    ranked = (
        contributions.sort_values("absolute_shap", ascending=False)
        .head(maximum_factors)
        .to_dict(orient="records")
    )

    return to_json_safe(
        {
            "stream": "injury_binary",
            "prediction": label,
            "explained_class": BINARY_CLASS_NAMES[positive_index],
            "probability_high_injury_risk": positive,
            "base_value": explanation["base_value"],
            "top_factors": ranked,
            # Built from the same helper as the three-class stream so the one
            # shared panel renders both without a per-stream branch.
            "attribution": build_attribution(
                bundle=bundle,
                base_value=float(explanation["base_value"]),
                predicted_probability=positive,
                contributions=contributions,
                named_factors=ranked,
                target_class=BINARY_CLASS_NAMES[positive_index],
            ),
            "review_areas": suggestions,
            "rules_available": bool(rules),
            "rules_source": str(BINARY_RULES.relative_to(PROJECT_ROOT)),
            "model_version": bundle.get("model_version"),
            "target_definition": _target_definition(bundle),
            "additivity_check": {
                "base_value_plus_contributions": reconstructed,
                "model_probability": positive,
                "absolute_difference": additivity_gap,
                # Permutation SHAP is exact for the model it wraps, so the gap
                # here is floating-point noise plus the masker's sampling. A
                # visible gap means the explanation is not describing the served
                # model, which is worth surfacing rather than hiding.
                "additive": bool(additivity_gap < 0.05),
            },
            "evidence_note": (
                "SHAP describes how this model reached its output for these "
                "inputs. It measures association with the model's response, not "
                "causation, and it does not establish that changing a value "
                "would change an athlete's outcome."
            ),
            "disclaimer": (
                "Review areas are declared in advance in the rule file and are "
                "model-informed prompts for a qualified person. They are not a "
                "diagnosis, a clearance decision or a treatment recommendation."
            ),
        }
    )


def run_injury_binary_what_if(
    bundle: dict,
    payload: dict[str, Any],
    changes: dict[str, Any],
) -> dict[str, Any]:
    """Bounded single-scenario sensitivity on the exact binary serving model.

    Only the bundle's declared mutable levers may be edited; immutable
    attributes and every non-lever measurement are refused by
    :func:`injury_what_if`, and requested values are clipped to the development
    1st-99th percentile range.
    """
    frame = build_injury_binary_frame(bundle, payload)
    result = injury_what_if(bundle, frame.iloc[0], changes)

    positive_name = BINARY_CLASS_NAMES[_positive_index(bundle)]
    current = float(result["current_probabilities"][positive_name])
    scenario = float(result["scenario_probabilities"][positive_name])
    delta = scenario - current

    outside_range = [
        change["feature"] for change in result["changes"]
        if change["clipped_to_training_range"]
    ]

    if delta < 0:
        direction = (
            "Under this hypothetical input change, the model's High Injury "
            "Risk score decreases."
        )
    elif delta > 0:
        direction = (
            "Under this hypothetical input change, the model's High Injury "
            "Risk score increases."
        )
    else:
        direction = (
            "Under this hypothetical input change, the model's High Injury "
            "Risk score is unchanged."
        )

    return to_json_safe(
        {
            "stream": "injury_binary",
            "model_version": bundle.get("model_version"),
            "threshold": float(bundle["threshold"]),
            "current_probability_high_injury_risk": current,
            "scenario_probability_high_injury_risk": scenario,
            "probability_change": delta,
            "current_prediction": _label_for(bundle, current),
            "scenario_prediction": _label_for(bundle, scenario),
            "model_sensitivity": direction,
            "changes": result["changes"],
            "editable_features": list(bundle["mutable_what_if_features"]),
            "locked_features": list(bundle.get("immutable_what_if_features", [])),
            "out_of_distribution_warnings": [
                {
                    "feature": feature,
                    "severity": "warning",
                    "reason": (
                        "requested value sits outside the development 1st-99th "
                        "percentile range and was clipped before scoring"
                    ),
                }
                for feature in outside_range
            ],
            "target_definition": _target_definition(bundle),
            "disclaimer": WHAT_IF_DISCLAIMER,
        }
    )


def binary_technical_details(bundle: dict) -> dict[str, Any]:
    """Everything the interface's technical-details section has to state."""
    evidence = bundle["grouped_validation_evidence"]
    return to_json_safe(
        {
            "stream": "injury_binary",
            "task": bundle.get("task", BINARY_TASK_DESCRIPTION),
            "original_label_mapping": bundle["binary_target_mapping"],
            "class_names": list(bundle["class_names"]),
            "result_explanations": bundle["result_explanations"],
            "model_family": bundle.get("champion_family"),
            "model_name": bundle.get("champion_name"),
            "model_version": bundle.get("model_version"),
            "imbalance_strategy": bundle.get("imbalance_strategy"),
            "selected_parameters": bundle.get("selected_parameters"),
            "selected_parameters_note": bundle.get("selected_parameters_note"),
            "threshold": float(bundle["threshold"]),
            "threshold_rule": bundle.get("threshold_rule"),
            "threshold_evidence": bundle.get("threshold_evidence"),
            "threshold_operating_point": bundle.get("threshold_operating_point"),
            "calibration_status": bundle.get("calibration_decision"),
            "calibration_mode": bundle.get("calibration_mode"),
            "development_evidence": evidence,
            "independent_evaluation_available": bool(
                evidence["independent_evaluation_available"]
            ),
            "training_period": bundle.get("training_period"),
            "training_rows": bundle.get("training_rows"),
            "training_athletes": bundle.get("training_athletes"),
            "high_injury_risk_prevalence": bundle.get("high_injury_risk_prevalence"),
            "limitations": list(bundle["limitations"]),
            "deployment_status": bundle.get("deployment_status"),
            "three_class_model_preserved": bundle.get("three_class_bundle_preserved"),
        }
    )
