from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.decision_support.what_if import injury_what_if
from src.serving.categorical_vocabulary import trained_categorical_vocabulary
from src.serving.conformal import decide, load_injury_calibration
from src.serving.errors import predict_probabilities
from src.serving.input_validation import build_feature_frame
from src.serving.json_safe import to_json_safe
from src.evaluation.robustness import out_of_distribution_flags


def _conformal_verdict(probabilities: dict[str, float]) -> dict | None:
    """The set of classes plausible at the calibrated coverage target.

    Healthy and Low Risk are not reliably separable by this model, so on many
    sessions the honest answer is both of them rather than a coin-flip between
    them. Absent when the calibration artefact has not been generated.

    The injury artefact deploys **class-conditional (Mondrian)** calibration, so
    each class is admitted at its own threshold and the coverage target holds
    within each class rather than only on average across them. That matters
    here because a pooled quantile was being met by the majority Healthy class
    while Low Risk ran under target - see docs/INJURY_CONFORMAL_PREDICTION.md.
    Which method is loaded is decided by the artefact's ``deployed_method`` key,
    not here; ``decide`` handles either, and the per-class thresholds it applied
    are reported back in the response under ``class_thresholds``.
    """
    calibration = load_injury_calibration()
    if calibration is None:
        return None
    return decide(probabilities, calibration)


def _uncertainty_status(conformal: dict | None) -> str:
    """One sentence about how settled this result is, taken from the conformal set.

    There is deliberately no probability cut-point here. The size of the
    conformal prediction set is the project's declared measure of ambiguity,
    and any second rule would eventually disagree with it on some request.
    """
    if conformal is None:
        return (
            "Conformal calibration is unavailable, so how settled this result "
            "is has not been established. Treat it as unverified."
        )
    if conformal.get("set_size", 0) == 0:
        return (
            "No class met the calibrated coverage threshold — the model has no "
            "supported answer for these inputs."
        )
    if conformal.get("confident"):
        return (
            "One class met the calibrated coverage threshold. This is a "
            "risk estimate, not clinical certainty or a guarantee."
        )
    retained = ", ".join(conformal.get("prediction_set") or [])
    return (
        f"{conformal.get('set_size')} classes remain plausible at the "
        f"calibrated coverage target ({retained}) — human review required."
    )


def build_injury_frame(bundle: dict, payload: dict[str, Any]) -> pd.DataFrame:
    return build_feature_frame(
        payload=payload,
        feature_columns=list(bundle["feature_columns"]),
        numeric_features=list(bundle["numeric_features"]),
        categorical_features=list(bundle["categorical_features"]),
    )


def predict_injury(bundle: dict, payload: dict[str, Any]) -> dict[str, Any]:
    frame = build_injury_frame(bundle, payload)
    model = bundle.get("serving_model", bundle.get("calibrated_model"))
    probabilities = predict_probabilities(model, frame, "injury")[0]
    class_names = list(bundle["class_names"])
    predicted_index = int(np.argmax(probabilities))
    # Both halves of the boundary, not just the numeric one. The numeric
    # percentile check has always been here; the categorical one had not, so a
    # badminton session - the one case this project exists to screen, and the
    # one sport the multi-sport training data does not contain - came back
    # carrying no flag at all. The governance policy said so on the screening
    # response, but the prediction itself did not, and the interface builds its
    # caveat strip from the prediction.
    ood_flags = out_of_distribution_flags(
        payload,
        bundle.get("reference_quantiles", {}),
        categorical_options=trained_categorical_vocabulary(bundle),
    )

    class_probabilities = {
        class_names[index]: float(probability)
        for index, probability in enumerate(probabilities)
    }
    conformal = _conformal_verdict(class_probabilities)

    result = {
        "stream": "injury",
        "prediction": class_names[predicted_index],
        "predicted_class_index": predicted_index,
        "model_version": bundle.get("model_version", "1.0-baseline"),
        "out_of_distribution_flags": ood_flags,
        "probabilities": class_probabilities,
        # Derived from the conformal decision, never from a chosen probability
        # cut-point. An earlier version reported separation whenever the leading
        # probability cleared 0.55, which is exactly the arbitrary threshold
        # conformal prediction was introduced to replace — and it could report
        # "separated" on the same request where the conformal set said the
        # outcome was too close to call.
        "uncertainty_status": _uncertainty_status(conformal),
        "conformal": conformal,
        "human_review_required": True,
        "limitations": [
            "The model is a multi-sport methodological proxy.",
            # The deployed champion recovers roughly 16.7% Low Risk recall. The
            # superseded model recovered 0%, and stating that here described a
            # model no longer being served.
            "Low Risk is recovered weakly, so a result other than an elevated "
            "signal is never a clearance.",
            "The output is not a diagnosis or treatment recommendation.",
        ],
        "disclaimer": (
            "Experimental screening support only. This output does not diagnose "
            "an injury and does not replace a coach, sports scientist, or "
            "healthcare professional."
        ),
    }
    return to_json_safe(result)


def run_injury_what_if(
    bundle: dict,
    payload: dict[str, Any],
    changes: dict[str, Any],
) -> dict[str, Any]:
    frame = build_injury_frame(bundle, payload)
    return to_json_safe(
        injury_what_if(bundle, frame.iloc[0], changes)
    )
