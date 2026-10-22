from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.decision_support.what_if import bwf_context_what_if
from src.serving.bwf_state import build_prematch_features
from src.serving.conformal import decide, load_bwf_calibration
from src.serving.data_registry import load_bwf_context_options
from src.serving.errors import predict_probabilities
from src.serving.json_safe import to_json_safe
from src.evaluation.robustness import out_of_distribution_flags


def _uncertainty_status(conformal: dict | None) -> str:
    """One sentence about how settled this forecast is, from the conformal set.

    No probability cut-point appears here. The conformal set size is the
    project's declared measure of ambiguity, and a second rule alongside it
    would eventually contradict it on some fixture.
    """
    if conformal is None:
        return (
            "Conformal calibration is unavailable, so how settled this estimate "
            "is has not been established. Treat it as unverified."
        )
    if conformal.get("set_size", 0) == 0:
        return (
            "Neither side met the calibrated coverage threshold — the model has "
            "no supported call for this fixture."
        )
    if conformal.get("confident"):
        return (
            "One side met the calibrated coverage threshold. This is an "
            "estimate from historical patterns, not a guaranteed result."
        )
    return (
        "Both sides remain plausible at the calibrated coverage target — the "
        "model declines to call this fixture."
    )


def _conformal_verdict(probabilities: dict[str, float]) -> dict | None:
    """A calibrated call or refusal, when a calibration artefact is available."""
    calibration = load_bwf_calibration()
    if calibration is None:
        return None
    return decide(probabilities, calibration)


def _host_country_options() -> list[str]:
    """The host countries the encoder was fitted on, or nothing if unavailable.

    Read defensively: the applicability flag below is an addition to a
    prediction, and a missing option file must not be able to stop the
    prediction being served.
    """
    try:
        return list(load_bwf_context_options().get("host_countries") or [])
    except (OSError, ValueError, KeyError):
        return []


def predict_bwf_feature_row(
    bundle: dict,
    feature_row: pd.Series | dict[str, Any],
) -> dict[str, Any]:
    feature_columns = list(bundle["feature_columns"])
    if isinstance(feature_row, pd.Series):
        frame = feature_row.to_frame().T
    else:
        frame = pd.DataFrame([feature_row])
    missing = [feature for feature in feature_columns if feature not in frame]
    if missing:
        raise ValueError(f"BWF feature row is missing: {missing}")

    frame = frame[feature_columns].copy()
    model = bundle.get("serving_model", bundle.get("calibrated_model"))
    probabilities = predict_probabilities(model, frame, "bwf")[0]
    class_names = list(bundle["class_names"])
    predicted_index = int(np.argmax(probabilities))

    # Host country is deliberately *not* refused when it is unseen, unlike the
    # discipline, tier and round, which are closed vocabularies where an
    # unrecognised value can only be a mistake. The tour visits new countries
    # every season - the exported calendar already carries New Zealand,
    # Botswana, Peru, the Philippines and one venue still listed as "TBC" - so
    # refusing them would deny a prediction for real fixtures the model can
    # still largely answer, since host country is one contextual feature beside
    # the ratings, form and head-to-head that carry the estimate.
    #
    # What must not happen is the silent version: the encoder emits an all-zero
    # block for an unseen country and the caller receives a confident-looking
    # probability with nothing to say the context was outside training. So the
    # boundary is surfaced instead of hidden - as an out-of-distribution flag
    # and as an explicit `host_country_known: false` - which is the same stance
    # the calendar takes when it offers the tournament in the first place.
    row_values = frame.iloc[0].to_dict()
    host_countries = _host_country_options()
    ood_flags = out_of_distribution_flags(
        row_values,
        bundle.get("reference_quantiles", {}),
        categorical_options={"host_country": host_countries} if host_countries else None,
    )
    host_country_known = (
        str(row_values.get("host_country")) in {str(value) for value in host_countries}
        if host_countries
        else None
    )

    class_probabilities = {
        class_names[index]: float(probability)
        for index, probability in enumerate(probabilities)
    }
    conformal = _conformal_verdict(class_probabilities)

    result = {
        "stream": "bwf",
        "prediction": class_names[predicted_index],
        "predicted_class_index": predicted_index,
        "model_version": bundle.get("model_version", "1.0-baseline"),
        "out_of_distribution_flags": ood_flags,
        # Mirrors the flag the calendar attaches to a tournament, so a caller
        # that came in through the fixture picker sees the same statement about
        # the same event on the prediction itself. ``None`` only when the option
        # artefact could not be read, which is "not established" rather than
        # "known".
        "host_country_known": host_country_known,
        "applicability_note": (
            None
            if host_country_known is not False
            else (
                f"{row_values.get('host_country')!r} is not among the host "
                "countries this model was fitted on, so that context feature "
                "contributes nothing to the estimate. The prediction still "
                "rests on the ratings, recent form and head-to-head record, but "
                "it is an extrapolation on venue and should be read as one."
            )
        ),
        "probabilities": class_probabilities,
        "team1_win_probability": float(probabilities[1]),
        # Derived from the conformal decision rather than a chosen probability
        # cut-point. The previous 0.55 rule is the arbitrary threshold conformal
        # prediction replaced, and it could report "separated" on the same
        # request where the conformal set declined to call a winner.
        "uncertainty_status": _uncertainty_status(conformal),
        # A calibrated verdict alongside the raw probability: either a single
        # winner at the stated coverage target, or an explicit refusal to call.
        # Absent when the calibration artefact has not been generated.
        "conformal": conformal,
        "cold_start": bool(
            int(frame.iloc[0]["team1_cold_start"]) == 1
            or int(frame.iloc[0]["team2_cold_start"]) == 1
        ),
        "human_review_required": True,
        "disclaimer": (
            "This is pre-match decision support based on historical patterns. "
            "It does not guarantee a winner and must not be used as an "
            "autonomous or gambling decision."
        ),
    }
    return to_json_safe(result)


def predict_future_bwf_match(
    bundle: dict,
    state: dict,
    *,
    team1_player_ids: Iterable[int],
    team2_player_ids: Iterable[int],
    discipline: str,
    prediction_date: str | pd.Timestamp,
    tier_category: str,
    round_name: str,
    is_qualification: bool,
    host_country: str,
) -> dict[str, Any]:
    features = build_prematch_features(
        state,
        team1_player_ids=team1_player_ids,
        team2_player_ids=team2_player_ids,
        discipline=discipline,
        prediction_date=prediction_date,
        tier_category=tier_category,
        round_name=round_name,
        is_qualification=is_qualification,
        host_country=host_country,
        enforce_after_snapshot=True,
    )
    prediction = predict_bwf_feature_row(bundle, features)
    prediction["features"] = to_json_safe(features)
    prediction["state_snapshot_as_of"] = state["snapshot_as_of"]
    return to_json_safe(prediction)


def run_bwf_context_what_if(
    bundle: dict,
    feature_row: pd.Series | dict[str, Any],
    changes: dict[str, Any],
) -> dict[str, Any]:
    return to_json_safe(
        bwf_context_what_if(bundle, feature_row, changes)
    )
