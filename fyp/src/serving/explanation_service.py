from __future__ import annotations

from typing import Any

import pandas as pd

from src.explainability.shap_tools import (
    local_contributions,
    pipeline_level_shap,
)
from src.serving.json_safe import to_json_safe


BACKGROUND_SEED = 42


def explain_row(
    bundle: dict,
    background: pd.DataFrame,
    row: pd.DataFrame,
    target_class_index: int,
    maximum_background_rows: int = 80,
) -> dict[str, Any]:
    feature_columns = list(bundle["feature_columns"])
    # SHAP's base value is the model's average output over the background, so
    # the background has to look like the population. Both callers hand this
    # function a chronologically ordered export, where ``head(n)`` is the
    # oldest n matches or the oldest n sessions - a reference drawn from one
    # end of the data, against which every later row looks like a departure. A
    # seeded sample spans the whole export and still returns the same rows on
    # every call.
    background = background[feature_columns]
    if len(background) > maximum_background_rows:
        background = background.sample(
            n=maximum_background_rows, random_state=BACKGROUND_SEED
        )
    row = row[feature_columns].head(1)

    explanation, _ = pipeline_level_shap(
        model_bundle=bundle,
        background=background,
        explain_rows=row,
        target_class_index=target_class_index,
        maximum_rows=1,
        random_state=42,
    )
    local = local_contributions(explanation, row_index=0)
    return to_json_safe({
        "target_class": bundle["class_names"][target_class_index],
        "base_value": float(explanation.base_values[0]),
        "contributions": local.to_dict(orient="records"),
        "explained_model": _explained_model_note(bundle),
        "disclaimer": (
            "SHAP describes model behaviour, not causation. "
            + _explained_model_note(bundle)
        ),
    })


def build_attribution(
    bundle: dict,
    base_value: float,
    predicted_probability: float,
    contributions: pd.DataFrame,
    named_factors: list[dict[str, Any]],
    target_class: str,
) -> dict[str, Any]:
    """Reconcile the named factors against the probability actually displayed.

    The prediction function handed to the explainer is ``predict_proba``, so
    these SHAP values live in probability space and permutation SHAP is exact
    for the model it wraps. That makes one specific claim available that a
    ranking alone cannot make: *this measurement moved the score by this many
    percentage points*, and the parts sum to the whole.

    The claim is only safe while the explained estimator is the serving one.
    Both injury bundles currently store a single estimator under
    ``serving_model`` and ``base_model_for_explanation`` — the three-class
    bundle because its calibration audit rejected calibration, the binary
    bundle because it was frozen that way — so the sum reconciles to the number
    on screen. If a future bundle reintroduces a calibration wrapper, the two
    diverge; ``exact`` goes false, the interface stops printing percentage
    points and falls back to the ranking, and nothing here has to be remembered
    by whoever swaps the bundle.

    Only ``named_factors`` are itemised, because the SAS simplification
    feedback caps the interface at three. Everything the interface does not
    name is folded into ``remainder`` rather than dropped, so a reader can see
    that the itemised factors are a selection and how much was left out.
    """
    # The middle of the development data for each feature, so the interface can
    # say what an ordinary value looks like beside the one this session had.
    # A bare "Stress level 0.8" cannot be read: the reader has no way to know
    # whether the scale runs to 1, to 10 or to 100, and the provider's variable
    # dictionary documents no units for these columns. The quantiles are the
    # model's own reference data, so this states an observed range rather than
    # asserting a unit nobody wrote down.
    quantiles = bundle.get("reference_quantiles") or {}

    def typical_range(feature: str) -> dict[str, float] | None:
        bounds = quantiles.get(feature)
        if not isinstance(bounds, dict):
            return None
        low, high = bounds.get("q25"), bounds.get("q75")
        if low is None or high is None:
            return None
        return {"low": float(low), "high": float(high)}

    named_names = [str(factor["feature"]) for factor in named_factors]
    named_total = sum(float(factor["shap_value"]) for factor in named_factors)

    # The whole movement from the reference score to this session's score.
    # Derived from the two probabilities rather than from the contribution
    # column, so the remainder below absorbs any gap instead of hiding it.
    total_shift = float(predicted_probability) - float(base_value)
    remainder = total_shift - named_total

    rest = contributions[~contributions["feature"].isin(named_names)]
    reconstructed = float(base_value) + float(contributions["shap_value"].sum())
    gap = abs(reconstructed - float(predicted_probability))

    return {
        "target_class": target_class,
        # The model's mean probability for this class across the background
        # sample: what an unremarkable session in the reference data scores.
        "base_probability": float(base_value),
        "predicted_probability": float(predicted_probability),
        "total_shift": total_shift,
        "factors": [
            {
                "feature": str(factor["feature"]),
                "feature_value": factor["feature_value"],
                "contribution": float(factor["shap_value"]),
                "direction": factor["direction"],
                "typical_range": typical_range(str(factor["feature"])),
            }
            for factor in named_factors
        ],
        "remainder": {
            "feature_count": int(len(rest)),
            "contribution": remainder,
            # The unnamed factors themselves, so the interface can offer them
            # on demand instead of only asserting how many there were. The
            # three-name cap is a rule about what is shown by default, not
            # about what a reader checking the work is allowed to see.
            "factors": [
                {
                    "feature": str(row["feature"]),
                    "feature_value": row["feature_value"],
                    "contribution": float(row["shap_value"]),
                    "direction": row["direction"],
                    "typical_range": typical_range(str(row["feature"])),
                }
                for row in rest.sort_values(
                    "absolute_shap", ascending=False
                ).to_dict(orient="records")
            ],
        },
        # Permutation SHAP is exact for the model it wraps, so a visible gap
        # means the explanation is not describing the served model. Surfaced
        # rather than hidden, and the interface reads this before it prints a
        # single percentage point.
        "exact": bool(gap < 0.005),
        "additivity_gap": gap,
        "explained_model": _explained_model_note(bundle),
    }


def _explained_model_note(bundle: dict) -> str:
    """State plainly which model the contributions come from.

    The explanation runs on ``base_model_for_explanation``; the probability on
    screen comes from ``serving_model``. When the bundle's calibration decision
    left those as the same estimator the two agree exactly, and when it wrapped
    the estimator in a calibrator they no longer do - the wrapper is a monotone
    transform of the same scores, so the *ranking and direction* of the factors
    still describe the served probability, but the magnitudes and the base value
    are the uncalibrated model's. Claiming outright that SHAP explains "the same
    model that supplies the displayed probability" was true only in the first
    case, and the disclaimer is the last place to be approximately right.
    """
    serving = bundle.get("serving_model", bundle.get("calibrated_model"))
    base = bundle.get("base_model_for_explanation")
    if serving is base:
        return (
            "The contributions are computed on the same saved estimator that "
            "supplies the displayed probability."
        )
    return (
        "The contributions are computed on the bundle's uncalibrated base "
        "estimator, while the displayed probability comes from the calibrated "
        "wrapper around it. The direction and ranking of the factors carry "
        "over; the base value and the individual magnitudes are the "
        "uncalibrated model's and will not sum to the displayed probability."
    )
