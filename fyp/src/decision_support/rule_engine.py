from __future__ import annotations

from pathlib import Path
import json

import pandas as pd


MINIMUM_MATERIAL_CONTRIBUTION = 0.02


def load_rules(path: Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload["rules"] if isinstance(payload, dict) else payload


def generate_review_suggestions(
    local_contributions: pd.DataFrame,
    rules: list[dict],
    target_direction: str = "toward target class",
    maximum_suggestions: int = 5,
    minimum_absolute_shap: float = 0.0,
) -> list[dict]:
    rule_map = {rule["feature"]: rule for rule in rules}
    suggestions = []

    for row in local_contributions.itertuples(index=False):
        if row.feature not in rule_map:
            continue
        if row.direction != target_direction:
            continue
        if row.absolute_shap < minimum_absolute_shap:
            continue

        rule = rule_map[row.feature]
        suggestions.append(
            {
                "feature": row.feature,
                "feature_value": row.feature_value,
                "shap_value": row.shap_value,
                "factor_explanation": rule["factor_explanation"],
                "review_category": rule["review_category"],
                "review_suggestion": rule["review_suggestion"],
                "actionability": rule.get("actionability", "review"),
                "evidence_note": rule.get(
                    "evidence_note",
                    "Expert-reviewed rule required before real-world use.",
                ),
            }
        )

    suggestions = sorted(
        suggestions,
        key=lambda item: abs(item["shap_value"]),
        reverse=True,
    )[:maximum_suggestions]

    return suggestions


def build_decision_support_output(
    stream: str,
    prediction_label: str,
    probabilities: dict[str, float],
    local_contributions: pd.DataFrame,
    rules: list[dict],
    uncertainty_threshold: float = 0.55,
    conformal: dict | None = None,
) -> dict:
    """Assemble one offline worked example: prediction, factors, review areas.

    ``conformal`` is the calibrated decision the serving layer produces for the
    same probabilities. When it is supplied it governs the uncertainty
    statement, because the conformal prediction set is the project's declared
    measure of ambiguity and a second rule alongside it would eventually
    contradict it on some row.

    ``uncertainty_threshold`` is the pre-conformal 0.55 cut-point and remains
    only as the fallback for callers that have no calibration artefact to hand.
    Its output is now labelled as the arbitrary cut-point it is, so it cannot be
    read as the project's governing basis for uncertainty.
    """
    suggestions = generate_review_suggestions(
        local_contributions,
        rules,
        maximum_suggestions=5,
    )

    if conformal is not None:
        retained = ", ".join(conformal.get("prediction_set") or []) or "none"
        if conformal.get("set_size", 0) == 0:
            status = (
                "No class met the calibrated coverage threshold — the model has "
                "no supported answer for this row."
            )
        elif conformal.get("confident"):
            status = (
                "One class met the calibrated coverage threshold. This is a "
                "risk estimate, not a certainty."
            )
        else:
            status = (
                f"{conformal.get('set_size')} classes remain plausible at the "
                f"calibrated coverage target ({retained}) — human review required."
            )
        basis = "conformal prediction set"
    else:
        maximum_probability = max(probabilities.values())
        uncertain = maximum_probability < uncertainty_threshold
        status = (
            "Leading probability below the "
            f"{uncertainty_threshold:.2f} presentation cut-point — professional "
            "review recommended"
            if uncertain
            else "Leading probability above the "
            f"{uncertainty_threshold:.2f} presentation cut-point"
        )
        basis = (
            f"probability cut-point at {uncertainty_threshold:.2f}, an offline "
            "presentation heuristic rather than the project's governing "
            "conformal basis; no calibration artefact was supplied"
        )

    return {
        "stream": stream,
        "prediction": prediction_label,
        "probabilities": probabilities,
        "uncertainty_status": status,
        "uncertainty_basis": basis,
        "conformal": conformal,
        "top_factors": suggestions,
        "human_review_required": True,
        "disclaimer": (
            "This is decision support, not an autonomous decision. SHAP "
            "describes model contributions and does not establish causation."
        ),
    }
