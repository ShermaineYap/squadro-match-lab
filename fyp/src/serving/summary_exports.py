"""Plain-text research summaries, for export rather than for the interface."""

from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Any
import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=4)
def _measured_performance(policy_file: str) -> dict[str, Any]:
    """The measured figures a policy document records, or nothing.

    Read rather than restated. A summary that quotes a recall the policy file no
    longer records is worse than one that omits it, because the reader has no
    way to tell which of the two numbers is current.
    """
    path = PROJECT_ROOT / "rules" / policy_file
    try:
        with path.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    measured = document.get("measured_performance")
    return measured if isinstance(measured, dict) else {}


def _timestamp(value: str | None = None) -> str:
    if value:
        return value
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def build_injury_text_summary(
    *,
    result: Mapping[str, Any],
    inputs: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
    display_label: str | None = None,
    next_review_step: str | None = None,
    generated_at: str | None = None,
) -> str:
    probabilities = result.get("probabilities", {})
    metadata = metadata or {}
    original_class = str(result.get("prediction", "Not available"))
    shown_label = display_label or original_class
    lines = [
        "CourtSense AI — Athlete Screening Research Summary",
        "=" * 58,
        f"Generated: {_timestamp(generated_at)}",
        "",
        "ASSESSMENT CONTEXT",
        f"Mode: {metadata.get('assessment_mode', 'Not recorded')}",
        f"Session reference: {metadata.get('session_reference', 'Not supplied')}",
        f"Measurement date: {metadata.get('assessment_date', 'Not recorded')}",
        f"Measurement source: {metadata.get('measurement_source', 'Not recorded')}",
        f"Input completeness: {metadata.get('input_completeness', f'{len(inputs)} values supplied')}",
        "",
        "PLAIN-LANGUAGE RESULT",
        f"Predicted injury-risk class: {shown_label}",
        f"Original frozen-model class: {original_class}",
        "This model output is not an injury-free percentage, diagnosis, medical clearance or future-risk forecast.",
        "",
        "CLASS PROBABILITIES",
    ]
    for label, value in probabilities.items():
        lines.append(f"- {label}: {float(value):.1%}")

    if next_review_step:
        lines.extend(
            [
                "",
                "HUMAN-REVIEW NEXT STEP",
                next_review_step,
            ]
        )

    measured = _measured_performance("injury_policies.json")
    low_risk_recall = measured.get("low_risk_recall")
    lines.extend(
        [
            "",
            "HOW SETTLED THIS RESULT IS",
            # The conformal statement the prediction already carries. It is the
            # project's declared basis for ambiguity, so it is quoted rather
            # than re-derived from the probabilities here.
            str(
                result.get("uncertainty_status")
                or "Not recorded with this result."
            ),
            "",
            "IMPORTANT LIMITATIONS",
            "- Experimental multi-sport screening only; the source data are not badminton-specific.",
            "- The model classifies the entered session pattern; it does not estimate injury within the next session, 7 days or 28 days.",
            "- This output is not a diagnosis, medical clearance or instruction to change training.",
        ]
    )
    if low_risk_recall is not None:
        lines.append(
            f"- Low Risk recall is {float(low_risk_recall):.3f} on the active "
            f"{measured.get('model_version', 'unknown')} corrected audit "
            "(rules/injury_policies.json), so the middle class remains weakly "
            "detected."
        )
    lines.extend(
        [
            "- Human review is required for any real-world decision.",
            "",
            "ENTERED VALUES",
        ]
    )
    for feature, value in inputs.items():
        lines.append(f"- {feature.replace('_', ' ').title()}: {_format_value(value)}")

    limitations = result.get("limitations") or []
    if limitations:
        lines.extend(["", "MODEL-SUPPLIED LIMITATIONS"])
        for item in limitations:
            lines.append(f"- {item}")

    lines.extend(
        [
            "",
            "INTERPRETATION BOUNDARY",
            "SHAP explains model behaviour, not causation. What-if comparisons show model sensitivity, not intervention evidence.",
            "",
            "For research and decision-support demonstration only.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_bwf_text_summary(
    *,
    result: Mapping[str, Any],
    side_1: str,
    side_2: str,
    recorded_outcome: str | None = None,
    advantage_label: str | None = None,
    generated_at: str | None = None,
) -> str:
    side_1_probability = float(result.get("team1_win_probability", 0.0))
    side_2_probability = 1.0 - side_1_probability
    favoured_probability = max(side_1_probability, side_2_probability)
    # A description of how large the number is, for a reader who has just been
    # shown "62%". It is deliberately not a confidence claim and does not decide
    # anything: the conformal statement printed below is what governs whether
    # the model considers this fixture callable at all.
    interpretation = advantage_label or (
        "Nearly even matchup"
        if favoured_probability <= 0.55
        else "Slight estimated advantage"
        if favoured_probability <= 0.65
        else "Moderate estimated advantage"
        if favoured_probability <= 0.75
        else "Stronger estimated advantage"
    )
    lines = [
        "CourtSense AI — BWF Match Prediction Research Summary",
        "=" * 61,
        f"Generated: {_timestamp(generated_at)}",
        "",
        "MATCHUP",
        f"Side 1: {side_1}",
        f"Side 2: {side_2}",
        "",
        "MODEL OUTPUT",
        f"Frozen-model class: {result.get('prediction', 'Not available')}",
        f"Side 1 probability: {side_1_probability:.1%}",
        f"Side 2 probability: {side_2_probability:.1%}",
        f"Cold-start warning: {'Yes' if result.get('cold_start', False) else 'No'}",
        "",
        "HOW SETTLED THIS ESTIMATE IS",
        str(result.get("uncertainty_status") or "Not recorded with this result."),
        f"Size of the estimated advantage (descriptive only): {interpretation}",
    ]
    if recorded_outcome:
        lines.append(f"Recorded historical outcome: {recorded_outcome}")

    measured = _measured_performance("bwf_policies.json")
    roc_auc = measured.get("roc_auc")
    lines.extend(
        [
            "",
            "RECOMMENDED USE",
            "Use this estimate as one input to a human matchup review alongside coach and analyst observations.",
            "",
            "IMPORTANT LIMITATIONS",
            "- This is a pre-match probabilistic estimate, not a guaranteed winner.",
            "- The output must not be used for betting or gambling.",
            "- Cold-start players or partnerships have less historical evidence.",
            "- SHAP explains model behaviour, not why a side will win in reality.",
            "- What-if comparisons show model sensitivity, not a real-world guarantee.",
            "",
        ]
    )
    if roc_auc is not None:
        lines.append(
            f"The corrected-audit BWF ROC-AUC was {float(roc_auc):.4f} "
            "(rules/bwf_policies.json), narrowly below the original 0.80 target."
        )
    lines.append("For research and decision-support demonstration only.")
    return "\n".join(lines) + "\n"
