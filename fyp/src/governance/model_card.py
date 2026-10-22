from __future__ import annotations

from pathlib import Path
from typing import Any
import json


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _metric_lines(metrics: dict | None) -> str:
    if not metrics:
        return "- Final-test metrics are not available because the final test remains locked."
    lines = []
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            lines.append(f"- **{key}:** {value:.4f}")
        else:
            lines.append(f"- **{key}:** {value}")
    return "\n".join(lines)


def generate_model_card(
    stream: str,
    metadata: dict[str, Any],
    output_path: Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if stream == "bwf":
        intended_use = (
            "Estimate pre-match BWF match-outcome probabilities from historical "
            "and tournament-context information."
        )
        excluded_use = (
            "Not for gambling, guaranteed outcome claims, player selection without "
            "human review, or use of post-match score information."
        )
        data_scope = (
            "BWF matches from 2021–2025. 2021 is historical warm-up, 2022–2023 "
            "training, 2024 development/calibration and 2025 final test."
        )
        limitations = [
            "Exact within-day match ordering is unavailable for most records.",
            "Cold-start players and partnerships have limited history.",
            "The model does not observe real-time injuries, tactics or private training data.",
            "SHAP explanations describe model behaviour, not causation.",
        ]
    else:
        intended_use = (
            "Support athlete-session injury-risk screening using a multi-sport "
            "methodological proxy dataset."
        )
        excluded_use = (
            "Not a medical diagnosis, clinical recommendation, badminton-specific "
            "injury model or autonomous training decision."
        )
        data_scope = (
            "15,420 multi-sport athlete-session records. Development and final "
            "test athletes are disjoint."
        )
        limitations = [
            "The dataset is multi-sport and not clinically validated badminton evidence.",
            "The target timing does not establish causal or prospective injury prevention.",
            "High Risk/Injured is the smallest class.",
            "Scenario analysis shows model sensitivity, not real-world causal effects.",
        ]

    final_metrics = metadata.get("final_test_metrics")
    limitations_markdown = "\n".join(f"- {item}" for item in limitations)
    bundle_value = str(metadata.get("bundle_path", ""))
    bundle_path = Path(bundle_value) if bundle_value else Path()
    if bundle_value and bundle_path.is_absolute():
        try:
            bundle_value = str(bundle_path.relative_to(PROJECT_ROOT))
        except ValueError:
            bundle_value = bundle_path.name

    # This generator produces the version 1.0 baseline cards only. The active
    # serving models are the refits, whose cards are maintained separately as
    # outputs/model_cards/{stream}_model_card_v2.md. The banner is emitted here
    # rather than added by hand so that it survives any retraining run.
    content = f"""> # ⚠️ SUPERSEDED BASELINE — NOT THE ACTIVE DASHBOARD MODEL
>
> This is the **version 1.0 baseline** card, retained for the audit trail.
> Its metrics are the superseded calibrated figures and **must not be quoted as the
> project's results**. The deployed model is the uncalibrated refit.
>
> **Quote `outputs/model_cards/{stream}_model_card_v2.md` instead.**

# Model Card — {stream.upper()} Stream (superseded baseline)

## Model identity

- **Champion:** {metadata.get("champion_name", "Not selected")}
- **Model version:** 1.0
- **Final test evaluated:** {metadata.get("final_test_evaluated", False)}
- **Bundle:** `{bundle_value}`

## Intended use

{intended_use}

## Excluded uses

{excluded_use}

## Data and validation

{data_scope}

- Champion selection was completed without using the final-test results.
- Preprocessing was fitted inside chronological or athlete-grouped validation folds.
- Displayed probabilities are calibrated.
- The final model remains decision support and requires human review.

## Selection evidence

{metadata.get("selection_rule", "See the model-development outputs.")}

## Final-test metrics

{_metric_lines(final_metrics)}

## Known limitations

{limitations_markdown}

## Explainability

SHAP is used to describe which model inputs contributed to a prediction. SHAP
does not establish that changing a feature will cause the real-world outcome to
change. Recommendations are produced by a separate transparent rule layer.

## What-if analysis

What-if outputs are constrained model scenarios. They are not causal evidence,
medical advice or guaranteed performance improvements.

## Human oversight

- Users must review uncertainty and data-quality flags.
- Injury outputs require appropriate professional judgement.
- Match outputs must not be treated as guaranteed outcomes.
- Overrides and user feedback should be logged in a future implementation.

## Monitoring and future implementation

A production implementation should monitor:

- input data quality;
- missingness and cold-start rates;
- probability calibration;
- subgroup performance;
- concept and model drift;
- user overrides and feedback;
- model version and retraining date.

## Governance note

This prototype can later be aligned with SAS Model Manager, SAS Intelligent
Decisioning and SAS Visual Analytics for model registration, rule governance,
monitoring and non-technical reporting.
"""
    output_path.write_text(content, encoding="utf-8")
    return output_path
