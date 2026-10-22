"""Per-request review guidance for the injury stream."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.decision_support.rule_engine import (
    MINIMUM_MATERIAL_CONTRIBUTION,
    generate_review_suggestions,
    load_rules,
)
from src.serving.errors import predict_probabilities
from src.serving.explanation_service import build_attribution, explain_row
from src.serving.injury_service import build_injury_frame
from src.serving.json_safe import to_json_safe

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INJURY_RULES = PROJECT_ROOT / "rules" / "injury_rules.json"

# SHAP cost scales with the background sample. Sixty rows keeps a single
# request responsive while still giving a stable expected value. `explain_row`
# draws them as a seeded random sample rather than the first sixty: this
# development export is ordered, so the leading rows are not a population.
BACKGROUND_ROWS = 60


def explain_injury_session(
    bundle: dict,
    background: pd.DataFrame,
    payload: dict[str, Any],
    maximum_factors: int = 3,
) -> dict[str, Any]:
    """Explain one session and return ranked review areas for the leading class."""
    frame = build_injury_frame(bundle, payload)
    model = bundle.get("serving_model", bundle.get("calibrated_model"))
    probabilities = predict_probabilities(model, frame, "injury")[0]
    target_index = int(probabilities.argmax())
    class_names = list(bundle["class_names"])

    explanation = explain_row(
        bundle=bundle,
        background=background,
        row=frame,
        target_class_index=target_index,
        maximum_background_rows=BACKGROUND_ROWS,
    )

    contributions = pd.DataFrame(explanation["contributions"])
    rules = load_rules(INJURY_RULES) if INJURY_RULES.exists() else []

    # Review suggestions only come from the rule file, so the dashboard can
    # never surface guidance that has not been written down and reviewed.
    # A review area still only comes from a factor arguing *for* the explained
    # class — a measurement arguing the session is fine is not a reason to go
    # and look at it — but it must now have moved the estimate materially. See
    # `MINIMUM_MATERIAL_CONTRIBUTION`.
    suggestions = generate_review_suggestions(
        contributions,
        rules,
        maximum_suggestions=maximum_factors,
        minimum_absolute_shap=MINIMUM_MATERIAL_CONTRIBUTION,
    )

    # The largest movers, in either direction, whether or not a rule exists.
    # These are labelled as factors, never as advice.
    #
    # This used to keep only features pushing *toward* the leading class, which
    # quietly broke whenever the leading class was a low-risk one: the panel
    # would name three factors nudging the score up while the score had in fact
    # fallen, because everything that pushed it down was excluded from the list
    # and swept into the "all other measurements" remainder. The reader was
    # shown the three measurements that least accounted for the result. Ranking
    # on absolute contribution names the factors that actually decided it, and
    # the interface already draws the sign as direction rather than assuming
    # every bar points the same way.
    ranked = (
        contributions.sort_values("absolute_shap", ascending=False)
        .head(maximum_factors)
        .to_dict(orient="records")
    )

    attribution = build_attribution(
        bundle=bundle,
        base_value=float(explanation["base_value"]),
        predicted_probability=float(probabilities[target_index]),
        contributions=contributions,
        named_factors=ranked,
        target_class=class_names[target_index],
    )

    return to_json_safe(
        {
            "stream": "injury",
            "target_class": class_names[target_index],
            "base_value": explanation["base_value"],
            "predicted_probability": float(probabilities[target_index]),
            "top_factors": ranked,
            # The same ranked factors, reconciled against the probability the
            # reader is looking at. See `build_attribution` for why percentage
            # points are a claim this stream is allowed to make.
            "attribution": attribution,
            "review_areas": suggestions,
            "rules_available": bool(rules),
            "evidence_note": (
                "SHAP describes how this model reached its output for these "
                "inputs. It measures statistical association with the model's "
                "response, not causation, and it does not establish that "
                "changing a value would change an athlete's outcome."
            ),
            "disclaimer": (
                "Review areas are model-informed prompts for a qualified "
                "person. They are not a diagnosis, a clearance decision, or a "
                "treatment recommendation."
            ),
        }
    )
