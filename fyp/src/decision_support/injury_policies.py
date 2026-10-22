"""Injury policy engine."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from statistics import median
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY_FILE = PROJECT_ROOT / "rules" / "injury_policies.json"


class Level(str, Enum):
    """Bounded vocabulary. Nothing here asserts injury, health or clearance."""

    WITHIN_EXPECTED = "Within expected range"
    REVIEW_SUGGESTED = "Review suggested"
    HIGHER_PRIORITY = "Higher-priority review"
    NOT_EVALUATED = "Not evaluated"

    @property
    def rank(self) -> int:
        return {
            Level.NOT_EVALUATED: -1,
            Level.WITHIN_EXPECTED: 0,
            Level.REVIEW_SUGGESTED: 1,
            Level.HIGHER_PRIORITY: 2,
        }[self]


@dataclass
class PolicyResult:
    policy_id: str
    variable: str
    level: str
    observed_value: float | None
    thresholds: dict[str, float]
    screening_result: str | None
    user_interpretation: str | None
    limitations: str
    evidence_sources: list[dict[str, Any]]
    evidence_strength: str
    enabled: bool
    not_evaluated_reason: str | None = None


@dataclass
class InjuryScreeningOutcome:
    overall_level: str
    results: list[PolicyResult] = field(default_factory=list)
    measurement_boundary: str = ""
    calibration_note: str = ""
    measured_performance: dict[str, Any] = field(default_factory=dict)
    disclaimer: str = (
        "Decision support for review purposes. It estimates an injury-risk class "
        "from session data and compares that data against declared policies. It "
        "does not diagnose injury, predict injury, or clear an athlete to train "
        "or compete."
    )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["results"] = [asdict(result) for result in self.results]
        return payload


def load_policies(path: Path | None = None) -> dict[str, Any]:
    with Path(path or DEFAULT_POLICY_FILE).open("r", encoding="utf-8") as handle:
        return json.load(handle)


BOUNDARY_TOLERANCE = 1e-9


def _grade(value: float, thresholds: dict[str, float], direction: str) -> Level:
    """Grade against the policy's two comparison values.

    Boundaries are inclusive, so a value exactly at a comparison value raises
    the level. Flagging a borderline session for a human is the conservative
    direction, and the tests pin it explicitly.
    """
    higher = thresholds.get("higher_priority_review")
    review = thresholds.get("review_suggested")

    if direction == "decrease":
        # Smaller is worse: class separation shrinking towards a coin toss.
        if higher is not None and value <= higher + BOUNDARY_TOLERANCE:
            return Level.HIGHER_PRIORITY
        if review is not None and value <= review + BOUNDARY_TOLERANCE:
            return Level.REVIEW_SUGGESTED
        return Level.WITHIN_EXPECTED

    if higher is not None and value >= higher - BOUNDARY_TOLERANCE:
        return Level.HIGHER_PRIORITY
    if review is not None and value >= review - BOUNDARY_TOLERANCE:
        return Level.REVIEW_SUGGESTED
    return Level.WITHIN_EXPECTED


def baseline_deviation(
    current: float, history: Sequence[float], direction: str
) -> float | None:
    """How far this session sits from the athlete's own median, proportionally.

    Returned as a proportion rather than a raw difference because the underlying
    scores are derived scales with no defined unit: "40% above this athlete's
    norm" is meaningful where "23 points above" is not.

    Only movement in the policy's declared direction counts. A recovery score
    above an athlete's norm is not evidence of poor recovery.
    """
    usable = [value for value in history if value is not None]
    if not usable:
        return None

    reference = median(usable)
    if reference == 0:
        return None

    change = (current - reference) / abs(reference)
    if direction == "decrease":
        return max(0.0, -change)
    if direction == "increase":
        return max(0.0, change)
    return abs(change)


def evaluate_session(
    prediction: str,
    probabilities: dict[str, float],
    features: dict[str, Any],
    athlete_history: dict[str, Sequence[float]] | None = None,
    policy_file: Path | None = None,
) -> InjuryScreeningOutcome:
    """Evaluate one session against every declared injury policy.

    ``athlete_history`` holds this athlete's previous values per feature. Without
    it the within-athlete policies report *not evaluated* and say what is
    missing, rather than falling back to an absolute cut-point the data cannot
    support.
    """
    document = load_policies(policy_file)
    history = athlete_history or {}

    outcome = InjuryScreeningOutcome(
        overall_level=Level.NOT_EVALUATED.value,
        measurement_boundary=document.get("measurement_boundary", ""),
        calibration_note=document.get("calibration_note", ""),
        measured_performance=document.get("measured_performance", {}),
    )

    ordered = sorted(probabilities.values(), reverse=True)
    separation = (ordered[0] - ordered[1]) if len(ordered) > 1 else None

    for policy in document["policies"]:
        thresholds = policy.get("thresholds", {})
        common = {
            "policy_id": policy["policy_id"],
            "variable": policy["variable"],
            "thresholds": thresholds,
            "limitations": policy.get("limitations", ""),
            "evidence_sources": policy.get("evidence_sources", []),
            "evidence_strength": policy.get("evidence_strength", ""),
            "enabled": bool(policy.get("enabled", False)),
        }

        if not policy.get("enabled", False):
            outcome.results.append(
                PolicyResult(
                    level=Level.NOT_EVALUATED.value,
                    observed_value=None,
                    screening_result=None,
                    user_interpretation=None,
                    not_evaluated_reason=policy.get(
                        "disabled_reason", "Policy is disabled in the policy file."
                    ),
                    **common,
                )
            )
            continue

        comparator = policy["comparator"]
        level: Level
        observed: float | None = None
        reason: str | None = None

        if comparator == "model_output":
            # Governance policies describe the model, not the athlete. They are
            # reported whatever the inputs, because the moment they are easiest
            # to omit - a reassuring-looking output - is when they matter most.
            excluded = policy.get("applies_when_prediction_is_not")
            if excluded and prediction == excluded:
                level = Level.WITHIN_EXPECTED
                reason = None
            else:
                level = Level(policy.get("level_when_triggered", Level.REVIEW_SUGGESTED.value))

        elif comparator == "separation":
            if separation is None:
                level = Level.NOT_EVALUATED
                reason = (
                    "Only one class probability was supplied, so the gap to the "
                    "next class could not be measured."
                )
            else:
                observed = separation
                level = _grade(separation, thresholds, policy.get("direction", "decrease"))

        elif comparator == "distribution":
            value = features.get(policy["feature"])
            if value is None:
                level = Level.NOT_EVALUATED
                reason = f"'{policy['feature']}' was not supplied for this session."
            elif str(value) in policy.get("known_values", []):
                level = Level.WITHIN_EXPECTED
            else:
                level = Level(policy.get("level_when_triggered", Level.REVIEW_SUGGESTED.value))

        elif comparator == "within_athlete":
            feature = policy["feature"]
            current = features.get(feature)
            previous = list(history.get(feature, []))
            minimum = policy.get("minimum_sessions", 5)

            if current is None:
                level = Level.NOT_EVALUATED
                reason = f"'{feature}' was not supplied for this session."
            elif len(previous) < minimum:
                level = Level.NOT_EVALUATED
                reason = policy.get(
                    "not_measurable_reason",
                    f"Needs at least {minimum} previous sessions from this athlete.",
                )
            else:
                observed = baseline_deviation(
                    float(current), previous, policy.get("direction", "increase")
                )
                if observed is None:
                    level = Level.NOT_EVALUATED
                    reason = (
                        f"A baseline for '{feature}' could not be formed from this "
                        "athlete's previous sessions."
                    )
                else:
                    level = _grade(observed, thresholds, "increase")

        else:
            level = Level.NOT_EVALUATED
            reason = f"Unknown comparator '{comparator}'."

        flagged = level not in (Level.WITHIN_EXPECTED, Level.NOT_EVALUATED)
        outcome.results.append(
            PolicyResult(
                level=level.value,
                observed_value=round(observed, 4) if observed is not None else None,
                screening_result=policy.get("screening_result") if flagged else None,
                user_interpretation=(
                    policy.get("user_interpretation")
                    if flagged
                    else None
                    if level is Level.NOT_EVALUATED
                    else "This comparison sat within the configured screening range."
                ),
                not_evaluated_reason=reason,
                **common,
            )
        )

    graded = [
        Level(result.level)
        for result in outcome.results
        if result.level != Level.NOT_EVALUATED.value
    ]
    outcome.overall_level = (
        max(graded, key=lambda level: level.rank).value
        if graded
        else Level.NOT_EVALUATED.value
    )
    return outcome
