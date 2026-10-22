"""BWF policy engine: how reliable is *this* prediction?"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY_FILE = PROJECT_ROOT / "rules" / "bwf_policies.json"

SEGMENT_FIELDS = {
    "discipline": "discipline",
    "tier_category": "tier_category",
    "is_qualification": "is_qualification",
    "either_team_cold_start": "cold_start",
}

FIXTURE_SEGMENTS = {"discipline", "tier_category"}


class Level(str, Enum):
    """Bounded vocabulary. Nothing here asserts a result."""

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
    segment: str | None = None
    not_evaluated_reason: str | None = None


@dataclass
class MatchScreeningOutcome:
    overall_level: str
    results: list[PolicyResult] = field(default_factory=list)
    measurement_boundary: str = ""
    measured_performance: dict[str, Any] = field(default_factory=dict)
    segments_matched: list[dict[str, Any]] = field(default_factory=list)
    disclaimer: str = (
        "Pre-match decision support based on historical patterns. It estimates a "
        "probability, does not guarantee a winner, and must not be used as an "
        "autonomous or gambling decision."
    )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["results"] = [asdict(result) for result in self.results]
        return payload


def load_policies(path: Path | None = None) -> dict[str, Any]:
    with Path(path or DEFAULT_POLICY_FILE).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_segment_metrics(document: dict[str, Any]) -> list[dict[str, Any]]:
    """Per-subgroup held-out metrics, as written by the evaluation notebook."""
    path = PROJECT_ROOT / document["segment_source"]
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ("accuracy", "roc_auc", "ece", "brier", "rows"):
            if row.get(key) not in (None, ""):
                row[key] = float(row[key])
    return rows


def _matches(row: dict[str, Any], match: dict[str, Any]) -> bool:
    """Does this metrics row describe the segment the match belongs to?"""
    field = SEGMENT_FIELDS.get(row["subgroup_variable"])
    if field is None or field not in match:
        return False

    value = match[field]
    stated = str(row["subgroup_value"])
    # Boolean segments are written as True/False in the metrics file and arrive
    # as booleans or 0/1 from a caller.
    if isinstance(value, bool):
        return stated in (str(value), "1" if value else "0")
    return stated == str(value)


def segments_for(match: dict[str, Any], metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every measured segment this match belongs to, weakest accuracy first."""
    found = [
        {
            "segment": f"{row['subgroup_variable']} = {row['subgroup_value']}",
            "variable": row["subgroup_variable"],
            "value": row["subgroup_value"],
            "matches_evaluated": int(row.get("rows", 0)),
            "accuracy": row.get("accuracy"),
            "roc_auc": row.get("roc_auc"),
            "ece": row.get("ece"),
        }
        for row in metrics
        if _matches(row, match)
    ]
    return sorted(found, key=lambda entry: entry["accuracy"] or 1.0)


def _grade(value: float, thresholds: dict[str, float], direction: str) -> Level:
    """Grade a value, with the boundary inclusive in the cautious direction."""
    higher = thresholds.get("higher_priority_review")
    review = thresholds.get("review_suggested")
    tolerance = 1e-9

    if direction == "decrease":
        if higher is not None and value <= higher + tolerance:
            return Level.HIGHER_PRIORITY
        if review is not None and value <= review + tolerance:
            return Level.REVIEW_SUGGESTED
        return Level.WITHIN_EXPECTED

    if higher is not None and value >= higher - tolerance:
        return Level.HIGHER_PRIORITY
    if review is not None and value >= review - tolerance:
        return Level.REVIEW_SUGGESTED
    return Level.WITHIN_EXPECTED


def evaluate_match(
    match: dict[str, Any],
    team1_win_probability: float,
    cold_start: bool = False,
    policy_file: Path | None = None,
) -> MatchScreeningOutcome:
    """Screen one prediction against the declared performance policies.

    ``match`` carries the fixture's context - discipline, tier, whether it is a
    qualification round - which is what allows the reliability reported here to
    be this match's rather than the model's average.
    """
    document = load_policies(policy_file)
    metrics = load_segment_metrics(document)

    context = dict(match)
    context.setdefault("cold_start", cold_start)
    matched = segments_for(context, metrics)

    outcome = MatchScreeningOutcome(
        overall_level=Level.NOT_EVALUATED.value,
        measurement_boundary=document.get("measurement_boundary", ""),
        measured_performance=document.get("measured_performance", {}),
        segments_matched=matched,
    )

    margin = abs(team1_win_probability - 0.5)

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

        # A policy switched off in the file must not be able to flag, exactly as
        # in the injury engine. Without this branch the enabled flag was written
        # into the result and then ignored by the grading below, so retiring a
        # policy required deleting it - and a reader saw ``enabled: false`` next
        # to a level the disabled policy had just contributed to the overall
        # verdict.
        if not policy.get("enabled", False):
            outcome.results.append(
                PolicyResult(
                    level=Level.NOT_EVALUATED.value,
                    observed_value=None,
                    screening_result=None,
                    user_interpretation=None,
                    segment=None,
                    not_evaluated_reason=policy.get(
                        "disabled_reason", "Policy is disabled in the policy file."
                    ),
                    **common,
                )
            )
            continue

        comparator = policy["comparator"]
        level = Level.NOT_EVALUATED
        observed: float | None = None
        segment: str | None = None
        reason: str | None = None

        if comparator == "model_output":
            level = Level(policy.get("level_when_triggered", Level.REVIEW_SUGGESTED.value))

        elif comparator == "flag":
            if context.get(policy["feature"]):
                level = Level(policy.get("level_when_triggered", Level.REVIEW_SUGGESTED.value))
            else:
                level = Level.WITHIN_EXPECTED

        elif comparator == "separation":
            observed = margin
            level = _grade(margin, thresholds, policy.get("direction", "decrease"))

        elif comparator == "segment_lookup":
            metric = policy.get("metric", "accuracy")
            usable = [entry for entry in matched if entry.get(metric) is not None]
            identified = any(entry["variable"] in FIXTURE_SEGMENTS for entry in usable)
            if not usable or not identified:
                level = Level.NOT_EVALUATED
                reason = (
                    "No held-out subgroup metrics matched this fixture's context, "
                    "so the model's reliability for matches of this kind is "
                    "unknown. Check that discipline and tier were supplied and "
                    f"that {document['segment_source']} exists."
                )
            else:
                # The weakest matching segment governs: a prediction is no more
                # reliable than its worst dimension, and reporting the best
                # would flatter exactly the fixtures that need care.
                direction = policy.get("direction", "decrease")
                worst = (
                    min(usable, key=lambda e: e[metric])
                    if direction == "decrease"
                    else max(usable, key=lambda e: e[metric])
                )
                observed = worst[metric]
                segment = worst["segment"]
                level = _grade(observed, thresholds, direction)

        flagged = level not in (Level.WITHIN_EXPECTED, Level.NOT_EVALUATED)
        interpretation = policy.get("user_interpretation") if flagged else None
        if flagged and segment and comparator == "segment_lookup":
            interpretation = f"{interpretation} Weakest matching segment: {segment}."

        outcome.results.append(
            PolicyResult(
                level=level.value,
                observed_value=round(observed, 4) if observed is not None else None,
                screening_result=policy.get("screening_result") if flagged else None,
                user_interpretation=interpretation,
                segment=segment,
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


def describe_reliability(outcome: MatchScreeningOutcome) -> str:
    """One sentence naming this match's measured reliability, not the average."""
    if not outcome.segments_matched:
        return (
            "No measured reliability is available for this fixture's context, so "
            "only the model's overall figures apply."
        )

    weakest = outcome.segments_matched[0]
    overall = outcome.measured_performance.get("accuracy")
    comparison = ""
    if overall is not None and weakest["accuracy"] is not None:
        difference = weakest["accuracy"] - overall
        comparison = (
            f", {abs(difference):.0%} {'below' if difference < 0 else 'above'} its "
            f"overall {overall:.0%}"
        )
    return (
        f"On the held-out season the model called {weakest['accuracy']:.0%} of "
        f"{weakest['matches_evaluated']} matches correct in the weakest segment "
        f"this fixture belongs to ({weakest['segment']}){comparison}."
    )
