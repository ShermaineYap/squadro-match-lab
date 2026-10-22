"""Optimisation engine: which changes to a planned session, and how much."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEVER_FILE = PROJECT_ROOT / "rules" / "injury_levers.json"

TARGET_CLASS = "High Risk/Injured"

MINIMUM_USEFUL_REDUCTION = 0.02


def load_levers(path: Path | None = None) -> dict[str, Any]:
    with Path(path or DEFAULT_LEVER_FILE).open("r", encoding="utf-8") as handle:
        return json.load(handle)


@dataclass
class Change:
    """One adjustment to one lever."""

    feature: str
    label: str
    current_value: float
    proposed_value: float
    effort: float

    @property
    def direction(self) -> str:
        return "reduce" if self.proposed_value < self.current_value else "increase"

    def describe(self) -> str:
        percent = abs(self.proposed_value - self.current_value) / abs(self.current_value or 1)
        return (
            f"{self.direction} {self.label.lower()} from {self.current_value:.5g} "
            f"to {self.proposed_value:.5g} ({percent:.0%})"
        )


@dataclass
class Plan:
    """A set of changes and what the model estimates would follow."""

    changes: list[Change]
    baseline_probability: float
    projected_probability: float
    effort: float

    @property
    def reduction(self) -> float:
        return self.baseline_probability - self.projected_probability

    @property
    def reduction_per_effort(self) -> float:
        return self.reduction / self.effort if self.effort else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "changes": [asdict(change) for change in self.changes],
            "summary": [change.describe() for change in self.changes],
            "baseline_probability": round(self.baseline_probability, 4),
            "projected_probability": round(self.projected_probability, 4),
            "estimated_reduction": round(self.reduction, 4),
            "effort": round(self.effort, 2),
            "reduction_per_effort": round(self.reduction_per_effort, 4),
        }


def _candidate_values(
    lever: dict[str, Any],
    current: float,
    bounds: tuple[float, float] | None = None,
) -> list[float]:
    """The values this lever may take, from a small step to its practical limit.

    Bounded by a fraction of the athlete's *current* value rather than by the
    dataset's range: halving a heavy session and halving a light one are
    different acts, and the dataset's percentiles describe other sports.

    ``bounds`` is the one thing a fraction of the current value cannot express:
    where a scale ends. Hydration is recorded as a percentage that reaches 100
    in the development data, and a lever allowed to grow to 1.2x its current
    value proposed raising a session already at 100 to 120 — a recommendation
    to hydrate to 120%, which is not a smaller version of good advice but a
    number that cannot exist. Candidates outside the supplied range are dropped
    rather than clamped: clamping would silently turn several distinct steps
    into repeats of the same endpoint and let the search spend effort buying
    nothing.
    """
    if current is None or current == 0:
        return []

    step = lever.get("step_fraction", 0.1)
    helps = lever.get("direction_that_helps", "decrease")
    values: list[float] = []

    if helps == "decrease":
        limit = lever.get("minimum_fraction_of_current", 0.5)
        multiplier = 1.0 - step
        while multiplier >= limit - 1e-9:
            values.append(current * multiplier)
            multiplier -= step
    else:
        limit = lever.get("maximum_fraction_of_current", 1.3)
        multiplier = 1.0 + step
        while multiplier <= limit + 1e-9:
            values.append(current * multiplier)
            multiplier += step

    if bounds is not None:
        low, high = bounds
        values = [value for value in values if low - 1e-9 <= value <= high + 1e-9]

    return values


def _effort_for(lever: dict[str, Any], current: float, proposed: float) -> float:
    """Cost of a move, in steps times the lever's own cost per step."""
    step = lever.get("step_fraction", 0.1)
    if not current:
        return float("inf")
    steps = abs(proposed - current) / abs(current) / step
    return steps * lever.get("effort_per_step", 1.0)


def refused_features(requested: list[str], document: dict[str, Any]) -> list[dict[str, Any]]:
    """Which requested features are outcomes, and what to act on instead.

    Returned rather than silently dropped: an operator who asked to optimise
    fatigue deserves to be told why that is not an action, not to receive a plan
    that quietly ignores their request.
    """
    outcomes = {entry["feature"]: entry for entry in document.get("outcomes", [])}
    refused = []
    for feature in requested:
        entry = outcomes.get(feature)
        if entry is None:
            continue
        refused.append(
            {
                "feature": feature,
                "label": entry["label"],
                "reason": entry["rationale"],
                "act_on_instead": entry.get("act_on_instead", []),
            }
        )
    return refused


def optimise_session(
    predict: Callable[[dict[str, Any]], dict[str, float]],
    inputs: dict[str, Any],
    effort_budget: float | None = None,
    maximum_changes: int | None = None,
    lever_file: Path | None = None,
    target_class: str = TARGET_CLASS,
    threshold: float | None = None,
    feature_bounds: dict[str, tuple[float, float]] | None = None,
) -> dict[str, Any]:
    """Search for the smallest change that most lowers the estimated signal.

    ``predict`` takes a full feature payload and returns class probabilities, so
    the engine never touches the model directly and can be tested without one.

    ``target_class`` is the class whose probability a plan tries to lower. It
    defaults to the three-class name this engine was written for; the binary
    screen passes its own, which is the only thing that differed between the two
    models as far as this search is concerned.

    ``threshold`` is what makes a plan on the binary screen mean something
    sharper than "lower". That model does not predict by argmax — it flags
    whenever the probability reaches a locked cut-point — so a plan there has a
    definite target to clear rather than an open-ended reduction, and each plan
    reports whether it clears it. It stays ``None`` for the three-class stream,
    which has no cut-point to speak of and where inventing one would be exactly
    the arbitrary threshold the governance rules out.

    Clearing the threshold is a statement about the model's output under
    hypothetical inputs. It is not a claim that the session became safe, and
    nothing in the observational training data could support such a claim.

    The search is a bounded beam rather than an exhaustive sweep. Exhaustive
    search over five levers at ten steps each is fifty thousand model calls for
    a result no coach would read; a beam keeps the few most promising partial
    plans at each step, which is both fast and explainable - every plan in the
    output can be traced to the single change that started it.
    """
    document = load_levers(lever_file)
    settings = document.get("search", {})
    budget = effort_budget if effort_budget is not None else settings.get("default_effort_budget", 4.0)
    # ``or`` collapsed an explicit 0 into the default, so a caller asking for
    # "no changes at all" - which is a legitimate way to ask for the baseline
    # and nothing else - silently received three-change plans instead.
    max_changes = (
        maximum_changes
        if maximum_changes is not None
        else settings.get("maximum_changes", 3)
    )
    beam_width = settings.get("beam_width", 6)

    levers = [
        lever
        for lever in document["levers"]
        if lever["feature"] in inputs and inputs[lever["feature"]] is not None
    ]

    baseline = predict(inputs).get(target_class, 0.0)

    unavailable = [
        lever["feature"] for lever in document["levers"] if lever["feature"] not in inputs
    ]

    # Each beam entry is (payload, changes so far, effort spent, probability).
    beam: list[tuple[dict[str, Any], list[Change], float, float]] = [
        (dict(inputs), [], 0.0, baseline)
    ]
    completed: list[Plan] = []

    for _ in range(max_changes):
        expanded: list[tuple[dict[str, Any], list[Change], float, float]] = []

        for payload, changes, spent, _probability in beam:
            already = {change.feature for change in changes}
            for lever in levers:
                feature = lever["feature"]
                if feature in already:
                    continue

                current = float(inputs[feature])
                for value in _candidate_values(
                    lever, current, (feature_bounds or {}).get(feature)
                ):
                    effort = _effort_for(lever, current, value)
                    if spent + effort > budget + 1e-9:
                        continue

                    trial = dict(payload)
                    trial[feature] = value
                    probability = predict(trial).get(target_class, 0.0)

                    change = Change(
                        feature=feature,
                        label=lever["label"],
                        current_value=current,
                        proposed_value=value,
                        effort=effort,
                    )
                    expanded.append(
                        (trial, [*changes, change], spent + effort, probability)
                    )

        if not expanded:
            break

        # Rank by how much signal each plan removes per unit of effort, so a
        # cheap change that helps a little is not buried under an expensive one
        # that helps slightly more.
        expanded.sort(
            key=lambda entry: (baseline - entry[3]) / (entry[2] or 1e-9), reverse=True
        )
        beam = expanded[:beam_width]

        for payload, changes, spent, probability in beam:
            completed.append(Plan(list(changes), baseline, probability, spent))

    useful = [plan for plan in completed if plan.reduction >= MINIMUM_USEFUL_REDUCTION]
    useful.sort(key=lambda plan: (-plan.reduction_per_effort, plan.effort))

    # Keep the best plan for each distinct set of levers, so the list offers
    # genuinely different courses of action rather than ten variations of one.
    seen: set[tuple[str, ...]] = set()
    distinct: list[Plan] = []
    for plan in useful:
        signature = tuple(sorted(change.feature for change in plan.changes))
        if signature in seen:
            continue
        seen.add(signature)
        distinct.append(plan)

    def as_dict(plan: Plan) -> dict[str, Any]:
        payload = plan.to_dict()
        if threshold is not None:
            # Stated per plan rather than left to the reader to compare two
            # rounded figures. "Below the threshold" is the whole point of a
            # plan on a thresholded screen, and it is the one thing a reader
            # would otherwise have to work out for themselves.
            payload["clears_threshold"] = bool(plan.projected_probability < threshold)
        return payload

    return {
        "baseline_probability": round(baseline, 4),
        "target_class": target_class,
        "threshold": threshold,
        "baseline_above_threshold": (
            None if threshold is None else bool(baseline >= threshold)
        ),
        "effort_budget": budget,
        "plans": [as_dict(plan) for plan in distinct[:5]],
        "levers_considered": [lever["feature"] for lever in levers],
        "levers_unavailable": unavailable,
        "causal_note": document["causal_note"],
        "no_plan_reason": (
            None
            if distinct
            else (
                "No combination of the available levers lowered the estimated "
                f"signal by at least {MINIMUM_USEFUL_REDUCTION:.0%} within the "
                "effort budget. That is a result, not a failure: it means the "
                "model does not attribute this session's signal to anything a "
                "coach can change here."
            )
        ),
    }


def describe_plan(plan: dict[str, Any], baseline: float) -> str:
    """One human-readable sentence, the last step of the supervisor's chain.

    Deliberately phrased as a conditional about the model rather than a promise
    about the athlete: the difference between "this would lower the estimated
    signal" and "this would lower the risk of injury" is the difference between
    what was measured and what was not.
    """
    actions = "; ".join(plan["summary"])
    return (
        f"If this session were recorded with {actions}, the model would estimate "
        f"the higher-risk signal at {plan['projected_probability']:.0%} instead of "
        f"{baseline:.0%}. This is the model's sensitivity to those inputs, not "
        "evidence that the change would prevent an injury."
    )
