"""Why did the model favour this side?"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PHRASING_FILE = PROJECT_ROOT / "rules" / "bwf_factor_phrasing.json"

DIFFERENCE_SUFFIX = "_diff"


def load_phrasing(path: Path | None = None) -> dict[str, Any]:
    with Path(path or DEFAULT_PHRASING_FILE).open("r", encoding="utf-8") as handle:
        return json.load(handle)


@dataclass
class Factor:
    feature: str
    label: str
    contribution: float
    favours: str | None
    supports_prediction: bool
    sentence: str
    means: str
    does_not_mean: str
    value: Any = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MatchExplanation:
    predicted_side: str
    team1_win_probability: float
    supporting: list[Factor] = field(default_factory=list)
    opposing: list[Factor] = field(default_factory=list)
    negligible: list[str] = field(default_factory=list)
    attribution_note: str = ""
    caveats: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["supporting"] = [factor.to_dict() for factor in self.supporting]
        payload["opposing"] = [factor.to_dict() for factor in self.opposing]
        return payload


def _leader(feature: str, value: Any, team1: str, team2: str) -> str | None:
    """Which side a difference feature favours, by its sign."""
    if not feature.endswith(DIFFERENCE_SUFFIX):
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    if value > 0:
        return team1
    if value < 0:
        return team2
    return None


def _render(
    text: str,
    leader: str | None,
    value: Any,
    team1: str,
    team2: str,
) -> str:
    """Fill a declared template with this match's values.

    A number is rendered as a magnitude because the sign has already been spent
    naming the side that leads; writing "-145 points" after "Team 2 leads" says
    the same thing twice and reads as a contradiction. If a template and a value
    disagree in type, the phrasing file is at fault, so the label is used rather
    than raising in the middle of an explanation.
    """
    magnitude: Any = value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        magnitude = abs(value)
    try:
        return text.format(
            leader=leader or "", value=magnitude, team1=team1, team2=team2
        ).strip()
    except (KeyError, ValueError, TypeError):
        return text


side_the_value_favours = _leader
render_declared_sentence = _render


def explain_match(
    contributions: dict[str, float],
    feature_values: dict[str, Any],
    team1_win_probability: float,
    team1_name: str = "Team 1",
    team2_name: str = "Team 2",
    phrasing_file: Path | None = None,
) -> MatchExplanation:
    """Turn SHAP contributions into supporting and opposing factors.

    ``contributions`` are signed towards *team 1 winning*, matching the model's
    target. A positive contribution therefore pushes the estimate towards team 1
    whichever side is favoured overall, and whether that *supports* the
    prediction depends on which side the model picked.
    """
    document = load_phrasing(phrasing_file)
    factors = document["factors"]
    presentation = document["presentation"]
    floor = presentation["negligible_contribution"]
    limit = presentation["maximum_factors"]

    predicted_side = team1_name if team1_win_probability >= 0.5 else team2_name
    prediction_is_team1 = team1_win_probability >= 0.5

    supporting: list[Factor] = []
    opposing: list[Factor] = []
    negligible: list[str] = []
    unknown: list[str] = []

    for feature, contribution in contributions.items():
        entry = factors.get(feature)
        if entry is None:
            unknown.append(feature)
            continue

        if abs(contribution) < floor:
            negligible.append(entry["label"])
            continue

        value = feature_values.get(feature)
        leader = _leader(feature, value, team1_name, team2_name)

        # A positive contribution pushes towards team 1. It supports the
        # prediction when team 1 is the predicted side, and opposes it otherwise.
        pushes_to_team1 = contribution > 0
        supports = pushes_to_team1 == prediction_is_team1

        # A feature that is absent, level or switched off is described by its
        # negligible wording even when it moved the estimate; "Team 1 has no
        # rating history" would be a false statement about a player who has one.
        absent = leader is None and (value is None or value == 0 or value is False)
        template = entry["when_negligible"] if absent else entry["template"]
        sentence = _render(template, leader, value, team1_name, team2_name)

        factor = Factor(
            feature=feature,
            label=entry["label"],
            contribution=round(float(contribution), 4),
            favours=leader,
            supports_prediction=supports,
            sentence=sentence,
            means=entry["means"],
            does_not_mean=entry["does_not_mean"],
            value=(
                round(float(value), 3)
                if isinstance(value, (int, float)) and not isinstance(value, bool)
                else value
            ),
        )
        (supporting if supports else opposing).append(factor)

    supporting.sort(key=lambda item: abs(item.contribution), reverse=True)
    opposing.sort(key=lambda item: abs(item.contribution), reverse=True)

    caveats = [
        "Factors describe what moved the model's estimate, not what will decide "
        "the match.",
    ]
    if not opposing:
        caveats.append(
            "No factor of any size pointed the other way. That is unusual and "
            "worth a second look at the inputs rather than read as certainty."
        )
    if unknown:
        caveats.append(
            f"{len(unknown)} contributing feature(s) have no declared phrasing and "
            f"were omitted: {', '.join(sorted(unknown))}. Add them to "
            "rules/bwf_factor_phrasing.json rather than leaving them unexplained."
        )

    return MatchExplanation(
        predicted_side=predicted_side,
        team1_win_probability=round(float(team1_win_probability), 4),
        supporting=supporting[:limit],
        opposing=opposing[:limit],
        negligible=sorted(set(negligible)),
        attribution_note=document["attribution_note"],
        caveats=caveats,
    )


def describe_explanation(explanation: MatchExplanation) -> list[str]:
    """The explanation as lines a dashboard can render directly."""
    lines = [
        f"{explanation.predicted_side} is favoured, at "
        f"{explanation.team1_win_probability:.0%} for team 1."
    ]

    if explanation.supporting:
        lines.append("")
        lines.append("Factors behind the estimate:")
        lines.extend(f"  + {factor.sentence}" for factor in explanation.supporting)

    if explanation.opposing:
        lines.append("")
        lines.append("Factors pointing the other way:")
        lines.extend(f"  - {factor.sentence}" for factor in explanation.opposing)

    if explanation.negligible:
        lines.append("")
        lines.append(
            "Little influence here: " + ", ".join(explanation.negligible).lower() + "."
        )

    lines.append("")
    lines.extend(explanation.caveats)
    return lines
