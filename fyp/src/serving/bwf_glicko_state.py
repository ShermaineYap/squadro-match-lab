"""Glicko-2 serving state: what the champion needs to rate a fixture nobody has played.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Iterable

import pandas as pd

from src.models.glicko2 import (
    DEFAULT_DEVIATION, DEFAULT_RATING, DEFAULT_VOLATILITY, SCALE, Glicko2, Rating,
)

GLICKO_STATE_SCHEMA_VERSION = "1.0"
GLICKO_FEATURE_CODE_VERSION = "glicko_serving_features.v1"

SERVING_GLICKO_FEATURES = [
    "glicko_certainty_weighted_rating_diff",
    "glicko_deviation_diff",
    "glicko_volatility_diff",
    "glicko_volatility_mean",
    "glicko_deviation_mean",
]


class StaleStateError(RuntimeError):
    """The requested prediction date is not one this state can answer for."""


def player_key(discipline: str, player_id: int) -> str:
    return f"{discipline}|{int(player_id)}"


def build_glicko_state(matches: pd.DataFrame, target_column: str = "team1_win") -> dict[str, Any]:
    """Replay completed matches and return the final Glicko-2 state.

    The loop mirrors ``build_glicko_features``: read nothing, queue the whole
    date, then apply it. Only the terminal state is kept.
    """
    engine = Glicko2()
    ordered = matches.sort_values(["date", "match_id"])
    last_date: dict[str, pd.Timestamp] = {}
    played: dict[str, int] = {}
    snapshot: pd.Timestamp | None = None

    for match_date, batch in ordered.groupby("date", sort=True):
        period: dict[str, list[tuple[str, float]]] = {}
        for row in batch.sort_values("match_id").to_dict("records"):
            side1 = _keys(row, "team1")
            side2 = _keys(row, "team2")
            if not side1 or not side2:
                continue
            won = float(row[target_column])
            for key in side1:
                for opponent in side2:
                    period.setdefault(key, []).append((opponent, won))
            for key in side2:
                for opponent in side1:
                    period.setdefault(key, []).append((opponent, 1.0 - won))
            for key in side1 + side2:
                last_date[key] = match_date
                played[key] = played.get(key, 0) + 1
        if period:
            engine.update_period(period)
            engine.decay_absent(set(period))
            snapshot = match_date

    players = {
        key: {
            "discipline": key.split("|", 1)[0],
            "player_id": int(key.split("|", 1)[1]),
            "rating": rating.rating,
            "deviation": rating.deviation,
            "volatility": rating.volatility,
            "matches": played.get(key, 0),
            "last_update": str(pd.Timestamp(last_date[key]).date()) if key in last_date else None,
        }
        for key, rating in engine.ratings.items()
    }
    return {
        "schema_version": GLICKO_STATE_SCHEMA_VERSION,
        "feature_code_version": GLICKO_FEATURE_CODE_VERSION,
        "snapshot_as_of": str(pd.Timestamp(snapshot).date()) if snapshot is not None else None,
        "defaults": {
            "rating": DEFAULT_RATING,
            "deviation": DEFAULT_DEVIATION,
            "volatility": DEFAULT_VOLATILITY,
        },
        "players": players,
        "feature_columns": list(SERVING_GLICKO_FEATURES),
    }


def _keys(row: dict, side: str) -> list[str]:
    keys = []
    for slot in (1, 2):
        value = row.get(f"{side}_player{slot}_id")
        if value is not None and value == value and str(value) != "nan":
            keys.append(player_key(row["discipline"], int(value)))
    return keys


def _rating_for(state: dict, discipline: str, player_id: int) -> Rating:
    record = state["players"].get(player_key(discipline, player_id))
    if record is None:
        return Rating()
    return Rating(
        rating=record["rating"],
        deviation=record["deviation"],
        volatility=record["volatility"],
        matches=record["matches"],
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def glicko_features_for_sides(
    state: dict,
    *,
    discipline: str,
    team1_player_ids: Iterable[int],
    team2_player_ids: Iterable[int],
) -> dict[str, float]:
    """The five model features for one fixture, by the training formulas."""
    side1 = [_rating_for(state, discipline, p) for p in team1_player_ids]
    side2 = [_rating_for(state, discipline, p) for p in team2_player_ids]
    if not side1 or not side2:
        raise ValueError("both sides need at least one player")

    rating1, rating2 = _mean([r.rating for r in side1]), _mean([r.rating for r in side2])
    deviation1 = _mean([r.deviation for r in side1])
    deviation2 = _mean([r.deviation for r in side2])
    volatility1 = _mean([r.volatility for r in side1])
    volatility2 = _mean([r.volatility for r in side2])

    combined_phi = math.sqrt((deviation1 / SCALE) ** 2 + (deviation2 / SCALE) ** 2)
    certainty = 1.0 / math.sqrt(1.0 + 3.0 * combined_phi**2 / math.pi**2)

    return {
        "glicko_certainty_weighted_rating_diff": certainty * (rating1 - rating2),
        "glicko_deviation_diff": deviation1 - deviation2,
        "glicko_volatility_diff": volatility1 - volatility2,
        "glicko_volatility_mean": (volatility1 + volatility2) / 2.0,
        "glicko_deviation_mean": (deviation1 + deviation2) / 2.0,
    }


def assert_prediction_date(state: dict, prediction_date: Any) -> dict[str, Any]:
    """Refuse impossible dates; disclose how old the state is for possible ones."""
    snapshot = pd.Timestamp(state["snapshot_as_of"])
    requested = pd.Timestamp(prediction_date)
    if requested <= snapshot:
        raise StaleStateError(
            f"prediction date {requested.date()} is not after the Glicko state "
            f"snapshot ({snapshot.date()}). Predicting a match the state may "
            "already contain is not a forecast."
        )
    return {
        "snapshot_as_of": str(snapshot.date()),
        "prediction_date": str(requested.date()),
        "state_age_days": int((requested - snapshot).days),
        "rating_deviation_note": (
            "Glicko-2 widens a rating deviation for each rating period a player "
            "misses. The number of periods between this snapshot and the fixture "
            "is unknown, so no decay has been applied: the deviations here are "
            "those of the snapshot date and understate present uncertainty by an "
            "amount that grows with state age."
        ),
    }


def verify_against_training(
    state: dict,
    eligible: pd.DataFrame,
    sidecar: pd.DataFrame,
    sample: int = 200,
) -> dict[str, Any]:
    """Prove the serving formulas reproduce the training side-car.

    Rebuilds the state as it stood immediately before a sampled historical match
    and recomputes that match's features through the serving path. If the two
    disagree, the live model is being fed something other than what it was fitted
    on - the failure this function exists to catch.
    """
    ordered = eligible.sort_values(["date", "match_id"]).reset_index(drop=True)
    reference = sidecar.set_index("match_id")
    step = max(1, len(ordered) // sample)
    checked = ordered.iloc[::step].tail(sample)

    worst = 0.0
    compared = 0
    for row in checked.to_dict("records"):
        if row["match_id"] not in reference.index:
            continue
        history = ordered[ordered["date"] < row["date"]]
        if history.empty:
            continue
        partial = build_glicko_state(history)
        produced = glicko_features_for_sides(
            partial,
            discipline=row["discipline"],
            team1_player_ids=[int(row[f"team1_player{s}_id"]) for s in (1, 2)
                              if row.get(f"team1_player{s}_id") == row.get(f"team1_player{s}_id")],
            team2_player_ids=[int(row[f"team2_player{s}_id"]) for s in (1, 2)
                              if row.get(f"team2_player{s}_id") == row.get(f"team2_player{s}_id")],
        )
        expected = reference.loc[row["match_id"]]
        for name, value in produced.items():
            worst = max(worst, abs(value - float(expected[name])))
        compared += 1
    return {"matches_compared": compared, "max_absolute_difference": worst}


def state_sha256(state: dict) -> str:
    """A hash over the state's content, stable across dictionary ordering."""
    digest = hashlib.sha256()
    for key in sorted(state["players"]):
        record = state["players"][key]
        digest.update(
            f"{key}|{record['rating']:.10f}|{record['deviation']:.10f}|"
            f"{record['volatility']:.10f}|{record['matches']}|{record['last_update']}\0".encode()
        )
    digest.update(str(state["snapshot_as_of"]).encode())
    return digest.hexdigest()
