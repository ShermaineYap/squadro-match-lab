"""Candidate features for the BWF champion Stage 1 study: opponent quality and pair
stability.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque

import pandas as pd

DEFAULT_ELO = 1500.0
K_FACTOR = 32.0
RECENT_WINDOW_DAYS = 180
_DOUBLES = frozenset({"MD", "WD", "XD"})

OPPONENT_QUALITY_COLUMNS = [
    "oq_form5_diff",
    "oq_form10_diff",
    "oq_surprise5_diff",
]
PARTNERSHIP_COLUMNS = [
    "pair_tenure_days_diff",
    "pair_recent_matches_diff",
]
CANDIDATE_FEATURE_COLUMNS = OPPONENT_QUALITY_COLUMNS + PARTNERSHIP_COLUMNS


def _players(row: dict, side: str) -> tuple:
    names = []
    for slot in ("player1", "player2"):
        value = row.get(f"{side}_{slot}_id")
        if value is not None and not (isinstance(value, float) and math.isnan(value)):
            names.append(str(int(value)) if isinstance(value, float) else str(value))
    return tuple(sorted(names))


def _mean(values: list[float], default: float = 0.0) -> float:
    return sum(values) / len(values) if values else default


def build_candidate_features(matches: pd.DataFrame, target_column: str = "team1_win") -> pd.DataFrame:
    """Replay the frame and emit pre-match candidate features, one row per match."""
    frame = matches.sort_values(["date", "match_id"]).copy()
    frame["date"] = pd.to_datetime(frame["date"])

    elo: dict = defaultdict(lambda: DEFAULT_ELO)
    # (result, opponent_rating, expected) for the last ten matches.
    history: dict = defaultdict(lambda: deque(maxlen=10))
    pair_first_seen: dict = {}
    pair_dates: dict = defaultdict(lambda: deque())

    rows = []
    for match_date, day in frame.groupby("date", sort=True):
        elo_delta: dict = defaultdict(float)
        pending = []

        for record in day.sort_values("match_id").to_dict("records"):
            discipline = record["discipline"]
            is_doubles = discipline in _DOUBLES
            side1 = [(discipline, p) for p in _players(record, "team1")]
            side2 = [(discipline, p) for p in _players(record, "team2")]
            if not side1 or not side2:
                continue

            rating1 = _mean([elo[k] for k in side1], DEFAULT_ELO)
            rating2 = _mean([elo[k] for k in side2], DEFAULT_ELO)

            def quality_form(keys, window):
                weighted, weights, plain = 0.0, 0.0, []
                for key in keys:
                    for result, opponent_rating, _expected in list(history[key])[-window:]:
                        weight = opponent_rating / DEFAULT_ELO
                        weighted += weight * result
                        weights += weight
                        plain.append(result)
                if not plain:
                    return float("nan")
                return weighted / weights if weights else float("nan")

            def surprise(keys, window):
                values = []
                for key in keys:
                    for result, _opponent_rating, expected in list(history[key])[-window:]:
                        values.append(result - expected)
                return _mean(values, float("nan")) if values else float("nan")

            def pair_state(keys, players):
                if not is_doubles or len(players) != 2:
                    return 0.0, 0.0
                key = (discipline, players)
                first = pair_first_seen.get(key)
                tenure = (match_date - first).days if first is not None else 0.0
                cutoff = match_date - pd.Timedelta(days=RECENT_WINDOW_DAYS)
                recent = sum(1 for d in pair_dates[key] if d > cutoff)
                return float(tenure), float(recent)

            players1 = _players(record, "team1")
            players2 = _players(record, "team2")
            tenure1, recent1 = pair_state(side1, players1)
            tenure2, recent2 = pair_state(side2, players2)

            def difference(a, b):
                if pd.isna(a) and pd.isna(b):
                    return float("nan")
                return (0.0 if pd.isna(a) else a) - (0.0 if pd.isna(b) else b)

            rows.append({
                "match_id": record["match_id"],
                "oq_form5_diff": difference(quality_form(side1, 5), quality_form(side2, 5)),
                "oq_form10_diff": difference(quality_form(side1, 10), quality_form(side2, 10)),
                "oq_surprise5_diff": difference(surprise(side1, 5), surprise(side2, 5)),
                "pair_tenure_days_diff": tenure1 - tenure2,
                "pair_recent_matches_diff": recent1 - recent2,
            })

            team1_win = int(record[target_column])
            expected1 = 1.0 / (1.0 + 10 ** ((rating2 - rating1) / 400.0))
            delta = K_FACTOR * (team1_win - expected1)
            for key in side1:
                elo_delta[key] += delta
            for key in side2:
                elo_delta[key] -= delta

            pending.append({
                "side1": side1, "side2": side2,
                "players1": players1, "players2": players2,
                "discipline": discipline, "is_doubles": is_doubles,
                "team1_win": team1_win, "rating1": rating1, "rating2": rating2,
                "expected1": expected1, "date": match_date,
            })

        # Nothing above this line has changed state; everything below applies the
        # whole date at once, which is what makes same-day leakage impossible.
        for key, value in elo_delta.items():
            elo[key] += value

        for update in pending:
            for key in update["side1"]:
                history[key].append(
                    (update["team1_win"], update["rating2"], update["expected1"])
                )
            for key in update["side2"]:
                history[key].append(
                    (1 - update["team1_win"], update["rating1"], 1.0 - update["expected1"])
                )
            if update["is_doubles"]:
                for players in (update["players1"], update["players2"]):
                    if len(players) != 2:
                        continue
                    key = (update["discipline"], players)
                    pair_first_seen.setdefault(key, update["date"])
                    pair_dates[key].append(update["date"])

    return pd.DataFrame(rows)
