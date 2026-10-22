from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Any, Iterable

import numpy as np
import pandas as pd


STATE_SCHEMA_VERSION = "1.0"


def _safe_difference(a: Any, b: Any) -> float:
    if pd.isna(a) or pd.isna(b):
        return np.nan
    return float(a - b)


def _history_win_rate(history: list[int], window: int | None = None) -> float:
    if not history:
        return np.nan
    values = history[-window:] if window is not None else history
    return float(np.mean(values)) if values else np.nan


def _mean_without_missing(values: Iterable[Any]) -> float:
    usable = [float(value) for value in values if not pd.isna(value)]
    return float(np.mean(usable)) if usable else np.nan


def _team_players_from_row(row: Any, side: str) -> tuple[int, ...]:
    player1 = int(getattr(row, f"{side}_player1_id"))
    player2 = getattr(row, f"{side}_player2_id")
    if pd.isna(player2):
        return (player1,)
    return tuple(sorted((player1, int(player2))))


def _normalize_team(players: Iterable[int], discipline: str) -> tuple[int, ...]:
    normalized = tuple(sorted(int(player) for player in players))
    is_doubles = discipline in {"MD", "WD", "XD"}
    expected = 2 if is_doubles else 1
    if len(normalized) != expected:
        raise ValueError(
            f"{discipline} requires {expected} player(s) per team; "
            f"received {len(normalized)}."
        )
    if len(set(normalized)) != len(normalized):
        raise ValueError("A player cannot appear twice on the same team.")
    return normalized


def empty_bwf_state(
    default_elo: float = 1500.0,
    k_factor: float = 32.0,
    cold_start_threshold: int = 5,
) -> dict[str, Any]:
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "snapshot_as_of": None,
        "default_elo": float(default_elo),
        "k_factor": float(k_factor),
        "cold_start_threshold": int(cold_start_threshold),
        "player_elo": {},
        "partnership_elo": {},
        "player_history": {},
        "partnership_history": {},
        "player_matches": {},
        "player_wins": {},
        "partnership_matches": {},
        "player_last_date": {},
        "h2h_wins": {},
        "h2h_matches": {},
    }


def _get_elo(mapping: dict, key: Any, default_elo: float) -> float:
    return float(mapping.get(key, default_elo))


def _get_history(mapping: dict, key: Any) -> list[int]:
    return list(mapping.get(key, []))


def _get_count(mapping: dict, key: Any) -> int:
    return int(mapping.get(key, 0))


def _last_date(state: dict, key: Any) -> pd.Timestamp | None:
    value = state["player_last_date"].get(key)
    return None if value is None else pd.Timestamp(value)


def build_prematch_features(
    state: dict[str, Any],
    *,
    team1_player_ids: Iterable[int],
    team2_player_ids: Iterable[int],
    discipline: str,
    prediction_date: str | pd.Timestamp,
    tier_category: str,
    round_name: str,
    is_qualification: bool,
    host_country: str,
    enforce_after_snapshot: bool = True,
) -> dict[str, Any]:
    match_date = pd.Timestamp(prediction_date).normalize()
    snapshot_as_of = state.get("snapshot_as_of")
    if enforce_after_snapshot and snapshot_as_of is not None:
        snapshot_date = pd.Timestamp(snapshot_as_of).normalize()
        if match_date <= snapshot_date:
            raise ValueError(
                "Prediction date must be later than the BWF state snapshot "
                f"({snapshot_date.date()})."
            )

    discipline = str(discipline)
    team1_players = _normalize_team(team1_player_ids, discipline)
    team2_players = _normalize_team(team2_player_ids, discipline)

    if set(team1_players) & set(team2_players):
        raise ValueError("The same player cannot appear on both teams.")

    is_doubles = discipline in {"MD", "WD", "XD"}
    team1_player_keys = [(discipline, player) for player in team1_players]
    team2_player_keys = [(discipline, player) for player in team2_players]
    team1_key = (discipline, team1_players)
    team2_key = (discipline, team2_players)

    default_elo = float(state.get("default_elo", 1500.0))
    player_elo = state["player_elo"]
    partnership_elo = state["partnership_elo"]
    player_history = state["player_history"]
    partnership_history = state["partnership_history"]
    player_matches = state["player_matches"]
    player_wins = state["player_wins"]
    partnership_matches = state["partnership_matches"]
    h2h_wins = state["h2h_wins"]
    h2h_matches = state["h2h_matches"]

    team1_individual_elo = float(np.mean([
        _get_elo(player_elo, key, default_elo) for key in team1_player_keys
    ]))
    team2_individual_elo = float(np.mean([
        _get_elo(player_elo, key, default_elo) for key in team2_player_keys
    ]))

    if is_doubles:
        team1_partnership_elo = _get_elo(
            partnership_elo, team1_key, default_elo
        )
        team2_partnership_elo = _get_elo(
            partnership_elo, team2_key, default_elo
        )
        partnership_elo_diff = (
            team1_partnership_elo - team2_partnership_elo
        )
        team1_partnership_form_5 = _history_win_rate(
            _get_history(partnership_history, team1_key), 5
        )
        team2_partnership_form_5 = _history_win_rate(
            _get_history(partnership_history, team2_key), 5
        )
        partnership_form_5_diff = _safe_difference(
            team1_partnership_form_5,
            team2_partnership_form_5,
        )
        team1_partnership_match_count = _get_count(
            partnership_matches, team1_key
        )
        team2_partnership_match_count = _get_count(
            partnership_matches, team2_key
        )
        partnership_matches_diff = (
            team1_partnership_match_count
            - team2_partnership_match_count
        )
    else:
        partnership_elo_diff = 0.0
        partnership_form_5_diff = 0.0
        partnership_matches_diff = 0

    team1_form_5 = _mean_without_missing([
        _history_win_rate(_get_history(player_history, key), 5)
        for key in team1_player_keys
    ])
    team2_form_5 = _mean_without_missing([
        _history_win_rate(_get_history(player_history, key), 5)
        for key in team2_player_keys
    ])
    team1_form_10 = _mean_without_missing([
        _history_win_rate(_get_history(player_history, key), 10)
        for key in team1_player_keys
    ])
    team2_form_10 = _mean_without_missing([
        _history_win_rate(_get_history(player_history, key), 10)
        for key in team2_player_keys
    ])

    team1_experience = float(np.mean([
        _get_count(player_matches, key) for key in team1_player_keys
    ]))
    team2_experience = float(np.mean([
        _get_count(player_matches, key) for key in team2_player_keys
    ]))

    team1_career_win_rate = _mean_without_missing([
        _get_count(player_wins, key) / _get_count(player_matches, key)
        if _get_count(player_matches, key) > 0 else np.nan
        for key in team1_player_keys
    ])
    team2_career_win_rate = _mean_without_missing([
        _get_count(player_wins, key) / _get_count(player_matches, key)
        if _get_count(player_matches, key) > 0 else np.nan
        for key in team2_player_keys
    ])

    team1_rest_values = [
        (match_date - last_date).days if (
            last_date := _last_date(state, key)
        ) is not None else np.nan
        for key in team1_player_keys
    ]
    team2_rest_values = [
        (match_date - last_date).days if (
            last_date := _last_date(state, key)
        ) is not None else np.nan
        for key in team2_player_keys
    ]
    team1_rest_mean = _mean_without_missing(team1_rest_values)
    team2_rest_mean = _mean_without_missing(team2_rest_values)
    valid_team1_rest = [
        value for value in team1_rest_values if not pd.isna(value)
    ]
    valid_team2_rest = [
        value for value in team2_rest_values if not pd.isna(value)
    ]
    team1_rest_min = min(valid_team1_rest) if valid_team1_rest else np.nan
    team2_rest_min = min(valid_team2_rest) if valid_team2_rest else np.nan

    h2h_pair = tuple(sorted((team1_players, team2_players)))
    h2h_key = (discipline, h2h_pair)
    h2h_wins_for_pair = h2h_wins.get(h2h_key, {})
    team1_h2h_wins = int(h2h_wins_for_pair.get(team1_players, 0))
    team2_h2h_wins = int(h2h_wins_for_pair.get(team2_players, 0))

    threshold = int(state.get("cold_start_threshold", 5))
    features = {
        "is_doubles": int(is_doubles),
        "individual_elo_diff": (
            team1_individual_elo - team2_individual_elo
        ),
        "partnership_elo_diff": float(partnership_elo_diff),
        "recent_form_5_diff": _safe_difference(
            team1_form_5, team2_form_5
        ),
        "recent_form_10_diff": _safe_difference(
            team1_form_10, team2_form_10
        ),
        "partnership_form_5_diff": float(partnership_form_5_diff),
        "player_experience_diff": team1_experience - team2_experience,
        "career_win_rate_diff": _safe_difference(
            team1_career_win_rate, team2_career_win_rate
        ),
        "partnership_matches_diff": int(partnership_matches_diff),
        "rest_days_mean_diff": _safe_difference(
            team1_rest_mean, team2_rest_mean
        ),
        "rest_days_min_diff": _safe_difference(
            team1_rest_min, team2_rest_min
        ),
        "h2h_matches": _get_count(h2h_matches, h2h_key),
        "h2h_win_diff": team1_h2h_wins - team2_h2h_wins,
        "team1_cold_start": int(any(
            _get_count(player_matches, key) < threshold
            for key in team1_player_keys
        )),
        "team2_cold_start": int(any(
            _get_count(player_matches, key) < threshold
            for key in team2_player_keys
        )),
        "discipline": discipline,
        "tier_category": tier_category,
        "round": round_name,
        "is_qualification": bool(is_qualification),
        "host_country": host_country,
    }
    return features


def build_state_from_completed_matches(
    matches: pd.DataFrame,
    *,
    default_elo: float = 1500.0,
    k_factor: float = 32.0,
    cold_start_threshold: int = 5,
    collect_features: bool = False,
) -> tuple[dict[str, Any], pd.DataFrame | None]:
    required = {
        "match_id", "date", "discipline",
        "team1_player1_id", "team1_player2_id",
        "team2_player1_id", "team2_player2_id",
        "winner", "tier_category", "round",
        "is_qualification", "host_country",
    }
    missing = sorted(required - set(matches.columns))
    if missing:
        raise ValueError(f"Historical BWF data is missing columns: {missing}")

    state = empty_bwf_state(
        default_elo=default_elo,
        k_factor=k_factor,
        cold_start_threshold=cold_start_threshold,
    )
    sorted_matches = matches.copy()
    sorted_matches["date"] = pd.to_datetime(sorted_matches["date"])
    sorted_matches = sorted_matches.sort_values(["date", "match_id"])
    generated_rows: list[dict[str, Any]] = []

    for match_date, date_matches in sorted_matches.groupby("date", sort=True):
        player_elo_delta: dict[Any, float] = defaultdict(float)
        partnership_elo_delta: dict[Any, float] = defaultdict(float)
        date_updates: list[dict[str, Any]] = []

        for row in date_matches.sort_values("match_id").itertuples(index=False):
            discipline = row.discipline
            team1_players = _team_players_from_row(row, "team1")
            team2_players = _team_players_from_row(row, "team2")
            features = build_prematch_features(
                state,
                team1_player_ids=team1_players,
                team2_player_ids=team2_players,
                discipline=discipline,
                prediction_date=match_date,
                tier_category=row.tier_category,
                round_name=row.round,
                is_qualification=bool(row.is_qualification),
                host_country=row.host_country,
                enforce_after_snapshot=False,
            )
            if collect_features:
                generated_rows.append({"match_id": row.match_id, **features})

            is_doubles = discipline in {"MD", "WD", "XD"}
            team1_player_keys = [
                (discipline, player) for player in team1_players
            ]
            team2_player_keys = [
                (discipline, player) for player in team2_players
            ]
            team1_key = (discipline, team1_players)
            team2_key = (discipline, team2_players)

            team1_individual_elo = float(np.mean([
                _get_elo(state["player_elo"], key, default_elo)
                for key in team1_player_keys
            ]))
            team2_individual_elo = float(np.mean([
                _get_elo(state["player_elo"], key, default_elo)
                for key in team2_player_keys
            ]))
            team1_win = int(row.winner == 1)

            expected_team1_player = 1 / (
                1 + 10 ** (
                    (team2_individual_elo - team1_individual_elo) / 400
                )
            )
            player_delta = k_factor * (
                team1_win - expected_team1_player
            )
            for key in team1_player_keys:
                player_elo_delta[key] += player_delta
            for key in team2_player_keys:
                player_elo_delta[key] -= player_delta

            if is_doubles:
                team1_partnership_elo = _get_elo(
                    state["partnership_elo"], team1_key, default_elo
                )
                team2_partnership_elo = _get_elo(
                    state["partnership_elo"], team2_key, default_elo
                )
                expected_team1_partnership = 1 / (
                    1 + 10 ** (
                        (
                            team2_partnership_elo
                            - team1_partnership_elo
                        ) / 400
                    )
                )
                partnership_delta = k_factor * (
                    team1_win - expected_team1_partnership
                )
                partnership_elo_delta[team1_key] += partnership_delta
                partnership_elo_delta[team2_key] -= partnership_delta

            h2h_pair = tuple(sorted((team1_players, team2_players)))
            date_updates.append({
                "discipline": discipline,
                "is_doubles": is_doubles,
                "team1_players": team1_players,
                "team2_players": team2_players,
                "team1_player_keys": team1_player_keys,
                "team2_player_keys": team2_player_keys,
                "team1_key": team1_key,
                "team2_key": team2_key,
                "team1_win": team1_win,
                "h2h_key": (discipline, h2h_pair),
                "date": pd.Timestamp(match_date).isoformat(),
            })

        for key, delta in player_elo_delta.items():
            state["player_elo"][key] = (
                _get_elo(state["player_elo"], key, default_elo)
                + float(delta)
            )

        for key, delta in partnership_elo_delta.items():
            state["partnership_elo"][key] = (
                _get_elo(state["partnership_elo"], key, default_elo)
                + float(delta)
            )

        for update in date_updates:
            team1_win = update["team1_win"]
            for key in update["team1_player_keys"]:
                history = _get_history(state["player_history"], key)
                state["player_history"][key] = (history + [team1_win])[-10:]
                state["player_matches"][key] = (
                    _get_count(state["player_matches"], key) + 1
                )
                state["player_wins"][key] = (
                    _get_count(state["player_wins"], key) + team1_win
                )
                state["player_last_date"][key] = update["date"]

            for key in update["team2_player_keys"]:
                result = 1 - team1_win
                history = _get_history(state["player_history"], key)
                state["player_history"][key] = (history + [result])[-10:]
                state["player_matches"][key] = (
                    _get_count(state["player_matches"], key) + 1
                )
                state["player_wins"][key] = (
                    _get_count(state["player_wins"], key) + result
                )
                state["player_last_date"][key] = update["date"]

            if update["is_doubles"]:
                for key, result in [
                    (update["team1_key"], team1_win),
                    (update["team2_key"], 1 - team1_win),
                ]:
                    history = _get_history(
                        state["partnership_history"], key
                    )
                    state["partnership_history"][key] = (
                        history + [result]
                    )[-10:]
                    state["partnership_matches"][key] = (
                        _get_count(state["partnership_matches"], key) + 1
                    )

            h2h_wins = state["h2h_wins"].setdefault(
                update["h2h_key"], {}
            )
            h2h_wins[update["team1_players"]] = int(
                h2h_wins.get(update["team1_players"], 0) + team1_win
            )
            h2h_wins[update["team2_players"]] = int(
                h2h_wins.get(update["team2_players"], 0)
                + (1 - team1_win)
            )
            state["h2h_matches"][update["h2h_key"]] = (
                _get_count(state["h2h_matches"], update["h2h_key"]) + 1
            )

        state["snapshot_as_of"] = pd.Timestamp(match_date).date().isoformat()

    generated = (
        pd.DataFrame(generated_rows) if collect_features else None
    )
    return state, generated


def head_to_head_record(
    state: dict[str, Any],
    discipline: str,
    team1_player_ids: Iterable[int],
    team2_player_ids: Iterable[int],
) -> dict[str, Any]:
    """The completed record between two sides, read from the serving state.

    This is a read of exactly the counters ``build_prematch_features`` derives
    ``h2h_matches`` and ``h2h_win_diff`` from, so what the interface shows and
    what the model was given cannot drift apart: both come from this dictionary,
    accumulated by ``build_state_from_completed_matches`` over this project's own
    World Tour dataset.

    That provenance is the whole point, and the caller is expected to say so.
    This is **not** a career head-to-head and not BWF's published record: it
    counts meetings in individual World Tour events from January 2021 up to the
    state's snapshot date, so a pair with a long history before 2021 or outside
    the World Tour will show a smaller number here than a reader expects. A
    count presented without that qualification reads as a thin rivalry when it
    is really a narrow window on a long one.

    Nothing here is fetched, fitted or inferred: the state is frozen, and this
    function only looks a key up in it.
    """
    team1_players = _normalize_team(team1_player_ids, discipline)
    team2_players = _normalize_team(team2_player_ids, discipline)
    if set(team1_players) & set(team2_players):
        raise ValueError("The same player cannot appear on both teams.")

    key = (str(discipline), tuple(sorted((team1_players, team2_players))))
    wins_for_pair = state["h2h_wins"].get(key, {})

    return {
        "discipline": str(discipline),
        "team1_player_ids": list(team1_players),
        "team2_player_ids": list(team2_players),
        "team1_wins": int(wins_for_pair.get(team1_players, 0)),
        "team2_wins": int(wins_for_pair.get(team2_players, 0)),
        "matches": _get_count(state["h2h_matches"], key),
        "snapshot_as_of": state.get("snapshot_as_of"),
    }
