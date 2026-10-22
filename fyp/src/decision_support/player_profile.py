"""Player profiles and rating trends from completed matches."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VALIDATED_MATCHES = PROJECT_ROOT / "outputs" / "bwf" / "bwf_validated_with_flags.csv"
RATED_MATCHES = PROJECT_ROOT / "outputs" / "bwf" / "bwf_model_rows_2022_2025.csv"

MINIMUM_MATCHES = 10

MINIMUM_SUBSET = 5


@dataclass
class Profile:
    """How a player wins, over their completed matches."""

    player: str
    matches: int
    wins: int
    win_rate: float
    straight_games_win_rate: float | None
    decider_share: float | None
    decider_win_rate: float | None
    comeback_rate: float | None
    comeback_opportunities: int
    first_match: str | None = None
    last_match: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load(path: Path):
    import pandas as pd

    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run the BWF preparation notebooks first."
        )
    return pd.read_csv(path)


PLAYER_COLUMNS = (
    "team1_player1", "team1_player2", "team2_player1", "team2_player2",
)


def _player_matches(frame, player: str):
    """Every match this player appeared in, with a flag for which side.

    Requires the per-player name columns. Some derived tables carry only team
    names, so the requirement is checked explicitly - a missing column would
    otherwise produce a scalar mask and a bare ``KeyError: False`` several
    frames away from the cause.
    """
    missing = [name for name in PLAYER_COLUMNS if name not in frame.columns]
    if missing:
        raise KeyError(
            f"Player columns are missing from this table: {missing}. "
            "It probably holds team names rather than player names; join it to "
            "bwf_validated_with_flags.csv on match_id first."
        )

    on_team1 = (frame["team1_player1"] == player) | (frame["team1_player2"] == player)
    on_team2 = (frame["team2_player1"] == player) | (frame["team2_player2"] == player)
    keep = on_team1 | on_team2
    mine = frame[keep].copy()
    mine["is_team1"] = on_team1[keep]
    return mine


def _rated_matches_with_players():
    """Rated rows joined to the player names they belong to.

    The rating table identifies sides by team name; for doubles a team is a
    pair, so a player cannot be found in it directly. The validated table holds
    the names, and both share ``match_id``.
    """
    import pandas as pd

    rated = _load(RATED_MATCHES)
    validated = _load(VALIDATED_MATCHES)
    names = validated[["match_id", *PLAYER_COLUMNS]]
    return rated.merge(names, on="match_id", how="left")


def competitiveness_profile(
    player: str, matches=None, minimum: int = MINIMUM_MATCHES
) -> Profile:
    """How this player wins, from completed matches only.

    A rate computed over a handful of matches is noise, so each statistic is
    reported only when its own subset is large enough, and the reason is
    recorded when it is not.
    """
    frame = matches if matches is not None else _load(VALIDATED_MATCHES)

    # Checked together and up front, so passing the wrong table gives one
    # message naming everything absent rather than failing on whichever column
    # happens to be touched first.
    required = ("winner", "number_of_games", "game1_team1", "game1_team2", *PLAYER_COLUMNS)
    absent = [name for name in required if name not in frame.columns]
    if absent:
        raise KeyError(
            f"This table cannot support a profile; missing {absent}. Use "
            "outputs/bwf/bwf_validated_with_flags.csv, or join it to that file "
            "on match_id."
        )

    if "standard_completed_match" in frame:
        frame = frame[frame["standard_completed_match"].astype(bool)]
    frame = frame[frame["winner"].notna()]

    mine = _player_matches(frame, player)
    notes: list[str] = []

    if len(mine) < minimum:
        return Profile(
            player=player,
            matches=len(mine),
            wins=0,
            win_rate=0.0,
            straight_games_win_rate=None,
            decider_share=None,
            decider_win_rate=None,
            comeback_rate=None,
            comeback_opportunities=0,
            notes=[
                f"Only {len(mine)} completed matches found; at least {minimum} are "
                "needed before a profile describes the player rather than the sample."
            ],
        )

    won = (mine.is_team1 & (mine.winner == 1)) | (~mine.is_team1 & (mine.winner == 2))
    mine = mine.assign(won=won)

    straight = mine[mine.number_of_games == 2]
    deciders = mine[mine.number_of_games == 3]

    # Losing the opening game is the situation a comeback is measured from, so
    # the denominator is those matches, not every match.
    if len(deciders):
        lost_first = deciders.apply(
            lambda row: (row.game1_team1 > row.game1_team2) != row.is_team1, axis=1
        )
        chances = deciders[lost_first]
    else:
        chances = deciders

    def rate(subset, label: str) -> float | None:
        if len(subset) < MINIMUM_SUBSET:
            notes.append(
                f"{label} not reported: only {len(subset)} such matches, "
                f"below the minimum of {MINIMUM_SUBSET}."
            )
            return None
        return round(float(subset.won.mean()), 4)

    dates = mine["date"].dropna().sort_values() if "date" in mine else None

    return Profile(
        player=player,
        matches=len(mine),
        wins=int(won.sum()),
        win_rate=round(float(won.mean()), 4),
        straight_games_win_rate=rate(straight, "Straight-games win rate"),
        decider_share=round(len(deciders) / len(mine), 4),
        decider_win_rate=rate(deciders, "Deciding-game win rate"),
        comeback_rate=rate(chances, "Comeback rate"),
        comeback_opportunities=len(chances),
        first_match=str(dates.iloc[0]) if dates is not None and len(dates) else None,
        last_match=str(dates.iloc[-1]) if dates is not None and len(dates) else None,
        notes=notes,
    )


def describe_profile(profile: Profile) -> list[str]:
    """Readable observations, each tied to the figure that supports it.

    Deliberately observations rather than judgements: "closes out matches" is a
    description of a measured pattern, where "mentally strong" would be an
    inference the data cannot support.
    """
    if profile.matches < MINIMUM_MATCHES:
        return profile.notes

    lines = [
        f"{profile.wins} wins from {profile.matches} completed matches "
        f"({profile.win_rate:.0%})."
    ]

    if profile.decider_share is not None:
        lines.append(
            f"{profile.decider_share:.0%} of matches went to a deciding game."
        )
    if profile.decider_win_rate is not None:
        direction = "above" if profile.decider_win_rate >= profile.win_rate else "below"
        lines.append(
            f"Wins {profile.decider_win_rate:.0%} of deciders, {direction} their "
            f"overall rate - close matches {'hold up' if direction == 'above' else 'are a weaker point'}."
        )
    if profile.comeback_rate is not None:
        lines.append(
            f"Recovered to win {profile.comeback_rate:.0%} of the "
            f"{profile.comeback_opportunities} matches where they lost the first game."
        )
    lines.extend(profile.notes)
    return lines


@dataclass
class TrendPoint:
    date: str
    rating: float
    opponent_rating: float | None
    won: bool | None


def rating_trend(
    player: str, matches=None, rating: str = "individual"
) -> dict[str, Any]:
    """This player's Elo over time, with what it can and cannot represent.

    ``rating`` selects individual or partnership Elo. The distinction matters:
    a partnership rating belongs to a *pair*, so for a doubles player it resets
    when they change partner and is not a personal rating at all. Presenting one
    as the other would misattribute a partner's contribution.
    """
    frame = matches if matches is not None else _rated_matches_with_players()
    mine = _player_matches(frame, player)

    caveats: list[str] = []
    if mine.empty:
        return {
            "player": player,
            "rating_type": rating,
            "points": [],
            "caveats": [f"No rated matches found for {player}."],
        }

    column = f"{{side}}_{rating}_elo"
    points: list[TrendPoint] = []
    for _, row in mine.sort_values("date").iterrows():
        own = column.format(side="team1" if row.is_team1 else "team2")
        other = column.format(side="team2" if row.is_team1 else "team1")
        if own not in row or row[own] != row[own]:  # NaN check without numpy
            continue
        won = None
        if "team1_win" in row and row["team1_win"] == row["team1_win"]:
            won = bool(row["team1_win"]) == bool(row.is_team1)
        points.append(
            TrendPoint(
                date=str(row["date"]),
                rating=float(row[own]),
                opponent_rating=float(row[other]) if other in row and row[other] == row[other] else None,
                won=won,
            )
        )

    if rating == "partnership":
        caveats.append(
            "Partnership Elo belongs to a pair, not a player. It resets when the "
            "partnership changes, so this line is the partnership's history rather "
            "than this athlete's."
        )
    if any(mine.get("is_doubles", 0) == 1) and rating == "individual":
        caveats.append(
            "Some of these matches are doubles, where an individual rating "
            "reflects results achieved alongside a partner."
        )

    movement = None
    if len(points) >= 2:
        movement = round(points[-1].rating - points[0].rating, 1)

    return {
        "player": player,
        "rating_type": rating,
        "points": [asdict(point) for point in points],
        "matches": len(points),
        "starting_rating": round(points[0].rating, 1) if points else None,
        "latest_rating": round(points[-1].rating, 1) if points else None,
        "movement": movement,
        "caveats": caveats,
    }


def describe_trend(trend: dict[str, Any]) -> str:
    """One sentence on where the rating has gone."""
    if not trend["points"]:
        return f"No rating history available for {trend['player']}."
    if trend["movement"] is None:
        return (
            f"{trend['player']} has a single rated match, which shows a position "
            "but not a direction."
        )

    movement = trend["movement"]
    direction = "risen" if movement > 0 else "fallen" if movement < 0 else "held steady"
    return (
        f"{trend['player']}'s rating has {direction} by {abs(movement):.0f} points "
        f"across {trend['matches']} rated matches, from {trend['starting_rating']:.0f} "
        f"to {trend['latest_rating']:.0f}."
    )
