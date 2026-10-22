"""Elo variants: margin-aware K, time-decayed K, and season-boundary uncertainty."""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass

from src.models.glicko2 import (
    DEFAULT_DEVIATION,
    SCALE,
    Glicko2,
    Rating,
)

DEFAULT_ELO = 1500.0

DEFAULT_K = 32.0

_DOUBLES_DISCIPLINES = frozenset({"MD", "WD", "XD"})

DECAY_GRACE_DAYS = 30.0

_MARGIN_REFERENCE = {
    "games": 0.75,
    "points": 0.25,
    "normalised_points": 0.15,
}


# ---------------------------------------------------------------------------
# Margin measures
# ---------------------------------------------------------------------------
def _game_points(row: dict) -> tuple[int, int]:
    """Total points each side scored across the match."""
    team1 = 0
    team2 = 0
    for game in (1, 2, 3):
        a = row.get(f"game{game}_team1")
        b = row.get(f"game{game}_team2")
        if a is None or a != a or b is None or b != b:
            continue
        team1 += int(a)
        team2 += int(b)
    return team1, team2


def _games_won(row: dict) -> tuple[int, int]:
    team1 = 0
    team2 = 0
    for game in (1, 2, 3):
        a = row.get(f"game{game}_team1")
        b = row.get(f"game{game}_team2")
        if a is None or a != a or b is None or b != b:
            continue
        if a > b:
            team1 += 1
        elif b > a:
            team2 += 1
    return team1, team2


def signed_margin(row: dict) -> float:
    """Normalised point difference from team 1's perspective, in (-1, 1).

    Used for the *margin form* feature, which asks whether recent margins carry
    information directly rather than through a rating update.
    """
    team1, team2 = _game_points(row)
    total = team1 + team2
    if total <= 0:
        return 0.0
    return (team1 - team2) / total


def decisiveness(row: dict, team1_won: bool, measure: str) -> float:
    """How decisive the win was, from the *winner's* perspective, in [0, 1].

    Three measures, and the reason `normalised_points` is the default:

    - ``games``: 1.0 for a two-game win, 0.5 for a three-game win. Honest and
      robust, but it takes exactly two values, so it throws away the difference
      between 21-9 21-11 and 21-19 22-20.
    - ``points``: raw total point difference, capped at 40. Uses the whole
      scoreline, but is not comparable between formats: a three-game match has
      about 50% more points on the table, so the same dominance produces a larger
      raw difference.
    - ``normalised_points``: point difference divided by points played. Comparable
      across two- and three-game matches, and it uses every point. This is the
      default for those two reasons; the study scores all three rather than
      leaving the choice to the docstring.

    A win on fewer total points is possible (8-21 21-19 21-19 loses the point
    count and wins the match). That is a genuinely marginal win, so the value is
    floored just above zero rather than clipped to it - a zero multiplier would
    mean "this match tells us nothing", which is too strong.
    """
    if measure == "games":
        games1, games2 = _games_won(row)
        winner_games = games1 if team1_won else games2
        loser_games = games2 if team1_won else games1
        if winner_games + loser_games == 0:
            return 0.5
        return 1.0 if loser_games == 0 else 0.5

    team1, team2 = _game_points(row)
    winner = team1 if team1_won else team2
    loser = team2 if team1_won else team1

    if measure == "points":
        return max(0.005, min((winner - loser) / 40.0, 1.0))
    if measure == "normalised_points":
        total = winner + loser
        if total <= 0:
            return 0.005
        return max(0.005, (winner - loser) / total)
    raise ValueError(f"Unknown margin measure: {measure!r}")


def margin_multiplier(
    value: float,
    gamma: float,
    reference: float,
) -> float:
    """Turn decisiveness into a K multiplier: ``ln(1 + gamma d) / ln(1 + gamma r)``.

    Logarithmic rather than linear because the marginal evidence in an extra
    point falls away: the gap between a 2-point and a 12-point win says far more
    than the gap between a 22-point and a 32-point win. ``gamma`` sets how
    aggressively small margins are separated from large ones and is tuned on
    development data; ``reference`` fixes the margin at which the multiplier is
    exactly 1, so the variant is centred on the frozen Elo rather than being a
    disguised change of K.
    """
    return math.log1p(gamma * value) / math.log1p(gamma * reference)


def autocorrelation_correction(rating_gap_for_winner: float) -> float:
    """Shrink the margin multiplier when the winner was already the favourite.

    This is the guard against the double-counting failure named in the module
    docstring. Without it, a strong player who beats weak opponents by a lot is
    rewarded twice - once for winning, once for the margin that their rating
    already predicted - and their rating runs away. With it, a blowout by a
    heavy favourite is treated as roughly the expected result, and a blowout by
    an underdog counts for more.

    The functional form and the constant 2.2 are FiveThirtyEight's.
    """
    return 2.2 / (2.2 + 0.001 * rating_gap_for_winner)


# ---------------------------------------------------------------------------
# Variant configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EloConfig:
    """One rating variant. Every mechanism is off by default.

    An ``EloConfig()`` with no arguments is *exactly* the project's frozen Elo:
    K = 32, no margin scaling, no decay, no season handling. That is deliberate,
    and `scripts/evaluate_elo_variants.py` checks the reproduction numerically
    against `individual_elo_diff` before trusting anything else this module says.
    """

    name: str = "elo_baseline"
    k_factor: float = DEFAULT_K

    margin_measure: str | None = None
    margin_gamma: float = 10.0
    margin_reference: float | None = None
    correct_margin_autocorrelation: bool = False

    absence_k_beta: float = 0.0

    decay_half_life_days: float | None = None

    season_regression: float = 1.0

    def reference(self) -> float:
        if self.margin_reference is not None:
            return self.margin_reference
        return _MARGIN_REFERENCE[self.margin_measure or "normalised_points"]


def _side_player_keys(row: dict, side: str) -> list[str]:
    """Rating keys for one side, namespaced by discipline.

    Doubles pairs are rated as individuals and averaged, and ratings do not
    follow a player between disciplines. Both choices mirror the frozen Elo keys
    exactly, so a variant differs from the frozen feature only in the mechanisms
    under test.
    """
    keys = []
    for slot in (1, 2):
        value = row.get(f"{side}_player{slot}_id")
        if value is not None and value == value and str(value) != "nan":
            keys.append(f"{row['discipline']}|{int(value)}")
    return keys


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------
#
# Why this lives here and not in `src/data/preparation_workflow.py`
# -----------------------------------------------------------------
# Identical reasoning to `src/models/glicko2.py`. The prepared CSV
# `outputs/bwf/bwf_model_rows_2022_2025.csv` has its SHA-256 in the refreeze
# manifest, so regenerating it to carry research columns would invalidate the
# freeze for a candidate that is not being promoted. These features are a
# side-car keyed on `match_id` and joined onto the frozen rows.
#
# The leakage discipline of the data workflow is reproduced exactly:
#
#   1. sort by (date, match_id);
#   2. for every match on a date, read state that predates the date entirely;
#   3. accumulate the date's updates without applying them;
#   4. apply the whole date at once.
#
# The two time-dependent mechanisms - rating decay and season regression - are
# applied *lazily at read time* as a pure function of (stored rating, stored
# anchor date, current date). Both are multiplicative, so lazy evaluation is
# exact, and because they depend only on the current date they cannot smuggle in
# information from any match at all, let alone a same-day one.
#
# `assert_elo_variant_features_are_date_batched` turns that from a claim into a
# check.


@dataclass
class _State:
    rating: float = DEFAULT_ELO
    anchor: object = None
    last_played: object = None


def _read_rating(state: _State, date, config: EloConfig) -> float:
    """The player's rating as of ``date``, after decay and season regression.

    Depends only on stored state and the calendar, never on any match on
    ``date``. That is what makes the lazy evaluation leakage-safe.
    """
    rating = state.rating
    if state.anchor is None:
        return rating

    if config.season_regression != 1.0:
        boundaries = date.year - state.anchor.year
        if boundaries > 0:
            rating = DEFAULT_ELO + (rating - DEFAULT_ELO) * (
                config.season_regression**boundaries
            )

    if config.decay_half_life_days:
        away = (date - state.anchor).days - DECAY_GRACE_DAYS
        if away > 0:
            factor = 0.5 ** (away / config.decay_half_life_days)
            rating = DEFAULT_ELO + (rating - DEFAULT_ELO) * factor

    return rating


def replay_elo_variant(
    matches,
    config: EloConfig,
    target_column: str = "team1_win",
    emit_context: bool = False,
) -> dict:
    """Replay ``matches`` under one :class:`EloConfig` and return per-match state.

    Returns ``{match_id: {column: value}}``. The primary column is
    ``f"{config.name}_diff"`` - the pre-match rating difference, the direct
    analogue of `individual_elo_diff`.

    With ``emit_context`` the replay additionally emits three columns that do not
    belong to any one variant:

    - ``elo_staleness_mean`` - how stale the two sides' ratings are, on a 0-1
      scale, as the Elo analogue of Glicko's rating deviation;
    - ``elo_staleness_weighted_diff`` - the rating gap attenuated by that
      staleness, the Elo analogue of `glicko_certainty_weighted_rating_diff`,
      which was the single most valuable feature in the Glicko-2 study;
    - ``elo_recent_margin_diff`` - the difference in the two sides' mean signed
      margin over their last ten matches. This is the control that matters for
      question 1: if margin carries information, does it need to travel through
      a rating update at all, or is a plain margin-form feature enough?
    """
    ratings: dict[str, _State] = defaultdict(_State)
    recent_margins: dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
    reference = config.reference()
    records: dict[object, dict] = {}

    ordered = matches.sort_values(["date", "match_id"])

    for match_date, batch in ordered.groupby("date", sort=True):
        deltas: dict[str, float] = defaultdict(float)
        pending: list[tuple] = []

        for row in batch.sort_values("match_id").to_dict("records"):
            side1 = _side_player_keys(row, "team1")
            side2 = _side_player_keys(row, "team2")
            if not side1 or not side2:
                continue

            state1 = [ratings[key] for key in side1]
            state2 = [ratings[key] for key in side2]
            rating1 = _mean([_read_rating(s, match_date, config) for s in state1])
            rating2 = _mean([_read_rating(s, match_date, config) for s in state2])

            record = records.setdefault(row["match_id"], {})
            record[f"{config.name}_diff"] = rating1 - rating2

            if emit_context:
                # Staleness on a 0-1 scale: a player who last competed a year ago
                # scores 1. `rest_days_mean` already exists in the frozen feature
                # set, so this is deliberately *not* new information - it is a
                # different encoding of it, bounded and saturating, in the shape
                # that made RD useful for Glicko-2. §3.1 of the write-up reports
                # how redundant it turns out to be.
                def staleness(state: _State) -> float:
                    if state.last_played is None:
                        return 1.0
                    return min((match_date - state.last_played).days / 365.0, 1.0)

                stale1 = _mean([staleness(s) for s in state1])
                stale2 = _mean([staleness(s) for s in state2])
                mean_stale = (stale1 + stale2) / 2.0
                record["elo_staleness_mean"] = mean_stale
                record["elo_staleness_weighted_diff"] = (
                    (1.0 - mean_stale) * (rating1 - rating2)
                )

                def margin_form(keys: list[str]) -> float:
                    values = [
                        sum(recent_margins[key]) / len(recent_margins[key])
                        for key in keys
                        if recent_margins[key]
                    ]
                    return _mean(values) if values else 0.0

                record["elo_recent_margin_diff"] = (
                    margin_form(side1) - margin_form(side2)
                )

            team1_win = float(row[target_column])
            expected = 1.0 / (1.0 + 10.0 ** ((rating2 - rating1) / 400.0))

            multiplier = 1.0
            if config.margin_measure is not None:
                value = decisiveness(row, team1_win >= 0.5, config.margin_measure)
                multiplier = margin_multiplier(value, config.margin_gamma, reference)
                if config.correct_margin_autocorrelation:
                    gap = (rating1 - rating2) if team1_win >= 0.5 else (rating2 - rating1)
                    multiplier *= autocorrelation_correction(gap)

            surprise = team1_win - expected

            # The absence boost is per player, so team 1's gain need not equal
            # team 2's loss. Elo's zero-sum property is given up on purpose: the
            # claim being tested is that a returning player's rating should move
            # further on the same evidence than a fully graded opponent's, and
            # that claim is inherently asymmetric. Glicko-2 is not zero-sum
            # either, for the same reason.
            for keys, sign in ((side1, 1.0), (side2, -1.0)):
                for key in keys:
                    state = ratings[key]
                    boost = 1.0
                    if config.absence_k_beta and state.last_played is not None:
                        away = (match_date - state.last_played).days
                        boost = 1.0 + config.absence_k_beta * min(away / 365.0, 1.0)
                    elif config.absence_k_beta and state.last_played is None:
                        boost = 1.0 + config.absence_k_beta
                    deltas[key] += sign * config.k_factor * multiplier * boost * surprise

            pending.append((side1, side2, signed_margin(row)))

        # The whole date lands at once. Nothing above this line saw it.
        for key, delta in deltas.items():
            state = ratings[key]
            state.rating = _read_rating(state, match_date, config) + delta
            state.anchor = match_date

        for side1, side2, margin in pending:
            for key in side1:
                ratings[key].last_played = match_date
                if emit_context:
                    recent_margins[key].append(margin)
            for key in side2:
                ratings[key].last_played = match_date
                if emit_context:
                    recent_margins[key].append(-margin)

    return records


def replay_glicko_season_variant(
    matches,
    season_deviation_inflation: float,
    target_column: str = "team1_win",
) -> dict:
    """Glicko-2 with extra rating-deviation inflation at each season boundary.

    Question 3 of the study. Glicko-2 already widens RD during inactivity, but it
    treats 1 January as an ordinary day. If squads, form and fitness really do
    reset between BWF calendar years, then every rating should become less
    certain at the boundary regardless of whether the player was resting.

    The inflation is applied in quadrature - ``phi <- sqrt(phi^2 + c^2)``, capped
    at the default RD - which is how Glicko-2 adds uncertainty everywhere else,
    so this is the engine's own mechanism fired on a calendar event rather than a
    new one bolted on.

    `src/models/glicko2.py` is not modified: the engine is imported and driven
    from here.
    """
    engine = Glicko2()
    ordered = matches.sort_values(["date", "match_id"])
    records: dict[object, dict] = {}
    current_year: int | None = None

    for match_date, batch in ordered.groupby("date", sort=True):
        if current_year is not None and match_date.year > current_year:
            for _ in range(match_date.year - current_year):
                _inflate_all_deviations(engine, season_deviation_inflation)
        current_year = match_date.year

        period: dict[str, list[tuple[str, float]]] = {}

        for row in batch.sort_values("match_id").to_dict("records"):
            side1 = _side_player_keys(row, "team1")
            side2 = _side_player_keys(row, "team2")
            if not side1 or not side2:
                continue

            state1 = [engine.get(key) for key in side1]
            state2 = [engine.get(key) for key in side2]
            rating1 = _mean([s.rating for s in state1])
            rating2 = _mean([s.rating for s in state2])
            deviation1 = _mean([s.deviation for s in state1])
            deviation2 = _mean([s.deviation for s in state2])

            combined = math.sqrt((deviation1 / SCALE) ** 2 + (deviation2 / SCALE) ** 2)
            certainty = engine._g(combined)

            records[row["match_id"]] = {
                "glicko_season_deviation_mean": (deviation1 + deviation2) / 2.0,
                "glicko_season_deviation_diff": deviation1 - deviation2,
                "glicko_season_certainty_weighted_rating_diff": (
                    certainty * (rating1 - rating2)
                ),
            }

            won = float(row[target_column])
            for key in side1:
                for opponent in side2:
                    period.setdefault(key, []).append((opponent, won))
            for key in side2:
                for opponent in side1:
                    period.setdefault(key, []).append((opponent, 1.0 - won))

        if period:
            engine.update_period(period)
            engine.decay_absent(set(period))

    return records


def _inflate_all_deviations(engine: Glicko2, inflation: float) -> None:
    if inflation <= 0:
        return
    for key, state in list(engine.ratings.items()):
        widened = math.sqrt(state.deviation**2 + inflation**2)
        engine.ratings[key] = Rating(
            rating=state.rating,
            deviation=min(widened, DEFAULT_DEVIATION),
            volatility=state.volatility,
            matches=state.matches,
        )


CONTEXT_FEATURE_COLUMNS = [
    "elo_staleness_mean",
    "elo_staleness_weighted_diff",
    "elo_recent_margin_diff",
]

GLICKO_SEASON_FEATURE_COLUMNS = [
    "glicko_season_deviation_mean",
    "glicko_season_deviation_diff",
    "glicko_season_certainty_weighted_rating_diff",
]


def build_elo_variant_features(
    matches,
    configs: list[EloConfig],
    season_deviation_inflation: float | None = None,
    target_column: str = "team1_win",
    context_from: str | None = None,
) -> "pandas.DataFrame":  # noqa: F821
    """Replay ``matches`` under every config and return one side-car frame.

    ``matches`` needs `match_id`, `date`, `discipline`, the four player-id
    columns, the six game-score columns and ``target_column``. The returned frame
    is keyed on `match_id` and carries one column per config
    (``f"{config.name}_diff"``), the context columns if ``context_from`` names a
    config, and the Glicko season columns if an inflation is given.

    Every replay is independently date-batched, so the frame as a whole inherits
    the leakage rule from each of its parts.
    """
    import pandas as pd

    combined: dict[object, dict] = {}
    columns: list[str] = []

    for config in configs:
        emit_context = config.name == context_from
        records = replay_elo_variant(
            matches, config, target_column, emit_context=emit_context
        )
        columns.append(f"{config.name}_diff")
        if emit_context:
            columns.extend(CONTEXT_FEATURE_COLUMNS)
        for match_id, values in records.items():
            combined.setdefault(match_id, {}).update(values)

    if season_deviation_inflation is not None:
        records = replay_glicko_season_variant(
            matches, season_deviation_inflation, target_column
        )
        columns.extend(GLICKO_SEASON_FEATURE_COLUMNS)
        for match_id, values in records.items():
            combined.setdefault(match_id, {}).update(values)

    frame = pd.DataFrame(
        [{"match_id": match_id, **values} for match_id, values in combined.items()]
    )
    ordered_columns = ["match_id"] + [c for c in columns if c in frame.columns]
    return frame[ordered_columns].sort_values("match_id").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Leakage assertion
# ---------------------------------------------------------------------------
def _flip_same_day_results(frame, cut, target_column: str):
    """Flip outcome *and* scoreline for every match on ``cut`` bar the first.

    The scores are swapped as well as the target because these variants read the
    scoreline directly. Flipping only the label would leave the margin inputs
    untouched and the perturbation would be weaker than the thing it is meant to
    catch.
    """
    flipped = frame.copy()
    same_day = frame["date"] == cut
    keep_first = frame.loc[same_day, "match_id"].sort_values().iloc[0]
    to_flip = same_day & (frame["match_id"] != keep_first)

    flipped.loc[to_flip, target_column] = 1 - flipped.loc[to_flip, target_column]
    for game in (1, 2, 3):
        left = f"game{game}_team1"
        right = f"game{game}_team2"
        if left in flipped.columns and right in flipped.columns:
            a = flipped.loc[to_flip, left].copy()
            flipped.loc[to_flip, left] = flipped.loc[to_flip, right].to_numpy()
            flipped.loc[to_flip, right] = a.to_numpy()
    return flipped, to_flip


def assert_elo_variant_features_are_date_batched(
    matches,
    configs: list[EloConfig],
    season_deviation_inflation: float | None = None,
    sample_dates=None,
    target_column: str = "team1_win",
    context_from: str | None = None,
) -> dict:
    """Check the leakage rule empirically instead of trusting the loops.

    Two independent properties, because they fail in different ways:

    - **No future information.** Truncating the frame after a cut date must leave
      that date's feature values bit-identical.
    - **No same-day information.** Flipping the outcome *and scoreline* of every
      other match on a date must also leave that date's values bit-identical.
      This is the property a naive per-match update breaks, and truncation alone
      would not catch it.

    The lazy decay and season regression make the first property less obvious
    than it looks - a rating read at date D depends on D itself - so it is worth
    checking rather than arguing. Returns a summary dict; raises
    ``AssertionError`` on violation.
    """
    import numpy as np
    import pandas as pd

    frame = matches.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    unique_dates = np.array(sorted(frame["date"].unique()))
    if sample_dates is None:
        positions = [
            len(unique_dates) // 4,
            len(unique_dates) // 2,
            (3 * len(unique_dates)) // 4,
        ]
        sample_dates = [unique_dates[position] for position in positions]

    def build(subset):
        return build_elo_variant_features(
            subset,
            configs,
            season_deviation_inflation=season_deviation_inflation,
            target_column=target_column,
            context_from=context_from,
        ).set_index("match_id")

    full = build(frame)
    checks = []

    for cut in sample_dates:
        cut = pd.Timestamp(cut)
        on_date = frame.loc[frame["date"] == cut, "match_id"]
        on_date = on_date[on_date.isin(full.index)]
        if on_date.empty:
            continue

        truncated = build(frame.loc[frame["date"] <= cut])
        difference = full.loc[on_date].to_numpy() - truncated.loc[on_date].to_numpy()
        future_max = float(np.nanmax(np.abs(difference))) if difference.size else 0.0
        if future_max > 0.0:
            raise AssertionError(
                f"Elo variant features on {cut.date()} changed when later matches "
                f"were removed: a future result had leaked in (max delta {future_max})."
            )

        flipped, _ = _flip_same_day_results(frame, cut, target_column)
        perturbed = build(flipped.loc[flipped["date"] <= cut])
        difference = full.loc[on_date].to_numpy() - perturbed.loc[on_date].to_numpy()
        same_day_max = float(np.nanmax(np.abs(difference))) if difference.size else 0.0
        if same_day_max > 0.0:
            raise AssertionError(
                f"Elo variant features on {cut.date()} moved when other matches on "
                f"the same date were altered: within-day leakage (max delta "
                f"{same_day_max})."
            )

        checks.append(
            {
                "cut_date": str(cut.date()),
                "matches_on_date": int(len(on_date)),
                "max_absolute_change_when_future_removed": future_max,
                "max_absolute_change_when_same_day_outcomes_flipped": same_day_max,
            }
        )

    if not checks:
        raise AssertionError("No dates were checked; the leakage assertion is vacuous.")
    return {
        "rule": "features for a date use only strictly earlier dates",
        "columns_checked": [c for c in full.columns],
        "checks": checks,
    }


def mean_margin_multiplier(
    matches,
    measure: str,
    gamma: float,
    reference: float | None = None,
    target_column: str = "team1_win",
) -> float:
    """Average K multiplier the margin rule would apply over ``matches``.

    Needed for the k-matched control: if a margin-aware Elo helps only because
    its multipliers average to something other than 1 - that is, because it is a
    change of K wearing a costume - then a plain Elo at ``32 * this`` will
    capture the same gain. Depends only on scorelines, not on ratings, so it can
    be computed on the training seasons without a replay.
    """
    reference = reference if reference is not None else _MARGIN_REFERENCE[measure]
    values = [
        margin_multiplier(
            decisiveness(row, float(row[target_column]) >= 0.5, measure),
            gamma,
            reference,
        )
        for row in matches.to_dict("records")
    ]
    return float(sum(values) / len(values)) if values else 1.0
