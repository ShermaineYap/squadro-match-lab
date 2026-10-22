"""Glicko-2 ratings: a rating that carries its own uncertainty."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

SCALE = 173.7178
DEFAULT_RATING = 1500.0
DEFAULT_DEVIATION = 350.0
DEFAULT_VOLATILITY = 0.06

DEFAULT_TAU = 0.5

CONVERGENCE = 1e-6
MAX_ITERATIONS = 100


@dataclass
class Rating:
    """One player's Glicko-2 state, on the display scale."""

    rating: float = DEFAULT_RATING
    deviation: float = DEFAULT_DEVIATION
    volatility: float = DEFAULT_VOLATILITY
    matches: int = 0

    @property
    def mu(self) -> float:
        return (self.rating - DEFAULT_RATING) / SCALE

    @property
    def phi(self) -> float:
        return self.deviation / SCALE


@dataclass
class Glicko2:
    """Glicko-2 rating engine.

    Updates are applied per rating period rather than per match, which is what
    the specification assumes. The caller decides the period; for this project a
    period is one date batch, matching the leakage rule already used for Elo:
    every match on a date is rated from state that predates that date entirely.
    """

    tau: float = DEFAULT_TAU
    ratings: dict = field(default_factory=dict)

    def get(self, key) -> Rating:
        return self.ratings.get(key, Rating())

    def _g(self, phi: float) -> float:
        return 1.0 / math.sqrt(1.0 + 3.0 * phi**2 / math.pi**2)

    def _expected(self, mu: float, opponent_mu: float, opponent_phi: float) -> float:
        return 1.0 / (1.0 + math.exp(-self._g(opponent_phi) * (mu - opponent_mu)))

    def _new_volatility(self, phi: float, sigma: float, delta: float, v: float) -> float:
        """Illinois-algorithm root find from the specification, step 5."""
        a = math.log(sigma**2)
        phi_squared = phi**2

        def f(x: float) -> float:
            exp_x = math.exp(x)
            numerator = exp_x * (delta**2 - phi_squared - v - exp_x)
            denominator = 2.0 * (phi_squared + v + exp_x) ** 2
            return numerator / denominator - (x - a) / self.tau**2

        A = a
        if delta**2 > phi_squared + v:
            B = math.log(delta**2 - phi_squared - v)
        else:
            k = 1
            while f(a - k * self.tau) < 0 and k < MAX_ITERATIONS:
                k += 1
            B = a - k * self.tau

        fa, fb = f(A), f(B)
        for _ in range(MAX_ITERATIONS):
            if abs(B - A) <= CONVERGENCE:
                break
            C = A + (A - B) * fa / (fb - fa)
            fc = f(C)
            if fc * fb <= 0:
                A, fa = B, fb
            else:
                fa /= 2.0
            B, fb = C, fc
        return math.exp(A / 2.0)

    def update_period(self, results: dict[object, list[tuple[object, float]]]) -> None:
        """Apply one rating period.

        ``results`` maps a player key to the matches they played in the period,
        each as ``(opponent key, score)`` with score 1.0 for a win and 0.0 for a
        loss. Every player's new state is computed from the state at the start of
        the period, so results within a period cannot influence one another.
        """
        snapshot = {key: self.get(key) for key in results}
        opponents = {
            opponent
            for played in results.values()
            for opponent, _score in played
        }
        for opponent in opponents:
            snapshot.setdefault(opponent, self.get(opponent))

        updated: dict[object, Rating] = {}
        for key, played in results.items():
            current = snapshot[key]
            if not played:
                updated[key] = self._decay(current)
                continue

            mu, phi, sigma = current.mu, current.phi, current.volatility

            v_inverse = 0.0
            delta_sum = 0.0
            for opponent_key, score in played:
                opponent = snapshot[opponent_key]
                g = self._g(opponent.phi)
                expected = self._expected(mu, opponent.mu, opponent.phi)
                v_inverse += g**2 * expected * (1.0 - expected)
                delta_sum += g * (score - expected)

            if v_inverse <= 0:
                updated[key] = self._decay(current)
                continue

            v = 1.0 / v_inverse
            delta = v * delta_sum

            sigma_new = self._new_volatility(phi, sigma, delta, v)
            phi_star = math.sqrt(phi**2 + sigma_new**2)
            phi_new = 1.0 / math.sqrt(1.0 / phi_star**2 + 1.0 / v)
            mu_new = mu + phi_new**2 * delta_sum

            updated[key] = Rating(
                rating=mu_new * SCALE + DEFAULT_RATING,
                deviation=phi_new * SCALE,
                volatility=sigma_new,
                matches=current.matches + len(played),
            )

        self.ratings.update(updated)

    def _decay(self, current: Rating) -> Rating:
        """Inactivity widens the rating deviation: the rating is less known now."""
        phi_star = math.sqrt(current.phi**2 + current.volatility**2)
        return Rating(
            rating=current.rating,
            deviation=min(phi_star * SCALE, DEFAULT_DEVIATION),
            volatility=current.volatility,
            matches=current.matches,
        )

    def decay_absent(self, active: set) -> None:
        """Widen the deviation of every rated player who did not play."""
        for key in list(self.ratings):
            if key not in active:
                self.ratings[key] = self._decay(self.ratings[key])

    def expected_score(self, key, opponent_key) -> float:
        """Win probability for ``key``, accounting for both players' uncertainty."""
        player, opponent = self.get(key), self.get(opponent_key)
        combined = math.sqrt(player.phi**2 + opponent.phi**2)
        return self._expected(player.mu, opponent.mu, combined)


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------
#
# Why the builder lives here and not in `src/data/preparation_workflow.py`
# -----------------------------------------------------------------------
# The Elo features are produced by `build_date_batched_bwf_features` inside the
# data workflow, and the prepared CSV it writes
# (`outputs/bwf/bwf_model_rows_2022_2025.csv`) has its SHA-256 recorded in the
# refreeze manifest. Regenerating that file to add research features would
# invalidate the freeze for the sake of a candidate that is not being promoted.
#
# The Glicko-2 features are therefore built as a side-car keyed on `match_id`
# and joined onto the frozen rows. The *discipline* the data workflow enforces
# is reproduced here exactly rather than approximately:
#
#   1. sort by (date, match_id);
#   2. for every match on a date, read state that predates the date entirely;
#   3. queue the date's results;
#   4. close the rating period only once the whole date has been read.
#
# `assert_features_are_date_batched` below turns that from a claim into a check.

_DOUBLES_DISCIPLINES = frozenset({"MD", "WD", "XD"})

GLICKO_FEATURE_COLUMNS = [
    # The direct analogue of `individual_elo_diff`. Included so that the rest of
    # the block can be judged against a rating difference the model already
    # effectively has, rather than against nothing.
    "glicko_rating_diff",
    # Glicko-2's own win probability. It is not a monotone transform of the
    # rating difference: the g(phi) term flattens the logistic slope when either
    # side is poorly known, so the same 100-point gap implies a smaller edge
    # between two uncertain players. Handing this to the model saves it from
    # discovering that interaction from trees on two separate columns.
    "glicko_expected_score",
    # Each side's rating deviation, kept separately rather than only as a
    # difference. A match between two well-established players and a match
    # between two unknowns share an RD difference of zero but are very different
    # prediction problems, and only the levels distinguish them.
    "glicko_team1_deviation",
    "glicko_team2_deviation",
    # Mean RD: how well known the matchup is overall. This is the feature that
    # lets the model discount the rating gap globally - the "early-2022 ratings
    # carry no history" effect that motivated the whole experiment.
    "glicko_deviation_mean",
    # RD asymmetry: which side is the better-known quantity. The archetype is a
    # returning player with a stale high rating against a graded opponent, where
    # the rating difference overstates the favourite.
    "glicko_deviation_diff",
    # The certainty-weighted rating difference the engine docstring asks for:
    # the rating gap attenuated by Glickman's g() using the two sides' combined
    # deviation. A 100-point gap between two settled players survives almost
    # intact; the same gap between two unknowns is shrunk towards zero. A tree
    # can only approximate this product by splitting on the two columns, so
    # supplying it directly is a real addition rather than a restatement.
    "glicko_certainty_weighted_rating_diff",
    # Volatility is cheap - the engine already tracks it - and measures how
    # erratic recent results have been, which is a different quantity from how
    # much history exists. Included so the study can report whether it earns its
    # place; the honest prior is that it will not.
    "glicko_volatility_mean",
    "glicko_volatility_diff",
]


def _side_player_keys(row: dict, side: str) -> list[str]:
    """Rating keys for one side, namespaced by discipline.

    Doubles pairs are rated as individuals and averaged, which is how the
    existing `individual_elo_diff` is built. Namespacing by discipline matches
    the Elo keys too: a singles specialist's rating should not follow them into
    a mixed-doubles draw.
    """
    keys = []
    for slot in (1, 2):
        value = row.get(f"{side}_player{slot}_id")
        if value is not None and value == value and str(value) != "nan":
            keys.append(f"{row['discipline']}|{int(value)}")
    return keys


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def build_glicko_features(
    matches,
    target_column: str = "team1_win",
) -> "pandas.DataFrame":  # noqa: F821
    """Replay ``matches`` through Glicko-2 and emit pre-match features per match.

    ``matches`` needs `match_id`, `date`, `discipline`, the four player-id
    columns and ``target_column``. The frame is replayed in date order; the
    returned frame carries one row per rated match, keyed on `match_id`, holding
    :data:`GLICKO_FEATURE_COLUMNS`.

    The leakage rule is structural, not incidental. Features for every match on
    a date are read from the engine before any of that date's results have been
    applied, and the whole date is applied as a single Glicko-2 rating period -
    which is also what the specification assumes a period is. Two matches on the
    same day therefore cannot inform one another, which matters because the
    source data does not record within-day order and inventing one would be an
    unsupported assumption.
    """
    import pandas as pd

    engine = Glicko2()
    ordered = matches.sort_values(["date", "match_id"])
    records: list[dict] = []

    for _match_date, batch in ordered.groupby("date", sort=True):
        period: dict[str, list[tuple[str, float]]] = {}

        for row in batch.sort_values("match_id").to_dict("records"):
            side1 = _side_player_keys(row, "team1")
            side2 = _side_player_keys(row, "team2")
            if not side1 or not side2:
                continue

            state1 = [engine.get(key) for key in side1]
            state2 = [engine.get(key) for key in side2]

            rating1 = _mean([state.rating for state in state1])
            rating2 = _mean([state.rating for state in state2])
            deviation1 = _mean([state.deviation for state in state1])
            deviation2 = _mean([state.deviation for state in state2])
            volatility1 = _mean([state.volatility for state in state1])
            volatility2 = _mean([state.volatility for state in state2])

            # Glickman's g() applied to the two sides' combined deviation. This
            # is exactly the factor by which Glicko-2 flattens its own expected
            # score when a matchup is poorly known, reused here as a shrinkage
            # weight on the rating gap.
            combined_phi = math.sqrt(
                (deviation1 / SCALE) ** 2 + (deviation2 / SCALE) ** 2
            )
            certainty = engine._g(combined_phi)

            expected = _mean([
                engine.expected_score(a, b) for a in side1 for b in side2
            ])

            records.append(
                {
                    "match_id": row["match_id"],
                    "glicko_rating_diff": rating1 - rating2,
                    "glicko_expected_score": expected,
                    "glicko_team1_deviation": deviation1,
                    "glicko_team2_deviation": deviation2,
                    "glicko_deviation_mean": (deviation1 + deviation2) / 2.0,
                    "glicko_deviation_diff": deviation1 - deviation2,
                    "glicko_certainty_weighted_rating_diff": (
                        certainty * (rating1 - rating2)
                    ),
                    "glicko_volatility_mean": (volatility1 + volatility2) / 2.0,
                    "glicko_volatility_diff": volatility1 - volatility2,
                }
            )

            # Queued, not applied. The date's results are handed to the engine
            # only after every match on the date has been read.
            won = float(row[target_column])
            for key in side1:
                for opponent in side2:
                    period.setdefault(key, []).append((opponent, won))
            for key in side2:
                for opponent in side1:
                    period.setdefault(key, []).append((opponent, 1.0 - won))

        if period:
            engine.update_period(period)
            # Everyone who did not play today becomes a little less well known.
            # This is the mechanism that marks a returning player as uncertain,
            # and it is the reason RD is not simply an inverse match count.
            engine.decay_absent(set(period))

    return pd.DataFrame(records, columns=["match_id"] + GLICKO_FEATURE_COLUMNS)


def assert_features_are_date_batched(
    matches,
    sample_dates=None,
    target_column: str = "team1_win",
) -> dict:
    """Check the leakage rule empirically instead of trusting the loop.

    Two independent properties are checked, because they fail in different ways:

    - **No future information.** Truncating the frame after a cut date must
      leave that date's feature values bit-identical. If any later match had
      leaked in, they would move.
    - **No same-day information.** Flipping the outcome of every *other* match
      on a date must also leave that date's feature values bit-identical. This
      is the property a naive per-match update would break, and truncation alone
      would not catch it.

    Returns a summary dict so a caller can record the evidence rather than just
    assert and forget. Raises ``AssertionError`` on violation.
    """
    import numpy as np
    import pandas as pd

    frame = matches.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    unique_dates = np.array(sorted(frame["date"].unique()))
    if sample_dates is None:
        # Spread the probes across the replay: an early date, where almost every
        # rating is still at its default, is a weak test on its own.
        positions = [len(unique_dates) // 4, len(unique_dates) // 2,
                     (3 * len(unique_dates)) // 4]
        sample_dates = [unique_dates[position] for position in positions]

    full = build_glicko_features(frame, target_column).set_index("match_id")
    checks = []

    for cut in sample_dates:
        cut = pd.Timestamp(cut)
        on_date = frame.loc[frame["date"] == cut, "match_id"]
        # A match whose player structure is unusable produces no feature row, so
        # it cannot be compared; the check is over the rows that were rated.
        on_date = on_date[on_date.isin(full.index)]
        if on_date.empty:
            continue

        truncated = build_glicko_features(
            frame.loc[frame["date"] <= cut], target_column
        ).set_index("match_id")
        difference = (
            full.loc[on_date].to_numpy() - truncated.loc[on_date].to_numpy()
        )
        future_max = float(np.nanmax(np.abs(difference))) if difference.size else 0.0
        if future_max > 0.0:
            raise AssertionError(
                f"Glicko features on {cut.date()} changed when later matches were "
                f"removed: a future result had leaked in (max delta {future_max})."
            )

        # Flip every other match on the cut date. Any within-day dependence
        # would move the surviving rows.
        flipped = frame.copy()
        same_day = (flipped["date"] == cut)
        keep_first = flipped.loc[same_day, "match_id"].sort_values().iloc[0]
        to_flip = same_day & (flipped["match_id"] != keep_first)
        flipped.loc[to_flip, target_column] = 1 - flipped.loc[to_flip, target_column]
        perturbed = build_glicko_features(
            flipped.loc[flipped["date"] <= cut], target_column
        ).set_index("match_id")
        difference = (
            full.loc[on_date].to_numpy() - perturbed.loc[on_date].to_numpy()
        )
        same_day_max = float(np.nanmax(np.abs(difference))) if difference.size else 0.0
        if same_day_max > 0.0:
            raise AssertionError(
                f"Glicko features on {cut.date()} moved when other matches on the "
                f"same date were altered: within-day leakage (max delta "
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
    return {"rule": "features for a date use only strictly earlier dates", "checks": checks}
