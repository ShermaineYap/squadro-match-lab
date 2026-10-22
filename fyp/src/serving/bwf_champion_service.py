"""BWF champion serving: symmetric prediction and explanation for a fixture not yet
played.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.decision_support.match_explanation import (
    load_phrasing as load_factor_phrasing,
    render_declared_sentence,
    side_the_value_favours,
)
from src.serving.bwf_glicko_state import (
    assert_prediction_date, glicko_features_for_sides,
)
from src.serving.bwf_state import build_prematch_features

MODEL_DISPLAY_NAME = "BWF Symmetric Glicko Logistic Regression, uncalibrated"
PREDICTION_FUNCTION_VERSION = "symmetric_margin_averaging.v1"
SYMMETRY_TOLERANCE = 1e-9

PRODUCTION_MODEL_VERSION = "3.0-symmetric-glicko-lr-uncalibrated"
BUNDLE_INTERNAL_VERSION = "3.0-candidate-symmetric-uncalibrated"

DISCLAIMER = (
    "This is pre-match decision support based on historical patterns. It does "
    "not guarantee a winner and must not be used as an autonomous or gambling "
    "decision."
)

UNCERTAINTY_STATUS_SENTENCE = (
    "The served model is uncalibrated: no fitted calibrator is applied, no "
    "conformal prediction set exists, and no validated reliability operating "
    "point has been established. Descriptive calibration evidence does exist "
    "from the locked audit, namely Brier 0.1788 and ECE 0.0142 on 2025 and "
    "Brier 0.1849 and ECE 0.0382 on 2026. That evidence describes the model's "
    "probabilities in aggregate and is not an operating point for this fixture."
)


def build_champion_feature_row(
    elo_state: dict[str, Any],
    glicko_state: dict[str, Any],
    *,
    team1_player_ids: Iterable[int],
    team2_player_ids: Iterable[int],
    discipline: str,
    prediction_date: Any,
    tier_category: str,
    round_name: str,
    is_qualification: bool,
    host_country: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """One fixture's features, plus what the caller should be told about staleness."""
    team1 = [int(p) for p in team1_player_ids]
    team2 = [int(p) for p in team2_player_ids]
    staleness = assert_prediction_date(glicko_state, prediction_date)
    features = build_prematch_features(
        elo_state,
        team1_player_ids=team1, team2_player_ids=team2,
        discipline=discipline, prediction_date=prediction_date,
        tier_category=tier_category, round_name=round_name,
        is_qualification=is_qualification, host_country=host_country,
        enforce_after_snapshot=True,
    )
    features.update(glicko_features_for_sides(
        glicko_state, discipline=discipline,
        team1_player_ids=team1, team2_player_ids=team2,
    ))
    return features, staleness


def _frame(bundle: dict[str, Any], features: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([features])[bundle["feature_columns"]]


def predict_symmetric(bundle: dict[str, Any], features: dict[str, Any]) -> dict[str, Any]:
    """The served probability, with its own symmetry proof attached."""
    model = bundle["model"]
    frame = _frame(bundle, features)
    margin = float(model.decision_function(frame)[0])
    probability = float(model.predict_proba(frame)[0, 1])

    swapped = model.roles.swap(frame)
    reverse = float(model.predict_proba(swapped)[0, 1])
    violation = abs(probability - (1.0 - reverse))
    if violation > SYMMETRY_TOLERANCE:
        raise AssertionError(
            f"symmetry violated by {violation:.3e}; the served model must satisfy "
            "P(A beats B) = 1 - P(B beats A)"
        )
    return {
        "model_version": bundle.get("model_version"),
        "model_display_name": MODEL_DISPLAY_NAME,
        "prediction_function_version": bundle.get(
            "prediction_function_version", PREDICTION_FUNCTION_VERSION
        ),
        "symmetric_margin_log_odds": margin,
        "team1_win_probability": probability,
        "team2_win_probability": 1.0 - probability,
        "predicted_label": "Team 1 wins" if probability >= 0.5 else "Team 2 wins",
        "order_symmetry_violation": violation,
        "calibration": "uncalibrated",
    }


def _linear_contributions(pipeline, frame: pd.DataFrame) -> tuple[dict[str, float], float]:
    preprocess = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    transformed = preprocess.transform(frame)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    products = (transformed * model.coef_[0])[0]
    names = preprocess.get_feature_names_out()

    grouped: dict[str, float] = {}
    for index, raw in enumerate(names):
        stem = raw.split("__", 1)[1] if "__" in raw else raw
        source = next((c for c in frame.columns if stem == c or stem.startswith(f"{c}_")), stem)
        grouped[source] = grouped.get(source, 0.0) + float(products[index])
    return grouped, float(model.intercept_[0])


def explain_symmetric(bundle: dict[str, Any], features: dict[str, Any]) -> dict[str, Any]:
    """Contributions to the symmetric log-odds that is actually displayed."""
    model = bundle["model"]
    roles = model.roles
    frame = _frame(bundle, features)

    forward, _ = _linear_contributions(model.estimator_, frame)
    reverse_raw, _ = _linear_contributions(model.estimator_, roles.swap(frame))

    # Permutation only. The sign is carried by the subtraction below.
    reverse = dict(reverse_raw)
    for left, right in roles.side_pairs:
        if left in reverse_raw and right in reverse_raw:
            reverse[left], reverse[right] = reverse_raw[right], reverse_raw[left]

    contributions = {
        name: (forward.get(name, 0.0) - reverse.get(name, 0.0)) / 2.0
        for name in frame.columns
    }
    total = sum(contributions.values())
    margin = float(model.decision_function(frame)[0])
    probability = float(model.predict_proba(frame)[0, 1])
    reconstructed = 1.0 / (1.0 + math.exp(-total))

    additivity_error = abs(total - margin)
    probability_error = abs(reconstructed - probability)
    if additivity_error > SYMMETRY_TOLERANCE or probability_error > SYMMETRY_TOLERANCE:
        raise AssertionError(
            f"explanation does not reconstruct the displayed probability "
            f"(margin error {additivity_error:.3e}, probability error {probability_error:.3e})"
        )

    ranked = sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)
    return {
        "units": "log-odds",
        "units_note": (
            "Each value is a contribution to the symmetric log-odds. They sum to "
            "the margin whose sigmoid is the displayed probability. They are not "
            "probability points."
        ),
        "base_value_log_odds": 0.0,
        "base_value_note": (
            "Zero, which is a 50% probability. A symmetric model with no evidence "
            "favours neither side."
        ),
        "contributions": [{"feature": name, "log_odds": value} for name, value in ranked],
        "reconstructed_margin": total,
        "reconstructed_probability": reconstructed,
        "additivity_error": additivity_error,
        "probability_error": probability_error,
        "explains_displayed_probability": True,
    }


def assert_symmetric(bundle: dict[str, Any], rows: list[dict[str, Any]]) -> float:
    """Largest |P(A>B) - (1 - P(B>A))| over the supplied fixtures."""
    model = bundle["model"]
    frame = pd.DataFrame(rows)[bundle["feature_columns"]]
    forward = model.predict_proba(frame)[:, 1]
    reverse = model.predict_proba(model.roles.swap(frame))[:, 1]
    return float(np.max(np.abs(forward - (1.0 - reverse))))


CHAMPION_UNCERTAINTY_STATUS = {
    "calibrator_applied": False,
    "conformal_available": False,
    "reliability_operating_points_available": False,
    "descriptive_calibration_evidence_available": True,
    "status": (
        "uncalibrated; no conformal set and no validated operating point, "
        "descriptive audit calibration evidence available"
    ),
    "descriptive_evidence": {
        "source": "outputs/research/bwf_champion_audit/champion_audit_result.json",
        "2025": {"brier": 0.1788, "ece": 0.0142},
        "2026": {"brier": 0.1849, "ece": 0.0382},
        "note": (
            "Aggregate over a whole season. It describes the model's "
            "probabilities in general; it is not a per-fixture guarantee and "
            "must not be read as one."
        ),
    },
    "explanation": (
        "The served model applies no fitted calibrator, has no conformal "
        "prediction set and has no validated reliability operating point. The "
        "earlier XGBoost baseline's conformal thresholds and operating points "
        "were calibrated on a different model over a different feature set, so "
        "they do not describe the served model and are not reused. What the "
        "served model does have is descriptive calibration evidence from the "
        "locked audit, Brier and ECE on both seasons, which is reported as "
        "descriptive rather than as a guarantee."
    ),
}


def predict_future_bwf_match_champion(
    bundle: dict[str, Any],
    elo_state: dict[str, Any],
    glicko_state: dict[str, Any],
    *,
    team1_player_ids: Iterable[int],
    team2_player_ids: Iterable[int],
    discipline: str,
    prediction_date: Any,
    tier_category: str,
    round_name: str,
    is_qualification: bool,
    host_country: str,
    include_explanation: bool = False,
) -> dict[str, Any]:
    """The full served payload for one future fixture, shaped like the baseline response.

    Keys the dashboard already reads are preserved so the interface does not have
    to special-case a version. What is added is the champion-specific material: the
    symmetric margin, the measured symmetry violation, the state-age disclosure
    and an explicit statement that no calibrated uncertainty is available.
    """
    from src.serving.bwf_service import _host_country_options
    from src.evaluation.robustness import out_of_distribution_flags

    features, staleness = build_champion_feature_row(
        elo_state, glicko_state,
        team1_player_ids=team1_player_ids, team2_player_ids=team2_player_ids,
        discipline=discipline, prediction_date=prediction_date,
        tier_category=tier_category, round_name=round_name,
        is_qualification=is_qualification, host_country=host_country,
    )
    prediction = predict_symmetric(bundle, features)

    host_countries = _host_country_options()
    payload = {
        "stream": "bwf",
        "prediction": prediction["predicted_label"],
        "predicted_class_index": 1 if prediction["team1_win_probability"] >= 0.5 else 0,
        "model_version": PRODUCTION_MODEL_VERSION,
        "bundle_model_version": bundle.get("model_version", BUNDLE_INTERNAL_VERSION),
        "model_display_name": MODEL_DISPLAY_NAME,
        "prediction_function_version": prediction["prediction_function_version"],
        # ``probabilities`` is the name the dashboard reads; ``class_probabilities``
        # is kept as the alias so nothing that already consumed it breaks.
        "probabilities": {
            "Team 2 wins": prediction["team2_win_probability"],
            "Team 1 wins": prediction["team1_win_probability"],
        },
        "class_probabilities": {
            "Team 2 wins": prediction["team2_win_probability"],
            "Team 1 wins": prediction["team1_win_probability"],
        },
        "uncertainty_status": UNCERTAINTY_STATUS_SENTENCE,
        "disclaimer": DISCLAIMER,
        "human_review_required": True,
        "cold_start": bool(
            features.get("team1_cold_start") == 1 or features.get("team2_cold_start") == 1
        ),
        "team1_win_probability": prediction["team1_win_probability"],
        "symmetric_margin_log_odds": prediction["symmetric_margin_log_odds"],
        "order_symmetry_violation": prediction["order_symmetry_violation"],
        "order_symmetry_guaranteed": True,
        "calibration": "uncalibrated",
        "uncertainty": dict(CHAMPION_UNCERTAINTY_STATUS),
        "conformal": None,
        "glicko_state": staleness,
        "out_of_distribution_flags": out_of_distribution_flags(
            features, bundle.get("reference_quantiles", {}),
            categorical_options={"host_country": host_countries} if host_countries else None,
        ),
        "host_country_known": (
            str(features.get("host_country")) in {str(v) for v in host_countries}
            if host_countries else None
        ),
        # Carried over from the baseline unchanged. An unseen venue is flagged rather than
        # refused - the tour visits new countries every season - but it must not
        # be flagged silently, so the caller is told what the boundary means.
        "applicability_note": (
            None
            if (not host_countries
                or str(features.get("host_country")) in {str(v) for v in host_countries})
            else (
                f"{features.get('host_country')!r} is not among the host "
                "countries this model was fitted on, so that context feature "
                "contributes nothing to the estimate. The prediction still "
                "rests on the ratings, recent form and head-to-head record, but "
                "it is an extrapolation on venue and should be read as one."
            )
        ),
        "features": {k: (float(v) if isinstance(v, (int, float)) else v)
                     for k, v in features.items()},
        "state_snapshot_as_of": elo_state.get("snapshot_as_of"),
    }
    if include_explanation:
        payload["explanation"] = explain_symmetric(bundle, features)
    return payload


def run_bwf_champion_what_if(
    bundle: dict[str, Any],
    elo_state: dict[str, Any],
    glicko_state: dict[str, Any],
    *,
    base_request: dict[str, Any],
    scenarios: list[dict[str, Any]],
) -> dict[str, Any]:
    """Context what-if scenarios, on the same model and the same rating state.

    The point of a what-if is that only the thing being varied changes. Scoring
    the baseline with one model and the variants with another - or against a
    different rating snapshot - would attribute the difference between two
    models to the scenario.
    """
    baseline = predict_future_bwf_match_champion(bundle, elo_state, glicko_state, **base_request)
    rows = []
    for scenario in scenarios:
        request = {**base_request, **scenario}
        result = predict_future_bwf_match_champion(bundle, elo_state, glicko_state, **request)
        rows.append({
            "scenario": scenario,
            "team1_win_probability": result["team1_win_probability"],
            "change_from_baseline": (
                result["team1_win_probability"] - baseline["team1_win_probability"]
            ),
            "symmetric_margin_log_odds": result["symmetric_margin_log_odds"],
        })
    # Request fields are named for the caller; features are named for the model.
    request_to_feature = {"round_name": "round", "discipline": "discipline",
                          "tier_category": "tier_category",
                          "is_qualification": "is_qualification",
                          "host_country": "host_country"}
    invariant = set(bundle["model"].roles.invariant)
    varied = {request_to_feature.get(key, key) for scenario in scenarios for key in scenario}
    inert = sorted(varied & invariant)
    largest = max((abs(row["change_from_baseline"]) for row in rows), default=0.0)

    return {
        "model_version": bundle.get("model_version"),
        "model_display_name": MODEL_DISPLAY_NAME,
        "glicko_state_sha_source": glicko_state.get("state_sha256"),
        "baseline": {
            "team1_win_probability": baseline["team1_win_probability"],
            "symmetric_margin_log_odds": baseline["symmetric_margin_log_odds"],
        },
        "scenarios": rows,
        "largest_absolute_change": largest,
        "inert_fields": inert,
        "inert_explanation": (
            "Context fields cannot move this model's probability, and the zeros "
            "below are "
            "exact rather than approximate. Discipline, tier, round, qualification "
            "status and host country are unchanged when the two sides are "
            "exchanged, so they contribute the same amount to both orientations "
            "and cancel completely in m_sym = [m(A,B) - m(B,A)] / 2. This is the "
            "arithmetic price of exact order symmetry in an additive model, not a "
            "fault: any term that survived the swap would also break the guarantee "
            "that P(A beats B) = 1 - P(B beats A). Under the earlier XGBoost "
            "baseline these fields did move the answer, and that answer also "
            "depended on which player was typed first."
        ) if inert else None,
        "note": (
            "Every row is scored by the same served bundle against the same Glicko "
            "state as the baseline; only the named context field differs."
        ),
    }


_SUBGROUP_CACHE: dict[str, Any] = {}


def _audit_subgroups() -> dict[str, list[dict[str, Any]]]:
    from pathlib import Path

    if _SUBGROUP_CACHE:
        return _SUBGROUP_CACHE
    root = Path(__file__).resolve().parents[2] / "outputs/research/bwf_champion_audit"
    for season in ("2025", "2026"):
        path = root / f"champion_audit_{season}_subgroup_metrics.csv"
        if path.exists():
            _SUBGROUP_CACHE[season] = pd.read_csv(path).to_dict("records")
    return _SUBGROUP_CACHE


_AUDIT_OVERALL_CACHE: dict[str, Any] = {}


def _audit_overall() -> dict[str, Any]:
    """Overall audit metrics for the most recent audited season."""
    import json
    from pathlib import Path

    if _AUDIT_OVERALL_CACHE:
        return _AUDIT_OVERALL_CACHE
    path = (
        Path(__file__).resolve().parents[2]
        / "outputs/champion_evidence/bwf_champion_freeze_manifest.json"
    )
    if not path.exists():
        return {}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    results = manifest.get("audit_results") or {}
    if not results:
        return {}
    season = max(results)
    figures = results[season]
    _AUDIT_OVERALL_CACHE.update({
        "season": season,
        "accuracy": float(figures["accuracy"]),
        "roc_auc": float(figures["roc_auc"]),
        "ece": float(figures["ece"]),
        "brier": float(figures["brier"]),
        "matches_evaluated": int(figures["rows"]),
        "source": f"{season} temporal audit, {int(figures['rows']):,} unseen matches",
        "model_version": PRODUCTION_MODEL_VERSION,
        "note": (
            "Descriptive. An overall figure describes almost no individual "
            "fixture, which is why the matched segments below govern."
        ),
    })
    return _AUDIT_OVERALL_CACHE


MEASUREMENT_BOUNDARY = (
    "Descriptive subgroup evidence from the locked audit, not a reliability "
    "operating point. The served model has no fitted calibrator, no conformal "
    "prediction set and no validated operating point, so these figures do not "
    "permit the interface to accept or decline a call. They describe how the "
    "frozen model scored on matches like this one in the audited seasons, and "
    "nothing stronger."
)


def champion_descriptive_reliability(
    *, discipline: str, tier_category: str, is_qualification: bool, cold_start: bool
) -> dict[str, Any]:
    """Measured accuracy for the segments this fixture belongs to - descriptive only.

    This is not a reliability *operating point*. The baseline had one: thresholds fitted so
    a stated coverage was met, which licensed the interface to decline a call.
    The champion has none, and inventing one from audit subgroups would be exactly the
    fitted-on-the-audit move the whole study was governed to avoid.

    What this returns is the audit's own subgroup table, filtered to the segments
    that describe this match, labelled as description. It answers "how did the
    model do on matches like this one, in the two seasons we measured" and
    nothing stronger.

    Each row carries a ``segment`` key that names its season -
    ``2026:discipline=MS`` - because the same subgroup is measured once per
    audited season and a key without the season collides between them. React
    keys the segment list on it, and two rows sharing a key is how one season's
    figures silently replaced the other's.
    """
    seasons = _audit_subgroups()
    wanted = {
        "discipline": str(discipline),
        "tier_category": str(tier_category),
        "is_qualification": str(bool(is_qualification)),
        "either_team_cold_start": "1" if cold_start else "0",
    }
    matched: list[dict[str, Any]] = []
    for season, rows in seasons.items():
        for row in rows:
            variable = str(row["subgroup_variable"])
            if variable not in wanted:
                continue
            if str(row["subgroup_value"]) != wanted[variable]:
                continue
            value = str(row["subgroup_value"])
            matched.append({
                "segment": f"{season}:{variable}={value}",
                "season": season,
                "variable": variable,
                "value": value,
                "matches_evaluated": int(row["rows"]),
                "accuracy": float(row["accuracy"]),
                "roc_auc": float(row["roc_auc"]),
                "ece": float(row["ece"]),
                # Restated per row rather than only once at the top: a single
                # segment lifted out of this list into a UI or a report carries
                # its own statement of what it does not license.
                "is_operating_point": False,
            })
    matched.sort(key=lambda item: (item["season"], item["variable"], item["value"]))
    weakest = min(matched, key=lambda r: r["accuracy"]) if matched else None
    return {
        "kind": "descriptive_audit_subgroup_evidence",
        "overall_level": "descriptive_only",
        "is_operating_point": False,
        "measurement_boundary": MEASUREMENT_BOUNDARY,
        "source": "outputs/research/bwf_champion_audit/champion_audit_{season}_subgroup_metrics.csv",
        "segments_matched": matched,
        "weakest_segment": weakest,
        "measured_performance": _audit_overall(),
        "model_version": PRODUCTION_MODEL_VERSION,
        "statement": (
            "Descriptive only. These are measured accuracies for segments this "
            "fixture belongs to, from the two locked audit seasons. The served "
            "model has no validated reliability operating point, so nothing "
            "here licenses the interface to accept or decline a call."
        ),
        "unavailable_reason": None if matched else (
            "No audit subgroup matched this fixture's discipline, tier, "
            "qualification status or cold-start status, so no descriptive "
            "reliability evidence is available for it."
        ),
    }


def _endstop(sentence: str) -> str:
    text = sentence.strip()
    return text if text.endswith((".", "!", "?")) else f"{text}."


def _readable_factor(
    name: str,
    log_odds: float,
    favours_team1: bool,
    entry: dict[str, Any],
    value: Any,
    team1: str,
    team2: str,
) -> dict[str, Any]:
    """One factor, named and phrased from ``rules/bwf_factor_phrasing.json``.

    This used to write its own wording: the label was the column name with its
    underscores taken out, and the sentence, the meaning and the limit were one
    hard-coded set reused for all twenty-five features. That put the wording a
    reader sees outside the artefact that exists to make it auditable, and it
    named the sides "Team 1" and "Team 2" rather than the people in the fixture.

    Everything readable now comes from the declared entry. The exact symmetric
    log-odds contribution is untouched by any of it - it is computed upstream and
    only reported here, and the sentence states it rather than replacing it.
    """
    label = entry["label"]
    leader = side_the_value_favours(name, value, team1, team2)

    # A feature that is absent, level or switched off takes its declared
    # negligible wording even where it moved the estimate: "{team1} has no
    # rating history" would be a false statement about a player who has one.
    absent = leader is None and (value is None or value == 0 or value is False)
    template = entry["when_negligible"] if absent else entry["template"]
    declared = render_declared_sentence(template, leader, value, team1, team2)

    side = team1 if log_odds > 0 else team2
    return {
        "feature": name,
        "label": label,
        # Which side the *measurement* favours, where the feature is a
        # difference. Reported alongside the side the *contribution* favours
        # because for three features they are not the same side: the champion
        # carries negative coefficients on the form columns, so conditional on
        # rating the better recent record moves the estimate the other way. A
        # reader meeting that in one sentence needs to be able to see it is two
        # facts and not a contradiction.
        "value_favours": leader,
        "contribution": log_odds,
        "favours": side if abs(log_odds) > 0 else None,
        "supports_prediction": (log_odds > 0) == favours_team1,
        "sentence": (
            f"{_endstop(declared)} Contributes {log_odds:+.3f} to the symmetric "
            f"log-odds, favouring {side}."
        ),
        "means": entry["means"],
        "does_not_mean": entry["does_not_mean"],
        "value": (
            round(float(value), 3)
            if isinstance(value, (int, float)) and not isinstance(value, bool)
            else value
        ),
    }


def _undeclared_factor(
    name: str,
    log_odds: float,
    favours_team1: bool,
    team1: str,
    team2: str,
) -> dict[str, Any]:
    """A contributing feature with no declared phrasing.

    Should never happen - ``scripts/check_performance_endpoints.py`` compares the
    bundle's feature list against the phrasing file's keys and fails when they
    disagree. If a future bundle adds a column anyway, the factor is reported
    with its column name and flagged, because a contribution that is silently
    dropped is an explanation that does not add up to the number on screen.
    """
    side = team1 if log_odds > 0 else team2
    return {
        "feature": name,
        "label": name,
        "contribution": log_odds,
        "favours": side if abs(log_odds) > 0 else None,
        "supports_prediction": (log_odds > 0) == favours_team1,
        "sentence": (
            f"{name} contributes {log_odds:+.3f} to the symmetric log-odds, "
            f"favouring {side}. No phrasing has been declared for this feature."
        ),
        "means": (
            "A contribution to the log-odds of the probability shown. No entry "
            "for this feature exists in rules/bwf_factor_phrasing.json, so no "
            "declared meaning is available."
        ),
        "does_not_mean": (
            "Not a probability point, and not a cause. The model is fitted on "
            "historical association, never on a mechanism."
        ),
        "value": None,
        "phrasing_declared": False,
    }


def explain_champion_match(
    bundle: dict[str, Any],
    features: dict[str, Any],
    team1_name: str = "Team 1",
    team2_name: str = "Team 2",
) -> dict[str, Any]:
    """The readable explanation of the exact symmetric probability that is displayed.

    ``team1_name`` and ``team2_name`` are the people who filled the two slots.
    They default to the slot labels, which is what an unresolvable id gets: a
    caller that cannot say who is playing must still get its explanation, and
    "Team 1" is an honest statement of what the request contained. Every
    readable field - the sentences, the side each factor favours, the predicted
    side - is written with whichever of the two the caller could supply, so the
    stored audit payload reads the same as the screen.
    """
    explanation = explain_symmetric(bundle, features)
    probability = float(1.0 / (1.0 + math.exp(-explanation["reconstructed_margin"])))
    favours_team1 = probability >= 0.5
    declared = load_factor_phrasing()["factors"]

    factors, negligible, undeclared = [], [], []
    for item in explanation["contributions"]:
        feature = item["feature"]
        if abs(item["log_odds"]) < 1e-9:
            negligible.append(feature)
            continue
        entry = declared.get(feature)
        if entry is None:
            undeclared.append(feature)
            factors.append(
                _undeclared_factor(
                    feature, item["log_odds"], favours_team1, team1_name, team2_name
                )
            )
            continue
        factors.append(
            _readable_factor(
                feature,
                item["log_odds"],
                favours_team1,
                entry,
                features.get(feature),
                team1_name,
                team2_name,
            )
        )

    caveats = [
        "Swap-invariant context - discipline, tier, round, qualification, "
        "host country - contributes exactly zero to a symmetric margin and "
        "is listed as negligible for that reason, not because it was found "
        "unimportant.",
        "Association, not mechanism.",
    ]
    crossed = [
        factor for factor in factors
        if factor.get("value_favours") and factor["value_favours"] != factor["favours"]
    ]
    if crossed:
        caveats.append(
            "For "
            + ", ".join(sorted(factor["label"].lower() for factor in crossed))
            + ", the side the measurement favours is not the side the "
            "contribution favours. That is the model's fitted coefficient for "
            "the feature, not an error in the explanation: conditional on the "
            "other features it has learned to move the estimate in that "
            "direction. It is a property of a historical fit and is not a claim "
            "about what will happen."
        )
    if undeclared:
        caveats.append(
            f"{len(undeclared)} contributing feature(s) have no declared phrasing "
            f"and are reported by column name: {', '.join(sorted(undeclared))}. Add "
            "them to rules/bwf_factor_phrasing.json rather than leaving them "
            "unexplained."
        )

    return {
        "predicted_side": team1_name if favours_team1 else team2_name,
        "team1_win_probability": probability,
        "supporting": [f for f in factors if f["supports_prediction"]],
        "opposing": [f for f in factors if not f["supports_prediction"]],
        "negligible": negligible,
        "attribution_note": (
            "Contributions are exact for this model rather than sampled: the "
            "estimator is linear in its transformed inputs, so each feature's "
            "share of the margin is the product itself. They are combined across "
            "both orientations the same way the probability is, so what is "
            "explained is the number on screen and not one half of it."
        ),
        "units": "log-odds",
        "base_value_log_odds": explanation["base_value_log_odds"],
        "reconstruction": {
            "margin": explanation["reconstructed_margin"],
            "probability": explanation["reconstructed_probability"],
            "additivity_error": explanation["additivity_error"],
            "probability_error": explanation["probability_error"],
        },
        "caveats": caveats,
        "phrasing_source": "rules/bwf_factor_phrasing.json",
        "disclaimer": DISCLAIMER,
    }
