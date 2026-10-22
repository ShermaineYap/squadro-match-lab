"""Split conformal prediction: turn a probability into a set with a coverage guarantee.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CALIBRATION_PATH = PROJECT_ROOT / "artifacts" / "bwf_conformal_calibration.json"
INJURY_CALIBRATION_PATH = PROJECT_ROOT / "artifacts" / "injury_conformal_calibration.json"


@dataclass(frozen=True)
class ConformalCalibration:
    """A probability threshold that delivers a stated coverage target."""

    alpha: float
    q_hat: float
    probability_threshold: float
    calibration_size: int
    target_coverage: float

    def threshold_for(self, class_name: str) -> float:
        """The admission threshold for a class.

        One threshold serves every class under marginal calibration; the
        argument exists so the marginal and class-conditional calibrations can
        be used through the same call.
        """
        return self.probability_threshold

    def to_dict(self) -> dict:
        return {
            "alpha": self.alpha,
            "target_coverage": self.target_coverage,
            "q_hat": self.q_hat,
            "probability_threshold": self.probability_threshold,
            "calibration_size": self.calibration_size,
        }


def minimum_calibration_size(alpha: float) -> int:
    """Fewest calibration points that can certify ``1 - alpha`` coverage.

    The finite-sample rank is ceil((n + 1)(1 - alpha)); it only exists inside
    the sample when that rank is at most n, which requires n >= (1 - alpha) /
    alpha. At alpha = 0.10 that is 9 points. Below it no quantile of the
    calibration scores certifies the target, whatever value is reported.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between 0 and 1")
    return math.ceil((1.0 - alpha) / alpha)


@dataclass(frozen=True)
class ClassConditionalCalibration:
    """A separate threshold per true class: coverage guaranteed within each class.

    Marginal conformal spends its error budget wherever it is cheapest, which on
    an imbalanced problem means the majority class is covered generously and the
    minority classes absorb the shortfall. Calibrating inside each class removes
    that transfer: every class gets its own quantile from its own rows, so the
    1 - alpha statement is true class by class rather than only on average.

    ``unsupported_classes`` names classes whose calibration set was too small to
    certify the target. Their thresholds fall back to the most permissive
    behaviour available rather than implying a guarantee the data cannot carry.
    """

    alpha: float
    target_coverage: float
    per_class: Mapping[str, ConformalCalibration]
    unsupported_classes: tuple[str, ...] = ()
    default_threshold: float = 0.0
    method: str = field(default="class-conditional (Mondrian) split conformal", init=False)

    def threshold_for(self, class_name: str) -> float:
        calibration = self.per_class.get(class_name)
        if calibration is None:
            return self.default_threshold
        return calibration.probability_threshold

    @property
    def calibration_size(self) -> int:
        """Total calibration rows across the classes."""
        return sum(item.calibration_size for item in self.per_class.values())

    @property
    def probability_threshold(self) -> float:
        """The most permissive per-class threshold.

        Reported for compatibility with the marginal calibration's single-number
        summary. It is a floor, not the rule that is applied: admission is
        decided per class by ``threshold_for``, and quoting this number as
        though it governed every class would overstate how readily the strict
        classes are admitted.
        """
        if not self.per_class:
            return self.default_threshold
        return min(item.probability_threshold for item in self.per_class.values())

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "alpha": self.alpha,
            "target_coverage": self.target_coverage,
            "calibration_size": self.calibration_size,
            "minimum_calibration_size": minimum_calibration_size(self.alpha),
            "unsupported_classes": list(self.unsupported_classes),
            "default_threshold": self.default_threshold,
            "per_class": {
                name: {
                    **item.to_dict(),
                    "certifies_target": name not in self.unsupported_classes,
                }
                for name, item in self.per_class.items()
            },
        }


def calibrate(
    probabilities_of_true_class: Sequence[float], alpha: float = 0.1
) -> ConformalCalibration:
    """Derive the probability threshold meeting a ``1 - alpha`` coverage target.

    ``probabilities_of_true_class`` is, for each calibration example, the
    probability the model assigned to the outcome that actually happened.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between 0 and 1")

    scores = sorted(1.0 - float(p) for p in probabilities_of_true_class)
    n = len(scores)
    if n == 0:
        raise ValueError("At least one calibration example is required")

    # The finite-sample correction. Using the plain empirical quantile would
    # undercover slightly; the (n+1) form is what makes the guarantee exact
    # rather than asymptotic.
    rank = math.ceil((n + 1) * (1 - alpha))
    if rank > n:
        # Too few calibration points to certify this coverage at all. Falling
        # back to the largest observed score is the most conservative honest
        # answer: every class will be admitted rather than a guarantee implied
        # that the data cannot support.
        q_hat = scores[-1]
    else:
        q_hat = scores[rank - 1]

    return ConformalCalibration(
        alpha=alpha,
        q_hat=q_hat,
        probability_threshold=1.0 - q_hat,
        calibration_size=n,
        target_coverage=1.0 - alpha,
    )


def calibrate_class_conditional(
    probabilities_of_true_class_by_class: Mapping[str, Sequence[float]],
    alpha: float = 0.1,
) -> ClassConditionalCalibration:
    """Derive one threshold per true class, each meeting ``1 - alpha`` within that class.

    ``probabilities_of_true_class_by_class`` maps a class name to the
    probabilities the model gave that class on the calibration examples whose
    true label *was* that class. Grouping by the true label is what makes the
    guarantee conditional: each quantile is computed from, and applies to, one
    class's own score distribution.

    Each class carries its own n, so each gets its own finite-sample rank
    ceil((n_k + 1)(1 - alpha)). Reusing a pooled n here would be the obvious
    mistake and would overstate what a small class's rows can certify.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between 0 and 1")
    if not probabilities_of_true_class_by_class:
        raise ValueError("At least one class of calibration examples is required")

    floor = minimum_calibration_size(alpha)
    per_class: dict[str, ConformalCalibration] = {}
    unsupported: list[str] = []

    for class_name, scores in probabilities_of_true_class_by_class.items():
        scores = list(scores)
        if not scores:
            # No rows at all: no quantile exists. The class is left out of
            # per_class so it falls to default_threshold, and named as
            # unsupported so no caller can mistake silence for a guarantee.
            unsupported.append(class_name)
            continue
        per_class[class_name] = calibrate(scores, alpha=alpha)
        if len(scores) < floor:
            unsupported.append(class_name)

    return ClassConditionalCalibration(
        alpha=alpha,
        target_coverage=1.0 - alpha,
        per_class=per_class,
        unsupported_classes=tuple(unsupported),
    )


def _marginal_from_block(block: dict) -> ConformalCalibration:
    return ConformalCalibration(
        alpha=float(block["alpha"]),
        q_hat=float(block["q_hat"]),
        probability_threshold=float(block["probability_threshold"]),
        calibration_size=int(block["calibration_size"]),
        target_coverage=float(block["target_coverage"]),
    )


def _class_conditional_from_block(block: dict) -> ClassConditionalCalibration:
    per_class = {
        name: _marginal_from_block(item)
        for name, item in dict(block["per_class"]).items()
    }
    if not per_class:
        raise ValueError("a class-conditional block needs at least one class")
    return ClassConditionalCalibration(
        alpha=float(block["alpha"]),
        target_coverage=float(block["target_coverage"]),
        per_class=per_class,
        unsupported_classes=tuple(block.get("unsupported_classes", ())),
        default_threshold=float(block.get("default_threshold", 0.0)),
    )


def _load(path: Path) -> ConformalCalibration | ClassConditionalCalibration | None:
    """Read a calibration artefact, or None when it has not been generated.

    Returning None rather than raising keeps the model API usable on a checkout
    where the calibration scripts have not been run: callers fall back to
    reporting the probability alone. A missing calibration must never be
    silently treated as a confident call.

    Which method the artefact deploys is stated in the artefact itself, under
    ``deployed_method``. An artefact without that key is a marginal one from
    before class-conditional calibration existed, and is read as marginal - the
    stronger method is never assumed on a caller's behalf.
    """
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        method = str(payload.get("deployed_method", "marginal"))
        if method == "class_conditional":
            return _class_conditional_from_block(payload["deployed_class_conditional"])
        return _marginal_from_block(payload["deployed"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError, OSError):
        return None


@lru_cache(maxsize=1)
def load_bwf_calibration() -> ConformalCalibration | None:
    """The deployed BWF calibration, or None if it has not been generated."""
    return _load(CALIBRATION_PATH)


@lru_cache(maxsize=1)
def load_injury_calibration() -> ConformalCalibration | ClassConditionalCalibration | None:
    """The deployed injury calibration, or None if it has not been generated.

    The injury artefact deploys class-conditional calibration, so this normally
    returns a ``ClassConditionalCalibration``. Callers should treat the return
    as either kind and go through ``threshold_for`` / ``prediction_set`` /
    ``decide`` rather than reading ``probability_threshold`` directly.
    """
    return _load(INJURY_CALIBRATION_PATH)


def clear_calibration_cache() -> None:
    load_bwf_calibration.cache_clear()
    load_injury_calibration.cache_clear()


def prediction_set(
    probabilities: dict[str, float],
    calibration: ConformalCalibration | ClassConditionalCalibration,
) -> list[str]:
    """Every class plausible enough to be admitted at its calibrated threshold.

    Under marginal calibration every class is judged against the same number.
    Under class-conditional calibration each class is judged against the
    threshold derived from its own rows, which is the whole mechanism: a class
    the model scores timidly is admitted on a lower probability than one it
    scores boldly, and that is how per-class coverage is equalised.
    """
    return [
        name
        for name, probability in probabilities.items()
        if float(probability) >= calibration.threshold_for(name)
    ]


def decide(
    probabilities: dict[str, float],
    calibration: ConformalCalibration | ClassConditionalCalibration,
) -> dict:
    """A conformal verdict: a confident call, or an explicit refusal to call.

    A single admitted class is a call. Two or more means the evidence does not
    separate them at the requested confidence, which is reported as a refusal
    rather than as a low-confidence winner - the point of the exercise is that
    the system declines instead of guessing.

    An empty set can occur when the threshold exceeds every class probability.
    It means no outcome is plausible at this confidence, which for a two-outcome
    match is a statement about the model rather than the match, so it is
    reported as a refusal too and the leading class is carried for reference.
    """
    admitted = prediction_set(probabilities, calibration)
    leading = max(probabilities, key=lambda name: probabilities[name])
    confident = len(admitted) == 1

    if confident:
        status = (
            f"Confident call at {calibration.target_coverage:.0%} target coverage"
        )
    elif not admitted:
        status = (
            f"No outcome reaches the {calibration.target_coverage:.0%} confidence "
            "threshold — too close to call"
        )
    else:
        status = (
            f"Outcomes not separated at {calibration.target_coverage:.0%} target "
            "coverage — too close to call"
        )

    class_conditional = isinstance(calibration, ClassConditionalCalibration)
    if class_conditional:
        method = (
            "class-conditional (Mondrian) split conformal "
            "(least-ambiguous set-valued classifier)"
        )
        guarantee = (
            "Under exchangeability the true class falls inside the returned set "
            f"at least {calibration.target_coverage:.0%} of the time *within "
            "each class*, not merely on average across them. This is a "
            "guarantee about the long-run rate, not about this case."
        )
    else:
        method = "split conformal (least-ambiguous set-valued classifier)"
        guarantee = (
            "Under exchangeability the true outcome falls inside the returned "
            f"set at least {calibration.target_coverage:.0%} of the time. This "
            "is a guarantee about the long-run rate, not about this fixture."
        )

    return {
        "prediction_set": admitted or [],
        "set_size": len(admitted),
        "confident": confident,
        "call": admitted[0] if confident else None,
        "leading_class": leading,
        "status": status,
        # Under class-conditional calibration this is the most permissive of the
        # per-class thresholds; the rule actually applied is class_thresholds.
        "probability_threshold": calibration.probability_threshold,
        "class_thresholds": {
            name: calibration.threshold_for(name) for name in probabilities
        },
        "target_coverage": calibration.target_coverage,
        "method": method,
        "guarantee": guarantee,
    }
