"""Order symmetry for BWF match models: a correctness constraint, not a tweak."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression

EPSILON = 1e-12


class FeatureRoles:
    """How each column behaves when the two sides are exchanged.

    Every feature must be classified before it may enter a model. An unclassified
    column is not a small omission: if a side-specific feature is treated as
    invariant, the swapped row is silently wrong and the symmetry it appears to
    have is an artefact.
    """

    def __init__(
        self,
        antisymmetric: list[str],
        side_pairs: list[tuple[str, str]],
        invariant: list[str],
    ) -> None:
        self.antisymmetric = list(antisymmetric)
        self.side_pairs = [tuple(pair) for pair in side_pairs]
        self.invariant = list(invariant)

    @property
    def columns(self) -> list[str]:
        paired = [name for pair in self.side_pairs for name in pair]
        return self.antisymmetric + paired + self.invariant

    def validate(self, columns: list[str]) -> None:
        declared = self.columns
        duplicates = {name for name in declared if declared.count(name) > 1}
        if duplicates:
            raise ValueError(f"features classified more than once: {sorted(duplicates)}")
        missing = [name for name in columns if name not in declared]
        if missing:
            raise ValueError(
                "unclassified features - every column must be declared "
                f"antisymmetric, side-paired or invariant before use: {missing}"
            )

    def swap(self, frame: pd.DataFrame) -> pd.DataFrame:
        """The same fixture entered the other way round."""
        swapped = frame.copy()
        for name in self.antisymmetric:
            if name in swapped.columns:
                swapped[name] = -swapped[name]
        for left, right in self.side_pairs:
            if left in swapped.columns and right in swapped.columns:
                swapped[[left, right]] = swapped[[right, left]].to_numpy()
        return swapped

    def align_swapped_contributions(
        self, contributions: pd.DataFrame
    ) -> pd.DataFrame:
        """Map contributions computed on the swapped row back onto this row's columns.

        This is a **permutation and nothing else**: only the side-paired columns
        move. It is tempting to also negate the antisymmetric ones - the
        instruction "negate antisymmetric features" is right, but it applies to
        the *feature values* when building the swapped row, which
        :meth:`swap` already did. Applying the sign a second time here would
        double-count it, and it would break additivity outright: the aligned
        contributions would no longer sum to the swapped margin, so
        ``(forward - aligned) / 2`` would stop equalling the symmetric margin and
        the explanation would no longer add up to the number on screen.

        The negation is carried by the minus sign in that subtraction. Because a
        permutation leaves a sum unchanged, the aligned contributions still total
        the swapped margin exactly, which is what makes the symmetric attribution
        exact rather than approximate.
        """
        aligned = contributions.copy()
        for left, right in self.side_pairs:
            if left in aligned.columns and right in aligned.columns:
                aligned[[left, right]] = aligned[[right, left]].to_numpy()
        return aligned


def margins(model, X: pd.DataFrame) -> np.ndarray:
    """Log-odds of the positive class, for any classifier with predict_proba."""
    probability = np.clip(model.predict_proba(X)[:, 1], EPSILON, 1 - EPSILON)
    return np.log(probability / (1 - probability))


class SymmetricAveraging(BaseEstimator, ClassifierMixin):
    """Wrap any classifier so its answer cannot depend on which side is first."""

    def __init__(self, estimator=None, roles: FeatureRoles | None = None):
        self.estimator = estimator
        self.roles = roles

    def fit(self, X, y):
        self.roles.validate(list(X.columns))
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y)
        self.classes_ = np.array([0, 1])
        return self

    def decision_function(self, X):
        forward = margins(self.estimator_, X)
        reverse = margins(self.estimator_, self.roles.swap(X))
        return (forward - reverse) / 2.0

    def predict_proba(self, X):
        probability = 1.0 / (1.0 + np.exp(-self.decision_function(X)))
        return np.column_stack([1.0 - probability, probability])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class AntisymmetricLogistic(BaseEstimator, ClassifierMixin):
    """Logistic regression on a design matrix that negates under a swap."""

    def __init__(self, roles: FeatureRoles | None = None, C: float = 1.0,
                 interaction_column: str | None = "discipline", max_iter: int = 2000):
        self.roles = roles
        self.C = C
        self.interaction_column = interaction_column
        self.max_iter = max_iter

    def _design(self, X: pd.DataFrame) -> pd.DataFrame:
        base = pd.DataFrame(index=X.index)
        for name in self.roles.antisymmetric:
            base[name] = pd.to_numeric(X[name], errors="coerce")
        # A side-paired feature contributes only through its difference; the sum
        # is invariant and would break antisymmetry if added.
        for left, right in self.roles.side_pairs:
            base[f"{left}_minus_{right}"] = (
                pd.to_numeric(X[left], errors="coerce")
                - pd.to_numeric(X[right], errors="coerce")
            )
        base = base.fillna(0.0)

        if self.interaction_column and self.interaction_column in X.columns:
            indicator = pd.get_dummies(
                X[self.interaction_column].astype(str), prefix="ctx"
            )
            indicator = indicator.reindex(columns=self.context_levels_, fill_value=0)
            blocks = [base]
            for level in self.context_levels_:
                block = base.mul(indicator[level].to_numpy(), axis=0)
                block.columns = [f"{column}__{level}" for column in base.columns]
                blocks.append(block)
            return pd.concat(blocks, axis=1)
        return base

    def fit(self, X, y):
        self.roles.validate(list(X.columns))
        if self.interaction_column and self.interaction_column in X.columns:
            self.context_levels_ = sorted(X[self.interaction_column].astype(str).unique())
        else:
            self.context_levels_ = []
        design = self._design(X)
        self.design_columns_ = list(design.columns)
        self.scale_ = design.std().replace(0.0, 1.0)
        self.model_ = LogisticRegression(
            C=self.C, fit_intercept=False, max_iter=self.max_iter, solver="lbfgs"
        )
        self.model_.fit(design / self.scale_, y)
        self.classes_ = np.array([0, 1])
        return self

    def decision_function(self, X):
        design = self._design(X).reindex(columns=self.design_columns_, fill_value=0.0)
        return self.model_.decision_function(design / self.scale_)

    def predict_proba(self, X):
        probability = 1.0 / (1.0 + np.exp(-self.decision_function(X)))
        return np.column_stack([1.0 - probability, probability])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def symmetry_violation(model, X: pd.DataFrame, roles: FeatureRoles) -> float:
    """max |P(A>B) - (1 - P(B>A))| over the supplied fixtures."""
    forward = model.predict_proba(X)[:, 1]
    reverse = model.predict_proba(roles.swap(X))[:, 1]
    return float(np.max(np.abs(forward - (1.0 - reverse))))
