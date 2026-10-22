from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression


class OrdinalRiskClassifier(ClassifierMixin, BaseEstimator):
    """Three-class ordinal model using P(y > 0) and P(y > 1)."""

    def __init__(self, C: float = 1.0, class_weight: str | dict | None = "balanced", random_state: int = 42):
        self.C = C
        self.class_weight = class_weight
        self.random_state = random_state

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.array([0, 1, 2])
        self.lower_model_ = LogisticRegression(
            C=self.C, class_weight=self.class_weight, max_iter=3000,
            random_state=self.random_state,
        ).fit(X, (y > 0).astype(int))
        self.upper_model_ = LogisticRegression(
            C=self.C, class_weight=self.class_weight, max_iter=3000,
            random_state=self.random_state,
        ).fit(X, (y > 1).astype(int))
        return self

    def predict_proba(self, X):
        above_zero = self.lower_model_.predict_proba(X)[:, 1]
        above_one = self.upper_model_.predict_proba(X)[:, 1]
        # Enforce the ordinal constraint P(y > 1) <= P(y > 0).
        above_one = np.minimum(above_one, above_zero)
        probabilities = np.column_stack([
            1.0 - above_zero,
            above_zero - above_one,
            above_one,
        ])
        return np.clip(probabilities, 0.0, 1.0)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class TwoStageRiskClassifier(ClassifierMixin, BaseEstimator):
    """Healthy/elevated gate followed by Low Risk/High Risk separation."""

    def __init__(self, C: float = 1.0, class_weight: str | dict | None = "balanced", random_state: int = 42):
        self.C = C
        self.class_weight = class_weight
        self.random_state = random_state

    def _new_model(self):
        return LogisticRegression(
            C=self.C, class_weight=self.class_weight, max_iter=3000,
            random_state=self.random_state,
        )

    def _fit_stage(self, X, y_binary):
        """A fitted logistic stage, or the constant a degenerate subset supports.

        A grouped fold can contain no elevated sessions at all, or elevated
        sessions of only one kind. Logistic regression cannot be fitted on a
        single-class target, and raising would abort an entire cross-validation
        because of one degenerate fold. The honest fallback is the rate the
        subset actually shows - 0 or 1 when it is single-class, 0 when it is
        empty - which the fold metrics then score on its merits.
        """
        if y_binary.size == 0 or np.unique(y_binary).size < 2:
            return None, float(y_binary.mean()) if y_binary.size else 0.0
        return self._new_model().fit(X, y_binary), None

    @staticmethod
    def _stage_probability(model, rate, X, n_rows):
        if model is None:
            return np.full(n_rows, rate)
        return model.predict_proba(X)[:, 1]

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.array([0, 1, 2])
        self.elevated_model_, self.elevated_rate_ = self._fit_stage(
            X, (y > 0).astype(int)
        )
        elevated = y > 0
        self.high_model_, self.high_rate_ = self._fit_stage(
            X[elevated], (y[elevated] == 2).astype(int)
        )
        return self

    def predict_proba(self, X):
        n_rows = X.shape[0]
        elevated = self._stage_probability(
            self.elevated_model_, self.elevated_rate_, X, n_rows
        )
        high_given_elevated = self._stage_probability(
            self.high_model_, self.high_rate_, X, n_rows
        )
        return np.column_stack([
            1.0 - elevated,
            elevated * (1.0 - high_given_elevated),
            elevated * high_given_elevated,
        ])

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
