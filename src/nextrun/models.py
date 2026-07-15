"""
Success-prediction models for NextRun: Logistic Regression and GPC behind one
uniform interface. GPC is the primary model.

Uncertainty is PREDICTIVE ENTROPY for both models — declared in advance per
the v4 spec; do not switch to latent variance after seeing results. The GPC
wrapper exposes latent variance only as a secondary diagnostic
(latent_uncertainty), never through uncertainty().

Cold-start guard: with a tiny labeled set (e.g. 10 seed trials) the labels are
often single-class. Both wrappers detect this in fit() and fall back to a
maximally-uncertain prior: predict_proba = [0.5, 0.5], uncertainty = ln(2)
(the entropy of p=0.5, i.e. the maximum of the declared uncertainty measure).
They must never crash on single-class data.

Inputs are assumed already normalized to [0,1] by features.py.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.linear_model import LogisticRegression


@runtime_checkable
class SuccessModel(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> "SuccessModel": ...
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...
    def uncertainty(self, X: np.ndarray) -> np.ndarray: ...


def _entropy(p1: np.ndarray) -> np.ndarray:
    """Binary predictive entropy in nats; max ln(2) ~ 0.6931 at p=0.5."""
    p = np.clip(np.asarray(p1, dtype=float), 1e-9, 1 - 1e-9)
    return -(p * np.log(p) + (1 - p) * np.log1p(-p))


class _ColdStartMixin:
    """Single-class fallback shared by both wrappers.

    fit() must call _check_cold_start first; when it returns True the sklearn
    fitter is skipped and predictions come from the maximally-uncertain prior.
    """

    _single_class: float | None = None
    _fitted: bool = False

    def _check_cold_start(self, y: np.ndarray) -> bool:
        classes = np.unique(np.asarray(y))
        if classes.size < 2:
            self._single_class = float(classes[0]) if classes.size else None
            self._fitted = True
            return True
        self._single_class = None
        return False

    def _cold_proba(self, X: np.ndarray) -> np.ndarray:
        n = np.asarray(X).shape[0]
        return np.full((n, 2), 0.5)

    def _cold_uncertainty(self, X: np.ndarray) -> np.ndarray:
        # entropy at p=0.5 is ln(2) — the max of the declared uncertainty
        # measure, so cold-start uncertainty stays on the same scale
        return np.full(np.asarray(X).shape[0], np.log(2.0))

    @property
    def is_cold(self) -> bool:
        return self._fitted and self._single_class is not None


class LogRegModel(_ColdStartMixin):
    def __init__(self, seed: int):
        self.seed = seed
        self._clf = LogisticRegression(C=1.0, random_state=seed)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogRegModel":
        if not self._check_cold_start(y):
            self._clf.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.is_cold:
            return self._cold_proba(X)
        return self._clf.predict_proba(X)

    def uncertainty(self, X: np.ndarray) -> np.ndarray:
        if self.is_cold:
            return self._cold_uncertainty(X)
        return _entropy(self.predict_proba(X)[:, 1])


class GPCModel(_ColdStartMixin):
    def __init__(self, seed: int):
        self.seed = seed
        self._clf = GaussianProcessClassifier(
            kernel=RBF(length_scale=1.0) + WhiteKernel(),
            n_restarts_optimizer=5,
            random_state=seed,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GPCModel":
        if not self._check_cold_start(y):
            self._clf.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.is_cold:
            return self._cold_proba(X)
        return self._clf.predict_proba(X)

    def uncertainty(self, X: np.ndarray) -> np.ndarray:
        if self.is_cold:
            return self._cold_uncertainty(X)
        return _entropy(self.predict_proba(X)[:, 1])

    def latent_uncertainty(self, X: np.ndarray) -> np.ndarray:
        """Diagnostic only: latent GP posterior variance. NOT the declared
        uncertainty measure — uncertainty() is predictive entropy."""
        if self.is_cold:
            return self._cold_uncertainty(X)
        _, var = self._clf.base_estimator_.latent_mean_and_variance(X)
        return np.asarray(var)

    def fitted_kernel(self) -> str:
        """The kernel after hyperparameter optimization, for reporting."""
        if self.is_cold:
            return "cold-start (no kernel fitted)"
        return str(self._clf.kernel_)


def make_model(name: str, seed: int) -> SuccessModel:
    if name == "logreg":
        return LogRegModel(seed)
    if name == "gpc":
        return GPCModel(seed)
    raise ValueError(f"unknown model name: {name!r} (expected 'logreg' or 'gpc')")
