"""Tests for the success-prediction models (models.py)."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nextrun.models import make_model, GPCModel, LogRegModel, _entropy

LN2 = np.log(2.0)


def separable_set(n: int = 60, seed: int = 3):
    """Synthetic, obviously-fake set in normalized [0,1] feature space.
    5 features matching features.FEATURE_NAMES; success iff the last feature
    (offset magnitude) is below 0.5."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, size=(n, 5))
    y = (X[:, 4] < 0.5).astype(int)  # 1 = success at low offset
    return X, y


@pytest.mark.parametrize("name", ["logreg", "gpc"])
def test_fit_predict_shapes(name):
    X, y = separable_set()
    m = make_model(name, seed=0).fit(X, y)
    P = m.predict_proba(X)
    assert P.shape == (len(X), 2)
    np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-9)


@pytest.mark.parametrize("name", ["logreg", "gpc"])
def test_uncertainty_is_entropy_bounded(name):
    X, y = separable_set()
    m = make_model(name, seed=0).fit(X, y)
    u = m.uncertainty(X)
    assert u.shape == (len(X),)
    assert np.all(u >= 0.0)
    assert np.all(u <= LN2 + 1e-9)
    # uncertainty() must equal entropy of predict_proba — the declared measure
    np.testing.assert_allclose(u, _entropy(m.predict_proba(X)[:, 1]))


@pytest.mark.parametrize("name", ["logreg", "gpc"])
@pytest.mark.parametrize("label", [0, 1])
def test_cold_start_single_class(name, label):
    X, _ = separable_set(n=10)
    y = np.full(10, label)
    m = make_model(name, seed=0)
    m.fit(X, y)  # must not crash
    P = m.predict_proba(X)
    np.testing.assert_array_equal(P, np.full((10, 2), 0.5))
    np.testing.assert_array_equal(m.uncertainty(X), np.ones(10))


@pytest.mark.parametrize("name", ["logreg", "gpc"])
def test_determinism_same_seed(name):
    X, y = separable_set()
    a = make_model(name, seed=42).fit(X, y).predict_proba(X)
    b = make_model(name, seed=42).fit(X, y).predict_proba(X)
    np.testing.assert_array_equal(a, b)


def test_gpc_latent_uncertainty_diagnostic():
    X, y = separable_set()
    m = make_model("gpc", seed=0)
    assert isinstance(m, GPCModel)
    m.fit(X, y)
    lat = m.latent_uncertainty(X)
    assert lat.shape == (len(X),)
    assert np.all(np.isfinite(lat))
    # latent variance is a DIFFERENT quantity from the declared entropy
    assert not np.allclose(lat, m.uncertainty(X))


def test_gpc_learns_direction():
    """Success iff offset magnitude below threshold: fitted GPC must assign
    higher failure probability to high-offset points than low-offset points."""
    X, y = separable_set(n=80, seed=11)
    m = make_model("gpc", seed=0).fit(X, y)
    low = X[X[:, 4] < 0.25]
    high = X[X[:, 4] > 0.75]
    p_fail_low = m.predict_proba(low)[:, 0].mean()
    p_fail_high = m.predict_proba(high)[:, 0].mean()
    assert p_fail_high > p_fail_low + 0.2, \
        f"GPC did not learn direction: fail@high={p_fail_high}, fail@low={p_fail_low}"


def test_make_model_rejects_unknown():
    with pytest.raises(ValueError):
        make_model("random_forest", seed=0)
