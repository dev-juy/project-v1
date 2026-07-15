"""Tests for the canonical feature definition (features.py)."""

from __future__ import annotations

import os
import sys
from math import hypot

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nextrun.features import (
    FEATURE_NAMES, FEATURE_BOUNDS,
    scenario_to_features, normalize_features, scenario_to_normalized,
)


TASK_CFG = {
    "task": {
        "target": {"x_in": 72.0, "y_in": 72.0, "heading_deg": 0.0},
        "success": {
            "position_error_in_max": 4.0,
            "heading_error_deg_max": 8.0,
            "duration_s_max": 6.0,
        },
    }
}


def make_scenarios(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "start_x_in": 72.0 + rng.uniform(-24, 24, n),
        "start_y_in": 72.0 + rng.uniform(-24, 24, n),
        "start_heading_deg": rng.uniform(-45, 45, n),
        "max_power": rng.uniform(0.4, 0.7, n),
    })


def test_feature_names_stable():
    assert FEATURE_NAMES == [
        "start_x_offset_in",
        "start_y_offset_in",
        "start_heading_deg",
        "max_power",
        "offset_magnitude_in",
    ]
    assert set(FEATURE_BOUNDS) == set(FEATURE_NAMES)


def test_scenario_to_features_deterministic():
    df = make_scenarios(20)
    a = scenario_to_features(df, TASK_CFG)
    b = scenario_to_features(df, TASK_CFG)
    pd.testing.assert_frame_equal(a, b)
    assert list(a.columns) == FEATURE_NAMES


def test_offset_magnitude_known_example():
    df = pd.DataFrame([{
        "start_x_in": 72.0 - 3.0, "start_y_in": 72.0 + 4.0,
        "start_heading_deg": 10.0, "max_power": 0.5,
    }])
    feats = scenario_to_features(df, TASK_CFG)
    assert feats["start_x_offset_in"].iloc[0] == pytest.approx(-3.0)
    assert feats["start_y_offset_in"].iloc[0] == pytest.approx(4.0)
    assert feats["offset_magnitude_in"].iloc[0] == pytest.approx(5.0)


def test_normalize_stateless_anti_leakage():
    """A row normalized alone must equal the same row normalized in a batch
    of 100. This is the anti-leakage property: batch composition must not
    influence any row's features."""
    batch = make_scenarios(100, seed=7)
    feats = scenario_to_features(batch, TASK_CFG)
    Z_batch = normalize_features(feats)
    for i in (0, 42, 99):
        Z_alone = normalize_features(feats.iloc[[i]])
        np.testing.assert_array_equal(Z_alone[0], Z_batch[i])


def test_out_of_bounds_clips():
    df = pd.DataFrame([{
        "start_x_in": 72.0 + 500.0, "start_y_in": 72.0 - 500.0,
        "start_heading_deg": 400.0, "max_power": 1.5,
    }])
    Z = scenario_to_normalized(df, TASK_CFG)
    assert Z.min() >= 0.0
    assert Z.max() <= 1.0
    # extreme values pin to the boundary, not beyond
    assert Z[0, 0] == pytest.approx(1.0)   # x offset way over max
    assert Z[0, 1] == pytest.approx(0.0)   # y offset way under min


def test_at_target_normalizes_to_midpoints():
    df = pd.DataFrame([{
        "start_x_in": 72.0, "start_y_in": 72.0,
        "start_heading_deg": 0.0, "max_power": 0.55,
    }])
    Z = scenario_to_normalized(df, TASK_CFG)
    named = dict(zip(FEATURE_NAMES, Z[0]))
    assert named["start_x_offset_in"] == pytest.approx(0.5)
    assert named["start_y_offset_in"] == pytest.approx(0.5)
    assert named["start_heading_deg"] == pytest.approx(0.5)
    assert named["max_power"] == pytest.approx(0.5)
    # magnitude bound is [0, hypot(24,24)]: zero offset -> 0.0, not 0.5
    assert named["offset_magnitude_in"] == pytest.approx(0.0)
