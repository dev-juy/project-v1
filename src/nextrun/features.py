"""
Canonical scenario feature definition for NextRun.

Everything downstream (models, selectors, evaluation) imports feature names,
bounds, and transforms from HERE. A single definition is the defense against
the selector and the model silently disagreeing about what a scenario is.

LEAKAGE FIREWALL: normalization uses FIXED scenario bounds from the experiment
config, never data-derived statistics. A data-fit scaler refit per selection
step would leak pool composition into the features. Do not replace this with
StandardScaler.fit or any other fitted transform.
"""

from __future__ import annotations

from math import hypot

import numpy as np
import pandas as pd


# TODO: load from config/experiment.yaml once the config module exists.
# These are the frozen scenario-space bounds from the v4 spec.
FEATURE_NAMES = [
    "start_x_offset_in",    # start_x - target_x
    "start_y_offset_in",    # start_y - target_y
    "start_heading_deg",
    "max_power",
    "offset_magnitude_in",  # hypot(x_offset, y_offset) — engineered
]

FEATURE_BOUNDS = {
    "start_x_offset_in":   (-24.0, 24.0),
    "start_y_offset_in":   (-24.0, 24.0),
    "start_heading_deg":   (-45.0, 45.0),
    "max_power":           (0.4, 0.7),
    "offset_magnitude_in": (0.0, hypot(24.0, 24.0)),
}


def scenario_to_features(df, task_cfg: dict) -> pd.DataFrame:
    """Scenario rows -> DataFrame[FEATURE_NAMES]. Deterministic, no fitting.

    Accepts a DataFrame or an iterable of dicts with columns
    start_x_in, start_y_in, start_heading_deg, max_power. Offsets are computed
    relative to task_cfg['task']['target'].
    """
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(list(df))
    target = task_cfg["task"]["target"]
    tx, ty = float(target["x_in"]), float(target["y_in"])

    x_off = df["start_x_in"].astype(float) - tx
    y_off = df["start_y_in"].astype(float) - ty
    out = pd.DataFrame({
        "start_x_offset_in": x_off,
        "start_y_offset_in": y_off,
        "start_heading_deg": df["start_heading_deg"].astype(float),
        "max_power": df["max_power"].astype(float),
        "offset_magnitude_in": np.hypot(x_off, y_off),
    }, index=df.index)
    return out[FEATURE_NAMES]


def normalize_features(X: pd.DataFrame) -> np.ndarray:
    """Min-max to [0,1] against FIXED FEATURE_BOUNDS. No data-derived stats.

    Pure and stateless: a row's normalized vector is identical whether it is
    normalized alone or inside any batch. Out-of-bounds values are clipped
    to [0,1].
    """
    X = X[FEATURE_NAMES]
    lo = np.array([FEATURE_BOUNDS[n][0] for n in FEATURE_NAMES])
    hi = np.array([FEATURE_BOUNDS[n][1] for n in FEATURE_NAMES])
    Z = (X.to_numpy(dtype=float) - lo) / (hi - lo)
    return np.clip(Z, 0.0, 1.0)


def scenario_to_normalized(df, task_cfg: dict) -> np.ndarray:
    """Convenience: scenario_to_features -> normalize_features."""
    return normalize_features(scenario_to_features(df, task_cfg))
