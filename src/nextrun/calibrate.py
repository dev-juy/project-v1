"""
Staged calibration of SimParams from physical calibration runs (v4.1).

Calibration matrix (12 primitive runs, NO repetitions):
    forward: 12in & 36in at powers 0.4 & 0.7   (4 runs)
    strafe:  12in & 36in at powers 0.4 & 0.7   (4 runs)
    turn:    45deg & 135deg at powers 0.4 & 0.7 (4 runs)
Two magnitudes per (type,power) give the distance leverage needed to fit
error-per-magnitude slopes. Stochastic noise is estimated from the SIX
repeated path-calibration runs (3 poses x 2 reps), not from primitive reps.

Four staged least-squares problems, each <=6 params:
  1. speeds (6)      : through-origin distance/duration per (type, power)
  2. error slopes (3): through-origin residual-vs-commanded, per axis
  3. noise (2)       : residual std across path repeats
  4. path margin (2) : OLS real_margin ~ scale*sim_margin + bias

LEAKAGE FIREWALL: consumes ONLY calibration rows; never reads `trial` outcomes.
"""

from __future__ import annotations

from typing import Optional
from collections import defaultdict
import numpy as np

from .simulator import SimParams, Simulator, POWER_LOW, POWER_HIGH, failure_margin


def _rows(data) -> list[dict]:
    if hasattr(data, "to_dict"):
        return data.to_dict("records")
    return list(data)


def _close_power(p: float, target: float, tol: float = 0.05) -> bool:
    return abs(float(p) - target) <= tol


def _through_origin_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Least-squares slope of y = m*x with no intercept: m = <x,y>/<x,x>."""
    denom = float(np.dot(x, x))
    if denom <= 0:
        return 0.0
    return float(np.dot(x, y) / denom)


# --------------------------------------------------------------------------- #
# Stage 1: speeds (6)
# --------------------------------------------------------------------------- #
def _fit_speeds(prim: list[dict]) -> dict:
    """Speed per (type, power) = through-origin slope of distance vs duration.

    With two magnitudes per power this is a genuine 2-point regression through
    the origin (speed is constant across distance), more robust than a single
    ratio.
    """
    def speed(mtype: str, power: float, default: float) -> float:
        mags, durs = [], []
        for r in prim:
            if r["movement_type"] != mtype or not _close_power(r["power"], power):
                continue
            dur = float(r["duration_s"])
            if dur <= 0:
                continue
            mag = (abs(float(r["commanded_angle_deg"])) if mtype == "turn"
                   else abs(float(r["commanded_distance_in"])))
            mags.append(mag)
            durs.append(dur)
        if not mags:
            return default
        # distance = speed * duration  ->  speed = slope of distance vs duration
        return _through_origin_slope(np.asarray(durs), np.asarray(mags)) or default

    return {
        "forward_speed_low":  speed("forward", POWER_LOW, 20.0),
        "forward_speed_high": speed("forward", POWER_HIGH, 40.0),
        "strafe_speed_low":   speed("strafe",  POWER_LOW, 15.0),
        "strafe_speed_high":  speed("strafe",  POWER_HIGH, 30.0),
        "turn_rate_low":      speed("turn",    POWER_LOW, 90.0),
        "turn_rate_high":     speed("turn",    POWER_HIGH, 180.0),
    }


# --------------------------------------------------------------------------- #
# Stage 2: error-per-magnitude slopes (3), through origin
# --------------------------------------------------------------------------- #
def _fit_error_slopes(prim: list[dict]) -> dict:
    """error = slope * commanded_magnitude, fit through the origin per axis.

    error is defined as (commanded - actual) on the moving axis: a shortfall is
    positive error. Slope may come out any sign; we do not constrain it.
    """
    fwd_cmd, fwd_err = [], []
    str_cmd, str_err = [], []
    trn_cmd, trn_err = [], []
    for r in prim:
        mt = r["movement_type"]
        if mt == "forward":
            c = float(r["commanded_distance_in"])
            fwd_cmd.append(c); fwd_err.append(c - float(r["actual_dx_in"]))
        elif mt == "strafe":
            c = float(r["commanded_distance_in"])
            str_cmd.append(c); str_err.append(c - float(r["actual_dy_in"]))
        elif mt == "turn":
            c = float(r["commanded_angle_deg"])
            trn_cmd.append(abs(c))
            trn_err.append(abs(c) - abs(float(r["actual_turn_deg"])))
    return {
        "forward_error_per_in": _through_origin_slope(np.asarray(fwd_cmd), np.asarray(fwd_err)),
        "strafe_error_per_in":  _through_origin_slope(np.asarray(str_cmd), np.asarray(str_err)),
        "turn_error_per_deg":   _through_origin_slope(np.asarray(trn_cmd), np.asarray(trn_err)),
    }


# --------------------------------------------------------------------------- #
# Stage 3: noise (2), from the six repeated path-calibration runs
# --------------------------------------------------------------------------- #
def _fit_noise(path: list[dict]) -> dict:
    """Std of measured position / heading error within repeated path poses.

    Path rows are grouped by (rounded) start pose + power; reps within a group
    estimate stochastic spread. Floors keep rollouts non-degenerate.
    """
    groups = defaultdict(list)
    for r in path:
        key = (round(float(r["start_x_in"]), 1), round(float(r["start_y_in"]), 1),
               round(float(r["start_heading_deg"]), 1), round(float(r["max_power"]), 2))
        groups[key].append(r)

    pos_spreads, hdg_spreads = [], []
    for rs in groups.values():
        if len(rs) >= 2:
            pos_spreads.append(np.std([float(r["position_error_in"]) for r in rs], ddof=1))
            hdg_spreads.append(np.std([float(r["heading_error_deg"]) for r in rs], ddof=1))

    pos = float(np.mean(pos_spreads)) if pos_spreads else 0.0
    hdg = float(np.mean(hdg_spreads)) if hdg_spreads else 0.0
    return {
        "position_noise_std_in": max(pos, 0.1),
        "heading_noise_std_deg": max(hdg, 0.1),
        "_noise_groups_used": len(pos_spreads),
    }


# --------------------------------------------------------------------------- #
# Stage 4: path margin (2)
# --------------------------------------------------------------------------- #
def _fit_path_margin(path: list[dict], partial: SimParams, task_cfg: dict) -> dict:
    if len(path) < 2:
        return {"path_margin_scale": 1.0, "path_margin_bias": 0.0,
                "_path_fit_ok": False, "_path_n": len(path)}

    prim_params = SimParams.from_dict(
        {**partial.to_dict(), "path_margin_scale": 1.0, "path_margin_bias": 0.0})
    sim = Simulator(prim_params, task_cfg)

    sim_m, real_m = [], []
    for r in path:
        res = sim.simulate_trial(
            float(r["start_x_in"]), float(r["start_y_in"]),
            float(r["start_heading_deg"]), float(r["max_power"]), rng=None)
        sim_m.append(res.margin)
        real_m.append(failure_margin(
            float(r["position_error_in"]), float(r["heading_error_deg"]),
            float(r["duration_s"]), task_cfg))

    x = np.asarray(sim_m); y = np.asarray(real_m)
    if np.ptp(x) < 1e-6:
        return {"path_margin_scale": 1.0, "path_margin_bias": float(np.mean(y - x)),
                "_path_fit_ok": False, "_path_n": len(path),
                "_path_note": "sim margins ~constant; slope unidentifiable"}

    A = np.vstack([x, np.ones_like(x)]).T
    (scale, bias), *_ = np.linalg.lstsq(A, y, rcond=None)
    return {"path_margin_scale": float(scale), "path_margin_bias": float(bias),
            "_path_fit_ok": True, "_path_n": len(path)}


# --------------------------------------------------------------------------- #
# public entry point
# --------------------------------------------------------------------------- #
def fit_simulator(calibration_primitive, calibration_path,
                  task_cfg: dict, seed: int = 0) -> tuple[SimParams, dict]:
    """Fit SimParams via four staged least-squares problems.

    calibration_primitive : 12 primitive runs (2 magnitudes x 2 powers x 3 types)
    calibration_path       : 6 path runs (3 poses x 2 reps)
    Consumes ONLY calibration data.
    """
    prim = _rows(calibration_primitive)
    path = _rows(calibration_path)

    speeds = _fit_speeds(prim)
    slopes = _fit_error_slopes(prim)
    noise = _fit_noise(path)

    partial = SimParams.from_dict({
        **speeds, **slopes,
        "position_noise_std_in": noise["position_noise_std_in"],
        "heading_noise_std_deg": noise["heading_noise_std_deg"],
        "path_margin_scale": 1.0, "path_margin_bias": 0.0,
    })

    path_fit = _fit_path_margin(path, partial, task_cfg)
    params = SimParams.from_dict({
        **partial.to_dict(),
        "path_margin_scale": path_fit["path_margin_scale"],
        "path_margin_bias": path_fit["path_margin_bias"],
    })

    diagnostics = {
        "stage1_speeds": speeds,
        "stage2_error_slopes": slopes,
        "stage3_noise": noise,
        "stage4_path": path_fit,
        "n_primitive": len(prim),
        "n_path": len(path),
    }
    return params, diagnostics
