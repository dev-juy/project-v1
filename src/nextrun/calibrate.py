"""
Staged calibration of SimParams from physical calibration runs (v4.1).

Calibration matrix (12 primitive runs, NO repetitions):
    forward: 12in & 36in at powers 0.4 & 0.7   (4 runs)
    strafe:  12in & 36in at powers 0.4 & 0.7   (4 runs)
    turn:    45deg & 135deg at powers 0.4 & 0.7 (4 runs)
Two magnitudes per (type,power) give the distance leverage needed to fit
error-per-magnitude slopes. Stochastic noise is estimated from the SIX
repeated path-calibration runs (3 poses x 2 reps), not from primitive reps.

Path-calibration rows log RAW terminal pose (final_x_in, final_y_in,
final_heading_deg, duration_s) per Decision B: the robot logs pose, heading,
duration, and status only; errors, margins, and success are computed at
ingest against the frozen thresholds in task_cfg.

Four staged least-squares problems, each <=6 params:
  1. speeds (6)      : through-origin distance/duration per (type, power)
  2. error slopes (3): through-origin residual-vs-commanded, per axis
  3. noise (2)       : pooled std of SIGNED final-pose residuals within
                       repeated path groups (not absolute error magnitudes,
                       which would fold the sign and bias the estimate)
  4. path margin (2) : OLS real_margin ~ scale*sim_margin + bias, with
                       per-run binding-axis, RMSE, R^2 and margin-spread
                       diagnostics (Decision A)

Input data is VALIDATED up front; incomplete matrices or unrepeated path
poses raise actionable ValueErrors instead of silently falling back to
defaults.

LEAKAGE FIREWALL: consumes ONLY calibration rows; never reads `trial` outcomes.
"""

from __future__ import annotations

from collections import defaultdict
import numpy as np

from .simulator import (
    SimParams, Simulator, POWER_LOW, POWER_HIGH,
    failure_margin, binding_axis, wrap_to_180,
)


# Minimum peak-to-peak spread of simulated margins across path-calibration
# runs for the scale/bias regression to be identifiable.
MIN_MARGIN_SPREAD = 0.05

# The frozen calibration cells (preregistered; validation requires EXACTLY
# these magnitudes at each power, not just any two distinct values).
FROZEN_MAGNITUDES = {
    "forward": (12.0, 36.0),   # inches
    "strafe": (12.0, 36.0),    # inches
    "turn": (45.0, 135.0),     # degrees
}
MAGNITUDE_TOL = 0.1


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


def _path_group_key(r: dict) -> tuple:
    return (round(float(r["start_x_in"]), 1), round(float(r["start_y_in"]), 1),
            round(float(r["start_heading_deg"]), 1), round(float(r["max_power"]), 2))


def _group_path(path: list[dict]) -> dict:
    groups = defaultdict(list)
    for r in path:
        groups[_path_group_key(r)].append(r)
    return groups


# --------------------------------------------------------------------------- #
# Input validation — actionable errors, never silent defaults
# --------------------------------------------------------------------------- #
def validate_calibration_data(prim: list[dict], path: list[dict]) -> None:
    """Check the full 12-run primitive matrix and repeated path groups.

    Raises ValueError listing every problem found, with the expected layout,
    instead of letting downstream stages silently fall back to defaults.
    """
    problems = []

    for mtype in ("forward", "strafe", "turn"):
        expected = FROZEN_MAGNITUDES[mtype]
        for power in (POWER_LOW, POWER_HIGH):
            found = set()
            for r in prim:
                if r.get("movement_type") != mtype:
                    continue
                if not _close_power(r.get("power", -1.0), power):
                    continue
                mag = (abs(float(r["commanded_angle_deg"])) if mtype == "turn"
                       else abs(float(r["commanded_distance_in"])))
                found.add(round(mag, 3))
            missing = [m for m in expected
                       if not any(abs(f - m) <= MAGNITUDE_TOL for f in found)]
            unexpected = [f for f in sorted(found)
                          if not any(abs(f - m) <= MAGNITUDE_TOL for m in expected)]
            if missing:
                problems.append(
                    f"primitive matrix incomplete: {mtype} @ power {power} "
                    f"missing frozen magnitude(s) {missing} "
                    f"(found {sorted(found) or '{}'})")
            if unexpected:
                problems.append(
                    f"primitive matrix off-spec: {mtype} @ power {power} has "
                    f"unexpected magnitude(s) {unexpected} — the matrix is "
                    f"frozen at {expected}")

    groups = _group_path(path)
    if len(groups) < 3:
        problems.append(
            f"only {len(groups)} distinct path-calibration pose(s) — need >=3 "
            f"difficulty levels (easy / medium / near-boundary, Decision A)")
    for key, rs in sorted(groups.items()):
        if len(rs) < 2:
            problems.append(
                f"path-calibration pose {key} has {len(rs)} run — each pose "
                f"needs >=2 repetitions to estimate noise")

    required_path_cols = ("final_x_in", "final_y_in", "final_heading_deg", "duration_s")
    for r in path:
        missing = [c for c in required_path_cols if c not in r]
        if missing:
            problems.append(
                f"path row {r.get('trial_uid', '?')} missing raw-pose fields "
                f"{missing} (Decision B: robot logs raw pose; errors computed at ingest)")
            break  # one schema report is enough

    if problems:
        raise ValueError(
            "Calibration data invalid:\n- " + "\n- ".join(problems) +
            "\nExpected: 12 primitive runs (frozen cells: forward/strafe "
            "12&36in, turn 45&135deg, each at powers 0.4 & 0.7, no reps) and "
            ">=3 path poses with >=2 reps each, logging raw final pose.")


# --------------------------------------------------------------------------- #
# Stage 1: speeds (6)
# --------------------------------------------------------------------------- #
def _fit_speeds(prim: list[dict]) -> dict:
    """Speed per (type, power) = through-origin slope of distance vs duration.

    With two magnitudes per power this is a genuine 2-point regression through
    the origin (speed is constant across distance). Data completeness is
    guaranteed by validate_calibration_data; an empty cell here raises rather
    than falling back to a default.
    """
    def speed(mtype: str, power: float) -> float:
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
            raise ValueError(
                f"no usable {mtype} runs at power {power} (all durations <= 0?) — "
                f"cannot fit speed; check the calibration log")
        return _through_origin_slope(np.asarray(durs), np.asarray(mags))

    return {
        "forward_speed_low":  speed("forward", POWER_LOW),
        "forward_speed_high": speed("forward", POWER_HIGH),
        "strafe_speed_low":   speed("strafe",  POWER_LOW),
        "strafe_speed_high":  speed("strafe",  POWER_HIGH),
        "turn_rate_low":      speed("turn",    POWER_LOW),
        "turn_rate_high":     speed("turn",    POWER_HIGH),
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
# Stage 3: noise (2), from SIGNED final-pose residuals in repeated path groups
# --------------------------------------------------------------------------- #
def _fit_noise(path: list[dict]) -> dict:
    """Pooled std of signed final_x / final_y / final_heading residuals
    within repeated path poses.

    Signed residuals (deviation from the group mean pose) matter: the std of
    ABSOLUTE error magnitudes folds the distribution at zero and biases the
    estimate — e.g. finals of [+0.5, -0.5] around the target have constant
    absolute error and would report zero noise. Heading residuals use
    wrap_to_180 relative to the group mean (poses stay far from the +/-180
    seam by design).

    Pooled variance across groups: sum of squared deviations / (N - n_groups).
    x and y residuals share one isotropic position std. Floors keep rollouts
    non-degenerate (a modeling floor, not a missing-data default).
    """
    groups = _group_path(path)

    ss_pos, df_pos = 0.0, 0
    ss_hdg, df_hdg = 0.0, 0
    n_groups_used = 0
    for rs in groups.values():
        if len(rs) < 2:
            continue
        n_groups_used += 1
        xs = np.array([float(r["final_x_in"]) for r in rs])
        ys = np.array([float(r["final_y_in"]) for r in rs])
        hs = np.array([float(r["final_heading_deg"]) for r in rs])
        ss_pos += float(np.sum((xs - xs.mean()) ** 2) + np.sum((ys - ys.mean()) ** 2))
        df_pos += 2 * (len(rs) - 1)
        h_dev = np.array([wrap_to_180(h - hs.mean()) for h in hs])
        ss_hdg += float(np.sum(h_dev ** 2))
        df_hdg += len(rs) - 1

    pos = float(np.sqrt(ss_pos / df_pos)) if df_pos > 0 else 0.0
    hdg = float(np.sqrt(ss_hdg / df_hdg)) if df_hdg > 0 else 0.0
    return {
        "position_noise_std_in": max(pos, 0.1),
        "heading_noise_std_deg": max(hdg, 0.1),
        "_noise_groups_used": n_groups_used,
    }


# --------------------------------------------------------------------------- #
# Stage 4: path margin (2), with Decision-A diagnostics
# --------------------------------------------------------------------------- #
def _fit_path_margin(path: list[dict], partial: SimParams, task_cfg: dict) -> dict:
    """OLS fit real_margin ~ scale*sim_margin + bias across all path runs.

    Real errors/margins are computed at ingest from the logged raw pose
    (Decision B). Diagnostics report, per Decision A: the binding axis of
    each simulated and real run, how often they match, fit residual RMSE,
    descriptive R^2, and the simulated-margin spread. A degenerate spread
    raises instead of silently returning identity.
    """
    prim_params = SimParams.from_dict(
        {**partial.to_dict(), "path_margin_scale": 1.0, "path_margin_bias": 0.0})
    sim = Simulator(prim_params, task_cfg)
    target = task_cfg["task"]["target"]
    tx, ty, th = float(target["x_in"]), float(target["y_in"]), float(target["heading_deg"])

    sim_m, real_m = [], []
    sim_axes, real_axes = [], []
    for r in path:
        res = sim.simulate_trial(
            float(r["start_x_in"]), float(r["start_y_in"]),
            float(r["start_heading_deg"]), float(r["max_power"]), rng=None)
        sim_m.append(res.margin)
        sim_axes.append(binding_axis(
            res.position_error_in, res.heading_error_deg, res.duration_s, task_cfg))

        # ingest: raw pose -> errors -> margin, against frozen thresholds
        pos_err = float(np.hypot(float(r["final_x_in"]) - tx,
                                 float(r["final_y_in"]) - ty))
        hdg_err = abs(wrap_to_180(float(r["final_heading_deg"]) - th))
        dur = float(r["duration_s"])
        real_m.append(failure_margin(pos_err, hdg_err, dur, task_cfg))
        real_axes.append(binding_axis(pos_err, hdg_err, dur, task_cfg))

    x = np.asarray(sim_m)
    y = np.asarray(real_m)
    spread = float(np.ptp(x))
    if spread < MIN_MARGIN_SPREAD:
        raise ValueError(
            f"path-margin fit unidentifiable: simulated margins span only "
            f"{spread:.4f} (< {MIN_MARGIN_SPREAD}) across {len(path)} runs — "
            f"the path-calibration poses must span difficulty levels so the "
            f"primitive simulator predicts distinctly different margins "
            f"(margins: {np.round(x, 4).tolist()})")

    A = np.vstack([x, np.ones_like(x)]).T
    (scale, bias), *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = scale * x + bias
    resid = y - pred
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1.0 - np.sum(resid ** 2) / ss_tot) if ss_tot > 0 else float("nan")

    def _counts(axes):
        return {a: axes.count(a) for a in ("position", "heading", "duration")}

    return {
        "path_margin_scale": float(scale), "path_margin_bias": float(bias),
        "_path_fit_ok": True, "_path_n": len(path),
        "residuals": resid.tolist(),
        "rmse": rmse,
        "r2": r2,  # descriptive only at n=6: report it, never gate on it
        "sim_margin_spread": spread,
        "sim_binding_axes": sim_axes,
        "real_binding_axes": real_axes,
        "sim_binding_axis_counts": _counts(sim_axes),
        "real_binding_axis_counts": _counts(real_axes),
        "binding_axis_matches": int(sum(s == r for s, r in zip(sim_axes, real_axes))),
    }


# --------------------------------------------------------------------------- #
# public entry point
# --------------------------------------------------------------------------- #
def fit_simulator(calibration_primitive, calibration_path,
                  task_cfg: dict, seed: int = 0) -> tuple[SimParams, dict]:
    """Fit SimParams via four staged least-squares problems.

    calibration_primitive : 12 primitive runs (2 magnitudes x 2 powers x 3 types)
    calibration_path       : >=6 path runs (>=3 poses x >=2 reps), raw pose logged
    Consumes ONLY calibration data. Raises ValueError on incomplete inputs.
    """
    prim = _rows(calibration_primitive)
    path = _rows(calibration_path)
    validate_calibration_data(prim, path)

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
