"""
Round-trip identifiability tests for simulator calibration (v4.1).

Generate synthetic calibration data from a KNOWN SimParams with nonzero
positive error slopes, then fit and assert the five required properties:

  1. speed parameters recovered
  2. error-per-distance / error-per-angle slopes recovered
  3. simulated margins are NONCONSTANT across scenarios
  4. path-margin scale and bias recovered
  5. p_success differs across scenarios (discrimination)

Path-calibration rows log RAW terminal pose (Decision B); errors and margins
are computed at ingest by the fitter. The synthetic path runs deliberately
VARY their binding axis (position / heading / duration bound) per Decision A —
the scalar margin fit must work across mixed axes, and the fitter reports
binding-axis agreement, residuals, RMSE, and R^2 as diagnostics rather than
constraining the scenarios.

The p_success check verifies DISCRIMINATION under known nonzero slopes; it does
NOT enforce universal monotonicity on real fitted data.

If the fit cannot recover the slopes or path-margin params, the tests fail and
we stop and report an identifiability failure.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nextrun.simulator import (
    SimParams, Simulator, POWER_LOW, POWER_HIGH, failure_margin, binding_axis,
)
from nextrun.calibrate import fit_simulator, _fit_noise, validate_calibration_data


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

# Known params with nonzero POSITIVE slopes -> harder scenarios accumulate more
# error -> nonconstant margins. Slopes are what stage 2 must recover.
TRUE = SimParams(
    forward_speed_low=24.0, forward_speed_high=42.0,
    strafe_speed_low=18.0, strafe_speed_high=33.0,
    turn_rate_low=100.0, turn_rate_high=190.0,
    forward_error_per_in=0.06, strafe_error_per_in=0.05,
    turn_error_per_deg=0.04,
    position_noise_std_in=0.4, heading_noise_std_deg=1.0,
    path_margin_scale=1.3, path_margin_bias=0.1,
)

# Calibration matrix: 2 magnitudes x 2 powers x 3 types = 12, no reps.
PRIM_SPECS = [
    ("forward", 12.0, 0.0), ("forward", 36.0, 0.0),
    ("strafe", 12.0, 0.0), ("strafe", 36.0, 0.0),
    ("turn", 0.0, 45.0), ("turn", 0.0, 135.0),
]

# Path poses spanning easy -> near-boundary, each with a DIFFERENT axis binding
# the real margin (Decision A: vary translation and heading; do not constrain
# poses to be position-dominant).
PATH_POSES = [
    (66.0, 72.0, 0.0, 0.5),    # easy
    (60.0, 64.0, 20.0, 0.5),   # medium
    (52.0, 56.0, 45.0, 0.7),   # near-boundary
]
PATH_BIND_AXES = ["position", "heading", "duration"]


def make_primitive_rows(params: SimParams, rng=None) -> list[dict]:
    sim = Simulator(params, TASK_CFG)
    rows = []
    tid = 0
    for mtype, dist, ang in PRIM_SPECS:
        for power in (POWER_LOW, POWER_HIGH):
            out = sim.simulate_calibration(mtype, dist, ang, power, rng=rng)
            rows.append({
                "trial_uid": f"cal_{tid}", "movement_type": mtype,
                "commanded_distance_in": dist, "commanded_angle_deg": ang,
                "power": power, "duration_s": out["duration_s"],
                "actual_dx_in": out["dx"], "actual_dy_in": out["dy"],
                "actual_turn_deg": out["dturn"],
            })
            tid += 1
    return rows


def make_path_rows(params: SimParams, rng=None, reps=2) -> list[dict]:
    """Synthetic path-calibration runs logging RAW terminal pose (Decision B).

    Generative model of the "real robot": its terminal margin is the primitive
    prediction transformed by the true path scale/bias (that IS the sim-to-real
    gap stage 4 must recover), realized on a per-pose BINDING AXIS so the six
    runs mix position-, heading-, and duration-bound failures (Decision A).
    Gaussian pose noise (params' true stds) is added on top when rng is given,
    which is what stage 3 estimates from the repeats.
    """
    prim_params = SimParams.from_dict(
        {**params.to_dict(), "path_margin_scale": 1.0, "path_margin_bias": 0.0})
    prim_sim = Simulator(prim_params, TASK_CFG)
    s = TASK_CFG["task"]["success"]
    tx, ty, th = prim_sim.target

    rows = []
    i = 0
    for (sx, sy, sh, pw), axis in zip(PATH_POSES, PATH_BIND_AXES):
        base = prim_sim.simulate_trial(sx, sy, sh, pw, rng=None)
        m1 = params.path_margin_scale * base.margin + params.path_margin_bias
        # realize margin m1 on the chosen axis; keep other axes at 0.3*m1
        # (base duration ratio is <= base.margin < m1, so it never binds)
        if axis == "position":
            pos_err, hdg_err, dur = m1 * s["position_error_in_max"], \
                0.3 * m1 * s["heading_error_deg_max"], base.duration_s
        elif axis == "heading":
            pos_err, hdg_err, dur = 0.3 * m1 * s["position_error_in_max"], \
                m1 * s["heading_error_deg_max"], base.duration_s
        else:  # duration
            pos_err, hdg_err, dur = 0.3 * m1 * s["position_error_in_max"], \
                0.3 * m1 * s["heading_error_deg_max"], m1 * s["duration_s_max"]

        for _ in range(reps):
            fx, fy, fh = tx + pos_err, ty, th + hdg_err
            if rng is not None:
                fx += rng.normal(0.0, params.position_noise_std_in)
                fy += rng.normal(0.0, params.position_noise_std_in)
                fh += rng.normal(0.0, params.heading_noise_std_deg)
            rows.append({
                "trial_uid": f"path_{i}",
                "start_x_in": sx, "start_y_in": sy,
                "start_heading_deg": sh, "max_power": pw,
                "final_x_in": fx, "final_y_in": fy,
                "final_heading_deg": fh, "duration_s": dur,
            })
            i += 1
    return rows


# --------------------------------------------------------------------------- #
# THE round-trip test (five required properties), noise-free
# --------------------------------------------------------------------------- #
def test_round_trip():
    prim = make_primitive_rows(TRUE, rng=None)
    path = make_path_rows(TRUE, rng=None, reps=2)
    fitted, diag = fit_simulator(prim, path, TASK_CFG, seed=0)

    # (1) speeds
    for name in ["forward_speed_low", "forward_speed_high",
                 "strafe_speed_low", "strafe_speed_high",
                 "turn_rate_low", "turn_rate_high"]:
        assert getattr(fitted, name) == pytest.approx(getattr(TRUE, name), rel=1e-6), \
            f"(1) speed {name}: {getattr(fitted, name)} vs {getattr(TRUE, name)}"

    # (2) error slopes
    for name in ["forward_error_per_in", "strafe_error_per_in", "turn_error_per_deg"]:
        assert getattr(fitted, name) == pytest.approx(getattr(TRUE, name), rel=1e-6), \
            f"(2) slope {name}: {getattr(fitted, name)} vs {getattr(TRUE, name)}"

    # (3) nonconstant simulated margins across scenarios (primitive-only sim)
    prim_sim = Simulator(
        SimParams.from_dict({**fitted.to_dict(),
                             "path_margin_scale": 1.0, "path_margin_bias": 0.0}),
        TASK_CFG)
    margins = [prim_sim.simulate_trial(sx, sy, sh, pw, rng=None).margin
               for sx, sy, sh, pw in PATH_POSES]
    assert np.ptp(margins) > 0.05, \
        f"(3) simulated margins ~constant: {margins}"

    # (4) path-margin scale and bias, across MIXED binding axes
    assert diag["stage4_path"]["_path_fit_ok"], "(4) path margin fit did not run"
    assert fitted.path_margin_scale == pytest.approx(TRUE.path_margin_scale, rel=1e-3), \
        f"(4) path_margin_scale: {fitted.path_margin_scale} vs {TRUE.path_margin_scale}"
    assert fitted.path_margin_bias == pytest.approx(TRUE.path_margin_bias, abs=1e-3), \
        f"(4) path_margin_bias: {fitted.path_margin_bias} vs {TRUE.path_margin_bias}"

    # (5) p_success differs across scenarios. Use scenarios spanning from easy
    #     to hard enough to straddle the failure boundary, so discrimination is
    #     observable (an all-easy or all-hard set would be uniformly 1.0/0.0).
    sim = Simulator(fitted, TASK_CFG)
    rng = np.random.default_rng(7)
    tx, ty, _ = sim.target
    discrimination_poses = [
        (tx - 4.0, ty, 0.0, 0.5),      # very easy
        (tx - 40.0, ty - 30.0, 30.0, 0.6),  # moderate
        (tx - 68.0, ty - 60.0, 45.0, 0.7),  # hard
    ]
    ps = [sim.p_success({"start_x_in": sx, "start_y_in": sy,
                         "start_heading_deg": sh, "max_power": pw},
                        n=400, rng=rng)
          for sx, sy, sh, pw in discrimination_poses]
    assert np.ptp(ps) > 0.05, f"(5) p_success does not discriminate: {ps}"


def test_round_trip_with_noise():
    """With process noise and more path reps, structural params recover within
    tolerance and noise std is in the right ballpark."""
    rng = np.random.default_rng(20260714)
    prim = make_primitive_rows(TRUE, rng=rng)          # single run each (as real)
    path = make_path_rows(TRUE, rng=rng, reps=8)       # more reps for stable noise est
    fitted, diag = fit_simulator(prim, path, TASK_CFG, seed=0)

    for name in ["forward_speed_low", "forward_speed_high",
                 "strafe_speed_low", "strafe_speed_high",
                 "turn_rate_low", "turn_rate_high"]:
        assert getattr(fitted, name) == pytest.approx(getattr(TRUE, name), rel=0.10), \
            f"speed {name}: {getattr(fitted, name)} vs {getattr(TRUE, name)}"

    for name in ["forward_error_per_in", "strafe_error_per_in", "turn_error_per_deg"]:
        assert getattr(fitted, name) == pytest.approx(getattr(TRUE, name), abs=0.03), \
            f"slope {name}: {getattr(fitted, name)} vs {getattr(TRUE, name)}"

    assert fitted.position_noise_std_in == pytest.approx(TRUE.position_noise_std_in, rel=0.6)
    assert fitted.heading_noise_std_deg == pytest.approx(TRUE.heading_noise_std_deg, rel=0.6)


def test_slopes_not_forced_positive():
    """A robot that OVERSHOOTS has negative shortfall error. The fitter must
    recover a negative slope, proving we don't hardcode positive difficulty."""
    over = SimParams.from_dict({**TRUE.to_dict(),
                                "forward_error_per_in": -0.04})
    prim = make_primitive_rows(over, rng=None)
    fitted, _ = fit_simulator(prim, make_path_rows(over, rng=None), TASK_CFG)
    assert fitted.forward_error_per_in == pytest.approx(-0.04, rel=1e-6), \
        f"negative slope not recovered: {fitted.forward_error_per_in}"


def test_uncalibrated_differs_from_truth():
    unc = SimParams.uncalibrated()
    assert unc.forward_speed_low != pytest.approx(TRUE.forward_speed_low, rel=0.01)
    assert unc.path_margin_scale == pytest.approx(1.0)


def test_zero_magnitude_zero_error():
    """Through-origin design: zero commanded magnitude -> zero systematic error."""
    sim = Simulator(TRUE, TASK_CFG)
    out = sim.simulate_calibration("forward", 0.0, 0.0, 0.5, rng=None)
    assert out["dx"] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Patch 1 regressions: per-axis signed error components, never averaged
# --------------------------------------------------------------------------- #
def test_axis_slopes_not_averaged():
    """Anisotropic slopes must survive into simulate_trial: a pure-x approach
    uses ONLY the forward slope, a pure-y approach ONLY the strafe slope.
    (The old isotropic model averaged them, giving 0.05*20=1.0 for both.)"""
    p = SimParams.from_dict({**TRUE.to_dict(),
                             "forward_error_per_in": 0.1,
                             "strafe_error_per_in": 0.0,
                             "path_margin_scale": 1.0, "path_margin_bias": 0.0})
    sim = Simulator(p, TASK_CFG)
    rx = sim.simulate_trial(72.0 - 20.0, 72.0, 0.0, 0.5, rng=None)
    ry = sim.simulate_trial(72.0, 72.0 - 20.0, 0.0, 0.5, rng=None)
    assert rx.position_error_in == pytest.approx(0.1 * 20.0)
    assert ry.position_error_in == pytest.approx(0.0)


def test_signed_error_components_preserved():
    """Sign must be kept per component: a negative (overshoot) forward slope
    lands PAST the target on x; a positive turn slope undershoots the heading
    in the direction of the commanded turn."""
    p = SimParams.from_dict({**TRUE.to_dict(),
                             "forward_error_per_in": -0.05,
                             "path_margin_scale": 1.0, "path_margin_bias": 0.0})
    sim = Simulator(p, TASK_CFG)
    r = sim.simulate_trial(72.0 - 20.0, 72.0, 0.0, 0.5, rng=None)
    assert r.final_x_in == pytest.approx(73.0)  # 72 - (-0.05*20)
    assert r.final_x_in > 72.0

    # commanded turn of +40deg (start_heading -40 -> target 0), slope 0.04:
    # heading falls short by 1.6deg on the same side as the turn
    r2 = sim.simulate_trial(72.0 - 10.0, 72.0, -40.0, 0.5, rng=None)
    assert r2.final_heading_deg == pytest.approx(-1.6)


def test_translation_duration_anisotropic():
    """Equal-distance pure-forward and pure-strafe moves must take DIFFERENT
    times when the axis speeds differ — in both translation modes. (The old
    model averaged the speeds, giving identical durations.)"""
    # TRUE at power 0.4: forward 24 in/s, strafe 18 in/s
    pp = Simulator(TRUE, TASK_CFG)  # pose_to_pose default
    fwd = pp.simulate_trial(72.0 - 20.0, 72.0, 0.0, POWER_LOW, rng=None)
    stf = pp.simulate_trial(72.0, 72.0 - 20.0, 0.0, POWER_LOW, rng=None)
    assert fwd.duration_s == pytest.approx(20.0 / 24.0)
    assert stf.duration_s == pytest.approx(20.0 / 18.0)
    assert fwd.duration_s != pytest.approx(stf.duration_s)

    seq = Simulator(TRUE, TASK_CFG, translation_mode="sequential")
    fwd_s = seq.simulate_trial(72.0 - 20.0, 72.0, 0.0, POWER_LOW, rng=None)
    stf_s = seq.simulate_trial(72.0, 72.0 - 20.0, 0.0, POWER_LOW, rng=None)
    assert fwd_s.duration_s == pytest.approx(20.0 / 24.0)
    assert stf_s.duration_s == pytest.approx(20.0 / 18.0)


def test_translation_modes_diagonal():
    """Diagonal move: sequential sums per-axis times; pose_to_pose uses the
    elliptical envelope (quadrature), which is strictly faster."""
    diag_pose = (72.0 - 12.0, 72.0 - 9.0, 0.0, POWER_LOW)  # dx=12, dy=9
    seq = Simulator(TRUE, TASK_CFG, translation_mode="sequential")
    pp = Simulator(TRUE, TASK_CFG, translation_mode="pose_to_pose")
    d_seq = seq.simulate_trial(*diag_pose, rng=None).duration_s
    d_pp = pp.simulate_trial(*diag_pose, rng=None).duration_s
    assert d_seq == pytest.approx(12.0 / 24.0 + 9.0 / 18.0)
    assert d_pp == pytest.approx(np.hypot(12.0 / 24.0, 9.0 / 18.0))
    assert d_pp < d_seq

    with pytest.raises(ValueError, match="translation_mode"):
        Simulator(TRUE, TASK_CFG, translation_mode="teleport")


# --------------------------------------------------------------------------- #
# Patch 2 regression: noise from SIGNED residuals, not absolute magnitudes
# --------------------------------------------------------------------------- #
def test_noise_from_signed_residuals_not_abs():
    """Finals landing symmetrically about the group mean have CONSTANT absolute
    error — the old abs-magnitude estimator reported ~zero noise for them. The
    signed-residual estimator must see the real spread."""
    rows = []
    for g, (sx, sy, sh, pw) in enumerate(PATH_POSES):
        for sign in (+1.0, -1.0):
            rows.append({
                "trial_uid": f"n_{g}_{sign}",
                "start_x_in": sx, "start_y_in": sy,
                "start_heading_deg": sh, "max_power": pw,
                "final_x_in": 72.0 + sign * 0.5,   # |x error| constant 0.5
                "final_y_in": 72.0,
                "final_heading_deg": sign * 1.0,   # |heading error| constant 1.0
                "duration_s": 1.0,
            })
    noise = _fit_noise(rows)
    # x devs +-0.5 (ss=0.5/group), y flat: pooled = sqrt(3*0.5 / 6) = 0.5
    assert noise["position_noise_std_in"] == pytest.approx(0.5, rel=1e-6)
    # heading devs +-1.0 (ss=2.0/group): sqrt(3*2.0 / 3) = sqrt(2)
    assert noise["heading_noise_std_deg"] == pytest.approx(np.sqrt(2.0), rel=1e-6)
    assert noise["_noise_groups_used"] == 3


# --------------------------------------------------------------------------- #
# Patch 3 regressions: varied binding axes + Decision-A diagnostics
# --------------------------------------------------------------------------- #
def test_path_margin_diagnostics_varied_axes():
    prim = make_primitive_rows(TRUE, rng=None)
    path = make_path_rows(TRUE, rng=None, reps=2)
    _, diag = fit_simulator(prim, path, TASK_CFG, seed=0)
    d = diag["stage4_path"]

    # the six real runs mix binding axes — all three axes appear
    counts = d["real_binding_axis_counts"]
    assert sum(counts.values()) == 6
    assert counts == {"position": 2, "heading": 2, "duration": 2}
    assert set(d["sim_binding_axis_counts"]) == {"position", "heading", "duration"}
    assert 0 <= d["binding_axis_matches"] <= 6

    # fit-quality diagnostics are reported; noise-free synthetic fit is exact
    assert len(d["residuals"]) == 6
    assert d["rmse"] < 1e-9
    assert d["r2"] > 0.999
    assert d["sim_margin_spread"] > 0.05


def test_binding_axis_helper():
    assert binding_axis(4.0, 0.0, 0.0, TASK_CFG) == "position"
    assert binding_axis(0.0, 8.0, 0.0, TASK_CFG) == "heading"
    assert binding_axis(0.4, 0.8, 5.9, TASK_CFG) == "duration"


# --------------------------------------------------------------------------- #
# Patch 5 regressions: validation raises actionable errors, no silent defaults
# --------------------------------------------------------------------------- #
def test_incomplete_primitive_matrix_raises():
    prim = [r for r in make_primitive_rows(TRUE)
            if not (r["movement_type"] == "forward"
                    and r["commanded_distance_in"] == 36.0)]
    with pytest.raises(ValueError, match="forward.*missing frozen magnitude"):
        fit_simulator(prim, make_path_rows(TRUE), TASK_CFG)


def test_wrong_magnitude_rejected_even_if_distinct():
    """Two distinct magnitudes are NOT enough — the matrix is frozen at
    12&36in / 45&135deg. A 24in run in place of 36in must be rejected."""
    prim = make_primitive_rows(TRUE)
    for r in prim:
        if r["movement_type"] == "forward" and r["commanded_distance_in"] == 36.0:
            r["commanded_distance_in"] = 24.0
    with pytest.raises(ValueError, match="frozen"):
        fit_simulator(prim, make_path_rows(TRUE), TASK_CFG)


def test_unrepeated_path_pose_raises():
    path = make_path_rows(TRUE, reps=1)
    with pytest.raises(ValueError, match="repetitions"):
        fit_simulator(make_primitive_rows(TRUE), path, TASK_CFG)


def test_too_few_path_poses_raises():
    path = [r for r in make_path_rows(TRUE)
            if float(r["start_x_in"]) != PATH_POSES[2][0]]
    with pytest.raises(ValueError, match="difficulty levels"):
        fit_simulator(make_primitive_rows(TRUE), path, TASK_CFG)


def test_missing_raw_pose_fields_raises():
    path = make_path_rows(TRUE)
    for r in path:
        del r["final_x_in"]
    with pytest.raises(ValueError, match="raw-pose fields"):
        validate_calibration_data(make_primitive_rows(TRUE), path)


def test_degenerate_margin_spread_raises():
    """Three valid, repeated poses whose PREDICTED margins are nearly equal
    (same 6in offset from target) leave the scale unidentifiable — the fitter
    must raise with the measured spread, not silently return identity."""
    prim = make_primitive_rows(TRUE)
    degenerate_poses = [(78.0, 72.0, 0.0, 0.5), (66.0, 72.0, 0.0, 0.5),
                        (72.0, 66.0, 0.0, 0.5)]
    path = []
    for g, (sx, sy, sh, pw) in enumerate(degenerate_poses):
        for rep in range(2):
            path.append({
                "trial_uid": f"deg_{g}_{rep}",
                "start_x_in": sx, "start_y_in": sy,
                "start_heading_deg": sh, "max_power": pw,
                "final_x_in": 72.4, "final_y_in": 72.0,
                "final_heading_deg": 0.5, "duration_s": 0.3,
            })
    with pytest.raises(ValueError, match="unidentifiable"):
        fit_simulator(prim, path, TASK_CFG)
