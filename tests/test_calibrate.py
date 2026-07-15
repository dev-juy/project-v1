"""
Round-trip identifiability test for simulator calibration (v4.1).

Generate synthetic calibration data from a KNOWN SimParams with nonzero
positive error slopes, then fit and assert the five required properties:

  1. speed parameters recovered
  2. error-per-distance / error-per-angle slopes recovered
  3. simulated margins are NONCONSTANT across scenarios
  4. path-margin scale and bias recovered
  5. p_success differs across scenarios (discrimination)

The p_success check verifies DISCRIMINATION under known nonzero slopes; it does
NOT enforce universal monotonicity on real fitted data.

If the fit cannot recover the slopes or path-margin params, the test fails and
we stop and report an identifiability failure.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nextrun.simulator import (
    SimParams, Simulator, POWER_LOW, POWER_HIGH, failure_margin,
)
from nextrun.calibrate import fit_simulator


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

# Path poses spanning easy->hard; each repeated twice (6 runs).
PATH_POSES = [
    (66.0, 72.0, 0.0, 0.5),    # easy: 6in translation, no turn
    (60.0, 64.0, 20.0, 0.5),   # medium
    (52.0, 56.0, 45.0, 0.7),   # hard: large translation + turn
]


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
    """Path-calib runs modelling a genuine sim-to-real margin gap.

    The PHYSICAL robot's terminal margin is the primitive-predicted margin
    transformed by the true path scale/bias (that IS the sim-to-real gap we
    want stage 4 to recover). We must therefore store COMPONENT errors whose
    margin equals that corrected value -- otherwise the fitter recomputes the
    uncorrected margin from the components and the scale collapses to 1.0.

    We inflate the position error so that the resulting margin equals the
    corrected margin. (Heading/duration held; position is the dominant axis
    here.) This makes real_margin = scale*sim_margin + bias reconstructable
    from stored components, matching clarification #1.
    """
    # primitive-only sim = params with identity path correction
    prim_params = SimParams.from_dict(
        {**params.to_dict(), "path_margin_scale": 1.0, "path_margin_bias": 0.0})
    prim_sim = Simulator(prim_params, TASK_CFG)
    pos_t = TASK_CFG["task"]["success"]["position_error_in_max"]

    rows = []
    i = 0
    for sx, sy, sh, pw in PATH_POSES:
        for _ in range(reps):
            base = prim_sim.simulate_trial(sx, sy, sh, pw, rng=rng)
            sim_margin = base.margin
            real_margin = params.path_margin_scale * sim_margin + params.path_margin_bias
            # choose a position error that realizes real_margin on the pos axis,
            # provided pos is the binding axis (it is for these poses).
            real_pos_err = real_margin * pos_t
            rows.append({
                "trial_uid": f"path_{i}",
                "start_x_in": sx, "start_y_in": sy,
                "start_heading_deg": sh, "max_power": pw,
                "position_error_in": real_pos_err,
                "heading_error_deg": base.heading_error_deg,
                "duration_s": base.duration_s,
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

    # (4) path-margin scale and bias
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
