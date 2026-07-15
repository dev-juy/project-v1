"""
NextRun 2D behavioral simulator.

Outcome model, not a physics engine and not a path follower. Given a start pose
and power it predicts the terminal error distribution of the autonomous
controller. It does NOT simulate PedroPathing feedback; the path_margin_* terms
calibrate the primitive prediction against the controller's observed terminal
margin (calibrate.py stage 4).

KEY DESIGN (v4.1): terminal error SCALES with commanded magnitude.
Earlier the terminal error was a constant bias, which made simulated margins
nearly identical across scenarios -> path-margin unidentifiable and Method E's
p_success signal flat. Now:

    x displacement error (signed) = forward_error_per_in * dx_commanded
    y displacement error (signed) = strafe_error_per_in  * dy_commanded
    heading error       (signed)  = turn_error_per_deg   * heading_change

Error components are SIGNED and kept PER AXIS — forward and strafe slopes are
never averaged, so anisotropy between the two translation axes survives into
diagonal moves. Errors pass through the origin (zero commanded magnitude ->
zero error) so each slope is a single identifiable parameter. Slopes are FIT
from data and may be any sign; nothing here forces positive slopes or
monotonic difficulty.

Translation DURATION is also anisotropic (see Simulator.TRANSLATION_MODES):
the default "pose_to_pose" branch uses an elliptical speed-envelope
approximation hypot(dx/forward_speed, dy/strafe_speed); the "sequential"
branch (Path B fallback) sums the per-axis times.

Conventions
-----------
Field frame: x right, y up, heading degrees CCW from +x. Calibration primitives
start at heading 0, so 'forward' == +x, 'strafe' == +y.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, fields
from typing import Optional
import math

import numpy as np


POWER_LOW = 0.4
POWER_HIGH = 0.7


@dataclass
class SimParams:
    # -- Stage 1: primitive speeds (6). Fit from two distances x two powers.
    forward_speed_low: float
    forward_speed_high: float
    strafe_speed_low: float
    strafe_speed_high: float
    turn_rate_low: float
    turn_rate_high: float

    # -- Stage 2: error-per-magnitude slopes (3), through origin.
    forward_error_per_in: float
    strafe_error_per_in: float
    turn_error_per_deg: float

    # -- Stage 3: stochastic noise (2), from the six path repeats.
    position_noise_std_in: float
    heading_noise_std_deg: float

    # -- Stage 4: path-following margin correction (2).
    path_margin_scale: float
    path_margin_bias: float

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SimParams":
        names = {f.name for f in fields(cls)}
        return cls(**{k: float(v) for k, v in d.items() if k in names})

    @classmethod
    def param_names(cls) -> list[str]:
        return [f.name for f in fields(cls)]

    @classmethod
    def uncalibrated(cls) -> "SimParams":
        return cls(
            forward_speed_low=20.0, forward_speed_high=40.0,
            strafe_speed_low=15.0, strafe_speed_high=30.0,
            turn_rate_low=90.0, turn_rate_high=180.0,
            forward_error_per_in=0.02, strafe_error_per_in=0.02,
            turn_error_per_deg=0.02,
            position_noise_std_in=1.0, heading_noise_std_deg=2.0,
            path_margin_scale=1.0, path_margin_bias=0.0,
        )


@dataclass
class SimResult:
    final_x_in: float
    final_y_in: float
    final_heading_deg: float
    duration_s: float
    position_error_in: float
    heading_error_deg: float
    margin: float
    success: bool


def wrap_to_180(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def failure_margin(position_error_in: float,
                   heading_error_deg: float,
                   duration_s: float,
                   task_cfg: dict) -> float:
    """margin = max over axes of (value / threshold). <=1 success, >1 failure."""
    s = task_cfg["task"]["success"]
    return max(
        position_error_in / s["position_error_in_max"],
        heading_error_deg / s["heading_error_deg_max"],
        duration_s / s["duration_s_max"],
    )


def binding_axis(position_error_in: float,
                 heading_error_deg: float,
                 duration_s: float,
                 task_cfg: dict) -> str:
    """Which axis binds the margin: argmax of (value / threshold)."""
    s = task_cfg["task"]["success"]
    ratios = {
        "position": position_error_in / s["position_error_in_max"],
        "heading": heading_error_deg / s["heading_error_deg_max"],
        "duration": duration_s / s["duration_s_max"],
    }
    return max(ratios, key=ratios.get)


class Simulator:
    # How translation duration is modeled (forward/strafe anisotropy is
    # preserved in BOTH branches — the speeds are never averaged):
    #
    #   "pose_to_pose" (default, Path A): the holonomic controller drives both
    #     axes simultaneously toward the target. Duration uses the ELLIPTICAL
    #     SPEED-ENVELOPE approximation, hypot(dx/forward_speed,
    #     dy/strafe_speed): the drive's max velocity traces an axis-aligned
    #     ellipse with radii (forward_speed, strafe_speed), so straight-line
    #     speed along direction theta is 1/sqrt((cos/vf)^2 + (sin/vs)^2).
    #     Exact for pure-axis moves; a stated approximation for diagonals.
    #
    #   "sequential" (Path B, primitive-composite fallback): axes run one
    #     after another, so durations add:
    #     abs(dx)/forward_speed + abs(dy)/strafe_speed.
    TRANSLATION_MODES = ("pose_to_pose", "sequential")

    def __init__(self, params: SimParams, task_cfg: dict,
                 translation_mode: str = "pose_to_pose"):
        if translation_mode not in self.TRANSLATION_MODES:
            raise ValueError(
                f"unknown translation_mode: {translation_mode!r} "
                f"(expected one of {self.TRANSLATION_MODES})")
        self.p = params
        self.task_cfg = task_cfg
        self.translation_mode = translation_mode
        t = task_cfg["task"]["target"]
        self.target = (float(t["x_in"]), float(t["y_in"]), float(t["heading_deg"]))

    def _interp(self, lo: float, hi: float, power: float) -> float:
        p = min(max(power, POWER_LOW), POWER_HIGH)
        frac = (p - POWER_LOW) / (POWER_HIGH - POWER_LOW)
        return lo + (hi - lo) * frac

    def forward_speed(self, power: float) -> float:
        return self._interp(self.p.forward_speed_low, self.p.forward_speed_high, power)

    def strafe_speed(self, power: float) -> float:
        return self._interp(self.p.strafe_speed_low, self.p.strafe_speed_high, power)

    def turn_rate(self, power: float) -> float:
        return self._interp(self.p.turn_rate_low, self.p.turn_rate_high, power)

    def simulate_calibration(self,
                             movement_type: str,
                             commanded_distance_in: float,
                             commanded_angle_deg: float,
                             power: float,
                             rng: Optional[np.random.Generator] = None) -> dict:
        """Predict one primitive run. Returns {dx, dy, dturn, duration_s}.

        Terminal error is slope * commanded_magnitude on the moving axis.
        Actual displacement = commanded - error (a shortfall). Noise added
        only when rng is provided.
        """
        dx = dy = dturn = 0.0
        duration = 0.0

        if movement_type == "forward":
            speed = self.forward_speed(power)
            duration = commanded_distance_in / speed if speed > 0 else 0.0
            err = self.p.forward_error_per_in * commanded_distance_in
            dx = commanded_distance_in - err
        elif movement_type == "strafe":
            speed = self.strafe_speed(power)
            duration = commanded_distance_in / speed if speed > 0 else 0.0
            err = self.p.strafe_error_per_in * commanded_distance_in
            dy = commanded_distance_in - err
        elif movement_type == "turn":
            rate = self.turn_rate(power)
            duration = abs(commanded_angle_deg) / rate if rate > 0 else 0.0
            err = self.p.turn_error_per_deg * abs(commanded_angle_deg)
            dturn = commanded_angle_deg - math.copysign(err, commanded_angle_deg)
        else:
            raise ValueError(f"unknown movement_type: {movement_type!r}")

        if rng is not None:
            if movement_type in ("forward", "strafe"):
                dx += rng.normal(0.0, self.p.position_noise_std_in)
                dy += rng.normal(0.0, self.p.position_noise_std_in)
            else:
                dturn += rng.normal(0.0, self.p.heading_noise_std_deg)

        return {"dx": dx, "dy": dy, "dturn": dturn, "duration_s": duration}

    def _raw_terminal(self,
                      start_x: float, start_y: float, start_heading: float,
                      max_power: float,
                      rng: Optional[np.random.Generator]) -> tuple:
        tx, ty, th = self.target

        ddx = tx - start_x
        ddy = ty - start_y

        vf = self.forward_speed(max_power)
        vs = self.strafe_speed(max_power)
        tx_time = abs(ddx) / vf if vf > 0 else 0.0
        ty_time = abs(ddy) / vs if vs > 0 else 0.0
        if self.translation_mode == "sequential":
            trans_time = tx_time + ty_time
        else:  # pose_to_pose: elliptical speed-envelope (see class docstring)
            trans_time = math.hypot(tx_time, ty_time)

        dtheta = wrap_to_180(th - start_heading)
        rate = self.turn_rate(max_power)
        rot_time = abs(dtheta) / rate if rate > 0 else 0.0
        duration = trans_time + rot_time

        # Signed per-axis error components, matching the primitive shortfall
        # model (actual = commanded - slope*commanded). The x displacement uses
        # the forward slope, the y displacement the strafe slope — they are NOT
        # averaged, so anisotropy between forward and strafe is preserved and
        # a diagonal move gets each axis's own error.
        err_x = self.p.forward_error_per_in * ddx
        err_y = self.p.strafe_error_per_in * ddy
        err_h = self.p.turn_error_per_deg * dtheta

        fx = tx - err_x
        fy = ty - err_y
        fh = th - err_h

        if rng is not None:
            fx += rng.normal(0.0, self.p.position_noise_std_in)
            fy += rng.normal(0.0, self.p.position_noise_std_in)
            fh += rng.normal(0.0, self.p.heading_noise_std_deg)

        return fx, fy, fh, duration

    def _corrected_margin(self, raw_margin: float) -> float:
        return self.p.path_margin_scale * raw_margin + self.p.path_margin_bias

    def simulate_trial(self,
                       start_x: float, start_y: float, start_heading: float,
                       max_power: float,
                       rng: Optional[np.random.Generator] = None) -> SimResult:
        fx, fy, fh, duration = self._raw_terminal(
            start_x, start_y, start_heading, max_power, rng)
        tx, ty, th = self.target

        pos_err = math.hypot(fx - tx, fy - ty)
        hdg_err = abs(wrap_to_180(fh - th))

        raw_margin = failure_margin(pos_err, hdg_err, duration, self.task_cfg)
        margin = self._corrected_margin(raw_margin)
        success = margin <= 1.0

        return SimResult(
            final_x_in=fx, final_y_in=fy, final_heading_deg=fh,
            duration_s=duration,
            position_error_in=pos_err, heading_error_deg=hdg_err,
            margin=margin, success=success,
        )

    def p_success(self, scenario: dict, n: int, rng: np.random.Generator) -> float:
        succ = 0
        for _ in range(n):
            r = self.simulate_trial(
                scenario["start_x_in"], scenario["start_y_in"],
                scenario["start_heading_deg"], scenario["max_power"], rng=rng)
            succ += int(r.success)
        return succ / n
