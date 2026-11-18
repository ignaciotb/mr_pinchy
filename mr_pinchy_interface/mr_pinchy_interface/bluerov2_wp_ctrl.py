#!/usr/bin/env python3
"""Waypoint-tracking outer loop for the BlueROV.

Receives a goal pose and emits a body-velocity reference (TwistStamped) on
`cmd_topic`, for whichever velocity backend is subscribed to it. Two exist in
this package, and they do not have the same reach:

  bluerov2_vel_rc          Maps all six twist axes proportionally onto the
                           ArduSub RC channels (pitch, roll, throttle, yaw,
                           forward, lateral). Open loop -- no velocity
                           feedback. Roll and pitch only actuate through this
                           one. Used by brov_sim_bringup.launch.py.

  brov_ctrl_twist_simple   Closed loop on measured body velocity, but only on
                           surge, sway, heave and yaw; it reads angular.x/.y
                           and then drops them, and drives ArduSub
                           ManualControl, which has no roll or pitch axis.
                           Used by start_ctrl.launch.py.

Both subscribe to the same topic, so run one or the other -- not both.

Goals come from the 6-DOF waypoint tool in `bluerov_waypoint_rviz_plugin` (its
"Publish Goal" button), which publishes geometry_msgs/PoseStamped on
/bluerov/goal_pose. Any PoseStamped publisher works; point `goal_topic`
wherever you like.

Two control laws are available behind one interface, selected with the
`controller` parameter:

  mpc  (default)  Convex MPC over a kinematic integrator. Tracks x, y and yaw
                  in the odom frame; leaves depth alone. Handles the velocity
                  and acceleration envelope as hard constraints and can hold
                  soft tank bounds, at the cost of a QP solve per tick.

  pid             Independent PID per body axis, ported from
                  brov_ctrl_waypoint_wrench.py on the feat/pid branch. Tracks
                  all six DOF -- surge, sway, heave, roll, pitch, yaw -- so
                  unlike the MPC it closes the loop on depth and attitude.
                  Cheap, no solver dependency, but the envelope is enforced by
                  clamping rather than optimised over. Roll and pitch reach the
                  vehicle only under bluerov2_vel_rc (see above).

Both implement `WaypointController`: given a `TrackingState` (current and goal
pose in the odom frame) and a timestep, return a `VelocityCommand` tagged with
the frame it is expressed in. The node handles everything else -- goal
transport, the arrival deadband, TF rotation into the body frame, watchdogs --
so the two are genuinely interchangeable and share one downstream contract.

Model note (MPC) -- deliberately *not* the hydrodynamic model. The inner
velocity PID hides all the hydrodynamics, so the MPC only needs a kinematic
integrator in the odom (world) frame:

    eta = [x, y, psi]              (horizontal position + yaw, odom frame)
    u   = [vx, vy, omega]          (commanded velocity, odom frame)
    eta_{k+1} = eta_k + dt * u_k

The QP is solved in the odom frame (convention-agnostic: FRD/FLU, ENU/NED),
then TF-rotated into the body frame before publishing so RViz anchors the
velocity arrow on the vehicle. This does not change control: the inner loop
TF-transforms the command by its header frame anyway, and odom->base_link
composes to the same body-frame command. If TF is momentarily unavailable the
command falls back to the odom frame. The problem is convex, modeled once in
CVXPY (DPP-parametrized) and re-solved with OSQP every tick via warm starting.

Teleop handover -- `teleop_twist_joy` publishes onto the SAME `cmd_topic` as
this node, but only while its deadman (`enable_button`, RB in the bringup
launch files) is held. So this node watches that same button on `joy_topic`
and, while it is held, publishes NOTHING: not a command, not even the zero
twist it normally emits when disengaged, because a 30 Hz stream of zeros would
fight the pilot's twists on the shared topic. Set `teleop_enable_button` to
match the teleop node's `enable_button`, or to -1 to disable the arbitration
entirely. Taking the deadman down also CANCELS the current goal, so releasing
it leaves the node commanding zero and waiting for a new one -- the pilot has
moved the vehicle, and resuming a stale waypoint would drive it back.

To stop the limit-cycle chatter the kinematic model causes near the goal (it
ignores vehicle momentum / inner-loop lag, so it over-corrects on small errors),
the node (a) commands zero once inside an arrival deadband on position and yaw,
and (b) the MPC is detuned for a gentler approach (higher command-rate penalty,
lower terminal weight).
"""
from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import cvxpy as cp
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Odometry, Path
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Joy
from tf2_geometry_msgs import do_transform_pose_stamped, do_transform_vector3
from tf2_ros import Buffer, TransformException, TransformListener

from mr_pinchy_interface.brov_utils import (
    PIDAxis,
    quat_to_rotation_matrix,
    quat_to_rpy,
    rpy_from_rotation_matrix,
    rpy_to_quat,
    vector3_stamped,
    vector3_to_array,
    wrap_pi,
)

# Reporting-only thresholds: how much of a goal's untracked DOF is worth
# mentioning in the log. Not control parameters -- they change nothing but the
# noise level, which is why they are constants rather than ROS parameters.
IGNORED_ATTITUDE_TOL = 0.05   # rad (~3 deg) of goal roll/pitch
IGNORED_DEPTH_TOL = 0.10      # m between goal z and current depth


# ===========================================================  shared types  ==
class CommandFrame(Enum):
    """Which frame a controller's output is expressed in.

    Controllers name the frame by role, not by string, so they never need to
    know what the odom or base_link frames are actually called.
    """

    ODOM = "odom"
    BODY = "body"


@dataclass(frozen=True)
class VelocityCommand:
    """A velocity reference for the inner loop."""

    linear: np.ndarray    # (3,) m/s
    angular: np.ndarray   # (3,) rad/s
    frame: CommandFrame


@dataclass(frozen=True)
class TrackingState:
    """Everything a controller needs, all in the odom frame.

    goal_rpy[2] arrives already unwrapped onto the same branch as rpy[2], so
    yaw errors are the true (shortest-path) error and no controller has to
    redo that reasoning.
    """

    position: np.ndarray       # (3,) current position
    rpy: np.ndarray            # (3,) current roll, pitch, yaw
    orientation: np.ndarray    # (4,) current quaternion, x, y, z, w
    goal_position: np.ndarray     # (3,) goal position
    goal_rpy: np.ndarray          # (3,) goal roll, pitch, yaw
    goal_orientation: np.ndarray  # (4,) goal quaternion, x, y, z, w


# ===========================================================  controllers  ==
class WaypointController(ABC):
    """One waypoint-tracking control law.

    Subclasses own all of their internal state (integrators, warm starts) and
    are responsible for respecting the velocity/acceleration envelope they were
    constructed with. The node owns everything outside the loop.
    """

    #: Short name, matches the `controller` parameter value.
    name: str = "abstract"
    #: Does this law close the loop on z? Governs the arrival deadband and
    #: whether the node warns that a goal's depth is being ignored.
    tracks_depth: bool = False
    #: Does this law close the loop on roll/pitch? Nothing does today -- the
    #: inner loop actuates surge/sway/heave/yaw only.
    tracks_attitude: bool = False

    @abstractmethod
    def reset(self) -> None:
        """Drop accumulated state. Called whenever control is disengaged."""

    @abstractmethod
    def compute(self, state: TrackingState, dt: float) -> VelocityCommand | None:
        """Return the next command, or None if the law failed this tick."""

    def predicted_states(self) -> np.ndarray | None:
        """Optional (N+1, 3) preview of [x, y, yaw] for visualisation."""
        return None

    def describe(self) -> str:
        return self.name


class MpcWaypointController(WaypointController):
    """Convex MPC over a kinematic integrator in the odom frame."""

    name = "mpc"
    tracks_depth = False

    def __init__(
        self,
        *,
        horizon: int,
        dt: float,
        v_lin_max: float,
        v_yaw_max: float,
        a_lin_max: float,
        a_yaw_max: float,
        q_pos: float,
        q_yaw: float,
        qf_pos: float,
        qf_yaw: float,
        r_v: float,
        r_w: float,
        rd_v: float,
        rd_w: float,
        enable_tank: bool,
        tank_min,
        tank_max,
        tank_margin: float,
        tank_slack_weight: float,
    ) -> None:
        self._N = int(horizon)
        self._dt = float(dt)

        # Weights stored as their (diagonal) matrix square roots, so the cost
        # can be written with cp.sum_squares -- the DPP-friendly form that lets
        # CVXPY canonicalize the QP once and only swap parameter values.
        self._Qh = np.sqrt(np.diag([q_pos] * 2 + [q_yaw]))
        self._Qfh = np.sqrt(np.diag([qf_pos] * 2 + [qf_yaw]))
        self._Rh = np.sqrt(np.diag([r_v] * 2 + [r_w]))
        self._Rdh = np.sqrt(np.diag([rd_v] * 2 + [rd_w]))

        self._enable_tank = bool(enable_tank)
        self._tank_min = np.asarray(tank_min, dtype=float)
        self._tank_max = np.asarray(tank_max, dtype=float)
        self._tank_margin = float(tank_margin)
        self._tank_w = float(tank_slack_weight)

        # Control bounds in [vx, vy, omega] order.
        self._u_lo = np.array([-v_lin_max, -v_lin_max, -v_yaw_max])
        self._u_hi = -self._u_lo
        # Per-step acceleration bound on |u_k - u_{k-1}|.
        self._du = np.array([a_lin_max, a_lin_max, a_yaw_max]) * self._dt

        # Accel-continuity state (also seeds the OSQP warm start each tick).
        self._u_prev = np.zeros(3, dtype=float)
        self._X_sol: np.ndarray | None = None

        self._build_problem()

    # -- setup
    def _build_problem(self) -> None:
        """Build the parametrized QP once. Only x0/goal/u_prev change per tick,
        so CVXPY canonicalizes a single time (DPP) and re-solves with OSQP."""
        N, dt = self._N, self._dt

        self._X = cp.Variable((N + 1, 3), name="x")
        self._U = cp.Variable((N, 3), name="u")
        self._p_x0 = cp.Parameter(3, name="x0")
        self._p_goal = cp.Parameter(3, name="goal")
        self._p_uprev = cp.Parameter(3, name="u_prev")

        X, U = self._X, self._U
        cost = 0
        cons = [X[0] == self._p_x0]  # initial condition

        for k in range(N):
            # Kinematic integrator: x_{k+1} = x_k + dt * u_k.
            cons.append(X[k + 1] == X[k] + dt * U[k])
            # Velocity box.
            cons += [U[k] >= self._u_lo, U[k] <= self._u_hi]
            # Acceleration box on |u_k - u_{k-1}|.
            u_prev = U[k - 1] if k > 0 else self._p_uprev
            cons += [U[k] - u_prev >= -self._du, U[k] - u_prev <= self._du]
            # Effort + rate cost.
            cost += cp.sum_squares(self._Rh @ U[k])
            cost += cp.sum_squares(self._Rdh @ (U[k] - u_prev))
            # Stage tracking cost on X[1..N].
            cost += cp.sum_squares(self._Qh @ (X[k + 1] - self._p_goal))

        # Extra terminal weight on X[N] (on top of the stage cost above).
        cost += cp.sum_squares(self._Qfh @ (X[N] - self._p_goal))

        # Optional soft tank bounds (slack keeps the QP always feasible).
        if self._enable_tank:
            tmin = self._tank_min + self._tank_margin
            tmax = self._tank_max - self._tank_margin
            S = cp.Variable((N, 4), nonneg=True, name="tank_slack")
            for k in range(1, N + 1):
                for i in range(2):
                    cons.append(X[k][i] + S[k - 1, i] >= tmin[i])
                    cons.append(X[k][i] - S[k - 1, 2 + i] <= tmax[i])
            cost += self._tank_w * cp.sum_squares(S)

        self._problem = cp.Problem(cp.Minimize(cost), cons)

    # -- interface
    def reset(self) -> None:
        self._u_prev = np.zeros(3, dtype=float)
        self._X_sol = None

    def compute(self, state: TrackingState, dt: float) -> VelocityCommand | None:
        # dt is ignored: the horizon steps at the fixed mpc_dt the QP was built
        # with, independent of how often the node re-solves.
        del dt

        x0 = np.array([state.position[0], state.position[1], state.rpy[2]])
        goal = np.array(
            [state.goal_position[0], state.goal_position[1], state.goal_rpy[2]]
        )

        self._p_x0.value = x0
        self._p_goal.value = goal
        self._p_uprev.value = self._u_prev

        try:
            self._problem.solve(
                solver=cp.OSQP, warm_start=True, max_iter=4000, verbose=False
            )
        except cp.error.SolverError:
            return None

        if self._U.value is None or self._X.value is None:
            return None

        u0 = self._U.value[0]
        self._u_prev = u0
        self._X_sol = self._X.value

        return VelocityCommand(
            linear=np.array([u0[0], u0[1], 0.0]),
            angular=np.array([0.0, 0.0, u0[2]]),
            frame=CommandFrame.ODOM,
        )

    def predicted_states(self) -> np.ndarray | None:
        return self._X_sol

    def describe(self) -> str:
        return (
            f"mpc(N={self._N}, dt={self._dt:.2f}s, "
            f"horizon={self._N * self._dt:.1f}s, tank={self._enable_tank})"
        )


class PidWaypointController(WaypointController):
    """Independent PID per body axis: surge, sway, heave, yaw.

    Ported from brov_ctrl_waypoint_wrench.py (feat/pid). It emits a velocity
    reference instead of a wrench + RC override, so it shares the MPC's
    downstream contract. Each axis's `output_limit` is therefore a velocity
    limit and the gains map error -> velocity rather than error -> normalised
    thrust, which means the wrench version's gains are a starting point, not a
    drop-in.

    All six DOF are tracked. Whether roll and pitch actually reach the vehicle
    depends on which velocity backend is subscribed to `cmd_topic`:
    bluerov2_vel_rc maps all six axes onto RC channels, while
    brov_ctrl_twist_simple reads surge/sway/heave/yaw and silently drops the
    other two. See the module docstring.

    Errors are body-frame: the position error is the odom-frame error rotated
    by the vehicle's own orientation, and the attitude error is the relative
    rotation R_cur^T R_goal read back as roll/pitch/yaw. That is exactly what
    transforming the goal into base_link gives -- the reference's approach --
    without needing TF to be up, and it stays correct when the vehicle is not
    level, which differencing Euler angles per axis would not. (It assumes
    odom's child frame is the body frame the command is published in, the same
    assumption the rest of this node already makes.)
    """

    name = "pid"
    tracks_depth = True
    tracks_attitude = True

    def __init__(
        self,
        *,
        kp_xy: float,
        ki_xy: float,
        kd_xy: float,
        kp_z: float,
        ki_z: float,
        kd_z: float,
        kp_roll: float,
        ki_roll: float,
        kd_roll: float,
        kp_pitch: float,
        ki_pitch: float,
        kd_pitch: float,
        kp_yaw: float,
        ki_yaw: float,
        kd_yaw: float,
        integral_limit_lin: float,
        integral_limit_ang: float,
        v_lin_max: float,
        v_heave_max: float,
        v_roll_max: float,
        v_pitch_max: float,
        v_yaw_max: float,
        a_lin_max: float,
        a_ang_max: float,
    ) -> None:
        self._pid_x = PIDAxis(
            kp=kp_xy, ki=ki_xy, kd=kd_xy,
            integral_limit=integral_limit_lin, output_limit=v_lin_max,
        )
        self._pid_y = PIDAxis(
            kp=kp_xy, ki=ki_xy, kd=kd_xy,
            integral_limit=integral_limit_lin, output_limit=v_lin_max,
        )
        self._pid_z = PIDAxis(
            kp=kp_z, ki=ki_z, kd=kd_z,
            integral_limit=integral_limit_lin, output_limit=v_heave_max,
        )
        self._pid_roll = PIDAxis(
            kp=kp_roll, ki=ki_roll, kd=kd_roll,
            integral_limit=integral_limit_ang, output_limit=v_roll_max,
        )
        self._pid_pitch = PIDAxis(
            kp=kp_pitch, ki=ki_pitch, kd=kd_pitch,
            integral_limit=integral_limit_ang, output_limit=v_pitch_max,
        )
        self._pid_yaw = PIDAxis(
            kp=kp_yaw, ki=ki_yaw, kd=kd_yaw,
            integral_limit=integral_limit_ang, output_limit=v_yaw_max,
        )

        # Slew limit, so the velocity backend never sees a step. The MPC gets
        # this for free as a hard QP constraint; here it has to be applied
        # after the fact.
        self._a_max = np.array(
            [a_lin_max, a_lin_max, a_lin_max, a_ang_max, a_ang_max, a_ang_max]
        )
        self._u_prev = np.zeros(6, dtype=float)

    def _axes(self):
        return (
            self._pid_x, self._pid_y, self._pid_z,
            self._pid_roll, self._pid_pitch, self._pid_yaw,
        )

    def reset(self) -> None:
        for pid in self._axes():
            pid.reset()
        self._u_prev = np.zeros(6, dtype=float)

    def compute(self, state: TrackingState, dt: float) -> VelocityCommand | None:
        # R maps body vectors into odom, so R.T maps the odom-frame position
        # error into the body frame -- surge/sway/heave error, directly.
        rotation = quat_to_rotation_matrix(*state.orientation)
        e_body = rotation.T @ (state.goal_position - state.position)

        # Attitude error as the relative rotation, read back as roll/pitch/yaw.
        # Differencing Euler angles axis by axis would only be valid near level.
        goal_rotation = quat_to_rotation_matrix(*state.goal_orientation)
        e_roll, e_pitch, e_yaw = rpy_from_rotation_matrix(rotation.T @ goal_rotation)

        u = np.array(
            [
                self._pid_x.step(float(e_body[0]), dt),
                self._pid_y.step(float(e_body[1]), dt),
                self._pid_z.step(float(e_body[2]), dt),
                self._pid_roll.step(float(e_roll), dt, wrap_error=True),
                self._pid_pitch.step(float(e_pitch), dt, wrap_error=True),
                self._pid_yaw.step(float(e_yaw), dt, wrap_error=True),
            ]
        )

        du = self._a_max * dt
        u = np.clip(u, self._u_prev - du, self._u_prev + du)
        self._u_prev = u

        return VelocityCommand(
            linear=u[0:3].copy(),
            angular=u[3:6].copy(),
            frame=CommandFrame.BODY,
        )

    def describe(self) -> str:
        return (
            f"pid(6-DOF, kp_xy={self._pid_x.kp:.2f}, kp_z={self._pid_z.kp:.2f}, "
            f"kp_roll={self._pid_roll.kp:.2f}, kp_pitch={self._pid_pitch.kp:.2f}, "
            f"kp_yaw={self._pid_yaw.kp:.2f})"
        )


CONTROLLERS = {
    MpcWaypointController.name: MpcWaypointController,
    PidWaypointController.name: PidWaypointController,
}


# =================================================================  node  ==
class Bluerov2WpCtrl(Node):
    def __init__(self) -> None:
        super().__init__("bluerov2_wp_ctrl")

        # --- I/O ---
        self.declare_parameter("goal_topic", "/bluerov/goal_pose")
        self.declare_parameter("odom_topic", "/mocap/mr_pinchy/odom_ekf")
        self.declare_parameter("base_link_frame", "mr_pinchy/base_link_ekf_flu")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("cmd_topic", "/mr_pinchy/cmd_vel")
        self.declare_parameter("predicted_path_topic", "~/mpc_predicted_path")

        # --- teleop arbitration ---
        # teleop_twist_joy shares cmd_topic with this node and only publishes
        # while its deadman is held, so the same button decides who owns the
        # topic. Must match the teleop node's "enable_button"; -1 disables.
        self.declare_parameter("joy_topic", "/joy")
        self.declare_parameter("teleop_enable_button", 5)
        # joy_node republishes at autorepeat_rate (20 Hz in the bringup files)
        # even when nothing changes, so silence this long means the joystick is
        # gone -- treat the deadman as released and take the topic back rather
        # than leaving the vehicle with nobody driving it.
        self.declare_parameter("joy_timeout_sec", 1.0)

        # --- which control law ---
        self.declare_parameter("controller", "mpc")  # "mpc" | "pid"

        # --- MPC horizon / rate ---
        self.declare_parameter("horizon", 20)        # steps
        self.declare_parameter("mpc_dt", 0.2)        # s per step -> 4 s lookahead
        # Control (re-solve + publish) frequency; independent of mpc_dt. Warm-
        # started solves measure ~9ms mean / ~24ms worst, so 30 Hz leaves margin.
        self.declare_parameter("publish_rate_hz", 30.0)
        self.declare_parameter("odom_timeout", 0.3)

        # --- limits (shared by both controllers) ---
        self.declare_parameter("v_lin_max", 0.15)    # m/s on x, y
        self.declare_parameter("v_heave_max", 0.1)   # m/s on z (PID only)
        self.declare_parameter("v_roll_max", 0.2)    # rad/s (PID only)
        self.declare_parameter("v_pitch_max", 0.2)   # rad/s (PID only)
        self.declare_parameter("v_yaw_max", 0.3)     # rad/s
        self.declare_parameter("a_lin_max", 0.2)     # m/s^2
        self.declare_parameter("a_yaw_max", 0.5)     # rad/s^2, all angular axes

        # --- MPC cost weights ---
        # Detuned vs. a textbook tracker to avoid near-goal overshoot: stronger
        # command-rate penalty (rd) and a softer terminal pull (qf_pos) make the
        # approach gentler so the laggy plant doesn't ring around the setpoint.
        self.declare_parameter("q_pos", 10.0)        # stage position tracking
        self.declare_parameter("q_yaw", 2.0)         # stage yaw tracking
        self.declare_parameter("qf_pos", 20.0)       # terminal position
        self.declare_parameter("qf_yaw", 10.0)       # terminal yaw
        self.declare_parameter("r_v", 1.0)           # linear effort
        self.declare_parameter("r_w", 1.0)           # yaw effort
        self.declare_parameter("rd_v", 12.0)         # linear rate (smoothness)
        self.declare_parameter("rd_w", 12.0)         # yaw rate

        # --- PID gains (error -> velocity) ---
        # Seeded from brov_ctrl_waypoint_wrench.py on feat/pid, where they mapped
        # error to normalised thrust. The output units are different here, so
        # treat these as a starting point and retune.
        self.declare_parameter("pid_kp_xy", 3.0)
        self.declare_parameter("pid_ki_xy", 0.0)
        self.declare_parameter("pid_kd_xy", 0.1)
        self.declare_parameter("pid_kp_z", 3.0)
        self.declare_parameter("pid_ki_z", 0.0)
        self.declare_parameter("pid_kd_z", 0.1)
        self.declare_parameter("pid_kp_roll", 1.0)
        self.declare_parameter("pid_ki_roll", 0.0)
        self.declare_parameter("pid_kd_roll", 0.0)
        self.declare_parameter("pid_kp_pitch", 1.0)
        self.declare_parameter("pid_ki_pitch", 0.0)
        self.declare_parameter("pid_kd_pitch", 0.0)
        self.declare_parameter("pid_kp_yaw", 3.0)
        self.declare_parameter("pid_ki_yaw", 0.0)
        self.declare_parameter("pid_kd_yaw", 0.1)
        self.declare_parameter("pid_integral_limit_lin", 1.0)
        self.declare_parameter("pid_integral_limit_ang", 1.0)

        # --- arrival deadband (applies to both controllers) ---
        # Within this position+yaw tolerance of the goal, command zero so sensor
        # noise / tiny model errors can't drive a limit cycle. Re-engages on its
        # own if the vehicle drifts back outside the band.
        self.declare_parameter("arrival_pos_tol", 0.1)   # m
        self.declare_parameter("arrival_yaw_tol", 0.1)   # rad (~6 deg)

        # --- tank soft constraints (MPC only, DISABLED by default) ---
        # Horizontal (x, y) bounds only; see the brov_ctrl_twist_simple tank
        # range. Slack keeps the QP always feasible.
        self.declare_parameter("enable_tank_constraint", False)
        self.declare_parameter("tank_min", [1.0, -1.8])
        self.declare_parameter("tank_max", [6.0, 1.8])
        self.declare_parameter("tank_margin", 0.2)
        self.declare_parameter("tank_slack_weight", 1000.0)

        gp = self.get_parameter
        goal_topic = gp("goal_topic").get_parameter_value().string_value
        self._goal_topic = goal_topic
        odom_topic = gp("odom_topic").get_parameter_value().string_value
        cmd_topic = gp("cmd_topic").get_parameter_value().string_value
        path_topic = gp("predicted_path_topic").get_parameter_value().string_value
        self._base_link_frame = (
            gp("base_link_frame").get_parameter_value().string_value.lstrip("/")
        )
        self._map_frame = gp("map_frame").get_parameter_value().string_value

        joy_topic = gp("joy_topic").get_parameter_value().string_value
        self._teleop_button = int(gp("teleop_enable_button").value)
        self._joy_timeout = float(gp("joy_timeout_sec").value)

        self._publish_rate_hz = float(gp("publish_rate_hz").value)
        self._odom_timeout = float(gp("odom_timeout").value)
        self._arrival_pos_tol = float(gp("arrival_pos_tol").value)
        self._arrival_yaw_tol = float(gp("arrival_yaw_tol").value)

        self._controller = self._build_controller()

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self._cmd_pub = self.create_publisher(TwistStamped, cmd_topic, qos)
        self._path_pub = self.create_publisher(Path, path_topic, qos)
        self.create_subscription(PoseStamped, goal_topic, self._goal_cb, 10)
        self.create_subscription(Odometry, odom_topic, self._odom_cb, 10)
        if self._teleop_button >= 0:
            self.create_subscription(Joy, joy_topic, self._joy_cb, 10)

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._goal_pose: PoseStamped | None = None
        self._goal_in_odom: PoseStamped | None = None
        self._last_odom: Odometry | None = None
        self._last_odom_rx_sec: float = 0.0
        self._last_tick_sec: float = self.get_clock().now().nanoseconds / 1e9
        self._teleop_button_held = False
        self._last_joy_rx_sec: float = 0.0
        self._teleop_owns_topic = False

        self.create_timer(1.0 / self._publish_rate_hz, self._tick)
        self.get_logger().info(
            f"bluerov2_wp_ctrl: controller={self._controller.describe()}, "
            f"goal='{goal_topic}', odom='{odom_topic}', cmd='{cmd_topic}', "
            f"rate={self._publish_rate_hz:.1f}Hz, "
            + (
                f"teleop handover on '{joy_topic}' button "
                f"{self._teleop_button}"
                if self._teleop_button >= 0
                else "teleop handover disabled"
            )
        )

    # ---------------------------------------------------------------- setup
    def _build_controller(self) -> WaypointController:
        gp = self.get_parameter
        requested = gp("controller").get_parameter_value().string_value.strip().lower()

        if requested not in CONTROLLERS:
            self.get_logger().error(
                f"Unknown controller '{requested}'; expected one of "
                f"{sorted(CONTROLLERS)}. Falling back to 'mpc'."
            )
            requested = MpcWaypointController.name

        v_lin_max = float(gp("v_lin_max").value)
        v_yaw_max = float(gp("v_yaw_max").value)
        a_lin_max = float(gp("a_lin_max").value)
        a_yaw_max = float(gp("a_yaw_max").value)

        if requested == PidWaypointController.name:
            return PidWaypointController(
                kp_xy=float(gp("pid_kp_xy").value),
                ki_xy=float(gp("pid_ki_xy").value),
                kd_xy=float(gp("pid_kd_xy").value),
                kp_z=float(gp("pid_kp_z").value),
                ki_z=float(gp("pid_ki_z").value),
                kd_z=float(gp("pid_kd_z").value),
                kp_roll=float(gp("pid_kp_roll").value),
                ki_roll=float(gp("pid_ki_roll").value),
                kd_roll=float(gp("pid_kd_roll").value),
                kp_pitch=float(gp("pid_kp_pitch").value),
                ki_pitch=float(gp("pid_ki_pitch").value),
                kd_pitch=float(gp("pid_kd_pitch").value),
                kp_yaw=float(gp("pid_kp_yaw").value),
                ki_yaw=float(gp("pid_ki_yaw").value),
                kd_yaw=float(gp("pid_kd_yaw").value),
                integral_limit_lin=float(gp("pid_integral_limit_lin").value),
                integral_limit_ang=float(gp("pid_integral_limit_ang").value),
                v_lin_max=v_lin_max,
                v_heave_max=float(gp("v_heave_max").value),
                v_roll_max=float(gp("v_roll_max").value),
                v_pitch_max=float(gp("v_pitch_max").value),
                v_yaw_max=v_yaw_max,
                a_lin_max=a_lin_max,
                a_ang_max=a_yaw_max,
            )

        return MpcWaypointController(
            horizon=int(gp("horizon").value),
            dt=float(gp("mpc_dt").value),
            v_lin_max=v_lin_max,
            v_yaw_max=v_yaw_max,
            a_lin_max=a_lin_max,
            a_yaw_max=a_yaw_max,
            q_pos=float(gp("q_pos").value),
            q_yaw=float(gp("q_yaw").value),
            qf_pos=float(gp("qf_pos").value),
            qf_yaw=float(gp("qf_yaw").value),
            r_v=float(gp("r_v").value),
            r_w=float(gp("r_w").value),
            rd_v=float(gp("rd_v").value),
            rd_w=float(gp("rd_w").value),
            enable_tank=bool(gp("enable_tank_constraint").value),
            tank_min=gp("tank_min").value,
            tank_max=gp("tank_max").value,
            tank_margin=float(gp("tank_margin").value),
            tank_slack_weight=float(gp("tank_slack_weight").value),
        )

    # ------------------------------------------------------------------ I/O
    def _goal_cb(self, msg: PoseStamped) -> None:
        self._goal_pose = msg
        self._goal_in_odom = None

        roll, pitch, yaw = quat_to_rpy(
            msg.pose.orientation.x, msg.pose.orientation.y,
            msg.pose.orientation.z, msg.pose.orientation.w,
        )
        self.get_logger().info(
            f"New goal: frame='{msg.header.frame_id}', "
            f"pos=({msg.pose.position.x:.3f}, {msg.pose.position.y:.3f}, "
            f"{msg.pose.position.z:.3f}), yaw={yaw:.3f} rad"
        )

        # The waypoint plugin can set all six DOF, but no controller here tracks
        # roll/pitch and only the PID tracks depth. Report the parts the active
        # controller will drop, once per goal, so a waypoint placed at depth or
        # pitched down does not look like the controller misbehaving.
        if not self._controller.tracks_attitude and (
            abs(roll) > IGNORED_ATTITUDE_TOL or abs(pitch) > IGNORED_ATTITUDE_TOL
        ):
            self.get_logger().warn(
                f"Goal carries roll={roll:.3f} / pitch={pitch:.3f} rad; the "
                f"'{self._controller.name}' controller tracks yaw only and "
                "ignores them."
            )
        if not self._controller.tracks_depth and self._last_odom is not None:
            dz = msg.pose.position.z - self._last_odom.pose.pose.position.z
            if abs(dz) > IGNORED_DEPTH_TOL:
                self.get_logger().warn(
                    f"Goal is {dz:+.2f} m from the current depth; the "
                    f"'{self._controller.name}' controller holds the horizontal "
                    "plane only -- depth is a separate loop. Set "
                    "controller:=pid to close the loop on depth here."
                )

    def _joy_cb(self, msg: Joy) -> None:
        if self._teleop_button >= len(msg.buttons):
            self.get_logger().warn(
                f"Joy message has {len(msg.buttons)} buttons, no index "
                f"{self._teleop_button}; teleop handover inactive.",
                throttle_duration_sec=5.0,
            )
            self._teleop_button_held = False
            return
        self._teleop_button_held = msg.buttons[self._teleop_button] != 0
        self._last_joy_rx_sec = self.get_clock().now().nanoseconds / 1e9

    def _teleop_has_control(self, now_sec: float) -> bool:
        """True while the pilot holds the deadman teleop_twist_joy publishes on."""
        if self._teleop_button < 0 or not self._teleop_button_held:
            return False
        if now_sec - self._last_joy_rx_sec > self._joy_timeout:
            self.get_logger().warn(
                f"No joy message for {self._joy_timeout:.1f}s while the deadman "
                "was held; assuming it was released and resuming waypoint control.",
                throttle_duration_sec=5.0,
            )
            self._teleop_button_held = False
            return False
        return True

    def _odom_cb(self, msg: Odometry) -> None:
        self._last_odom = msg
        self._last_odom_rx_sec = self.get_clock().now().nanoseconds / 1e9

    def _resolve_goal_in_odom(self) -> bool:
        if self._goal_in_odom is not None:
            return True
        if self._goal_pose is None or self._last_odom is None:
            return False

        goal_frame = self._goal_pose.header.frame_id or self._map_frame
        odom_frame = self._last_odom.header.frame_id
        if not odom_frame:
            return False

        src = PoseStamped()
        src.header.frame_id = goal_frame
        src.pose = self._goal_pose.pose

        if goal_frame == odom_frame:
            self._goal_in_odom = src
            return True

        try:
            tf = self._tf_buffer.lookup_transform(
                odom_frame, goal_frame, rclpy.time.Time(),
                timeout=Duration(seconds=0.05),
            )
            self._goal_in_odom = do_transform_pose_stamped(src, tf)
            return True
        except TransformException as exc:
            self._goal_in_odom = None
            self.get_logger().warn(
                f"TF {goal_frame} -> {odom_frame} unavailable: {exc}",
                throttle_duration_sec=2.0,
            )
            return False

    def _disengage(self) -> None:
        """Publish a zero twist and drop the controller's accumulated state."""
        twist = TwistStamped()
        twist.header.stamp = self.get_clock().now().to_msg()
        twist.header.frame_id = self._base_link_frame
        self._cmd_pub.publish(twist)
        self._controller.reset()

    def _rotate_vec_to_base_link(
        self, vec, source_frame: str, stamp,
    ) -> np.ndarray | None:
        """Rotate a free vector (e.g. velocity) from source_frame into the body
        frame via TF. Returns None if the transform is unavailable."""
        source_frame = source_frame.lstrip("/")
        if source_frame == self._base_link_frame:
            return vector3_to_array(vec)
        try:
            tf = self._tf_buffer.lookup_transform(
                self._base_link_frame, source_frame, rclpy.time.Time(),
                timeout=Duration(seconds=0.05),
            )
        except TransformException:
            return None
        vs = vector3_stamped(vec, frame_id=source_frame, stamp=stamp)
        return vector3_to_array(do_transform_vector3(vs, copy.deepcopy(tf)).vector)

    def _publish_cmd(self, cmd: VelocityCommand, odom_frame: str) -> None:
        """Publish a controller's command as a TwistStamped.

        A body-frame command (PID) goes out as-is. An odom-frame command (MPC)
        is re-expressed in the body frame so RViz anchors the velocity arrow on
        the vehicle. Control is unchanged either way: the inner loop
        TF-transforms the command by its header frame, and odom->base_link
        composes to the same body-frame command. Falls back to the odom frame
        if TF is momentarily unavailable, so control is never interrupted."""
        stamp = self.get_clock().now().to_msg()
        twist = TwistStamped()
        twist.header.stamp = stamp
        twist.twist.linear.x, twist.twist.linear.y, twist.twist.linear.z = (
            map(float, cmd.linear)
        )
        twist.twist.angular.x, twist.twist.angular.y, twist.twist.angular.z = (
            map(float, cmd.angular)
        )

        if cmd.frame is CommandFrame.BODY:
            twist.header.frame_id = self._base_link_frame
            self._cmd_pub.publish(twist)
            return

        lin = self._rotate_vec_to_base_link(twist.twist.linear, odom_frame, stamp)
        ang = self._rotate_vec_to_base_link(twist.twist.angular, odom_frame, stamp)
        if lin is not None and ang is not None:
            twist.twist.linear.x, twist.twist.linear.y, twist.twist.linear.z = map(float, lin)
            twist.twist.angular.x, twist.twist.angular.y, twist.twist.angular.z = map(float, ang)
            twist.header.frame_id = self._base_link_frame
        else:
            twist.header.frame_id = odom_frame  # odom-frame fallback (still valid)
            self.get_logger().warn(
                f"TF {odom_frame} -> {self._base_link_frame} unavailable; "
                "publishing cmd in odom frame.",
                throttle_duration_sec=2.0,
            )
        self._cmd_pub.publish(twist)

    def _publish_predicted_path(
        self, X_sol: np.ndarray, frame_id: str, z: float
    ) -> None:
        if self._path_pub.get_subscription_count() == 0:
            return
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = frame_id
        for state in X_sol:
            ps = PoseStamped()
            ps.header.frame_id = frame_id
            ps.pose.position.x = float(state[0])
            ps.pose.position.y = float(state[1])
            ps.pose.position.z = z
            qx, qy, qz, qw = rpy_to_quat(0.0, 0.0, float(state[2]))
            ps.pose.orientation.x = qx
            ps.pose.orientation.y = qy
            ps.pose.orientation.z = qz
            ps.pose.orientation.w = qw
            path.poses.append(ps)
        self._path_pub.publish(path)

    # ----------------------------------------------------------------- loop
    def _build_state(self) -> TrackingState:
        odom_pose = self._last_odom.pose.pose
        goal = self._goal_in_odom.pose

        cur_roll, cur_pitch, cur_yaw = quat_to_rpy(
            odom_pose.orientation.x, odom_pose.orientation.y,
            odom_pose.orientation.z, odom_pose.orientation.w,
        )
        goal_roll, goal_pitch, goal_yaw = quat_to_rpy(
            goal.orientation.x, goal.orientation.y,
            goal.orientation.z, goal.orientation.w,
        )
        # Unwrap goal yaw onto the same branch as the current yaw so no control
        # law ever "winds up" the long way around.
        goal_yaw = cur_yaw + wrap_pi(goal_yaw - cur_yaw)

        return TrackingState(
            position=np.array(
                [odom_pose.position.x, odom_pose.position.y, odom_pose.position.z]
            ),
            rpy=np.array([cur_roll, cur_pitch, cur_yaw]),
            orientation=np.array(
                [
                    odom_pose.orientation.x, odom_pose.orientation.y,
                    odom_pose.orientation.z, odom_pose.orientation.w,
                ]
            ),
            goal_position=np.array(
                [goal.position.x, goal.position.y, goal.position.z]
            ),
            goal_rpy=np.array([goal_roll, goal_pitch, goal_yaw]),
            goal_orientation=np.array(
                [
                    goal.orientation.x, goal.orientation.y,
                    goal.orientation.z, goal.orientation.w,
                ]
            ),
        )

    def _has_arrived(self, state: TrackingState) -> bool:
        """Arrival deadband, over exactly the DOFs the active controller tracks.

        Including an axis the controller cannot command would let an error it
        can never correct keep it engaged forever -- so depth only counts for a
        controller that commands heave, and roll/pitch only for one that
        commands them."""
        error = state.goal_position - state.position
        if self._controller.tracks_depth:
            pos_err = float(np.linalg.norm(error))
        else:
            pos_err = float(np.hypot(error[0], error[1]))
        if pos_err > self._arrival_pos_tol:
            return False

        # goal_rpy[2] is already unwrapped onto the current branch.
        ang_errs = [abs(state.goal_rpy[2] - state.rpy[2])]
        if self._controller.tracks_attitude:
            ang_errs.append(abs(wrap_pi(state.goal_rpy[0] - state.rpy[0])))
            ang_errs.append(abs(wrap_pi(state.goal_rpy[1] - state.rpy[1])))
        return max(ang_errs) <= self._arrival_yaw_tol

    def _tick(self) -> None:
        now_sec = self.get_clock().now().nanoseconds / 1e9
        dt = max(now_sec - self._last_tick_sec, 1e-3)
        self._last_tick_sec = now_sec

        # The pilot owns cmd_topic while the deadman is down. Publish nothing at
        # all -- _disengage()'s zero twists would interleave with the teleop's
        # commands on the shared topic and stutter the vehicle.
        if self._teleop_has_control(now_sec):
            if not self._teleop_owns_topic:
                self._teleop_owns_topic = True
                self._controller.reset()
                # Cancel the goal: the pilot is about to move the vehicle, and
                # resuming a now-stale waypoint on release would drive it back.
                had_goal = self._goal_pose is not None
                self._goal_pose = None
                self._goal_in_odom = None
                self.get_logger().info(
                    "Teleop deadman held; yielding cmd_topic to the pilot"
                    + (" and cancelling the current goal." if had_goal else ".")
                )
            return
        if self._teleop_owns_topic:
            self._teleop_owns_topic = False
            self.get_logger().info(
                "Teleop deadman released; commanding zero until a new goal "
                f"arrives on '{self._goal_topic}'."
            )

        if self._goal_pose is None:
            self._disengage()
            return

        if now_sec - self._last_odom_rx_sec > self._odom_timeout:
            self._disengage()
            self.get_logger().warn(
                f"Odometry timeout ({self._odom_timeout:.2f}s); forcing zero control.",
                throttle_duration_sec=2.0,
            )
            return

        if not self._resolve_goal_in_odom():
            self._disengage()
            return

        state = self._build_state()
        odom_frame = self._last_odom.header.frame_id

        if self._has_arrived(state):
            self._disengage()
            return

        cmd = self._controller.compute(state, dt)
        if cmd is None:
            self._disengage()
            self.get_logger().warn(
                f"'{self._controller.name}' controller failed to produce a "
                "command; forcing zero control.",
                throttle_duration_sec=2.0,
            )
            return

        self._publish_cmd(cmd, odom_frame)

        predicted = self._controller.predicted_states()
        if predicted is not None:
            self._publish_predicted_path(
                predicted, odom_frame, self._last_odom.pose.pose.position.z
            )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = Bluerov2WpCtrl()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
