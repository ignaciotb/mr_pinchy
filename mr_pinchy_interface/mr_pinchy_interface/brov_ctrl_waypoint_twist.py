#!/usr/bin/env python3
"""Waypoint-tracking MPC for the BlueROV.

Replaces the previous P-controller outer loop. Receives a goal pose (e.g. the
RViz "2D Goal Pose" tool publishing on /goal_pose) and tracks it with a convex
MPC, emitting body-velocity references (TwistStamped) to the inner velocity
controller (brov_ctrl_twist_simple).

Model — deliberately *not* the hydrodynamic model. The inner velocity PID hides
all the hydrodynamics, so the MPC only needs a kinematic integrator in the odom
(world) frame. Depth is left to a separate controller; this node tracks the
horizontal plane and heading only:

    eta = [x, y, psi]              (horizontal position + yaw, odom frame)
    u   = [vx, vy, omega]          (commanded velocity, odom frame)
    eta_{k+1} = eta_k + dt * u_k

The emitted twist therefore leaves linear.z at zero. The QP is solved in the
odom frame (convention-agnostic: FRD/FLU, ENU/NED), then the resulting velocity
command is TF-rotated into the body frame before publishing so RViz anchors the
velocity arrow on the vehicle. This does not change control: the inner loop
TF-transforms the command by its header frame anyway, and odom->base_link
composes to the same body-frame command. If TF is momentarily unavailable the
command falls back to the odom frame, exactly as the previous P controller
relied on. The resulting problem is a convex QP, modeled once in CVXPY
(DPP-parametrized) and re-solved with OSQP every tick via warm starting.

To stop the limit-cycle chatter the kinematic model causes near the goal (it
ignores vehicle momentum / inner-loop lag, so it over-corrects on small errors),
the controller (a) commands zero once inside an arrival deadband on position and
yaw, and (b) is detuned for a gentler approach (higher command-rate penalty,
lower terminal weight).
"""
from __future__ import annotations

import copy

import cvxpy as cp
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Odometry, Path
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from tf2_geometry_msgs import do_transform_pose_stamped, do_transform_vector3
from tf2_ros import Buffer, TransformException, TransformListener

from mr_pinchy_interface.brov_utils import (
    quat_to_rpy,
    rpy_to_quat,
    vector3_stamped,
    vector3_to_array,
    wrap_pi,
)


class BrovWaypointMpcController(Node):
    def __init__(self) -> None:
        super().__init__("brov_ctrl_waypoint_twist")

        # --- I/O (kept identical to the previous P-controller node) ---
        self.declare_parameter("goal_topic", "/goal_pose")
        self.declare_parameter("odom_topic", "/mocap/mr_pinchy/odom_ekf")
        self.declare_parameter("base_link_frame", "mr_pinchy/base_link_ekf_flu")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("cmd_topic", "/mr_pinchy/cmd_vel")
        self.declare_parameter("predicted_path_topic", "~/mpc_predicted_path")

        # --- MPC horizon / rate ---
        self.declare_parameter("horizon", 20)        # steps
        self.declare_parameter("mpc_dt", 0.2)        # s per step -> 4 s lookahead
        # Control (re-solve + publish) frequency; independent of mpc_dt. Warm-
        # started solves measure ~9ms mean / ~24ms worst, so 30 Hz leaves margin.
        self.declare_parameter("publish_rate_hz", 30.0)
        self.declare_parameter("odom_timeout", 0.3)

        # --- limits (odom frame; isotropic horizontal, matches old P node) ---
        self.declare_parameter("v_lin_max", 0.15)    # m/s on x, y
        self.declare_parameter("v_yaw_max", 0.3)     # rad/s
        self.declare_parameter("a_lin_max", 0.2)     # m/s^2
        self.declare_parameter("a_yaw_max", 0.5)     # rad/s^2

        # --- cost weights ---
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

        # --- arrival deadband ---
        # Within this position+yaw tolerance of the goal, command zero so sensor
        # noise / tiny model errors can't drive a limit cycle. Re-engages on its
        # own if the vehicle drifts back outside the band.
        self.declare_parameter("arrival_pos_tol", 0.1)   # m
        self.declare_parameter("arrival_yaw_tol", 0.1)   # rad (~3 deg)

        # --- tank soft constraints (DISABLED by default) ---
        # Horizontal (x, y) bounds only; see the brov_ctrl_twist_simple tank
        # range. Slack keeps the QP always feasible.
        self.declare_parameter("enable_tank_constraint", False)
        self.declare_parameter("tank_min", [1.0, -1.8])
        self.declare_parameter("tank_max", [6.0, 1.8])
        self.declare_parameter("tank_margin", 0.2)
        self.declare_parameter("tank_slack_weight", 1000.0)

        gp = self.get_parameter
        goal_topic = gp("goal_topic").get_parameter_value().string_value
        odom_topic = gp("odom_topic").get_parameter_value().string_value
        cmd_topic = gp("cmd_topic").get_parameter_value().string_value
        path_topic = gp("predicted_path_topic").get_parameter_value().string_value
        self._base_link_frame = (
            gp("base_link_frame").get_parameter_value().string_value.lstrip("/")
        )
        self._map_frame = gp("map_frame").get_parameter_value().string_value

        self._N = int(gp("horizon").value)
        self._dt = float(gp("mpc_dt").value)
        self._publish_rate_hz = float(gp("publish_rate_hz").value)
        self._odom_timeout = float(gp("odom_timeout").value)

        self._v_lin_max = float(gp("v_lin_max").value)
        self._v_yaw_max = float(gp("v_yaw_max").value)
        self._a_lin_max = float(gp("a_lin_max").value)
        self._a_yaw_max = float(gp("a_yaw_max").value)

        # Weights stored as their (diagonal) matrix square roots, so the cost can
        # be written with cp.sum_squares -- the DPP-friendly form that lets CVXPY
        # canonicalize the QP once and only swap parameter values each tick.
        self._Qh = np.sqrt(np.diag([gp("q_pos").value] * 2 + [gp("q_yaw").value]))
        self._Qfh = np.sqrt(np.diag([gp("qf_pos").value] * 2 + [gp("qf_yaw").value]))
        self._Rh = np.sqrt(np.diag([gp("r_v").value] * 2 + [gp("r_w").value]))
        self._Rdh = np.sqrt(np.diag([gp("rd_v").value] * 2 + [gp("rd_w").value]))

        self._arrival_pos_tol = float(gp("arrival_pos_tol").value)
        self._arrival_yaw_tol = float(gp("arrival_yaw_tol").value)

        self._enable_tank = bool(gp("enable_tank_constraint").value)
        self._tank_min = np.asarray(gp("tank_min").value, dtype=float)
        self._tank_max = np.asarray(gp("tank_max").value, dtype=float)
        self._tank_margin = float(gp("tank_margin").value)
        self._tank_w = float(gp("tank_slack_weight").value)

        # Control bounds in [vx, vy, omega] order.
        self._u_lo = np.array(
            [-self._v_lin_max, -self._v_lin_max, -self._v_yaw_max]
        )
        self._u_hi = -self._u_lo
        # Per-step acceleration bound on |u_k - u_{k-1}|.
        self._du = np.array(
            [self._a_lin_max, self._a_lin_max, self._a_yaw_max]
        ) * self._dt

        # --- solver: build the parametrized QP once, re-solve every tick ---
        self._build_problem()

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self._cmd_pub = self.create_publisher(TwistStamped, cmd_topic, qos)
        self._path_pub = self.create_publisher(Path, path_topic, qos)
        self.create_subscription(PoseStamped, goal_topic, self._goal_cb, 10)
        self.create_subscription(Odometry, odom_topic, self._odom_cb, 10)

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._goal_pose: PoseStamped | None = None
        self._goal_in_odom: PoseStamped | None = None
        self._last_odom: Odometry | None = None
        self._last_odom_rx_sec: float = 0.0

        # Accel-continuity state (also seeds the OSQP warm start each tick).
        self._u_prev = np.zeros(3, dtype=float)

        self.create_timer(1.0 / self._publish_rate_hz, self._tick)
        self.get_logger().info(
            f"brov_ctrl_waypoint MPC: goal='{goal_topic}', odom='{odom_topic}', "
            f"cmd='{cmd_topic}', N={self._N}, dt={self._dt:.2f}s "
            f"(horizon {self._N * self._dt:.1f}s), tank_constraint={self._enable_tank}"
        )

    # ------------------------------------------------------------------ I/O
    def _goal_cb(self, msg: PoseStamped) -> None:
        self._goal_pose = msg
        self._goal_in_odom = None
        self.get_logger().info(
            f"New goal: frame='{msg.header.frame_id}', "
            f"pos=({msg.pose.position.x:.3f}, {msg.pose.position.y:.3f})"
        )

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

    def _publish_zero_twist(self) -> None:
        twist = TwistStamped()
        twist.header.stamp = self.get_clock().now().to_msg()
        twist.header.frame_id = self._base_link_frame
        self._cmd_pub.publish(twist)
        self._u_prev = np.zeros(3, dtype=float)

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

    def _publish_cmd(self, u0: np.ndarray, odom_frame: str) -> None:
        """Publish the MPC command. The solver works in the odom frame, but we
        re-express the twist in the body frame so RViz anchors the velocity
        arrow on the vehicle. Control is unchanged: the inner loop TF-transforms
        the command by its header frame, and odom->base_link composes to the
        same body-frame command either way. Falls back to the odom frame if TF
        is momentarily unavailable, so control is never interrupted."""
        stamp = self.get_clock().now().to_msg()
        twist = TwistStamped()
        twist.header.stamp = stamp
        twist.twist.linear.x = float(u0[0])
        twist.twist.linear.y = float(u0[1])
        twist.twist.angular.z = float(u0[2])

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

    # ------------------------------------------------------------------ MPC
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

    def _solve_mpc(
        self, x0: np.ndarray, goal: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Return (u0, predicted_states) or None on solver failure."""
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
        return self._U.value[0], self._X.value

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

    # ------------------------------------------------------------------ loop
    def _tick(self) -> None:
        now_sec = self.get_clock().now().nanoseconds / 1e9

        if self._goal_pose is None:
            self._publish_zero_twist()
            return

        if now_sec - self._last_odom_rx_sec > self._odom_timeout:
            self._publish_zero_twist()
            self.get_logger().warn(
                f"Odometry timeout ({self._odom_timeout:.2f}s); forcing zero control.",
                throttle_duration_sec=2.0,
            )
            return

        if not self._resolve_goal_in_odom():
            self._publish_zero_twist()
            return

        odom_pose = self._last_odom.pose.pose
        goal = self._goal_in_odom.pose
        odom_frame = self._last_odom.header.frame_id

        _, _, cur_yaw = quat_to_rpy(
            odom_pose.orientation.x, odom_pose.orientation.y,
            odom_pose.orientation.z, odom_pose.orientation.w,
        )
        _, _, goal_yaw = quat_to_rpy(
            goal.orientation.x, goal.orientation.y,
            goal.orientation.z, goal.orientation.w,
        )
        # Unwrap goal yaw to the same branch as the current yaw so the linear
        # model never "winds up" the long way around.
        goal_yaw = cur_yaw + wrap_pi(goal_yaw - cur_yaw)

        x0 = np.array([odom_pose.position.x, odom_pose.position.y, cur_yaw])
        goal_vec = np.array([goal.position.x, goal.position.y, goal_yaw])

        # Arrival deadband: once within tolerance, command zero so noise / small
        # model errors can't drive a limit cycle around the setpoint. goal_yaw is
        # already unwrapped onto cur_yaw's branch, so the diff is the true error.
        pos_err = float(np.hypot(goal_vec[0] - x0[0], goal_vec[1] - x0[1]))
        yaw_err = abs(goal_vec[2] - x0[2])
        if pos_err <= self._arrival_pos_tol and yaw_err <= self._arrival_yaw_tol:
            self._publish_zero_twist()
            return

        solution = self._solve_mpc(x0, goal_vec)
        if solution is None:
            self._publish_zero_twist()
            self.get_logger().warn("MPC solve failed; forcing zero control.",
                                   throttle_duration_sec=2.0)
            return

        u0, X_sol = solution
        self._u_prev = u0

        self._publish_cmd(u0, odom_frame)

        self._publish_predicted_path(X_sol, odom_frame, odom_pose.position.z)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BrovWaypointMpcController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
