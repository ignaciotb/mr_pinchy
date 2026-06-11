#!/usr/bin/env python3
"""Waypoint-tracking MPC for the BlueROV.

Replaces the previous P-controller outer loop. Receives a goal pose (e.g. the
RViz "2D Goal Pose" tool publishing on /goal_pose) and tracks it with a convex
MPC, emitting body-velocity references (TwistStamped) to the inner velocity
controller (brov_ctrl_twist_simple).

Model — deliberately *not* the hydrodynamic model. The inner velocity PID hides
all the hydrodynamics, so the MPC only needs a kinematic integrator in the odom
(world) frame:

    eta = [x, y, z, psi]            (position + yaw, odom frame)
    u   = [vx, vy, vz, omega]       (commanded velocity, odom frame)
    eta_{k+1} = eta_k + dt * u_k

Because the command and the error both live in the odom frame, this is
convention-agnostic (FRD/FLU, ENU/NED): the inner loop's TF turns the odom-frame
twist into the body frame, exactly as the previous P controller relied on. The
resulting problem is a convex QP solved with OSQP every tick.
"""
from __future__ import annotations

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Odometry, Path
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from tf2_geometry_msgs import do_transform_pose_stamped
from tf2_ros import Buffer, TransformException, TransformListener

from pydrake.solvers import (
    MathematicalProgram,
    OsqpSolver,
    SolverOptions,
)

from mr_pinchy_interface.brov_utils import (
    quat_to_rpy,
    rpy_to_quat,
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
        self.declare_parameter("cmd_topic", "/mr_pinchy/cmd_vel_ref")
        self.declare_parameter("predicted_path_topic", "~/mpc_predicted_path")

        # --- MPC horizon / rate ---
        self.declare_parameter("horizon", 20)        # steps
        self.declare_parameter("mpc_dt", 0.2)        # s per step -> 4 s lookahead
        self.declare_parameter("publish_rate_hz", 20.0)
        self.declare_parameter("odom_timeout", 0.3)

        # --- limits (odom frame; isotropic horizontal, matches old P node) ---
        self.declare_parameter("v_lin_max", 0.15)    # m/s on x, y, z
        self.declare_parameter("v_yaw_max", 0.3)     # rad/s
        self.declare_parameter("a_lin_max", 0.2)     # m/s^2
        self.declare_parameter("a_yaw_max", 0.5)     # rad/s^2

        # --- cost weights ---
        self.declare_parameter("q_pos", 10.0)        # stage position tracking
        self.declare_parameter("q_yaw", 2.0)         # stage yaw tracking
        self.declare_parameter("qf_pos", 50.0)       # terminal position
        self.declare_parameter("qf_yaw", 10.0)       # terminal yaw
        self.declare_parameter("r_v", 1.0)           # linear effort
        self.declare_parameter("r_w", 1.0)           # yaw effort
        self.declare_parameter("rd_v", 5.0)          # linear rate (smoothness)
        self.declare_parameter("rd_w", 5.0)          # yaw rate

        # --- goal depth override (preserves previous behavior) ---
        # NOTE: the previous node hard-set goal z = -1.2. RViz "2D Goal Pose"
        # has no z, so a fixed depth is needed. Kept as a parameter.
        self.declare_parameter("use_fixed_goal_z", True)
        self.declare_parameter("fixed_goal_z", -1.2)

        # --- tank soft constraints (DISABLED by default) ---
        # See the brov_ctrl_twist_simple tank range. The sign of the Z bound vs.
        # fixed_goal_z is inconsistent across the two nodes, so confirm the odom
        # Z convention before enabling these.
        self.declare_parameter("enable_tank_constraint", False)
        self.declare_parameter("tank_min", [1.0, -1.8, 0.3])
        self.declare_parameter("tank_max", [6.0, 1.8, 3.0])
        self.declare_parameter("tank_margin", 0.2)
        self.declare_parameter("tank_slack_weight", 1000.0)

        gp = self.get_parameter
        goal_topic = gp("goal_topic").get_parameter_value().string_value
        odom_topic = gp("odom_topic").get_parameter_value().string_value
        cmd_topic = gp("cmd_topic").get_parameter_value().string_value
        path_topic = gp("predicted_path_topic").get_parameter_value().string_value
        self._base_link_frame = gp("base_link_frame").get_parameter_value().string_value
        self._map_frame = gp("map_frame").get_parameter_value().string_value

        self._N = int(gp("horizon").value)
        self._dt = float(gp("mpc_dt").value)
        self._publish_rate_hz = float(gp("publish_rate_hz").value)
        self._odom_timeout = float(gp("odom_timeout").value)

        self._v_lin_max = float(gp("v_lin_max").value)
        self._v_yaw_max = float(gp("v_yaw_max").value)
        self._a_lin_max = float(gp("a_lin_max").value)
        self._a_yaw_max = float(gp("a_yaw_max").value)

        self._Q = np.diag([gp("q_pos").value] * 3 + [gp("q_yaw").value]).astype(float)
        self._Qf = np.diag([gp("qf_pos").value] * 3 + [gp("qf_yaw").value]).astype(float)
        self._R = np.diag([gp("r_v").value] * 3 + [gp("r_w").value]).astype(float)
        self._Rd = np.diag([gp("rd_v").value] * 3 + [gp("rd_w").value]).astype(float)

        self._use_fixed_goal_z = bool(gp("use_fixed_goal_z").value)
        self._fixed_goal_z = float(gp("fixed_goal_z").value)

        self._enable_tank = bool(gp("enable_tank_constraint").value)
        self._tank_min = np.asarray(gp("tank_min").value, dtype=float)
        self._tank_max = np.asarray(gp("tank_max").value, dtype=float)
        self._tank_margin = float(gp("tank_margin").value)
        self._tank_w = float(gp("tank_slack_weight").value)

        # Control bounds in [vx, vy, vz, omega] order.
        self._u_lo = np.array(
            [-self._v_lin_max, -self._v_lin_max, -self._v_lin_max, -self._v_yaw_max]
        )
        self._u_hi = -self._u_lo
        # Per-step acceleration bound on |u_k - u_{k-1}|.
        self._du = np.array(
            [self._a_lin_max, self._a_lin_max, self._a_lin_max, self._a_yaw_max]
        ) * self._dt

        # --- solver ---
        self._solver = OsqpSolver()
        self._solver_opts = SolverOptions()
        self._solver_opts.SetOption(OsqpSolver.id(), "max_iter", 4000)
        self._solver_opts.SetOption(OsqpSolver.id(), "verbose", 0)

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

        # Warm-start / accel-continuity state.
        self._u_prev = np.zeros(4, dtype=float)
        self._x_guess: np.ndarray | None = None
        self._u_guess: np.ndarray | None = None

        self.create_timer(1.0 / self._publish_rate_hz, self._tick)
        self.get_logger().info(
            f"brov_ctrl_waypoint MPC: goal='{goal_topic}', odom='{odom_topic}', "
            f"cmd='{cmd_topic}', N={self._N}, dt={self._dt:.2f}s "
            f"(horizon {self._N * self._dt:.1f}s), tank_constraint={self._enable_tank}"
        )

    # ------------------------------------------------------------------ I/O
    def _goal_cb(self, msg: PoseStamped) -> None:
        self._goal_pose = msg
        if self._use_fixed_goal_z:
            self._goal_pose.pose.position.z = self._fixed_goal_z
        self._goal_in_odom = None
        self._x_guess = None
        self._u_guess = None
        self.get_logger().info(
            f"New goal: frame='{msg.header.frame_id}', "
            f"pos=({msg.pose.position.x:.3f}, {msg.pose.position.y:.3f}, "
            f"{self._goal_pose.pose.position.z:.3f})"
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
        self._u_prev = np.zeros(4, dtype=float)
        self._x_guess = None
        self._u_guess = None

    # ------------------------------------------------------------------ MPC
    def _solve_mpc(
        self, x0: np.ndarray, goal: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Return (u0, predicted_states) or None on solver failure."""
        N, dt = self._N, self._dt
        prog = MathematicalProgram()
        X = prog.NewContinuousVariables(N + 1, 4, "x")
        U = prog.NewContinuousVariables(N, 4, "u")

        # Initial condition.
        prog.AddLinearEqualityConstraint(X[0], x0)

        for k in range(N):
            # Kinematic integrator: x_{k+1} = x_k + dt * u_k.
            prog.AddLinearEqualityConstraint(
                X[k + 1] - X[k] - dt * U[k], np.zeros(4)
            )
            # Velocity box.
            prog.AddBoundingBoxConstraint(self._u_lo, self._u_hi, U[k])
            # Acceleration box on |u_k - u_{k-1}|.
            if k == 0:
                prog.AddBoundingBoxConstraint(
                    self._u_prev - self._du, self._u_prev + self._du, U[0]
                )
            else:
                prog.AddLinearConstraint(U[k] - U[k - 1], -self._du, self._du)
            # Effort + rate cost.
            prog.AddQuadraticErrorCost(self._R, np.zeros(4), U[k])
            du = U[k] - (U[k - 1] if k > 0 else self._u_prev)
            prog.AddQuadraticCost(du @ self._Rd @ du, is_convex=True)
            # Stage tracking cost on X[1..N].
            prog.AddQuadraticErrorCost(self._Q, goal, X[k + 1])

        # Extra terminal weight on X[N] (on top of the stage cost above).
        prog.AddQuadraticErrorCost(self._Qf, goal, X[N])

        # Optional soft tank bounds (slack keeps the QP always feasible).
        if self._enable_tank:
            tmin = self._tank_min + self._tank_margin
            tmax = self._tank_max - self._tank_margin
            s = prog.NewContinuousVariables(N, 6, "tank_slack")
            sv = s.reshape(-1)
            prog.AddBoundingBoxConstraint(np.zeros(sv.size), np.full(sv.size, np.inf), sv)
            for k in range(1, N + 1):
                for i in range(3):
                    prog.AddLinearConstraint(X[k][i] + s[k - 1, i] >= tmin[i])
                    prog.AddLinearConstraint(X[k][i] - s[k - 1, 3 + i] <= tmax[i])
            prog.AddQuadraticCost(self._tank_w * (sv @ sv), is_convex=True)

        # Warm start from previous (shifted) solution.
        if self._x_guess is not None and self._u_guess is not None:
            prog.SetInitialGuess(X, self._x_guess)
            prog.SetInitialGuess(U, self._u_guess)

        result = self._solver.Solve(prog, None, self._solver_opts)
        if not result.is_success():
            return None

        X_sol = result.GetSolution(X)
        U_sol = result.GetSolution(U)
        # Shift for next warm start.
        self._x_guess = np.vstack([X_sol[1:], X_sol[-1]])
        self._u_guess = np.vstack([U_sol[1:], U_sol[-1]])
        return U_sol[0], X_sol

    def _publish_predicted_path(self, X_sol: np.ndarray, frame_id: str) -> None:
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
            ps.pose.position.z = float(state[2])
            qx, qy, qz, qw = rpy_to_quat(0.0, 0.0, float(state[3]))
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

        x0 = np.array(
            [odom_pose.position.x, odom_pose.position.y, odom_pose.position.z, cur_yaw]
        )
        goal_vec = np.array([goal.position.x, goal.position.y, goal.position.z, goal_yaw])

        solution = self._solve_mpc(x0, goal_vec)
        if solution is None:
            self._publish_zero_twist()
            self.get_logger().warn("MPC solve failed; forcing zero control.",
                                   throttle_duration_sec=2.0)
            return

        u0, X_sol = solution
        self._u_prev = u0

        twist = TwistStamped()
        twist.header.stamp = self.get_clock().now().to_msg()
        twist.header.frame_id = odom_frame  # odom-frame velocity, like the old node
        twist.twist.linear.x = float(u0[0])
        twist.twist.linear.y = float(u0[1])
        twist.twist.linear.z = float(u0[2])
        twist.twist.angular.z = float(u0[3])
        self._cmd_pub.publish(twist)

        self._publish_predicted_path(X_sol, odom_frame)


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
