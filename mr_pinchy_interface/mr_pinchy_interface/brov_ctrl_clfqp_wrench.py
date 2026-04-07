#!/usr/bin/env python3
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, WrenchStamped
from mavros_msgs.msg import OverrideRCIn
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from tf2_geometry_msgs import do_transform_pose_stamped
from tf2_ros import Buffer, TransformException, TransformListener

from mr_pinchy_interface.brov_ctrl_wrench import wrench_to_rc_override
from mr_pinchy_interface.brov_dyn import bluerov2_dynamics

_VECTOR_SIGN = np.array([1.0, -1.0, -1.0], dtype=float)
_ANGLE_SIGN = np.array([1.0, -1.0, -1.0], dtype=float)
_STATE_SIGN = np.concatenate((_VECTOR_SIGN, _ANGLE_SIGN, _VECTOR_SIGN, _ANGLE_SIGN))
_ACTION_SIGN = np.array([1.0, -1.0, -1.0, 1.0, -1.0, -1.0], dtype=float)


def _wrap_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _rpy_from_quat(x: float, y: float, z: float, w: float) -> tuple[float, float, float]:
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ],
        dtype=float,
    )


def _as_numpy(values, *, expected_len: int | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if expected_len is not None and arr.shape != (expected_len,):
        raise ValueError(f"expected shape ({expected_len},), got {arr.shape}")
    return arr


def _copy_pose_stamped(msg: PoseStamped) -> PoseStamped:
    out = PoseStamped()
    out.header = msg.header
    out.pose.position.x = msg.pose.position.x
    out.pose.position.y = msg.pose.position.y
    out.pose.position.z = msg.pose.position.z
    out.pose.orientation.x = msg.pose.orientation.x
    out.pose.orientation.y = msg.pose.orientation.y
    out.pose.orientation.z = msg.pose.orientation.z
    out.pose.orientation.w = msg.pose.orientation.w
    return out


def frd_to_flu_vector(vec: np.ndarray) -> np.ndarray:
    return _VECTOR_SIGN * _as_numpy(vec, expected_len=3)


def flu_to_frd_vector(vec: np.ndarray) -> np.ndarray:
    return _VECTOR_SIGN * _as_numpy(vec, expected_len=3)


def frd_to_flu_rpy(rpy: np.ndarray) -> np.ndarray:
    return _ANGLE_SIGN * _as_numpy(rpy, expected_len=3)


def flu_to_frd_rpy(rpy: np.ndarray) -> np.ndarray:
    return _ANGLE_SIGN * _as_numpy(rpy, expected_len=3)


def frd_state_to_flu(x_frd: np.ndarray) -> np.ndarray:
    return _STATE_SIGN * _as_numpy(x_frd, expected_len=12)


def flu_state_to_frd(x_flu: np.ndarray) -> np.ndarray:
    return _STATE_SIGN * _as_numpy(x_flu, expected_len=12)


def frd_action_to_flu(tau_frd: np.ndarray) -> np.ndarray:
    return _ACTION_SIGN * _as_numpy(tau_frd, expected_len=6)


def flu_action_to_frd(tau_flu: np.ndarray) -> np.ndarray:
    return _ACTION_SIGN * _as_numpy(tau_flu, expected_len=6)


def body_wrench_from_tau(
    tau: np.ndarray,
    *,
    frame_id: str = "",
    stamp=None,
) -> WrenchStamped:
    tau = _as_numpy(tau, expected_len=6)
    msg = WrenchStamped()
    if hasattr(msg, "header"):
        msg.header.frame_id = frame_id
        if stamp is not None:
            msg.header.stamp = stamp
    msg.wrench.force.x = float(tau[0])
    msg.wrench.force.y = float(tau[1])
    msg.wrench.force.z = float(tau[2])
    msg.wrench.torque.x = float(tau[3])
    msg.wrench.torque.y = float(tau[4])
    msg.wrench.torque.z = float(tau[5])
    return msg


def tau_from_wrench_message(msg: WrenchStamped) -> np.ndarray:
    return np.array(
        [
            msg.wrench.force.x,
            msg.wrench.force.y,
            msg.wrench.force.z,
            msg.wrench.torque.x,
            msg.wrench.torque.y,
            msg.wrench.torque.z,
        ],
        dtype=float,
    )


def neutral_rc_override(*, neutral_pwm: int) -> OverrideRCIn:
    msg = OverrideRCIn()
    channels = [OverrideRCIn.CHAN_NOCHANGE] * 18
    for idx in range(6):
        channels[idx] = neutral_pwm
    msg.channels = channels
    return msg


def odom_to_frd_state(msg: Odometry) -> np.ndarray:
    pose = msg.pose.pose
    twist = msg.twist.twist
    roll, pitch, yaw = _rpy_from_quat(
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    )
    return np.array(
        [
            pose.position.x,
            pose.position.y,
            pose.position.z,
            roll,
            pitch,
            yaw,
            twist.linear.x,
            twist.linear.y,
            twist.linear.z,
            twist.angular.x,
            twist.angular.y,
            twist.angular.z,
        ],
        dtype=float,
    )


def pose_error_from_goal(goal_pose: PoseStamped) -> np.ndarray:
    roll, pitch, yaw = _rpy_from_quat(
        goal_pose.pose.orientation.x,
        goal_pose.pose.orientation.y,
        goal_pose.pose.orientation.z,
        goal_pose.pose.orientation.w,
    )
    return np.array(
        [
            goal_pose.pose.position.x,
            goal_pose.pose.position.y,
            goal_pose.pose.position.z,
            _wrap_pi(roll),
            _wrap_pi(pitch),
            _wrap_pi(yaw),
        ],
        dtype=float,
    )


def tracking_error_state(goal_pose: PoseStamped, body_twist_frd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    e = pose_error_from_goal(goal_pose)
    twist = _as_numpy(body_twist_frd, expected_len=6)
    vel = twist[:3]
    rates = twist[3:]
    e_dot_pos = -vel - np.cross(rates, e[:3])
    e_dot_att = -rates
    e_dot = np.concatenate((e_dot_pos, e_dot_att))
    return e, e_dot


def dynamics_frd(x_frd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_flu = frd_state_to_flu(x_frd)
    f_flu, g_flu = bluerov2_dynamics(x_flu)
    f_flu_np = np.asarray(f_flu, dtype=float)
    g_flu_np = np.asarray(g_flu, dtype=float)
    f_frd = flu_state_to_frd(f_flu_np)
    g_frd = _STATE_SIGN[:, None] * g_flu_np * _ACTION_SIGN[None, :]
    return f_frd, g_frd


def reduced_error_dynamics(
    goal_pose: PoseStamped,
    body_twist_frd: np.ndarray,
    f_frd: np.ndarray,
    g_frd: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    e, e_dot = tracking_error_state(goal_pose, body_twist_frd)
    body_twist = _as_numpy(body_twist_frd, expected_len=6)
    rates = body_twist[3:]

    f_acc = _as_numpy(f_frd[6:], expected_len=6)
    g_acc = np.asarray(g_frd[6:, :], dtype=float)
    f_lin = f_acc[:3]
    f_ang = f_acc[3:]
    g_lin = g_acc[:3, :]
    g_ang = g_acc[3:, :]

    a_pos = -f_lin + _skew(e[:3]) @ f_ang - np.cross(rates, e_dot[:3])
    b_pos = -g_lin + _skew(e[:3]) @ g_ang
    a_att = -f_ang
    b_att = -g_ang

    z = np.concatenate((e, e_dot))
    a = np.concatenate((a_pos, a_att))
    b = np.vstack((b_pos, b_att))
    return z, a, b


def build_second_order_clf_matrix(kp: np.ndarray, kd: np.ndarray) -> np.ndarray:
    kp = _as_numpy(kp, expected_len=6)
    kd = _as_numpy(kd, expected_len=6)
    p = np.zeros((12, 12), dtype=float)
    for idx, (kp_i, kd_i) in enumerate(zip(kp, kd)):
        if kp_i <= 0.0 or kd_i <= 0.0:
            raise ValueError("all kp/kd entries must be positive")
        p12 = 1.0 / (2.0 * kp_i)
        p22 = (1.0 + 1.0 / kp_i) / (2.0 * kd_i)
        p11 = kd_i * p12 + kp_i * p22
        block = np.array([[p11, p12], [p12, p22]], dtype=float)
        start = 2 * idx
        p[start : start + 2, start : start + 2] = block
    perm = [0, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9, 11]
    return p[np.ix_(perm, perm)]


def clf_terms(
    z: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    p: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    z = _as_numpy(z, expected_len=12)
    a = _as_numpy(a, expected_len=6)
    b = np.asarray(b, dtype=float)
    p = np.asarray(p, dtype=float)
    grad_v = 2.0 * (p @ z)
    v = float(z @ p @ z)
    lf_v = float(grad_v[:6] @ z[6:] + grad_v[6:] @ a)
    lg_v = np.asarray(grad_v[6:] @ b, dtype=float)
    return v, lf_v, lg_v


def tau_reference(
    e: np.ndarray,
    e_dot: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    kp: np.ndarray,
    kd: np.ndarray,
    tau_min: np.ndarray,
    tau_max: np.ndarray,
) -> np.ndarray:
    e = _as_numpy(e, expected_len=6)
    e_dot = _as_numpy(e_dot, expected_len=6)
    a = _as_numpy(a, expected_len=6)
    b = np.asarray(b, dtype=float)
    kp = _as_numpy(kp, expected_len=6)
    kd = _as_numpy(kd, expected_len=6)
    tau_min = _as_numpy(tau_min, expected_len=6)
    tau_max = _as_numpy(tau_max, expected_len=6)
    desired_accel = -(kp * e + kd * e_dot)
    tau = np.linalg.pinv(b) @ (desired_accel - a)
    return np.clip(np.asarray(tau, dtype=float), tau_min, tau_max)


@dataclass
class CLFQPResult:
    tau: np.ndarray
    slack: float
    value: float
    status: str


class CLFQPProblem:
    def __init__(
        self,
        *,
        tau_min: np.ndarray,
        tau_max: np.ndarray,
        clf_decay: float,
        slack_weight: float,
        solver_eps_abs: float,
        solver_eps_rel: float,
        solver_max_iter: int,
    ) -> None:
        if clf_decay <= 0.0:
            raise ValueError("clf_decay must be positive")
        if slack_weight <= 0.0:
            raise ValueError("slack_weight must be positive")

        self._tau = cp.Variable(6)
        self._delta = cp.Variable(nonneg=True)
        self._tau_ref = cp.Parameter(6)
        self._tau_min = cp.Parameter(6)
        self._tau_max = cp.Parameter(6)
        self._lg_v = cp.Parameter(6)
        self._lf_v = cp.Parameter()
        self._rhs = cp.Parameter()

        objective = 0.5 * cp.sum_squares(self._tau - self._tau_ref) + 0.5 * slack_weight * cp.square(
            self._delta
        )
        constraints = [
            self._lg_v @ self._tau + self._lf_v <= self._rhs + self._delta,
            self._tau >= self._tau_min,
            self._tau <= self._tau_max,
        ]
        self._problem = cp.Problem(cp.Minimize(objective), constraints)
        self._tau_min.value = _as_numpy(tau_min, expected_len=6)
        self._tau_max.value = _as_numpy(tau_max, expected_len=6)
        self._clf_decay = float(clf_decay)
        self._solver_eps_abs = float(solver_eps_abs)
        self._solver_eps_rel = float(solver_eps_rel)
        self._solver_max_iter = int(solver_max_iter)

    def solve(
        self,
        *,
        v: float,
        lf_v: float,
        lg_v: np.ndarray,
        tau_ref: np.ndarray,
    ) -> CLFQPResult | None:
        self._lf_v.value = float(lf_v)
        self._lg_v.value = _as_numpy(lg_v, expected_len=6)
        self._rhs.value = -self._clf_decay * float(v)
        self._tau_ref.value = _as_numpy(tau_ref, expected_len=6)
        try:
            self._problem.solve(
                solver=cp.OSQP,
                warm_start=True,
                eps_abs=self._solver_eps_abs,
                eps_rel=self._solver_eps_rel,
                max_iter=self._solver_max_iter,
            )
        except Exception:
            return None

        if self._problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            return None
        if self._tau.value is None or self._delta.value is None:
            return None

        tau = np.asarray(self._tau.value, dtype=float).reshape(6)
        tau = np.clip(tau, self._tau_min.value, self._tau_max.value)
        return CLFQPResult(
            tau=tau,
            slack=float(self._delta.value),
            value=float(self._problem.value),
            status=str(self._problem.status),
        )


class BrovCtrlCLFQPWrench(Node):
    def __init__(self) -> None:
        super().__init__("brov_ctrl_clfqp_wrench")

        self.declare_parameter("goal_topic", "/goal_pose")
        self.declare_parameter("odom_topic", "/mocap/mr_pinchy/odom")
        self.declare_parameter("rc_topic", "/mavros/rc/override")
        self.declare_parameter("wrench_topic", "/cmd_wrench")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_link_frame", "mr_pinchy/base_link")

        goal_topic = self.get_parameter("goal_topic").value
        odom_topic = self.get_parameter("odom_topic").value
        rc_topic = self.get_parameter("rc_topic").value
        wrench_topic = self.get_parameter("wrench_topic").value
        self._map_frame = self.get_parameter("map_frame").value
        self._base_link_frame = self.get_parameter("base_link_frame").value

        self._publish_rate_hz = 20.0
        self._period = 1.0 / self._publish_rate_hz
        self._odom_timeout = 0.3
        self._neutral_pwm = 1500
        self._debug_log = True
        self._clf_decay = 1.0
        self._slack_weight = 1000.0
        self._solver_eps_abs = 1e-5
        self._solver_eps_rel = 1e-5
        self._solver_max_iter = 10000
        self._kp = np.array([2.0, 2.0, 2.5, 4.0, 4.0, 2.5], dtype=float)
        self._kd = np.array([2.0, 2.0, 2.5, 2.0, 2.0, 1.5], dtype=float)
        force_bounds = np.array([20.0, 20.0, 30.0], dtype=float)
        torque_bounds = np.array([1.5, 1.5, 1.0], dtype=float)
        self._tau_max = np.concatenate((force_bounds, torque_bounds))
        self._tau_min = -self._tau_max
        self._clf_p = build_second_order_clf_matrix(self._kp, self._kd)
        self._qp = CLFQPProblem(
            tau_min=self._tau_min,
            tau_max=self._tau_max,
            clf_decay=self._clf_decay,
            slack_weight=self._slack_weight,
            solver_eps_abs=self._solver_eps_abs,
            solver_eps_rel=self._solver_eps_rel,
            solver_max_iter=self._solver_max_iter,
        )

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self._wrench_pub = self.create_publisher(WrenchStamped, wrench_topic, qos)
        self._rc_pub = self.create_publisher(OverrideRCIn, rc_topic, qos)
        self.create_subscription(PoseStamped, goal_topic, self._goal_cb, 10)
        self.create_subscription(Odometry, odom_topic, self._odom_cb, 10)

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._goal_pose: PoseStamped | None = None
        self._last_odom: Odometry | None = None
        self._last_odom_rx_sec = 0.0
        self._last_tick_sec = time.monotonic()

        self.create_timer(self._period, self._tick)
        self.get_logger().info(
            f"brov_ctrl_clfqp_wrench: goal='{goal_topic}', odom='{odom_topic}', "
            f"wrench='{wrench_topic}', rc='{rc_topic}', map_frame='{self._map_frame}', "
            f"base_link_frame='{self._base_link_frame}', rate={self._publish_rate_hz:.1f}Hz, "
            f"odom_timeout={self._odom_timeout:.2f}s"
        )

    def _goal_cb(self, msg: PoseStamped) -> None:
        self._goal_pose = _copy_pose_stamped(msg)
        self._goal_pose.pose.orientation.x = 0.0
        self._goal_pose.pose.orientation.y = 0.0
        self._goal_pose.pose.orientation.z = 0.0
        self._goal_pose.pose.orientation.w = 1.0

    def _odom_cb(self, msg: Odometry) -> None:
        self._last_odom = msg
        self._last_odom_rx_sec = time.monotonic()

    def _goal_in_base_link(self) -> PoseStamped | None:
        if self._goal_pose is None:
            return None

        src = _copy_pose_stamped(self._goal_pose)
        if not src.header.frame_id:
            src.header.frame_id = self._map_frame

        if src.header.frame_id == self._base_link_frame:
            return src

        try:
            tf = self._tf_buffer.lookup_transform(
                self._base_link_frame,
                src.header.frame_id,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.05),
            )
            return do_transform_pose_stamped(src, tf)
        except TransformException as exc:
            self.get_logger().warn(
                f"TF {src.header.frame_id} -> {self._base_link_frame} unavailable: {exc}",
                throttle_duration_sec=2.0,
            )
            return None

    def _publish_wrench_and_rc(self, wrench_msg: WrenchStamped, rc_msg: OverrideRCIn) -> None:
        self._wrench_pub.publish(wrench_msg)
        self._rc_pub.publish(rc_msg)

    def _publish_zero_control(self) -> None:
        stamp = self.get_clock().now().to_msg()
        wrench_msg = body_wrench_from_tau(np.zeros(6), frame_id=self._base_link_frame, stamp=stamp)
        rc_msg = neutral_rc_override(neutral_pwm=self._neutral_pwm)
        self._publish_wrench_and_rc(wrench_msg, rc_msg)

    def _compute_control(self, goal_in_base: PoseStamped) -> CLFQPResult | None:
        if self._last_odom is None:
            return None

        x_frd = odom_to_frd_state(self._last_odom)
        body_twist_frd = x_frd[6:]
        f_frd, g_frd = dynamics_frd(x_frd)
        z, a, b = reduced_error_dynamics(goal_in_base, body_twist_frd, f_frd, g_frd)
        e = z[:6]
        e_dot = z[6:]
        v, lf_v, lg_v = clf_terms(z, a, b, self._clf_p)
        tau_ref = tau_reference(e, e_dot, a, b, self._kp, self._kd, self._tau_min, self._tau_max)
        return self._qp.solve(v=v, lf_v=lf_v, lg_v=lg_v, tau_ref=tau_ref)

    def _tick(self) -> None:
        now = time.monotonic()
        dt = max(now - self._last_tick_sec, 1e-3)
        self._last_tick_sec = now

        if self._goal_pose is None:
            self._publish_zero_control()
            return

        if self._last_odom is None or (now - self._last_odom_rx_sec) > self._odom_timeout:
            self._publish_zero_control()
            self.get_logger().warn(
                f"Odometry timeout ({self._odom_timeout:.2f}s); forcing zero control.",
                throttle_duration_sec=2.0,
            )
            return

        goal_in_base = self._goal_in_base_link()
        if goal_in_base is None:
            self._publish_zero_control()
            return

        result = self._compute_control(goal_in_base)
        if result is None:
            self._publish_zero_control()
            self.get_logger().warn(
                "CLF-QP solve failed; forcing zero control.",
                throttle_duration_sec=2.0,
            )
            return

        stamp = self.get_clock().now().to_msg()
        wrench_msg = body_wrench_from_tau(result.tau, frame_id=self._base_link_frame, stamp=stamp)
        rc_msg = wrench_to_rc_override(wrench_msg)

        if self._debug_log:
            e, e_dot = tracking_error_state(goal_in_base, odom_to_frd_state(self._last_odom)[6:])
            self.get_logger().info(
                f"err=({e[0]:.3f}, {e[1]:.3f}, {e[2]:.3f}, {e[3]:.3f}, {e[4]:.3f}, {e[5]:.3f}) "
                f"err_dot=({e_dot[0]:.3f}, {e_dot[1]:.3f}, {e_dot[2]:.3f}, {e_dot[3]:.3f}, {e_dot[4]:.3f}, {e_dot[5]:.3f}) "
                f"tau=({result.tau[0]:.3f}, {result.tau[1]:.3f}, {result.tau[2]:.3f}, "
                f"{result.tau[3]:.3f}, {result.tau[4]:.3f}, {result.tau[5]:.3f}) "
                f"slack={result.slack:.3e} status={result.status} dt={dt:.3f}"
            )

        self._publish_wrench_and_rc(wrench_msg, rc_msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BrovCtrlCLFQPWrench()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
