from __future__ import annotations

import math
import sys
import time
import types
from pathlib import Path

import numpy as np


class _Header:
    def __init__(self) -> None:
        self.frame_id = ""
        self.stamp = None


class _Vector3:
    def __init__(self) -> None:
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0


class _Quaternion:
    def __init__(self) -> None:
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.w = 1.0


class _Pose:
    def __init__(self) -> None:
        self.position = _Vector3()
        self.orientation = _Quaternion()


class _PoseWithCovariance:
    def __init__(self) -> None:
        self.pose = _Pose()


class _Twist:
    def __init__(self) -> None:
        self.linear = _Vector3()
        self.angular = _Vector3()


class _TwistWithCovariance:
    def __init__(self) -> None:
        self.twist = _Twist()


class PoseStamped:
    def __init__(self) -> None:
        self.header = _Header()
        self.pose = _Pose()


class _Wrench:
    def __init__(self) -> None:
        self.force = _Vector3()
        self.torque = _Vector3()


class WrenchStamped:
    def __init__(self) -> None:
        self.header = _Header()
        self.wrench = _Wrench()


class OverrideRCIn:
    CHAN_NOCHANGE = 65535

    def __init__(self) -> None:
        self.channels = []


class Odometry:
    def __init__(self) -> None:
        self.pose = _PoseWithCovariance()
        self.twist = _TwistWithCovariance()


class _Publisher:
    def __init__(self) -> None:
        self.messages = []

    def publish(self, msg) -> None:
        self.messages.append(msg)


class _DummyClock:
    def now(self):
        return types.SimpleNamespace(to_msg=lambda: "stamp")


class Node:
    def __init__(self, *args, **kwargs) -> None:
        self._params = {}
        self._logger = types.SimpleNamespace(info=lambda *a, **k: None, warn=lambda *a, **k: None)

    def declare_parameter(self, name, default_value=None):
        self._params.setdefault(name, default_value)
        return types.SimpleNamespace(value=self._params[name])

    def get_parameter(self, name):
        return types.SimpleNamespace(value=self._params[name])

    def create_publisher(self, *args, **kwargs):
        return _Publisher()

    def create_subscription(self, *args, **kwargs):
        return None

    def create_timer(self, *args, **kwargs):
        return None

    def get_clock(self):
        return _DummyClock()

    def get_logger(self):
        return self._logger

    def destroy_node(self) -> None:
        return None


class Duration:
    def __init__(self, *, seconds: float) -> None:
        self.seconds = seconds


class QoSProfile:
    def __init__(self, *args, **kwargs) -> None:
        pass


class ReliabilityPolicy:
    RELIABLE = object()


class Buffer:
    def lookup_transform(self, *args, **kwargs):
        return None


class TransformListener:
    def __init__(self, *args, **kwargs) -> None:
        pass


class TransformException(Exception):
    pass


class Time:
    pass


geometry_msgs = types.ModuleType("geometry_msgs")
geometry_msgs_msg = types.ModuleType("geometry_msgs.msg")
geometry_msgs_msg.PoseStamped = PoseStamped
geometry_msgs_msg.WrenchStamped = WrenchStamped
geometry_msgs.msg = geometry_msgs_msg
sys.modules["geometry_msgs"] = geometry_msgs
sys.modules["geometry_msgs.msg"] = geometry_msgs_msg

mavros_msgs = types.ModuleType("mavros_msgs")
mavros_msgs_msg = types.ModuleType("mavros_msgs.msg")
mavros_msgs_msg.OverrideRCIn = OverrideRCIn
mavros_msgs.msg = mavros_msgs_msg
sys.modules["mavros_msgs"] = mavros_msgs
sys.modules["mavros_msgs.msg"] = mavros_msgs_msg

nav_msgs = types.ModuleType("nav_msgs")
nav_msgs_msg = types.ModuleType("nav_msgs.msg")
nav_msgs_msg.Odometry = Odometry
nav_msgs.msg = nav_msgs_msg
sys.modules["nav_msgs"] = nav_msgs
sys.modules["nav_msgs.msg"] = nav_msgs_msg

rclpy = types.ModuleType("rclpy")
rclpy.init = lambda *args, **kwargs: None
rclpy.shutdown = lambda *args, **kwargs: None
rclpy.spin = lambda *args, **kwargs: None
rclpy.time = types.SimpleNamespace(Time=Time)
sys.modules["rclpy"] = rclpy

rclpy_duration = types.ModuleType("rclpy.duration")
rclpy_duration.Duration = Duration
sys.modules["rclpy.duration"] = rclpy_duration

rclpy_node = types.ModuleType("rclpy.node")
rclpy_node.Node = Node
sys.modules["rclpy.node"] = rclpy_node

rclpy_qos = types.ModuleType("rclpy.qos")
rclpy_qos.QoSProfile = QoSProfile
rclpy_qos.ReliabilityPolicy = ReliabilityPolicy
sys.modules["rclpy.qos"] = rclpy_qos

tf2_geometry_msgs = types.ModuleType("tf2_geometry_msgs")
tf2_geometry_msgs.do_transform_pose_stamped = lambda pose, tf: pose
sys.modules["tf2_geometry_msgs"] = tf2_geometry_msgs

tf2_ros = types.ModuleType("tf2_ros")
tf2_ros.Buffer = Buffer
tf2_ros.TransformException = TransformException
tf2_ros.TransformListener = TransformListener
sys.modules["tf2_ros"] = tf2_ros

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mr_pinchy_interface.brov_ctrl_clfqp_wrench import (  # noqa: E402
    BrovCtrlCLFQPWrench,
    CLFQPProblem,
    body_wrench_from_tau,
    build_second_order_clf_matrix,
    clf_terms,
    dynamics_frd,
    flu_to_frd_vector,
    frd_action_to_flu,
    frd_state_to_flu,
    frd_to_flu_rpy,
    frd_to_flu_vector,
    neutral_rc_override,
    odom_to_frd_state,
    reduced_error_dynamics,
    tau_from_wrench_message,
    tau_reference,
    tracking_error_state,
)


def _quat_from_rpy(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def _make_goal(
    *,
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
    roll: float = 0.0,
    pitch: float = 0.0,
    yaw: float = 0.0,
    frame_id: str = "mr_pinchy/base_link",
) -> PoseStamped:
    msg = PoseStamped()
    msg.header.frame_id = frame_id
    msg.pose.position.x = x
    msg.pose.position.y = y
    msg.pose.position.z = z
    qx, qy, qz, qw = _quat_from_rpy(roll, pitch, yaw)
    msg.pose.orientation.x = qx
    msg.pose.orientation.y = qy
    msg.pose.orientation.z = qz
    msg.pose.orientation.w = qw
    return msg


def _make_odom(
    *,
    pos=(0.0, 0.0, 0.0),
    rpy=(0.0, 0.0, 0.0),
    linear=(0.0, 0.0, 0.0),
    angular=(0.0, 0.0, 0.0),
) -> Odometry:
    msg = Odometry()
    msg.pose.pose.position.x = pos[0]
    msg.pose.pose.position.y = pos[1]
    msg.pose.pose.position.z = pos[2]
    qx, qy, qz, qw = _quat_from_rpy(*rpy)
    msg.pose.pose.orientation.x = qx
    msg.pose.pose.orientation.y = qy
    msg.pose.pose.orientation.z = qz
    msg.pose.pose.orientation.w = qw
    msg.twist.twist.linear.x = linear[0]
    msg.twist.twist.linear.y = linear[1]
    msg.twist.twist.linear.z = linear[2]
    msg.twist.twist.angular.x = angular[0]
    msg.twist.twist.angular.y = angular[1]
    msg.twist.twist.angular.z = angular[2]
    return msg


def test_frd_flu_conversions_round_trip() -> None:
    vec = np.array([1.0, 2.0, -3.0])
    rpy = np.array([0.1, -0.2, 0.3])
    state = np.arange(1.0, 13.0)
    tau = np.array([1.0, -2.0, 3.0, -4.0, 5.0, -6.0])

    assert np.allclose(frd_to_flu_vector(vec), [1.0, -2.0, 3.0])
    assert np.allclose(flu_to_frd_vector(frd_to_flu_vector(vec)), vec)
    assert np.allclose(frd_to_flu_rpy(rpy), [0.1, 0.2, -0.3])
    assert np.allclose(frd_state_to_flu(state), state * np.array([1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1]))
    assert np.allclose(frd_action_to_flu(tau), tau * np.array([1, -1, -1, 1, -1, -1]))


def test_body_wrench_round_trip_matches_tau_order() -> None:
    tau = np.array([1.0, -2.0, 3.0, -4.0, 5.0, -6.0])
    msg = body_wrench_from_tau(tau, frame_id="mr_pinchy/base_link", stamp="stamp")

    assert msg.header.frame_id == "mr_pinchy/base_link"
    assert msg.header.stamp == "stamp"
    assert np.allclose(tau_from_wrench_message(msg), tau)


def test_odom_to_frd_state_uses_requested_state_ordering() -> None:
    odom = _make_odom(
        pos=(1.0, 2.0, 3.0),
        rpy=(0.1, -0.2, 0.3),
        linear=(4.0, 5.0, 6.0),
        angular=(0.4, 0.5, 0.6),
    )
    state = odom_to_frd_state(odom)

    assert np.allclose(state, [1.0, 2.0, 3.0, 0.1, -0.2, 0.3, 4.0, 5.0, 6.0, 0.4, 0.5, 0.6])


def test_tracking_error_state_matches_body_frame_transport_terms() -> None:
    goal = _make_goal(x=1.0, y=-2.0, z=0.5, roll=0.2, pitch=-0.1, yaw=0.3)
    twist = np.array([0.4, -0.2, 0.1, 0.3, -0.4, 0.2])
    e, e_dot = tracking_error_state(goal, twist)

    assert np.allclose(e, [1.0, -2.0, 0.5, 0.2, -0.1, 0.3], atol=1e-6)
    assert np.allclose(e_dot[:3], -twist[:3] - np.cross(twist[3:], e[:3]))
    assert np.allclose(e_dot[3:], -twist[3:])


def test_reduced_error_model_gives_expected_input_map_structure() -> None:
    goal = _make_goal(x=1.0, y=0.0, z=0.0, roll=0.2, pitch=0.0, yaw=0.0)
    body_twist = np.zeros(6)
    f = np.zeros(12)
    g = np.zeros((12, 6))
    g[6:, :] = np.eye(6)

    z, a, b = reduced_error_dynamics(goal, body_twist, f, g)

    assert np.allclose(z[:6], [1.0, 0.0, 0.0, 0.2, 0.0, 0.0], atol=1e-6)
    assert np.allclose(a, np.zeros(6))
    assert np.allclose(b[3:, :3], np.zeros((3, 3)))
    assert np.allclose(b[3:, 3:], -np.eye(3))
    assert np.allclose(b[0], [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert np.allclose(b[1], [0.0, -1.0, 0.0, 0.0, 0.0, -1.0])
    assert np.allclose(b[2], [0.0, 0.0, -1.0, 0.0, 1.0, 0.0])


def test_tau_reference_pushes_toward_goal_with_reduced_model_signs() -> None:
    e = np.array([1.0, -2.0, 0.5, 0.2, -0.3, 0.4])
    e_dot = np.zeros(6)
    a = np.zeros(6)
    b = -np.eye(6)
    kp = np.array([2.0, 2.0, 2.0, 3.0, 3.0, 3.0])
    kd = np.ones(6)
    tau = tau_reference(e, e_dot, a, b, kp, kd, -10.0 * np.ones(6), 10.0 * np.ones(6))

    assert np.allclose(tau, kp * e)


def test_second_order_clf_matrix_has_expected_axis_layout() -> None:
    kp = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    kd = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
    p = build_second_order_clf_matrix(kp, kd)

    assert p.shape == (12, 12)
    assert np.allclose(p, p.T)
    assert np.all(np.linalg.eigvals(p) > 0.0)
    assert p[0, 6] > 0.0
    assert p[0, 7] == 0.0


def test_clf_qp_respects_bounds_and_satisfies_constraint() -> None:
    kp = np.ones(6)
    kd = np.ones(6)
    p = build_second_order_clf_matrix(kp, kd)
    z = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    a = np.zeros(6)
    b = -np.eye(6)
    v, lf_v, lg_v = clf_terms(z, a, b, p)
    solver = CLFQPProblem(
        tau_min=-0.25 * np.ones(6),
        tau_max=0.25 * np.ones(6),
        clf_decay=1.0,
        slack_weight=1000.0,
        solver_eps_abs=1e-6,
        solver_eps_rel=1e-6,
        solver_max_iter=10000,
    )

    result = solver.solve(v=v, lf_v=lf_v, lg_v=lg_v, tau_ref=np.ones(6))

    assert result is not None
    assert np.all(result.tau <= 0.25 + 1e-8)
    assert np.all(result.tau >= -0.25 - 1e-8)
    lhs = float(lg_v @ result.tau + lf_v)
    rhs = -v + result.slack
    assert lhs <= rhs + 1e-6


def test_dynamics_frd_applies_frame_conversion_on_drift_and_input_map() -> None:
    import mr_pinchy_interface.brov_ctrl_clfqp_wrench as mod

    original = mod.bluerov2_dynamics

    def fake_dyn(x):
        f = np.arange(1.0, 13.0)
        g = np.ones((12, 6))
        return f, g

    mod.bluerov2_dynamics = fake_dyn
    try:
        f_frd, g_frd = dynamics_frd(np.zeros(12))
    finally:
        mod.bluerov2_dynamics = original

    assert np.allclose(f_frd, np.array([1, -2, -3, 4, -5, -6, 7, -8, -9, 10, -11, -12]))
    assert g_frd[0, 0] == 1.0
    assert g_frd[0, 1] == -1.0
    assert g_frd[1, 0] == -1.0
    assert g_frd[1, 1] == 1.0


def test_neutral_rc_override_has_expected_shape() -> None:
    msg = neutral_rc_override(neutral_pwm=1500)

    assert msg.channels[:6] == [1500] * 6
    assert msg.channels[6:] == [OverrideRCIn.CHAN_NOCHANGE] * 12


def test_node_missing_goal_publishes_zero_control() -> None:
    node = BrovCtrlCLFQPWrench()

    node._tick()

    wrench_msg = node._wrench_pub.messages[-1]
    rc_msg = node._rc_pub.messages[-1]
    assert np.allclose(tau_from_wrench_message(wrench_msg), np.zeros(6))
    assert rc_msg.channels[:6] == [1500] * 6


def test_node_stale_odom_publishes_zero_control() -> None:
    node = BrovCtrlCLFQPWrench()
    node._goal_pose = _make_goal()
    node._last_odom = _make_odom()
    node._last_odom_rx_sec = time.monotonic() - 10.0
    node._odom_timeout = 0.1

    node._tick()

    assert np.allclose(tau_from_wrench_message(node._wrench_pub.messages[-1]), np.zeros(6))


def test_node_tf_failure_publishes_zero_control() -> None:
    node = BrovCtrlCLFQPWrench()
    node._goal_pose = _make_goal(frame_id="map")
    node._last_odom = _make_odom()
    node._last_odom_rx_sec = time.monotonic()

    def fail_lookup(*args, **kwargs):
        raise TransformException("no tf")

    node._tf_buffer.lookup_transform = fail_lookup
    node._tick()

    assert np.allclose(tau_from_wrench_message(node._wrench_pub.messages[-1]), np.zeros(6))


def test_node_qp_failure_publishes_zero_control() -> None:
    node = BrovCtrlCLFQPWrench()
    node._goal_pose = _make_goal()
    node._last_odom = _make_odom()
    node._last_odom_rx_sec = time.monotonic()
    node._compute_control = lambda goal_in_base: None

    node._tick()

    assert np.allclose(tau_from_wrench_message(node._wrench_pub.messages[-1]), np.zeros(6))
