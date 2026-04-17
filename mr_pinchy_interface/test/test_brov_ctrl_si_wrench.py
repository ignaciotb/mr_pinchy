from __future__ import annotations

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


class TwistStamped:
    def __init__(self) -> None:
        self.header = _Header()
        self.twist = _Twist()


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


class _Logger:
    def info(self, *args, **kwargs) -> None:
        return None

    def warn(self, *args, **kwargs) -> None:
        return None


class Node:
    def __init__(self, *args, **kwargs) -> None:
        self._params = {}
        self._logger = _Logger()

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


class QoSProfile:
    def __init__(self, *args, **kwargs) -> None:
        pass


class ReliabilityPolicy:
    RELIABLE = object()


class Duration:
    def __init__(self, *, seconds: float) -> None:
        self.seconds = seconds


class Time:
    pass


class TransformException(Exception):
    pass


class Buffer:
    def __init__(self) -> None:
        self.lookup_result = None
        self.lookup_exception = None

    def lookup_transform(self, *args, **kwargs):
        if self.lookup_exception is not None:
            raise self.lookup_exception
        if self.lookup_result is None:
            raise TransformException("missing transform")
        return self.lookup_result


class TransformListener:
    def __init__(self, *args, **kwargs) -> None:
        pass


geometry_msgs = types.ModuleType("geometry_msgs")
geometry_msgs_msg = types.ModuleType("geometry_msgs.msg")
geometry_msgs_msg.TwistStamped = TwistStamped
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

tf2_ros = types.ModuleType("tf2_ros")
tf2_ros.Buffer = Buffer
tf2_ros.TransformException = TransformException
tf2_ros.TransformListener = TransformListener
sys.modules["tf2_ros"] = tf2_ros

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mr_pinchy.mr_pinchy_interface.mr_pinchy_interface.brov_ctrl_twist import (  # noqa: E402
    BrovCtrlSIWrench,
    body_wrench_message,
    clamp_body_wrench,
    odom_to_body_state,
    static_wrench_feedforward,
    twist_to_body_velocity,
    velocity_feedback_wrench,
)
from mr_pinchy.mr_pinchy_interface.mr_pinchy_interface.brov_wrench_to_pwm import wrench_to_rc_override  # noqa: E402
from mr_pinchy.mr_pinchy_interface.mr_pinchy_interface.brov_dynamics import (  # noqa: E402
    bluerov2_dynamics,
    bluerov2_model_terms,
)


def _quat_from_rpy(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def _make_transform(
    *,
    translation=(0.0, 0.0, 0.0),
    quaternion=(0.0, 0.0, 0.0, 1.0),
):
    return types.SimpleNamespace(
        transform=types.SimpleNamespace(
            translation=types.SimpleNamespace(
                x=translation[0],
                y=translation[1],
                z=translation[2],
            ),
            rotation=types.SimpleNamespace(
                x=quaternion[0],
                y=quaternion[1],
                z=quaternion[2],
                w=quaternion[3],
            ),
        )
    )


def _make_twist_stamped(
    *,
    linear=(0.0, 0.0, 0.0),
    angular=(0.0, 0.0, 0.0),
    frame_id: str = "mr_pinchy/base_link",
) -> TwistStamped:
    msg = TwistStamped()
    msg.header.frame_id = frame_id
    msg.twist.linear.x = linear[0]
    msg.twist.linear.y = linear[1]
    msg.twist.linear.z = linear[2]
    msg.twist.angular.x = angular[0]
    msg.twist.angular.y = angular[1]
    msg.twist.angular.z = angular[2]
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


def _wrench_vector(msg: WrenchStamped) -> np.ndarray:
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


def test_bluerov2_model_terms_reconstruct_dynamics() -> None:
    x = np.array([1.0, -2.0, 0.5, 0.2, -0.1, 0.3, 0.4, -0.3, 0.2, 0.1, -0.05, 0.07])

    f, g = bluerov2_dynamics(x)
    terms = bluerov2_model_terms(x[:6], x[6:])
    f_expected = np.concatenate(
        [
            np.asarray(terms["J"] @ terms["nu"], dtype=float),
            np.asarray(
                terms["M_inv"]
                @ (-terms["C"] @ terms["nu"] - terms["D"] @ terms["nu_rel"] - terms["g"]),
                dtype=float,
            ),
        ]
    )

    assert np.allclose(np.asarray(f, dtype=float), f_expected)
    assert np.allclose(np.asarray(g[:6, :], dtype=float), np.zeros((6, 6)))
    assert np.allclose(np.asarray(g[6:, :], dtype=float), np.asarray(terms["M_inv"], dtype=float))


def test_twist_to_body_velocity_uses_frd_ordering() -> None:
    msg = _make_twist_stamped(linear=(1.0, 2.0, 3.0), angular=(4.0, 5.0, 6.0))

    assert np.allclose(twist_to_body_velocity(msg), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])


def test_odom_to_body_state_uses_pose_and_body_twist_ordering() -> None:
    odom = _make_odom(
        pos=(1.0, 2.0, 3.0),
        rpy=(0.1, -0.2, 0.3),
        linear=(4.0, 5.0, 6.0),
        angular=(0.4, 0.5, 0.6),
    )

    eta, nu = odom_to_body_state(odom)

    assert np.allclose(eta, [1.0, 2.0, 3.0, 0.1, -0.2, 0.3], atol=1e-6)
    assert np.allclose(nu, [4.0, 5.0, 6.0, 0.4, 0.5, 0.6], atol=1e-6)


def test_zero_reference_feedforward_reduces_to_restoring_term() -> None:
    eta = np.array([0.0, 0.0, 0.0, 0.2, -0.1, 0.3], dtype=float)
    nu_ref = np.zeros(6)

    tau_ff = static_wrench_feedforward(eta, nu_ref)
    restoring = np.asarray(bluerov2_model_terms(eta, nu_ref)["g"], dtype=float)

    assert np.allclose(tau_ff, restoring)


def test_velocity_feedback_wrench_has_expected_sign() -> None:
    nu = np.array([1.0, -2.0, 3.0, -4.0, 5.0, -6.0])
    nu_ref = np.zeros(6)
    gains = np.array([2.0, 2.0, 2.0, 3.0, 3.0, 3.0])

    tau_fb = velocity_feedback_wrench(nu, nu_ref, gains)

    assert np.allclose(tau_fb, [-2.0, 4.0, -6.0, 12.0, -15.0, 18.0])


def test_static_feedforward_uses_reference_velocity_not_measured_velocity() -> None:
    node = BrovCtrlSIWrench()
    node._gains = np.zeros(6)
    node._nu_ref = np.array([0.3, -0.2, 0.1, 0.05, -0.04, 0.03], dtype=float)
    node._last_odom = _make_odom(rpy=(0.1, -0.05, 0.2), linear=(0.0, 0.0, 0.0))

    tau_a = node._compute_control()

    node._last_odom = _make_odom(
        rpy=(0.1, -0.05, 0.2),
        linear=(3.0, 2.0, 1.0),
        angular=(0.5, 0.4, 0.3),
    )
    tau_b = node._compute_control()

    assert tau_a is not None
    assert tau_b is not None
    assert np.allclose(tau_a, tau_b)


def test_compute_control_matches_feedforward_when_tracking_exact_reference() -> None:
    node = BrovCtrlSIWrench()
    eta = np.array([0.0, 0.0, 0.0, 0.1, -0.05, 0.2], dtype=float)
    nu_ref = np.array([0.2, -0.1, 0.05, 0.02, -0.03, 0.04], dtype=float)
    node._nu_ref = nu_ref.copy()
    node._last_odom = _make_odom(
        rpy=(eta[3], eta[4], eta[5]),
        linear=(nu_ref[0], nu_ref[1], nu_ref[2]),
        angular=(nu_ref[3], nu_ref[4], nu_ref[5]),
    )

    tau = node._compute_control()
    tau_ff = static_wrench_feedforward(eta, nu_ref)
    tau_ff = clamp_body_wrench(
        tau_ff,
        force_limits=node._force_limits,
        torque_limits=node._torque_limits,
    )

    assert tau is not None
    assert np.allclose(velocity_feedback_wrench(nu_ref, nu_ref, node._gains), np.zeros(6))
    assert np.allclose(tau, tau_ff)


def test_command_in_si_link_rotates_flu_to_frd() -> None:
    node = BrovCtrlSIWrench()
    node._tf_buffer.lookup_result = _make_transform(quaternion=(1.0, 0.0, 0.0, 0.0))

    nu_ref = node._command_in_base_link(
        _make_twist_stamped(
            linear=(1.0, 2.0, 3.0),
            angular=(4.0, 5.0, 6.0),
            frame_id="mr_pinchy/si_link",
        )
    )

    assert nu_ref is not None
    assert np.allclose(nu_ref, [1.0, -2.0, -3.0, 4.0, -5.0, -6.0])


def test_command_in_base_link_is_used_unchanged() -> None:
    node = BrovCtrlSIWrench()
    nu_ref = node._command_in_base_link(
        _make_twist_stamped(
            linear=(1.0, 2.0, 3.0),
            angular=(4.0, 5.0, 6.0),
            frame_id="mr_pinchy/base_link",
        )
    )

    assert nu_ref is not None
    assert np.allclose(nu_ref, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])


def test_missing_tf_forces_zero_output() -> None:
    node = BrovCtrlSIWrench()
    node._tf_buffer.lookup_exception = TransformException("missing transform")
    node._cmd_cb(_make_twist_stamped(frame_id="mr_pinchy/si_link", linear=(0.3, 0.0, 0.0)))
    node._odom_cb(_make_odom())

    node._tick()

    wrench_msg = node._wrench_pub.messages[-1]
    rc_msg = node._rc_pub.messages[-1]
    assert np.allclose(_wrench_vector(wrench_msg), np.zeros(6))
    assert rc_msg.channels[:6] == [1500] * 6


def test_timeout_on_stale_command_publishes_zero_wrench_and_neutral_rc() -> None:
    node = BrovCtrlSIWrench()
    node._nu_ref = np.ones(6)
    node._last_cmd_rx_sec = time.monotonic() - 10.0
    node._last_odom = _make_odom()
    node._last_odom_rx_sec = time.monotonic()
    node._tick()

    wrench_msg = node._wrench_pub.messages[-1]
    rc_msg = node._rc_pub.messages[-1]
    assert np.allclose(_wrench_vector(wrench_msg), np.zeros(6))
    assert rc_msg.channels[:6] == [1500] * 6


def test_timeout_on_stale_odom_publishes_zero_wrench_and_neutral_rc() -> None:
    node = BrovCtrlSIWrench()
    node._last_cmd = _make_twist_stamped(linear=(0.2, 0.0, 0.0))
    node._last_cmd_rx_sec = time.monotonic()
    node._last_odom = _make_odom()
    node._last_odom_rx_sec = time.monotonic() - 10.0
    node._tick()

    wrench_msg = node._wrench_pub.messages[-1]
    rc_msg = node._rc_pub.messages[-1]
    assert np.allclose(_wrench_vector(wrench_msg), np.zeros(6))
    assert rc_msg.channels[:6] == [1500] * 6


def test_body_wrench_message_composes_with_wrench_to_rc_override() -> None:
    wrench_msg = body_wrench_message(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]))
    rc_msg = wrench_to_rc_override(wrench_msg)

    assert rc_msg.channels[3] == 1523
    assert rc_msg.channels[0] == 1500
    assert rc_msg.channels[1] == 1500
    assert rc_msg.channels[2] == 1500
    assert rc_msg.channels[4] == 1500
    assert rc_msg.channels[5] == 1500


def test_node_with_fresh_command_and_odom_publishes_nonzero_control() -> None:
    node = BrovCtrlSIWrench()
    node._cmd_cb(_make_twist_stamped(linear=(0.2, 0.0, 0.0), angular=(0.0, 0.0, 0.1)))
    node._odom_cb(_make_odom())

    node._tick()

    wrench_msg = node._wrench_pub.messages[-1]
    rc_msg = node._rc_pub.messages[-1]
    assert not np.allclose(_wrench_vector(wrench_msg), np.zeros(6))
    assert rc_msg.channels[:6] != [1500] * 6
