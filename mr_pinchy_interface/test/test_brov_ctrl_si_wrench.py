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


class PoseStamped:
    def __init__(self) -> None:
        self.header = _Header()
        self.pose = _Pose()


class Vector3Stamped:
    def __init__(self) -> None:
        self.header = _Header()
        self.vector = _Vector3()


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


class _Transform:
    def __init__(self) -> None:
        self.translation = _Vector3()
        self.rotation = _Quaternion()


class TransformStamped:
    def __init__(self) -> None:
        self.header = _Header()
        self.child_frame_id = ""
        self.transform = _Transform()


class Odometry:
    def __init__(self) -> None:
        self.header = _Header()
        self.child_frame_id = ""
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

    def lookup_transform(self, target_frame, source_frame, *args, **kwargs):
        key = (target_frame, source_frame)
        if isinstance(self.lookup_exception, dict):
            if key in self.lookup_exception:
                raise self.lookup_exception[key]
        elif self.lookup_exception is not None:
            raise self.lookup_exception

        if isinstance(self.lookup_result, dict):
            if key in self.lookup_result:
                return self.lookup_result[key]
            raise TransformException(f"missing transform {key}")

        if self.lookup_result is None:
            raise TransformException("missing transform")
        return self.lookup_result


class TransformListener:
    def __init__(self, *args, **kwargs) -> None:
        pass


geometry_msgs = types.ModuleType("geometry_msgs")
geometry_msgs_msg = types.ModuleType("geometry_msgs.msg")
geometry_msgs_msg.PoseStamped = PoseStamped
geometry_msgs_msg.TransformStamped = TransformStamped
geometry_msgs_msg.TwistStamped = TwistStamped
geometry_msgs_msg.Vector3Stamped = Vector3Stamped
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


def _quat_multiply(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def _quat_to_rotation_matrix(
    quaternion: tuple[float, float, float, float],
) -> np.ndarray:
    x, y, z, w = quaternion
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=float,
    )


def _rotate_vector(
    quaternion: tuple[float, float, float, float],
    vector: tuple[float, float, float],
) -> np.ndarray:
    return _quat_to_rotation_matrix(quaternion) @ np.asarray(vector, dtype=float)


def do_transform_pose_stamped(pose: PoseStamped, transform: TransformStamped) -> PoseStamped:
    rotation = (
        transform.transform.rotation.x,
        transform.transform.rotation.y,
        transform.transform.rotation.z,
        transform.transform.rotation.w,
    )
    translated = _rotate_vector(
        rotation,
        (pose.pose.position.x, pose.pose.position.y, pose.pose.position.z),
    ) + np.array(
        [
            transform.transform.translation.x,
            transform.transform.translation.y,
            transform.transform.translation.z,
        ],
        dtype=float,
    )
    orientation = _quat_multiply(
        rotation,
        (
            pose.pose.orientation.x,
            pose.pose.orientation.y,
            pose.pose.orientation.z,
            pose.pose.orientation.w,
        ),
    )

    out = PoseStamped()
    out.header = transform.header
    out.pose.position.x = translated[0]
    out.pose.position.y = translated[1]
    out.pose.position.z = translated[2]
    out.pose.orientation.x = orientation[0]
    out.pose.orientation.y = orientation[1]
    out.pose.orientation.z = orientation[2]
    out.pose.orientation.w = orientation[3]
    return out


def do_transform_vector3(vector: Vector3Stamped, transform: TransformStamped) -> Vector3Stamped:
    rotation = (
        transform.transform.rotation.x,
        transform.transform.rotation.y,
        transform.transform.rotation.z,
        transform.transform.rotation.w,
    )
    rotated = _rotate_vector(rotation, (vector.vector.x, vector.vector.y, vector.vector.z))
    out = Vector3Stamped()
    out.header = transform.header
    out.vector.x = rotated[0]
    out.vector.y = rotated[1]
    out.vector.z = rotated[2]
    return out


tf2_geometry_msgs = types.ModuleType("tf2_geometry_msgs")
tf2_geometry_msgs.do_transform_pose_stamped = do_transform_pose_stamped
tf2_geometry_msgs.do_transform_vector3 = do_transform_vector3
sys.modules["tf2_geometry_msgs"] = tf2_geometry_msgs

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
    header_frame: str = "mr_pinchy/base_link",
    child_frame: str = "",
) -> TransformStamped:
    msg = TransformStamped()
    msg.header.frame_id = header_frame
    msg.child_frame_id = child_frame
    msg.transform.translation.x = translation[0]
    msg.transform.translation.y = translation[1]
    msg.transform.translation.z = translation[2]
    msg.transform.rotation.x = quaternion[0]
    msg.transform.rotation.y = quaternion[1]
    msg.transform.rotation.z = quaternion[2]
    msg.transform.rotation.w = quaternion[3]
    return msg


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
    frame_id: str = "map_ned",
    child_frame_id: str = "mr_pinchy/base_link",
) -> Odometry:
    msg = Odometry()
    msg.header.frame_id = frame_id
    msg.child_frame_id = child_frame_id
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


def test_bluerov2_model_terms_keep_only_body_force_terms() -> None:
    eta = np.array([1.0, -2.0, 0.5, 0.2, -0.1, 0.3], dtype=float)
    nu = np.array([0.4, -0.3, 0.2, 0.1, -0.05, 0.07], dtype=float)

    terms = bluerov2_model_terms(eta, nu)

    assert "R" not in terms
    assert "T" not in terms
    assert "J" not in terms
    assert np.allclose(np.asarray(terms["eta"], dtype=float), eta)
    assert np.allclose(np.asarray(terms["nu"], dtype=float), nu)
    assert np.allclose(np.asarray(terms["nu_rel"], dtype=float), nu)
    assert np.asarray(terms["M"], dtype=float).shape == (6, 6)
    assert np.asarray(terms["M_inv"], dtype=float).shape == (6, 6)
    assert np.asarray(terms["C"], dtype=float).shape == (6, 6)
    assert np.asarray(terms["D"], dtype=float).shape == (6, 6)
    assert np.asarray(terms["g"], dtype=float).shape == (6,)


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
        frame_id="map_ned",
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


def test_command_in_base_link_flu_rotates_flu_to_frd() -> None:
    node = BrovCtrlSIWrench()
    node._tf_buffer.lookup_result = _make_transform(quaternion=(1.0, 0.0, 0.0, 0.0))

    nu_ref = node._command_in_base_link(
        _make_twist_stamped(
            linear=(1.0, 2.0, 3.0),
            angular=(4.0, 5.0, 6.0),
            frame_id="mr_pinchy/base_link_flu",
        )
    )

    assert nu_ref is not None
    assert np.allclose(nu_ref, [1.0, -2.0, -3.0, 4.0, -5.0, -6.0])


def test_odom_in_base_link_transforms_base_link_flu_pose_and_twist() -> None:
    node = BrovCtrlSIWrench()
    node._tf_buffer.lookup_result = {
        ("mr_pinchy/base_link", "mr_pinchy/base_link_flu"): _make_transform(
            quaternion=(1.0, 0.0, 0.0, 0.0),
            header_frame="mr_pinchy/base_link",
            child_frame="mr_pinchy/base_link_flu",
        ),
        ("mr_pinchy/base_link_flu", "mr_pinchy/base_link"): _make_transform(
            quaternion=(1.0, 0.0, 0.0, 0.0),
            header_frame="mr_pinchy/base_link_flu",
            child_frame="mr_pinchy/base_link",
        ),
    }

    state = node._odom_in_base_link(
        _make_odom(
            pos=(1.0, 2.0, 3.0),
            linear=(4.0, 5.0, 6.0),
            angular=(0.4, 0.5, 0.6),
            frame_id="map",
            child_frame_id="mr_pinchy/base_link_flu",
        )
    )

    assert state is not None
    eta, nu = state
    assert np.allclose(eta[:3], [1.0, 2.0, 3.0], atol=1e-6)
    assert np.allclose(eta[3:], [0.0, 0.0, np.pi / 2.0], atol=1e-6)
    assert np.allclose(nu, [4.0, -5.0, -6.0, 0.4, -0.5, -0.6], atol=1e-6)


def test_odom_in_base_link_requires_child_frame_id() -> None:
    node = BrovCtrlSIWrench()

    try:
        node._odom_in_base_link(_make_odom(child_frame_id=""))
    except ValueError as exc:
        assert "child_frame_id" in str(exc)
    else:
        raise AssertionError("expected ValueError when odometry child_frame_id is missing")


def test_odom_in_base_link_converts_enu_frd_attitude_to_ned_angles() -> None:
    node = BrovCtrlSIWrench()

    state = node._odom_in_base_link(
        _make_odom(
            rpy=(np.pi, 0.0, 0.0),
            frame_id="map",
            child_frame_id="mr_pinchy/base_link",
        )
    )

    assert state is not None
    eta, _ = state
    assert np.allclose(eta[3:], [0.0, 0.0, np.pi / 2.0], atol=1e-6)
    assert static_wrench_feedforward(eta, np.zeros(6))[2] < 0.0


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
    node._cmd_cb(
        _make_twist_stamped(frame_id="mr_pinchy/base_link_flu", linear=(0.3, 0.0, 0.0))
    )
    node._odom_cb(_make_odom())

    node._tick()

    wrench_msg = node._wrench_pub.messages[-1]
    rc_msg = node._rc_pub.messages[-1]
    assert np.allclose(_wrench_vector(wrench_msg), np.zeros(6))
    assert rc_msg.channels[:6] == [1500] * 6


def test_timeout_on_stale_odom_publishes_zero_wrench_and_neutral_rc() -> None:
    node = BrovCtrlSIWrench()
    node._last_cmd = _make_twist_stamped(linear=(0.2, 0.0, 0.0))
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
