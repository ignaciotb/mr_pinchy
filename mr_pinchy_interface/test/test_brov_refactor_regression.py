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


class TransformStamped:
    def __init__(self) -> None:
        self.header = _Header()
        self.child_frame_id = ""
        self.transform = types.SimpleNamespace(
            translation=_Vector3(),
            rotation=_Quaternion(),
        )


class TwistStamped:
    def __init__(self) -> None:
        self.header = _Header()
        self.twist = _Twist()


class Vector3Stamped:
    def __init__(self) -> None:
        self.header = _Header()
        self.vector = _Vector3()


class WrenchStamped:
    def __init__(self) -> None:
        self.header = _Header()
        self.wrench = types.SimpleNamespace(force=_Vector3(), torque=_Vector3())


class OverrideRCIn:
    CHAN_NOCHANGE = 65535

    def __init__(self) -> None:
        self.channels = []


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


class _ParameterValue:
    def __init__(self, value) -> None:
        self.string_value = value if isinstance(value, str) else ""
        self.integer_value = int(value) if isinstance(value, int) and not isinstance(value, bool) else 0
        self.double_value = float(value) if isinstance(value, (float, int)) and not isinstance(value, bool) else 0.0


class _Parameter:
    def __init__(self, value) -> None:
        self.value = value

    def get_parameter_value(self) -> _ParameterValue:
        return _ParameterValue(self.value)


class _Logger:
    def info(self, *args, **kwargs) -> None:
        return None

    def warn(self, *args, **kwargs) -> None:
        return None

    def error(self, *args, **kwargs) -> None:
        return None


class _DummyClock:
    def now(self):
        return types.SimpleNamespace(to_msg=lambda: "stamp")


class Node:
    def __init__(self, *args, **kwargs) -> None:
        self._params = {}
        self._logger = _Logger()

    def declare_parameter(self, name, default_value=None):
        self._params.setdefault(name, default_value)
        return _Parameter(self._params[name])

    def get_parameter(self, name):
        return _Parameter(self._params[name])

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
        raise RuntimeError("unexpected TF lookup in regression test")


class TransformListener:
    def __init__(self, *args, **kwargs) -> None:
        pass


class StaticTransformBroadcaster:
    sent_transforms = []

    def __init__(self, *args, **kwargs) -> None:
        pass

    def sendTransform(self, transform) -> None:
        self.__class__.sent_transforms.append(transform)


class TransformBroadcaster:
    sent_transforms = []

    def __init__(self, *args, **kwargs) -> None:
        pass

    def sendTransform(self, transform) -> None:
        self.__class__.sent_transforms.append(transform)


class TransformException(Exception):
    pass


class Time:
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

tf2_geometry_msgs = types.ModuleType("tf2_geometry_msgs")
tf2_geometry_msgs.do_transform_pose_stamped = lambda pose, transform: pose
tf2_geometry_msgs.do_transform_vector3 = lambda vector, transform: vector
sys.modules["tf2_geometry_msgs"] = tf2_geometry_msgs

tf2_ros = types.ModuleType("tf2_ros")
tf2_ros.Buffer = Buffer
tf2_ros.StaticTransformBroadcaster = StaticTransformBroadcaster
tf2_ros.TransformBroadcaster = TransformBroadcaster
tf2_ros.TransformException = TransformException
tf2_ros.TransformListener = TransformListener
sys.modules["tf2_ros"] = tf2_ros

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mr_pinchy_interface import brov_utils  # noqa: E402
from mr_pinchy_interface.brov_ctrl_twist import (  # noqa: E402
    BrovCtrlSIWrench,
    BrovTwistController,
    body_wrench_message,
    odom_to_body_state,
    twist_to_body_velocity,
)
from mr_pinchy_interface.brov_ctrl_waypoint import (  # noqa: E402
    PIDAxis,
    _rpy_from_quat,
    _wrap_pi,
    build_wrench_message,
    neutral_rc_override,
    pid_outputs_to_wrench,
)
from mr_pinchy_interface.brov_kf_odom import (  # noqa: E402
    BrovKfOdom,
    FilterTuning,
    QuaternionCvEkf,
    base_link_z_axis_angle_rad,
    orientation_residual,
    quat_exp,
    quat_multiply,
    quat_normalize,
)


def _make_twist_stamped(
    *,
    linear=(0.0, 0.0, 0.0),
    angular=(0.0, 0.0, 0.0),
) -> TwistStamped:
    msg = TwistStamped()
    msg.twist.linear.x = linear[0]
    msg.twist.linear.y = linear[1]
    msg.twist.linear.z = linear[2]
    msg.twist.angular.x = angular[0]
    msg.twist.angular.y = angular[1]
    msg.twist.angular.z = angular[2]
    return msg


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


def _make_pose_stamped(
    *,
    pos=(0.0, 0.0, 0.0),
    quat=(0.0, 0.0, 0.0, 1.0),
    frame_id="mocap",
) -> PoseStamped:
    msg = PoseStamped()
    msg.header.frame_id = frame_id
    msg.pose.position.x = pos[0]
    msg.pose.position.y = pos[1]
    msg.pose.position.z = pos[2]
    msg.pose.orientation.x = quat[0]
    msg.pose.orientation.y = quat[1]
    msg.pose.orientation.z = quat[2]
    msg.pose.orientation.w = quat[3]
    return msg


def _make_odom(
    *,
    pos=(0.0, 0.0, 0.0),
    rpy=(0.0, 0.0, 0.0),
    linear=(0.0, 0.0, 0.0),
    angular=(0.0, 0.0, 0.0),
) -> Odometry:
    quat = _quat_from_rpy(*rpy)

    msg = Odometry()
    msg.pose.pose.position.x = pos[0]
    msg.pose.pose.position.y = pos[1]
    msg.pose.pose.position.z = pos[2]
    msg.pose.pose.orientation.x = quat[0]
    msg.pose.pose.orientation.y = quat[1]
    msg.pose.pose.orientation.z = quat[2]
    msg.pose.pose.orientation.w = quat[3]
    msg.twist.twist.linear.x = linear[0]
    msg.twist.twist.linear.y = linear[1]
    msg.twist.twist.linear.z = linear[2]
    msg.twist.twist.angular.x = angular[0]
    msg.twist.twist.angular.y = angular[1]
    msg.twist.twist.angular.z = angular[2]
    return msg


def test_waypoint_module_reexports_shared_helpers() -> None:
    assert PIDAxis is brov_utils.PIDAxis
    assert _wrap_pi is brov_utils.wrap_pi
    assert _rpy_from_quat is brov_utils.quat_to_rpy
    assert build_wrench_message is brov_utils.build_wrench_message
    assert neutral_rc_override is brov_utils.neutral_rc_override
    assert pid_outputs_to_wrench is brov_utils.pid_outputs_to_wrench


def test_waypoint_helper_behavior_is_preserved() -> None:
    msg = pid_outputs_to_wrench(
        surge_pid_output=0.5,
        sway_pid_output=-0.25,
        heave_pid_output=1.0,
        roll_pid_output=0.75,
        pitch_pid_output=-0.5,
        yaw_pid_output=-0.4,
        force_x_scale=10.0,
        force_y_scale=20.0,
        force_z_scale=30.0,
        torque_x_scale=4.0,
        torque_y_scale=6.0,
        torque_z_scale=40.0,
        frame_id="mr_pinchy/base_link",
        stamp="stamp",
    )

    assert msg.header.frame_id == "mr_pinchy/base_link"
    assert msg.header.stamp == "stamp"
    assert msg.wrench.force.x == 5.0
    assert msg.wrench.force.y == -5.0
    assert msg.wrench.force.z == 30.0
    assert msg.wrench.torque.x == 3.0
    assert msg.wrench.torque.y == -3.0
    assert msg.wrench.torque.z == -16.0
    assert _wrap_pi(math.pi + 0.1) < 0.0

    pid = PIDAxis(kp=1.0, ki=0.0, kd=0.0, integral_limit=1.0, output_limit=10.0)
    assert pid.step(math.pi + 0.1, 0.1, wrap_error=True) < 0.0


def test_twist_module_reexports_shared_helpers_and_alias() -> None:
    assert BrovCtrlSIWrench is BrovTwistController
    assert twist_to_body_velocity is brov_utils.twist_to_body_velocity
    assert odom_to_body_state is brov_utils.odom_to_body_state
    assert body_wrench_message is brov_utils.body_wrench_message


def test_twist_conversion_helpers_and_zero_tick_behavior() -> None:
    StaticTransformBroadcaster.sent_transforms.clear()

    twist = _make_twist_stamped(linear=(1.0, 2.0, 3.0), angular=(4.0, 5.0, 6.0))
    odom = _make_odom(
        pos=(1.0, 2.0, 3.0),
        rpy=(0.1, -0.2, 0.3),
        linear=(4.0, 5.0, 6.0),
        angular=(0.4, 0.5, 0.6),
    )

    assert np.allclose(twist_to_body_velocity(twist), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    eta, nu = odom_to_body_state(odom)
    assert np.allclose(eta, [1.0, 2.0, 3.0, 0.1, -0.2, 0.3], atol=1e-6)
    assert np.allclose(nu, [4.0, 5.0, 6.0, 0.4, 0.5, 0.6], atol=1e-6)

    node = BrovTwistController()
    assert len(StaticTransformBroadcaster.sent_transforms) == 1
    transform = StaticTransformBroadcaster.sent_transforms[0]
    assert transform.header.frame_id == "mr_pinchy/base_link"
    assert transform.child_frame_id == "mr_pinchy/base_link_flu"
    assert transform.transform.translation.x == 0.0
    assert transform.transform.translation.y == 0.0
    assert transform.transform.translation.z == 0.0
    assert transform.transform.rotation.x == 1.0
    assert transform.transform.rotation.y == 0.0
    assert transform.transform.rotation.z == 0.0
    assert transform.transform.rotation.w == 0.0

    node._tick()

    wrench_msg = node._wrench_pub.messages[-1]
    rc_msg = node._rc_pub.messages[-1]
    assert np.allclose(
        [
            wrench_msg.wrench.force.x,
            wrench_msg.wrench.force.y,
            wrench_msg.wrench.force.z,
            wrench_msg.wrench.torque.x,
            wrench_msg.wrench.torque.y,
            wrench_msg.wrench.torque.z,
        ],
        np.zeros(6),
    )
    assert rc_msg.channels[:6] == [1500] * 6


def test_kf_quaternion_helpers_use_shortest_orientation_error() -> None:
    yaw_90 = quat_exp(np.array([0.0, 0.0, math.pi / 2.0]))
    combined = quat_multiply(yaw_90, yaw_90)
    yaw_180 = quat_exp(np.array([0.0, 0.0, math.pi]))

    assert np.allclose(quat_normalize(combined), yaw_180, atol=1e-6)
    assert np.allclose(orientation_residual(yaw_90, -yaw_90), np.zeros(3), atol=1e-6)

    residual = orientation_residual(
        quat_exp(np.array([0.0, 0.0, -math.pi + 0.1])),
        quat_exp(np.array([0.0, 0.0, math.pi - 0.1])),
    )
    assert np.allclose(residual, [0.0, 0.0, 0.2], atol=1e-6)


def test_kf_base_link_z_axis_angle_ignores_yaw_and_tracks_tilt() -> None:
    assert base_link_z_axis_angle_rad(np.array([0.0, 0.0, 0.0, 1.0])) == 0.0
    assert math.isclose(
        base_link_z_axis_angle_rad(_quat_from_rpy(0.8, 0.0, 0.0)),
        0.8,
        abs_tol=1e-6,
    )
    assert math.isclose(
        base_link_z_axis_angle_rad(_quat_from_rpy(0.0, -0.8, 0.0)),
        0.8,
        abs_tol=1e-6,
    )
    assert base_link_z_axis_angle_rad(_quat_from_rpy(0.0, 0.0, 1.2)) == 0.0


def test_quaternion_cv_ekf_estimates_constant_velocity_motion() -> None:
    ekf = QuaternionCvEkf(
        FilterTuning(
            position_std=0.01,
            orientation_std=0.01,
            linear_accel_std=0.2,
            angular_accel_std=0.2,
        )
    )
    linear_velocity = np.array([0.4, -0.2, 0.1], dtype=float)
    angular_velocity = np.array([0.0, 0.0, 0.3], dtype=float)
    dt = 0.05

    ekf.initialize(np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]))
    for step in range(1, 101):
        t = step * dt
        ekf.predict(dt)
        ekf.update(linear_velocity * t, quat_exp(angular_velocity * t))

    assert np.allclose(ekf.position, linear_velocity * 5.0, atol=0.02)
    assert np.allclose(ekf.linear_velocity, linear_velocity, atol=0.03)
    assert np.allclose(ekf.angular_velocity, angular_velocity, atol=0.03)
    orientation_error = orientation_residual(quat_exp(angular_velocity * 5.0), ekf.orientation)
    assert np.linalg.norm(orientation_error) < 0.02


def test_quaternion_cv_ekf_rejects_full_pose_on_absurd_orientation_outlier() -> None:
    ekf = QuaternionCvEkf()
    ekf.initialize(np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]))

    result = ekf.update(
        np.array([0.02, 0.0, 0.0]),
        quat_exp(np.array([0.0, 0.0, math.pi])),
        max_position_innovation_m=0.25,
        max_orientation_innovation_rad=0.75,
    )

    assert result.position_accepted
    assert not result.orientation_accepted
    assert not result.accepted
    assert result.orientation_innovation_rad > 0.75
    orientation_error = orientation_residual(
        np.array([0.0, 0.0, 0.0, 1.0]),
        ekf.orientation,
    )
    assert np.linalg.norm(orientation_error) < 1e-6
    assert np.allclose(ekf.position, np.zeros(3))


def test_quaternion_cv_ekf_rejects_absolute_z_axis_outlier() -> None:
    ekf = QuaternionCvEkf()
    tilted_orientation = _quat_from_rpy(0.8, 0.0, 0.0)
    ekf.initialize(np.zeros(3), tilted_orientation)

    result = ekf.update(
        np.array([0.02, 0.0, 0.0]),
        tilted_orientation,
        max_position_innovation_m=0.25,
        max_orientation_innovation_rad=0.75,
        max_base_link_z_axis_angle_rad=0.75,
    )

    assert result.position_accepted
    assert result.orientation_accepted
    assert not result.z_axis_accepted
    assert not result.accepted
    assert result.z_axis_angle_rad > 0.75
    assert np.allclose(ekf.position, np.zeros(3))
    assert np.allclose(ekf.orientation, tilted_orientation)


def test_kf_odom_node_publishes_expected_frames_and_tf() -> None:
    TransformBroadcaster.sent_transforms.clear()
    node = BrovKfOdom()
    quat = _quat_from_rpy(0.0, 0.0, math.pi / 2.0)

    node._pose_cb(_make_pose_stamped(pos=(1.0, 2.0, 3.0), quat=quat))
    node._tick()

    odom = node._odom_pub.messages[-1]
    assert odom.header.frame_id == "mocap"
    assert odom.child_frame_id == "mr_pinchy/base_link_kf"
    assert odom.pose.pose.position.x == 1.0
    assert odom.pose.pose.position.y == 2.0
    assert odom.pose.pose.position.z == 3.0
    assert np.allclose(
        [
            odom.pose.pose.orientation.x,
            odom.pose.pose.orientation.y,
            odom.pose.pose.orientation.z,
            odom.pose.pose.orientation.w,
        ],
        quat,
    )

    transform = TransformBroadcaster.sent_transforms[-1]
    assert transform.header.frame_id == "mocap"
    assert transform.child_frame_id == "mr_pinchy/base_link_kf"
    assert transform.transform.translation.x == 1.0
    assert transform.transform.translation.y == 2.0
    assert transform.transform.translation.z == 3.0


def test_kf_odom_node_does_not_initialize_from_absolute_z_axis_outlier() -> None:
    node = BrovKfOdom()
    bad_quat = _quat_from_rpy(0.8, 0.0, 0.0)

    node._pose_cb(_make_pose_stamped(pos=(1.0, 2.0, 3.0), quat=bad_quat))

    assert not node._filter.initialized
    assert node._consecutive_full_rejections == 1


def test_kf_odom_node_does_not_reinitialize_from_absolute_z_axis_outliers() -> None:
    node = BrovKfOdom()
    node._max_orientation_innovation_rad = 1.0
    node._max_rejected_samples = 2
    good_quat = _quat_from_rpy(0.0, 0.0, 0.0)
    bad_quat = _quat_from_rpy(0.8, 0.0, 0.0)
    initial_position = np.array([1.0, 2.0, 3.0])

    node._pose_cb(_make_pose_stamped(pos=initial_position, quat=good_quat))
    node._pose_cb(_make_pose_stamped(pos=initial_position + np.array([0.01, 0.0, 0.0]), quat=bad_quat))
    node._pose_cb(_make_pose_stamped(pos=initial_position + np.array([0.02, 0.0, 0.0]), quat=bad_quat))

    assert node._filter.initialized
    assert np.allclose(node._filter.orientation, good_quat)
    assert not np.allclose(node._filter.orientation, bad_quat)
    assert node._consecutive_full_rejections == 2


def test_kf_odom_node_reinitializes_after_coast_timeout() -> None:
    node = BrovKfOdom()
    node._pose_cb(_make_pose_stamped(pos=(1.0, 2.0, 3.0)))

    assert node._filter.initialized
    node._last_pose_rx_sec = time.monotonic() - node._max_coast_sec - 0.1
    node._tick()

    assert not node._filter.initialized
    assert node._last_filter_sec is None
    assert node._last_pose_rx_sec is None


def test_setup_entry_points_reference_current_modules() -> None:
    setup_text = Path(__file__).resolve().parents[1].joinpath("setup.py").read_text()
    assert "brov_ctrl_waypoint = mr_pinchy_interface.brov_ctrl_waypoint:main" in setup_text
    assert "brov_ctrl_twist = mr_pinchy_interface.brov_ctrl_twist:main" in setup_text
    assert "brov_kf_odom = mr_pinchy_interface.brov_kf_odom:main" in setup_text
