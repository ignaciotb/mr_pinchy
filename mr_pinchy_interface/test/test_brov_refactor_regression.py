from __future__ import annotations

import math
import sys
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


def _make_odom(
    *,
    pos=(0.0, 0.0, 0.0),
    rpy=(0.0, 0.0, 0.0),
    linear=(0.0, 0.0, 0.0),
    angular=(0.0, 0.0, 0.0),
) -> Odometry:
    roll, pitch, yaw = rpy
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    msg = Odometry()
    msg.pose.pose.position.x = pos[0]
    msg.pose.pose.position.y = pos[1]
    msg.pose.pose.position.z = pos[2]
    msg.pose.pose.orientation.x = sr * cp * cy - cr * sp * sy
    msg.pose.pose.orientation.y = cr * sp * cy + sr * cp * sy
    msg.pose.pose.orientation.z = cr * cp * sy - sr * sp * cy
    msg.pose.pose.orientation.w = cr * cp * cy + sr * sp * sy
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
    assert transform.header.frame_id == "/mr_pinchy/base_link"
    assert transform.child_frame_id == "/mr_pinchy/base_link_flu"
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


def test_setup_entry_points_reference_current_modules() -> None:
    setup_text = Path(__file__).resolve().parents[1].joinpath("setup.py").read_text()
    assert "brov_ctrl_waypoint = mr_pinchy_interface.brov_ctrl_waypoint:main" in setup_text
    assert "brov_ctrl_twist = mr_pinchy_interface.brov_ctrl_twist:main" in setup_text
