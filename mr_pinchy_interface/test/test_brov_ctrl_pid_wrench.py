from __future__ import annotations

import math
import sys
import types
from pathlib import Path


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
    pass


class _DummyClock:
    def now(self):
        return types.SimpleNamespace(to_msg=lambda: "stamp")


class Node:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def create_publisher(self, *args, **kwargs):
        return types.SimpleNamespace(publish=lambda msg: None)

    def create_subscription(self, *args, **kwargs):
        return None

    def create_timer(self, *args, **kwargs):
        return None

    def get_clock(self):
        return _DummyClock()

    def get_logger(self):
        return types.SimpleNamespace(info=lambda *a, **k: None, warn=lambda *a, **k: None)


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

from mr_pinchy_interface.brov_ctrl_pid_wrench import (
    PIDAxis,
    _rpy_from_quat,
    _wrap_pi,
    build_wrench_message,
    neutral_rc_override,
    pid_outputs_to_wrench,
)
from mr_pinchy_interface.brov_ctrl_wrench import wrench_to_rc_override


def test_pid_outputs_to_wrench_applies_axis_scales() -> None:
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


def test_pid_wrench_composes_with_wrench_to_rc_override() -> None:
    wrench_msg = pid_outputs_to_wrench(
        surge_pid_output=0.0,
        sway_pid_output=0.0,
        heave_pid_output=0.0,
        yaw_pid_output=0.5,
        force_x_scale=1.0,
        force_y_scale=1.0,
        force_z_scale=1.0,
        torque_z_scale=2.0,
    )

    rc_msg = wrench_to_rc_override(wrench_msg)

    assert rc_msg.channels[3] == 1523
    assert rc_msg.channels[0] == 1500
    assert rc_msg.channels[1] == 1500
    assert rc_msg.channels[2] == 1500
    assert rc_msg.channels[4] == 1500
    assert rc_msg.channels[5] == 1500


def test_zero_wrench_message_has_expected_shape() -> None:
    msg = build_wrench_message(frame_id="mr_pinchy/base_link", stamp="stamp")

    assert msg.header.frame_id == "mr_pinchy/base_link"
    assert msg.header.stamp == "stamp"
    assert msg.wrench.force.x == 0.0
    assert msg.wrench.force.y == 0.0
    assert msg.wrench.force.z == 0.0
    assert msg.wrench.torque.x == 0.0
    assert msg.wrench.torque.y == 0.0
    assert msg.wrench.torque.z == 0.0


def test_neutral_rc_override_has_expected_shape() -> None:
    msg = neutral_rc_override(neutral_pwm=1500)

    assert msg.channels[:6] == [1500] * 6
    assert msg.channels[6:] == [OverrideRCIn.CHAN_NOCHANGE] * 12


def test_wrap_pi_keeps_yaw_error_on_shortest_path() -> None:
    assert _wrap_pi(math.pi + 0.1) < 0.0
    assert _wrap_pi(-math.pi - 0.1) > 0.0


def test_rpy_from_quat_returns_expected_roll_pitch_yaw() -> None:
    roll, pitch, yaw = _rpy_from_quat(0.0, 0.0, 0.0, 1.0)

    assert roll == 0.0
    assert pitch == 0.0
    assert yaw == 0.0


def test_yaw_pid_derivative_wraps_across_branch_cut() -> None:
    pid = PIDAxis(kp=0.0, ki=0.0, kd=1.0, integral_limit=1.0, output_limit=100.0)

    pid.step(math.pi - 0.01, 0.1, wrap_error=True)
    out = pid.step(-math.pi + 0.01, 0.1, wrap_error=True)

    assert abs(out) < 1.0


def test_angle_pid_wraps_proportional_error_to_shortest_path() -> None:
    pid = PIDAxis(kp=1.0, ki=0.0, kd=0.0, integral_limit=1.0, output_limit=10.0)

    out = pid.step(math.pi + 0.1, 0.1, wrap_error=True)

    assert out < 0.0
    assert abs(out - (-math.pi + 0.1)) < 1e-9
