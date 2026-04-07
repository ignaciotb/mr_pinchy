from __future__ import annotations

import sys
import types
from pathlib import Path


class PoseStamped:
    pass


class Odometry:
    pass


class OverrideRCIn:
    CHAN_NOCHANGE = 65535

    def __init__(self) -> None:
        self.channels = []


class Node:
    def __init__(self, *args, **kwargs) -> None:
        pass


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

from mr_pinchy_interface.brov_ctrl_pid_rc import thrust_factors_to_rc_channels


def test_thrust_factors_to_rc_channels_respects_frd_axis_signs() -> None:
    channels = thrust_factors_to_rc_channels(
        surge_factor=0.5,
        sway_factor=0.25,
        heave_factor=0.75,
        yaw_factor=0.4,
        pitch_factor=0.2,
        roll_factor=0.3,
        neutral=1500,
        delta=100,
    )

    assert channels[0] == 1480
    assert channels[1] == 1530
    assert channels[2] == 1425
    assert channels[3] == 1540
    assert channels[4] == 1550
    assert channels[5] == 1525
    assert channels[6:] == [OverrideRCIn.CHAN_NOCHANGE] * 12
