#!/usr/bin/env python3
"""
ROS 2 node: ``TwistStamped`` → MAVROS ``OverrideRCIn`` (same channel layout as keyboard teleop).

RC channel mapping (ArduSub-style, 0-indexed, same as keyboard node):

  ch0 pitch   ← twist.angular.y
  ch1 roll    ← twist.angular.x
  ch2 heave   ← twist.linear.z   (ascend / descend)
  ch3 yaw     ← twist.angular.z
  ch4 surge   ← twist.linear.x   (forward / back)
  ch5 sway    ← twist.linear.y   (strafe)

If no twist is received for ``cmd_timeout_sec``, publishes neutral on channels 0–5 (failsafe).

Optional: subscribe to ``std_msgs/Bool`` on ``arming_bool_topic`` (non-empty param): ``data=True``
arms, ``data=False`` disarms (one-shot per message).

Run::

    ros2 run mr_pinchy_interface brov_twist_stamped_teleop --ros-args \\
        -p twist_topic:=/cmd_vel_stamped
"""

from __future__ import annotations

import threading
import time

import rclpy
from geometry_msgs.msg import Twist, TwistStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from mavros_msgs.msg import OverrideRCIn
from mavros_msgs.srv import CommandBool
from std_msgs.msg import Bool


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def twist_to_rc_channels(
    twist: Twist,
    *,
    neutral: int,
    delta: int,
    max_linear_x: float,
    max_linear_y: float,
    max_linear_z: float,
    max_angular_x: float,
    max_angular_y: float,
    max_angular_z: float,
) -> list[int]:
    """Map twist scalars to 18 RC uint16; channels 0–5 filled, rest CHAN_NOCHANGE."""

    def pwm_axis(val: float, vmax: float) -> int:
        if vmax <= 1e-9:
            return neutral
        s = _clamp(val / vmax, -1.0, 1.0)
        return int(round(neutral + s * delta))

    nc = OverrideRCIn.CHAN_NOCHANGE
    ch = [nc] * 18
    ch[0] = pwm_axis(twist.angular.y, max_angular_y)  # pitch
    ch[1] = pwm_axis(twist.angular.x, max_angular_x)  # roll
    ch[2] = pwm_axis(twist.linear.z, max_linear_z)  # vertical
    ch[3] = pwm_axis(twist.angular.z, max_angular_z)  # yaw
    ch[4] = pwm_axis(twist.linear.x, max_linear_x)  # forward
    ch[5] = pwm_axis(twist.linear.y, max_linear_y)  # lateral
    return ch


class BrovTwistStampedTeleop(Node):
    def __init__(self) -> None:
        super().__init__("brov_twist_stamped_teleop")

        self.declare_parameter("twist_topic", "/pinchy/cmd_vel")
        self.declare_parameter("arming_bool_topic", "/pinchy/arm")
        self.declare_parameter("odom_topic", "/mocap/mr_pinchy/odom")
        self.declare_parameter("rc_topic", "/mavros/rc/override")
        self.declare_parameter("arming_service", "/mavros/cmd/arming")
        self.declare_parameter("publish_rate_hz", 20.0)
        self.declare_parameter("neutral_pwm", 1500)
        self.declare_parameter("delta_pwm", 100)
        self.declare_parameter("cmd_timeout_sec", 0.1)

        self.declare_parameter("max_linear_x", 1.0)
        self.declare_parameter("max_linear_y", 1.0)
        self.declare_parameter("max_linear_z", 1.0)
        self.declare_parameter("max_angular_x", 1.0)
        self.declare_parameter("max_angular_y", 1.0)
        self.declare_parameter("max_angular_z", 1.0)

        twist_topic = self.get_parameter("twist_topic").get_parameter_value().string_value
        rc_topic = self.get_parameter("rc_topic").get_parameter_value().string_value
        arm_srv = self.get_parameter("arming_service").get_parameter_value().string_value
        arm_bool_topic = (
            self.get_parameter("arming_bool_topic").get_parameter_value().string_value.strip()
        )
        rate_hz = self.get_parameter("publish_rate_hz").get_parameter_value().double_value
        if rate_hz <= 0.0:
            rate_hz = 20.0

        self._neutral = int(
            self.get_parameter("neutral_pwm").get_parameter_value().integer_value
        )
        if self._neutral <= 0:
            self._neutral = 1500
        self._delta = int(
            self.get_parameter("delta_pwm").get_parameter_value().integer_value
        )
        if self._delta <= 0:
            self._delta = 100

        self._cmd_timeout = float(
            self.get_parameter("cmd_timeout_sec").get_parameter_value().double_value
        )
        if self._cmd_timeout <= 0.0:
            self._cmd_timeout = 0.1

        self._max_lin = (
            float(self.get_parameter("max_linear_x").get_parameter_value().double_value),
            float(self.get_parameter("max_linear_y").get_parameter_value().double_value),
            float(self.get_parameter("max_linear_z").get_parameter_value().double_value),
        )
        self._max_ang = (
            float(self.get_parameter("max_angular_x").get_parameter_value().double_value),
            float(self.get_parameter("max_angular_y").get_parameter_value().double_value),
            float(self.get_parameter("max_angular_z").get_parameter_value().double_value),
        )

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self._rc_pub = self.create_publisher(OverrideRCIn, rc_topic, qos)

        self._arming_client = self.create_client(CommandBool, arm_srv)
        self._arming_in_flight = False
        self._armed = False
        self._arming_service_name = arm_srv

        self._lock = threading.Lock()
        self._last_twist: Twist | None = None
        self._last_twist_stamp_sec: float = 0.0

        self.create_subscription(TwistStamped, twist_topic, self._twist_cb, 10)
        if arm_bool_topic:
            self.create_subscription(Bool, arm_bool_topic, self._arming_bool_cb, 10)
            self.get_logger().info(f"Arming Bool subscription: '{arm_bool_topic}'")

        period = 1.0 / rate_hz
        self.create_timer(period, self._tick)

        self.get_logger().info(
            f"brov_twist_stamped_teleop: TwistStamped '{twist_topic}' -> "
            f"OverrideRCIn '{rc_topic}' @ {rate_hz:.1f} Hz; timeout={self._cmd_timeout:.2f}s; "
            f"arming service '{arm_srv}'"
        )

    def _twist_cb(self, msg: TwistStamped) -> None:
        with self._lock:
            self._last_twist = msg.twist
            self._last_twist_stamp_sec = time.monotonic()

    def _arming_bool_cb(self, msg: Bool) -> None:
        self._call_arming(bool(msg.data))

    def _call_arming(self, arm: bool) -> None:
        if self._arming_in_flight:
            self.get_logger().warn("Arming request already in progress; ignored.")
            return
        if not self._arming_client.service_is_ready():
            self.get_logger().warn(
                f"Arming service '{self._arming_service_name}' not ready (is MAVROS running?)"
            )
            return
        self._arming_in_flight = True
        req = CommandBool.Request()
        req.value = arm
        fut = self._arming_client.call_async(req)
        fut.add_done_callback(lambda f: self._arming_done(f, arm))

    def _arming_done(self, future, requested_arm: bool) -> None:
        self._arming_in_flight = False
        try:
            resp = future.result()
            if resp.success:
                self._armed = requested_arm
                state = "ARMED" if requested_arm else "DISARMED"
                self.get_logger().info(f"cmd/arming success -> {state}")
            else:
                self.get_logger().error("cmd/arming returned success=False")
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"cmd/arming failed: {exc}")

    def _tick(self) -> None:
        now = time.monotonic()
        with self._lock:
            twist = self._last_twist
            stamp = self._last_twist_stamp_sec

        stale = twist is None or (now - stamp) > self._cmd_timeout
        t = Twist() if stale else twist

        if stale and twist is not None:
            self.get_logger().warn(
                f"Twist command timeout ({self._cmd_timeout:.2f}s) — publishing neutral RC",
                throttle_duration_sec=2.0,
            )

        channels = twist_to_rc_channels(
            t,
            neutral=self._neutral,
            delta=self._delta,
            max_linear_x=self._max_lin[0],
            max_linear_y=self._max_lin[1],
            max_linear_z=self._max_lin[2],
            max_angular_x=self._max_ang[0],
            max_angular_y=self._max_ang[1],
            max_angular_z=self._max_ang[2],
        )

        msg = OverrideRCIn()
        msg.channels = channels
        self._rc_pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BrovTwistStampedTeleop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
