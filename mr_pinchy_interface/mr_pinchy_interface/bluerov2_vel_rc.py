#!/usr/bin/env python3
"""
ROS 2 node: translate ``geometry_msgs/msg/TwistStamped`` velocity commands from a
standard ``teleop_twist_joy`` node into ``mavros_msgs/msg/OverrideRCIn`` RC
override messages, using the same channel layout, PWM span and neutral
behaviour as ``bluerov2_keyboard_teleop``.

- Subscribes ``geometry_msgs/msg/TwistStamped`` on ``/mr_pinchy/cmd_vel``
  (param ``cmd_topic``) — this is what ``teleop_twist_joy`` publishes when
  ``publish_stamped_twist`` is true.
- Publishes ``mavros_msgs/msg/OverrideRCIn`` on ``/mavros/rc/override``
  (param ``rc_topic``) at a fixed rate (param ``publish_rate_hz``).
- Publishes the human-readable labels of the motions currently driving the RC
  override as a comma-separated ``std_msgs/msg/String`` on
  ``/bluerov2_vel_rc/axis_labels`` (param ``axis_labels_topic``), one
  message per RC override message. Empty string means "all axes neutral".

Where the keyboard node moves a channel a full ``delta_pwm`` off neutral for as
long as a key is held, this node moves it proportionally: a commanded velocity
of ``max_*`` (the ``scale_*`` values given to ``teleop_twist_joy``) maps to a
full ``delta_pwm`` deflection, and anything in between is linear. Commands are
clamped, so an oversized ``scale_*`` can never push a channel past the same
``neutral ± delta_pwm`` span the keyboard teleop uses.

Frames: ``teleop_twist_joy`` publishes in a REP-103 FLU body frame (+x forward,
+y left, +z up, +yaw CCW), while the RC channels are ArduSub body FRD. The
conversion is done here (param ``input_is_flu``, default true); set it false if
the incoming twist is already FRD.

If no command arrives within ``cmd_timeout_sec``, every channel is driven back
to neutral — the joystick equivalent of releasing all keys.
"""

from __future__ import annotations

import rclpy
from geometry_msgs.msg import TwistStamped
from mavros_msgs.msg import OverrideRCIn
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

# Same motion labels as bluerov2_keyboard_teleop, keyed by the sign of each axis.
_AXIS_LABELS: dict[str, tuple[str, str]] = {
    # axis: (label when negative, label when positive)
    "pitch": ("pitch_down", "pitch_up"),
    "roll": ("roll_left", "roll_right"),
    "heave": ("down", "up"),
    "yaw": ("yaw_left", "yaw_right"),
    "surge": ("back", "fwd"),
    "sway": ("left", "right"),
}

# RC channel index (0-based) per axis — the ArduSub layout the keyboard node uses:
# 1 pitch, 2 roll, 3 throttle, 4 yaw, 5 forward, 6 lateral.
_AXIS_CHANNEL: dict[str, int] = {
    "pitch": 0,
    "roll": 1,
    "heave": 2,
    "yaw": 3,
    "surge": 4,
    "sway": 5,
}

# Order used when building the label string, matching the keyboard node's order.
_LABEL_ORDER = ("heave", "yaw", "surge", "sway", "roll", "pitch")


def _clamp_unit(value: float) -> float:
    return max(-1.0, min(1.0, value))


class Bluerov2VelRc(Node):
    def __init__(self) -> None:
        super().__init__("bluerov2_vel_rc")

        self.declare_parameter("cmd_topic", "/mr_pinchy/cmd_vel")
        self.declare_parameter("rc_topic", "/mavros/rc/override")
        self.declare_parameter("axis_labels_topic", "/bluerov2_vel_rc/axis_labels")
        self.declare_parameter("publish_rate_hz", 20.0)
        self.declare_parameter("neutral_pwm", 1500)
        self.declare_parameter("delta_pwm", 100)
        self.declare_parameter("cmd_timeout_sec", 0.4)
        self.declare_parameter("input_is_flu", True)
        # Velocity that maps to a full delta_pwm deflection. Defaults match the
        # scale_* parameters given to teleop_twist_joy in start_ctrl.launch.py.
        self.declare_parameter("max_linear_x", 0.2)
        self.declare_parameter("max_linear_y", 0.2)
        self.declare_parameter("max_linear_z", 0.2)
        self.declare_parameter("max_angular_x", 0.3)
        self.declare_parameter("max_angular_y", 0.3)
        self.declare_parameter("max_angular_z", 0.3)
        # Commands below this fraction of the corresponding max_* are treated as
        # zero, so joystick noise does not keep the channels off neutral.
        self.declare_parameter("deadband", 0.02)

        cmd_topic = str(self.get_parameter("cmd_topic").value)
        rc_topic = str(self.get_parameter("rc_topic").value)
        labels_topic = str(self.get_parameter("axis_labels_topic").value)

        rate_hz = float(self.get_parameter("publish_rate_hz").value)
        if rate_hz <= 0.0:
            rate_hz = 20.0

        self._neutral = int(self.get_parameter("neutral_pwm").value)
        if self._neutral <= 0:
            self._neutral = 1500
        self._delta = int(self.get_parameter("delta_pwm").value)
        if self._delta <= 0:
            self._delta = 100

        self._cmd_timeout = float(self.get_parameter("cmd_timeout_sec").value)
        if self._cmd_timeout <= 0.0:
            self._cmd_timeout = 0.4

        self._input_is_flu = bool(self.get_parameter("input_is_flu").value)
        self._deadband = abs(float(self.get_parameter("deadband").value))

        self._max = {
            "linear_x": self._positive("max_linear_x", 0.2),
            "linear_y": self._positive("max_linear_y", 0.2),
            "linear_z": self._positive("max_linear_z", 0.2),
            "angular_x": self._positive("max_angular_x", 0.3),
            "angular_y": self._positive("max_angular_y", 0.3),
            "angular_z": self._positive("max_angular_z", 0.3),
        }

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self._rc_pub = self.create_publisher(OverrideRCIn, rc_topic, qos)
        self._labels_pub = self.create_publisher(String, labels_topic, qos)
        self.create_subscription(TwistStamped, cmd_topic, self._cmd_cb, 10)

        self._last_cmd: TwistStamped | None = None
        self._last_cmd_rx_sec: float = 0.0
        self._timed_out = True

        self.create_timer(1.0 / rate_hz, self._tick)
        self.get_logger().info(
            f"bluerov2_vel_rc: cmd='{cmd_topic}' -> OverrideRCIn on '{rc_topic}' "
            f"at {rate_hz:.1f} Hz; labels on '{labels_topic}'; "
            f"pwm {self._neutral}±{self._delta}; "
            f"cmd_timeout={self._cmd_timeout:.2f}s; "
            f"input frame = {'FLU' if self._input_is_flu else 'FRD'}"
        )

    def _positive(self, param: str, fallback: float) -> float:
        value = float(self.get_parameter(param).value)
        if value <= 0.0:
            self.get_logger().warn(f"'{param}' must be > 0, using {fallback}")
            return fallback
        return value

    def _cmd_cb(self, msg: TwistStamped) -> None:
        self._last_cmd = msg
        self._last_cmd_rx_sec = self.get_clock().now().nanoseconds * 1e-9

    def _normalized_axes(self, msg: TwistStamped) -> dict[str, float]:
        """Twist -> per-axis command in [-1, 1], in the RC (FRD) sign convention."""
        lin = msg.twist.linear
        ang = msg.twist.angular

        surge = float(lin.x) / self._max["linear_x"]
        sway = float(lin.y) / self._max["linear_y"]
        heave = float(lin.z) / self._max["linear_z"]
        roll = float(ang.x) / self._max["angular_x"]
        pitch = float(ang.y) / self._max["angular_y"]
        yaw = float(ang.z) / self._max["angular_z"]

        if self._input_is_flu:
            # FLU -> FRD: +y is left, +z is up, +yaw/+pitch follow the FLU axes.
            #   sway:  +y left  -> RC lateral is +right
            #   heave: +z up    -> RC throttle is +up, so the sign is kept
            #   yaw:   +z CCW   -> RC yaw is +CW (right)
            #   pitch: +y rot   -> nose down in FLU, RC pitch is +up
            #   roll:  +x rot   -> roll right in FLU, same as RC
            sway = -sway
            yaw = -yaw
            pitch = -pitch
        else:
            # FRD -> RC: +z is down while RC throttle is +up.
            heave = -heave

        axes = {
            "surge": surge,
            "sway": sway,
            "heave": heave,
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
        }
        return {
            name: 0.0 if abs(value) < self._deadband else _clamp_unit(value)
            for name, value in axes.items()
        }

    def _publish(self, axes: dict[str, float]) -> None:
        nc = OverrideRCIn.CHAN_NOCHANGE
        channels = [nc] * 18
        for idx in range(6):
            channels[idx] = self._neutral

        for name, value in axes.items():
            channels[_AXIS_CHANNEL[name]] = self._neutral + int(round(value * self._delta))

        msg = OverrideRCIn()
        msg.channels = channels
        self._rc_pub.publish(msg)

        # One label message per RC override message, so a subscriber can line the
        # two streams up. Empty data = every axis at neutral.
        active = [
            _AXIS_LABELS[name][1 if axes[name] > 0.0 else 0]
            for name in _LABEL_ORDER
            if axes[name] != 0.0
        ]
        labels = String()
        labels.data = ",".join(active)
        self._labels_pub.publish(labels)

    def _publish_neutral(self) -> None:
        self._publish({name: 0.0 for name in _AXIS_CHANNEL})

    def _tick(self) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9

        if self._last_cmd is None or (now - self._last_cmd_rx_sec) > self._cmd_timeout:
            if not self._timed_out and self._last_cmd is not None:
                self.get_logger().warn(
                    f"Command timeout ({self._cmd_timeout:.2f}s); holding all channels neutral.",
                    throttle_duration_sec=2.0,
                )
            self._timed_out = True
            self._publish_neutral()
            return

        self._timed_out = False
        self._publish(self._normalized_axes(self._last_cmd))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = Bluerov2VelRc()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
