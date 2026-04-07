#!/usr/bin/env python3
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import rclpy
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import OverrideRCIn
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from tf2_geometry_msgs import do_transform_pose_stamped
from tf2_ros import Buffer, TransformException, TransformListener


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _wrap_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _unwrap_angle(curr_wrapped: float, prev_unwrapped: float | None) -> float:
    """Keep angular signal continuous across the -pi/pi branch cut."""
    if prev_unwrapped is None:
        return curr_wrapped
    return prev_unwrapped + _wrap_pi(curr_wrapped - prev_unwrapped)


@dataclass
class PIDAxis:
    kp: float
    ki: float
    kd: float
    integral_limit: float
    output_limit: float
    integral: float = 0.0
    prev_error: float = 0.0
    has_prev: bool = False

    def reset(self) -> None:
        self.integral = 0.0
        self.prev_error = 0.0
        self.has_prev = False

    def step(self, error: float, dt: float) -> float:
        if dt <= 1e-6:
            deriv = 0.0
        else:
            self.integral += error * dt
            self.integral = _clamp(self.integral, -self.integral_limit, self.integral_limit)
            if self.has_prev:
                deriv = (error - self.prev_error) / dt
            else:
                deriv = 0.0
        out = self.kp * error + self.ki * self.integral + self.kd * deriv
        out = _clamp(out, -self.output_limit, self.output_limit)
        self.prev_error = error
        self.has_prev = True
        return out


def thrust_factors_to_rc_channels(
    *,
    surge_factor: float,
    sway_factor: float,
    heave_factor: float,
    yaw_factor: float,
    neutral: int,
    delta: int,
) -> list[int]:
    """Map normalized thrust factors to 18 RC uint16 channels."""

    def pwm_axis(factor: float) -> int:
        return int(round(neutral + factor * delta))

    nc = OverrideRCIn.CHAN_NOCHANGE
    ch = [nc] * 18
    ch[0] = neutral  # pitch unused
    ch[1] = neutral  # roll unused
    ch[2] = pwm_axis(heave_factor)  # vertical
    ch[3] = pwm_axis(-yaw_factor)  # yaw (match teleop: yaw_left -> negative PWM)
    ch[4] = pwm_axis(surge_factor)  # forward
    ch[5] = pwm_axis(-sway_factor)  # lateral (match teleop: left -> negative PWM)
    return ch


class BrovCtrlPID(Node):
    def __init__(self) -> None:
        super().__init__("brov_ctrl_pid")

        self.declare_parameter("goal_topic", "/goal_pose")
        self.declare_parameter("odom_topic", "/mocap/mr_pinchy/odom")
        self.declare_parameter("rc_topic", "/mavros/rc/override")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_link_frame", "mr_pinchy/base_link")

        goal_topic = self.get_parameter("goal_topic").get_parameter_value().string_value
        odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value
        rc_topic = self.get_parameter("rc_topic").get_parameter_value().string_value
        self._map_frame = self.get_parameter("map_frame").get_parameter_value().string_value
        self._base_link_frame = (
            self.get_parameter("base_link_frame").get_parameter_value().string_value
        )

        self._debug_log = True
        self._publish_rate_hz = 20.0
        self._period = 1.0 / self._publish_rate_hz
        self._odom_timeout = 0.3
        self._neutral = 1500
        self._delta = 100

        self._pid_x = PIDAxis(kp=1.0, ki=0.0, kd=0.0, integral_limit=1.0, output_limit=1.0)
        self._pid_y = PIDAxis(kp=1.0, ki=0.0, kd=0.0, integral_limit=1.0, output_limit=1.0)
        self._pid_z = PIDAxis(kp=1.0, ki=0.0, kd=0.0, integral_limit=1.0, output_limit=1.0)
        self._pid_yaw = PIDAxis(
            kp=1.0,
            ki=0.0,
            kd=0.0,
            integral_limit=1.0,
            output_limit=1.0,
        )

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self._rc_pub = self.create_publisher(OverrideRCIn, rc_topic, qos)
        self.create_subscription(PoseStamped, goal_topic, self._goal_cb, 10)
        self.create_subscription(Odometry, odom_topic, self._odom_cb, 10)

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._goal_pose: PoseStamped | None = None
        self._last_odom_rx_sec: float = 0.0
        self._last_tick_sec: float = time.monotonic()
        self._yaw_err_unwrapped: float | None = None

        self.create_timer(self._period, self._tick)
        self.get_logger().info(
            f"brov_ctrl_pid: goal='{goal_topic}', odom='{odom_topic}', rc='{rc_topic}', "
            f"map_frame='{self._map_frame}', base_link_frame='{self._base_link_frame}', "
            f"rate={self._publish_rate_hz:.1f}Hz, odom_timeout={self._odom_timeout:.2f}s"
        )

    def _goal_cb(self, msg: PoseStamped) -> None:
        self._goal_pose = msg
        self._goal_pose.pose.position.z = -2.0

    def _odom_cb(self, _msg: Odometry) -> None:
        # Odom is used as a freshness watchdog so control is neutralized on stream loss.
        self._last_odom_rx_sec = time.monotonic()

    def _goal_in_base_link(self) -> PoseStamped | None:
        if self._goal_pose is None:
            return None

        src = PoseStamped()
        src.header = self._goal_pose.header
        src.pose = self._goal_pose.pose
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

    def _reset_pid(self) -> None:
        self._pid_x.reset()
        self._pid_y.reset()
        self._pid_z.reset()
        self._pid_yaw.reset()
        self._yaw_err_unwrapped = None

    def _publish_neutral(self) -> None:
        msg = OverrideRCIn()
        ch = [OverrideRCIn.CHAN_NOCHANGE] * 18
        for i in range(6):
            ch[i] = self._neutral
        msg.channels = ch
        self._rc_pub.publish(msg)

    def _tick(self) -> None:
        now = time.monotonic()
        dt = now - self._last_tick_sec
        self._last_tick_sec = now
        dt = max(dt, 1e-3)

        if self._goal_pose is None:
            self._reset_pid()
            self._publish_neutral()
            return

        if (now - self._last_odom_rx_sec) > self._odom_timeout:
            self._reset_pid()
            self._publish_neutral()
            self.get_logger().warn(
                f"Odometry timeout ({self._odom_timeout:.2f}s); forcing zero control.",
                throttle_duration_sec=2.0,
            )
            return

        goal_in_base = self._goal_in_base_link()
        if goal_in_base is None:
            self._reset_pid()
            self._publish_neutral()
            return
        goal = goal_in_base.pose

        ex = goal.position.x
        ey = goal.position.y
        ez = goal.position.z

        yaw_goal = _yaw_from_quat(
            goal.orientation.x,
            goal.orientation.y,
            goal.orientation.z,
            goal.orientation.w,
        )
        eyaw_wrapped = _wrap_pi(yaw_goal)
        self._yaw_err_unwrapped = _unwrap_angle(eyaw_wrapped, self._yaw_err_unwrapped)
        eyaw = self._yaw_err_unwrapped

        surge_factor = self._pid_x.step(ex, dt)
        sway_factor = self._pid_y.step(ey, dt)
        heave_factor = self._pid_z.step(ez, dt)
        yaw_factor = self._pid_yaw.step(eyaw, dt)

        if self._debug_log:
            self.get_logger().info(
                f"err_body=({ex:.3f}, {ey:.3f}, {ez:.3f}) eyaw={eyaw:.3f} "
                "thrust_factors="
                f"({surge_factor:.3f}, {sway_factor:.3f}, {heave_factor:.3f}) "
                f"yaw_factor={yaw_factor:.3f}"
            )

        msg = OverrideRCIn()
        msg.channels = thrust_factors_to_rc_channels(
            surge_factor=surge_factor,
            sway_factor=sway_factor,
            heave_factor=heave_factor,
            yaw_factor=yaw_factor,
            neutral=self._neutral,
            delta=self._delta,
        )
        self._rc_pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BrovCtrlPID()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
