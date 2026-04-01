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


def acc_to_rc_channels(
    *,
    ax: float,
    ay: float,
    az: float,
    ayaw: float,
    neutral: int,
    delta: int,
    max_acc_x: float,
    max_acc_y: float,
    max_acc_z: float,
    max_acc_yaw: float,
) -> list[int]:
    """Map accel commands to 18 RC uint16 channels, with ch0/ch1 held neutral."""

    def pwm_axis(val: float, vmax: float) -> int:
        if vmax <= 1e-9:
            return neutral
        s = _clamp(val / vmax, -1.0, 1.0)
        return int(round(neutral + s * delta))

    nc = OverrideRCIn.CHAN_NOCHANGE
    ch = [nc] * 18
    ch[0] = neutral  # pitch unused
    ch[1] = neutral  # roll unused
    ch[2] = pwm_axis(az, max_acc_z)  # vertical
    ch[3] = pwm_axis(-ayaw, max_acc_yaw)  # yaw (match teleop: yaw_left -> negative PWM)
    ch[4] = pwm_axis(ax, max_acc_x)  # forward
    ch[5] = pwm_axis(-ay, max_acc_y)  # lateral (match teleop: left -> negative PWM)
    return ch


class BrovCtrlPID(Node):
    def __init__(self) -> None:
        super().__init__("brov_ctrl_pid")

        self.declare_parameter("goal_topic", "/goal_pose")
        self.declare_parameter("odom_topic", "/mocap/mr_pinchy/odom")
        self.declare_parameter("rc_topic", "/mavros/rc/override")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_link_frame", "mr_pinchy/base_link")
        self.declare_parameter("debug_log", True)
        self.declare_parameter("publish_rate_hz", 20.0)
        self.declare_parameter("odom_timeout_sec", 0.3)
        self.declare_parameter("neutral_pwm", 1500)
        self.declare_parameter("delta_pwm", 100)

        self.declare_parameter("kp_x", 1.0)
        self.declare_parameter("ki_x", 0.0)
        self.declare_parameter("kd_x", 0.0)
        self.declare_parameter("i_limit_x", 1.0)
        self.declare_parameter("max_acc_x", 1.0)

        self.declare_parameter("kp_y", 1.0)
        self.declare_parameter("ki_y", 0.0)
        self.declare_parameter("kd_y", 0.0)
        self.declare_parameter("i_limit_y", 1.0)
        self.declare_parameter("max_acc_y", 1.0)

        self.declare_parameter("kp_z", 1.0)
        self.declare_parameter("ki_z", 0.0)
        self.declare_parameter("kd_z", 0.0)
        self.declare_parameter("i_limit_z", 1.0)
        self.declare_parameter("max_acc_z", 1.0)

        self.declare_parameter("kp_yaw", 1.0)
        self.declare_parameter("ki_yaw", 0.0)
        self.declare_parameter("kd_yaw", 0.0)
        self.declare_parameter("i_limit_yaw", 1.0)
        self.declare_parameter("max_acc_yaw", 1.0)

        goal_topic = self.get_parameter("goal_topic").get_parameter_value().string_value
        odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value
        rc_topic = self.get_parameter("rc_topic").get_parameter_value().string_value
        self._map_frame = self.get_parameter("map_frame").get_parameter_value().string_value
        self._base_link_frame = (
            self.get_parameter("base_link_frame").get_parameter_value().string_value
        )
        self._debug_log = self.get_parameter("debug_log").get_parameter_value().bool_value

        rate_hz = self.get_parameter("publish_rate_hz").get_parameter_value().double_value
        if rate_hz <= 0.0:
            rate_hz = 20.0
        self._period = 1.0 / rate_hz

        self._odom_timeout = float(
            self.get_parameter("odom_timeout_sec").get_parameter_value().double_value
        )
        if self._odom_timeout <= 0.0:
            self._odom_timeout = 0.3

        self._neutral = int(
            self.get_parameter("neutral_pwm").get_parameter_value().integer_value
        )
        if self._neutral <= 0:
            self._neutral = 1500

        self._delta = int(self.get_parameter("delta_pwm").get_parameter_value().integer_value)
        if self._delta <= 0:
            self._delta = 100

        self._max_acc_x = float(
            self.get_parameter("max_acc_x").get_parameter_value().double_value
        )
        self._max_acc_y = float(
            self.get_parameter("max_acc_y").get_parameter_value().double_value
        )
        self._max_acc_z = float(
            self.get_parameter("max_acc_z").get_parameter_value().double_value
        )
        self._max_acc_yaw = float(
            self.get_parameter("max_acc_yaw").get_parameter_value().double_value
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

        self._pid_x = PIDAxis(
            kp=float(self.get_parameter("kp_x").get_parameter_value().double_value),
            ki=float(self.get_parameter("ki_x").get_parameter_value().double_value),
            kd=float(self.get_parameter("kd_x").get_parameter_value().double_value),
            integral_limit=float(
                self.get_parameter("i_limit_x").get_parameter_value().double_value
            ),
            output_limit=self._max_acc_x,
        )
        self._pid_y = PIDAxis(
            kp=float(self.get_parameter("kp_y").get_parameter_value().double_value),
            ki=float(self.get_parameter("ki_y").get_parameter_value().double_value),
            kd=float(self.get_parameter("kd_y").get_parameter_value().double_value),
            integral_limit=float(
                self.get_parameter("i_limit_y").get_parameter_value().double_value
            ),
            output_limit=self._max_acc_y,
        )
        self._pid_z = PIDAxis(
            kp=float(self.get_parameter("kp_z").get_parameter_value().double_value),
            ki=float(self.get_parameter("ki_z").get_parameter_value().double_value),
            kd=float(self.get_parameter("kd_z").get_parameter_value().double_value),
            integral_limit=float(
                self.get_parameter("i_limit_z").get_parameter_value().double_value
            ),
            output_limit=self._max_acc_z,
        )
        self._pid_yaw = PIDAxis(
            kp=float(self.get_parameter("kp_yaw").get_parameter_value().double_value),
            ki=float(self.get_parameter("ki_yaw").get_parameter_value().double_value),
            kd=float(self.get_parameter("kd_yaw").get_parameter_value().double_value),
            integral_limit=float(
                self.get_parameter("i_limit_yaw").get_parameter_value().double_value
            ),
            output_limit=self._max_acc_yaw,
        )

        self.create_timer(self._period, self._tick)
        self.get_logger().info(
            f"brov_ctrl_pid: goal='{goal_topic}', odom='{odom_topic}', rc='{rc_topic}', "
            f"map_frame='{self._map_frame}', base_link_frame='{self._base_link_frame}', "
            f"rate={rate_hz:.1f}Hz, odom_timeout={self._odom_timeout:.2f}s"
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

        ax_body = self._pid_x.step(ex, dt)
        ay_body = self._pid_y.step(ey, dt)
        az_body = self._pid_z.step(ez, dt)
        # ayaw = self._pid_yaw.step(eyaw, dt)
        ayaw = 0.0

        if self._debug_log:
            self.get_logger().info(
                f"err_body=({ex:.3f}, {ey:.3f}, {ez:.3f}) eyaw={eyaw:.3f} "
                f"acc_body=({ax_body:.3f}, {ay_body:.3f}, {az_body:.3f}) ayaw={ayaw:.3f}"
            )

        msg = OverrideRCIn()
        msg.channels = acc_to_rc_channels(
            ax=ax_body,
            ay=ay_body,
            az=az_body,
            ayaw=ayaw,
            neutral=self._neutral,
            delta=self._delta,
            max_acc_x=self._max_acc_x,
            max_acc_y=self._max_acc_y,
            max_acc_z=self._max_acc_z,
            max_acc_yaw=self._max_acc_yaw,
        )
        self.get_logger().info(f"msg: {msg}")  
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