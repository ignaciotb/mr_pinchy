#!/usr/bin/env python3
from __future__ import annotations

import time

import rclpy
from geometry_msgs.msg import PoseStamped, WrenchStamped
from mavros_msgs.msg import OverrideRCIn
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from tf2_geometry_msgs import do_transform_pose_stamped
from tf2_ros import Buffer, TransformException, TransformListener

from mr_pinchy_interface.brov_utils import (
    PIDAxis,
    build_wrench_message,
    neutral_rc_override,
    pid_outputs_to_wrench,
    quat_to_rpy,
    wrap_pi,
    wrench_to_rc_override,
)


class BrovWayPointController(Node):
    def __init__(self) -> None:
        super().__init__("brov_ctrl_pid_wrench")

        self.declare_parameter("goal_topic", "/goal_pose")
        self.declare_parameter("odom_topic", "/mocap/mr_pinchy/odom")
        self.declare_parameter("rc_topic", "/mavros/rc/override")
        self.declare_parameter("wrench_topic", "/mr_pinchy/cmd_wrench")
        self.declare_parameter("base_link_frame", "/mr_pinchy/base_link")
        self.declare_parameter("map_frame", "/map")

        goal_topic = self.get_parameter("goal_topic").get_parameter_value().string_value
        odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value
        rc_topic = self.get_parameter("rc_topic").get_parameter_value().string_value
        wrench_topic = (self.get_parameter("wrench_topic").get_parameter_value().string_value)
        self._base_link_frame = self.get_parameter("base_link_frame").get_parameter_value().string_value
        self._map_frame = self.get_parameter("map_frame").get_parameter_value().string_value

        self._neutral_pwm = 1500
        self._force_x_scale = 20.0
        self._force_y_scale = 20.0
        self._force_z_scale = 20.0
        self._torque_x_scale = 1.0
        self._torque_y_scale = 1.0
        self._torque_z_scale = 1.0

        self._debug_log = True
        self._publish_rate_hz = 20.0
        self._period = 1.0 / self._publish_rate_hz
        self._odom_timeout = 0.3

        self._pid_x = PIDAxis(
            kp=3.0, ki=0.0, kd=0.1, integral_limit=1.0, output_limit=1.0
        )
        self._pid_y = PIDAxis(
            kp=3.0, ki=0.0, kd=0.1, integral_limit=1.0, output_limit=1.0
        )
        self._pid_z = PIDAxis(
            kp=1.0, ki=0.0, kd=0.1, integral_limit=1.0, output_limit=1.0
        )
        self._pid_roll = PIDAxis(
            kp=1.0,
            ki=0.0,
            kd=0.0,
            integral_limit=1.0,
            output_limit=1.0,
        )
        self._pid_pitch = PIDAxis(
            kp=1.0,
            ki=0.0,
            kd=0.0,
            integral_limit=1.0,
            output_limit=1.0,
        )
        self._pid_yaw = PIDAxis(
            kp=3.0,
            ki=0.0,
            kd=0.1,
            integral_limit=1.0,
            output_limit=1.0,
        )

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self._wrench_pub = self.create_publisher(WrenchStamped, wrench_topic, qos)
        self._rc_pub = self.create_publisher(OverrideRCIn, rc_topic, qos)
        self.create_subscription(PoseStamped, goal_topic, self._goal_cb, 10)
        self.create_subscription(Odometry, odom_topic, self._odom_cb, 10)

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._goal_pose: PoseStamped | None = None
        self._last_odom_rx_sec: float = 0.0
        self._last_tick_sec: float = time.monotonic()

        self.create_timer(self._period, self._tick)
        self.get_logger().info(
            f"brov_ctrl_pid_wrench: goal='{goal_topic}', odom='{odom_topic}', "
            f"wrench='{wrench_topic}', rc='{rc_topic}', map_frame='{self._map_frame}', "
            f"base_link_frame='{self._base_link_frame}', rate={self._publish_rate_hz:.1f}Hz, "
            f"odom_timeout={self._odom_timeout:.2f}s"
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
        self.get_logger().info(
            f"Received new goal pose: frame_id='{src.header.frame_id}', "
            f"position=({src.pose.position.x:.3f}, {src.pose.position.y:.3f}, {src.pose.position.z:.3f}), "
            f"orientation=({src.pose.orientation.x:.3f}, {src.pose.orientation.y:.3f}, {src.pose.orientation.z:.3f}, {src.pose.orientation.w:.3f})"
        )

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
            try:
                base_link_in_map = self._tf_buffer.lookup_transform(
                    self._map_frame,
                    self._base_link_frame,
                    rclpy.time.Time(),
                    timeout=Duration(seconds=0.05),
                )
                translation = base_link_in_map.transform.translation
                rotation = base_link_in_map.transform.rotation
                self.get_logger().info(
                    f"base_link in {self._map_frame}: "
                    f"position=({translation.x:.3f}, {translation.y:.3f}, {translation.z:.3f}), "
                    f"orientation=({rotation.x:.3f}, {rotation.y:.3f}, {rotation.z:.3f}, {rotation.w:.3f})"
                )
            except TransformException as exc:
                self.get_logger().warn(
                    f"TF {self._base_link_frame} -> {self._map_frame} unavailable: {exc}",
                    throttle_duration_sec=2.0,
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
        self._pid_roll.reset()
        self._pid_pitch.reset()
        self._pid_yaw.reset()

    def _publish_wrench_and_rc(
        self, wrench_msg: WrenchStamped, rc_msg: OverrideRCIn
    ) -> None:
        self._wrench_pub.publish(wrench_msg)
        self._rc_pub.publish(rc_msg)

    def _publish_zero_control(self) -> None:
        stamp = self.get_clock().now().to_msg()
        wrench_msg = build_wrench_message(frame_id=self._base_link_frame, stamp=stamp)
        rc_msg = neutral_rc_override(neutral_pwm=self._neutral_pwm)
        self._publish_wrench_and_rc(wrench_msg, rc_msg)

    def _tick(self) -> None:
        now = time.monotonic()
        dt = now - self._last_tick_sec
        self._last_tick_sec = now
        dt = max(dt, 1e-3)

        if self._goal_pose is None:
            self._reset_pid()
            self._publish_zero_control()
            return

        if (now - self._last_odom_rx_sec) > self._odom_timeout:
            self._reset_pid()
            self._publish_zero_control()
            self.get_logger().warn(
                f"Odometry timeout ({self._odom_timeout:.2f}s); forcing zero control.",
                throttle_duration_sec=2.0,
            )
            return

        goal_in_base = self._goal_in_base_link()
        if goal_in_base is None:
            self._reset_pid()
            self._publish_zero_control()
            return
        goal = goal_in_base.pose

        ex = goal.position.x
        ey = goal.position.y
        ez = goal.position.z

        eroll, epitch, eyaw = quat_to_rpy(
            goal.orientation.x,
            goal.orientation.y,
            goal.orientation.z,
            goal.orientation.w,
        )

        surge_pid = self._pid_x.step(ex, dt)
        sway_pid = self._pid_y.step(ey, dt)
        heave_pid = self._pid_z.step(ez, dt)
        roll_pid = self._pid_roll.step(eroll, dt, wrap_error=True)
        pitch_pid = self._pid_pitch.step(epitch, dt, wrap_error=True)
        yaw_pid = self._pid_yaw.step(eyaw, dt, wrap_error=True)

        stamp = self.get_clock().now().to_msg()
        wrench_msg = pid_outputs_to_wrench(
            surge_pid_output=surge_pid,
            sway_pid_output=sway_pid,
            heave_pid_output=heave_pid,
            roll_pid_output=roll_pid,
            pitch_pid_output=pitch_pid,
            yaw_pid_output=yaw_pid,
            force_x_scale=self._force_x_scale,
            force_y_scale=self._force_y_scale,
            force_z_scale=self._force_z_scale,
            torque_x_scale=self._torque_x_scale,
            torque_y_scale=self._torque_y_scale,
            torque_z_scale=self._torque_z_scale,
            frame_id=self._base_link_frame,
            stamp=stamp,
        )

        rc_msg = wrench_to_rc_override(
            wrench_msg,
            delta_pwm=150,
        )

        if self._debug_log:
            self.get_logger().info(
                f"err_body=({ex:.3f}, {ey:.3f}, {ez:.3f}) "
                f"err_rpy=({eroll:.3f}, {epitch:.3f}, {eyaw:.3f}) "
                "pid="
                f"({surge_pid:.3f}, {sway_pid:.3f}, {heave_pid:.3f}, "
                f"{roll_pid:.3f}, {pitch_pid:.3f}, {yaw_pid:.3f}) "
                "wrench="
                f"(Fx={wrench_msg.wrench.force.x:.3f}, Fy={wrench_msg.wrench.force.y:.3f}, "
                f"Fz={wrench_msg.wrench.force.z:.3f}, Tx={wrench_msg.wrench.torque.x:.3f}, "
                f"Ty={wrench_msg.wrench.torque.y:.3f}, Tz={wrench_msg.wrench.torque.z:.3f})"
                " rc="
                f"({rc_msg.channels[0]}, {rc_msg.channels[1]}, {rc_msg.channels[2]}, {rc_msg.channels[3]}, {rc_msg.channels[4]}, {rc_msg.channels[5]})"
            )

        self._publish_wrench_and_rc(wrench_msg, rc_msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BrovWayPointController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
