#!/usr/bin/env python3
from __future__ import annotations

import copy

import numpy as np
import rclpy
from geometry_msgs.msg import TwistStamped
from mavros_msgs.msg import ManualControl
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from tf2_geometry_msgs import do_transform_vector3
from tf2_ros import Buffer, TransformException, TransformListener

from mr_pinchy_interface.brov_utils import (
    vector3_stamped,
    vector3_to_array,
)


def normalize_tf_frame_id(frame_id: str) -> str:
    return str(frame_id).lstrip("/")


class BrovTwistSimpleController(Node):
    def __init__(self) -> None:
        super().__init__("brov_ctrl_twist_simple")

        self.declare_parameter("cmd_topic", "/mr_pinchy/cmd_vel")
        self.declare_parameter("odom_topic", "/mocap/mr_pinchy/odom_ekf")
        self.declare_parameter("manual_control_topic", "/mavros/manual_control/send")
        self.declare_parameter("base_link_frame", "mr_pinchy/base_link_ekf")

        cmd_topic = str(self.get_parameter("cmd_topic").value)
        odom_topic = str(self.get_parameter("odom_topic").value)
        mc_topic = str(self.get_parameter("manual_control_topic").value)
        self._base_link_frame = normalize_tf_frame_id(self.get_parameter("base_link_frame").value)

        self._debug_log = True
        self._publish_rate_hz = 50.0
        self._odom_timeout_sec = 0.3
        self._tf_timeout = Duration(seconds=0.05)
        self._cmd_timeout_sec = 0.3

        # # PID gains for [surge, sway, heave, yaw]
        # self._kp = np.array([2, 2, 0.8, 0.8], dtype=float)
        # self._ki = np.array([0.1, 0.1, 0.4, 0.1], dtype=float)
        # self._kd = np.array([0.0, 0.0, 0.2, 0.0], dtype=float)
        # self._integral = np.zeros(4, dtype=float)
        # # Heave needs more integral authority to absorb the constant buoyancy bias.
        # self._integral_limit = np.array([0.5, 0.5, 1.0, 0.5], dtype=float)

        # # Feedforward: linear mapping from commanded velocity to control output
        # self._kff = np.array([1.0 / 0.5, 1.0 / 0.4, 1.0 / 0.6, 1.0 / 1.3], dtype=float)

        # PID gains for [surge, sway, heave, yaw]
        self._kp = np.array([7., 8, 0.5, 1.4], dtype=float)
        self._ki = np.array([0.5, 0.5, 0.0, 0.3], dtype=float)
        self._kd = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)
        self._integral = np.zeros(4, dtype=float)
        self._integral_limit = np.array([0.5, 0.5, 0.5, 0.5], dtype=float)
        # Feedforward: linear mapping from commanded velocity to control output
        self._kff = np.array([1.0 / 0.3, 1.0 / 0.25, 1.0 / 0.3, 1.0 / 0.65], dtype=float)

        # Tank safety bounds: [[x_min, y_min, z_min], [x_max, y_max, z_max]]
        self._tank_range = np.array([[1.0, -1.8, 0.3], [6.0, 1.8, 3.0]])

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self._mc_pub = self.create_publisher(ManualControl, mc_topic, qos)
        self.create_subscription(TwistStamped, cmd_topic, self._cmd_cb, 10)
        self.create_subscription(Odometry, odom_topic, self._odom_cb, 10)

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._last_cmd: TwistStamped | None = None
        self._last_odom: Odometry | None = None
        self._last_odom_rx_sec: float = 0.0
        self._last_cmd_rx_sec: float = 0.0
        self._prev_meas: np.ndarray | None = None
        self._prev_feedback_sec: float | None = None
        self._last_ctrl_cmd = np.array([0.0, 0.0, 500.0, 0.0], dtype=float)
        self.create_timer(1.0 / self._publish_rate_hz, self._tick)
        self.get_logger().info(
            f"brov_ctrl_twist_simple: cmd='{cmd_topic}', odom='{odom_topic}', "
            f"manual_control='{mc_topic}', base_link_frd='{self._base_link_frame}', "
            f"rate={self._publish_rate_hz:.1f}Hz"
        )

    def _cmd_cb(self, msg: TwistStamped) -> None:
        self._last_cmd = msg
        self._last_cmd_rx_sec = self.get_clock().now().nanoseconds * 1e-9

    def _odom_cb(self, msg: Odometry) -> None:
        self._last_odom = msg
        self._last_odom_rx_sec = self.get_clock().now().nanoseconds * 1e-9

    def _transform_vector_to_base_link(
        self,
        msg,
        *,
        source_frame: str,
        stamp=None,
    ) -> np.ndarray | None:
        source_frame = normalize_tf_frame_id(source_frame)
        if source_frame == self._base_link_frame:
            return vector3_to_array(msg)

        try:
            transform = self._tf_buffer.lookup_transform(
                self._base_link_frame,
                source_frame,
                rclpy.time.Time(),
                timeout=self._tf_timeout,
            )
        except TransformException as exc:
            self.get_logger().warn(
                f"TF {source_frame} -> {self._base_link_frame} unavailable: {exc}",
                throttle_duration_sec=2.0,
            )
            return None

        vector = vector3_stamped(msg, frame_id=source_frame, stamp=stamp)
        transformed = do_transform_vector3(vector, copy.deepcopy(transform))
        return vector3_to_array(transformed.vector)

    def _get_body_velocity(self, odom: Odometry) -> np.ndarray | None:
        source_frame = normalize_tf_frame_id(getattr(odom, "child_frame_id", ""))
        if not source_frame:
            return None
        stamp = getattr(getattr(odom, "header", None), "stamp", None)

        linear = self._transform_vector_to_base_link(
            odom.twist.twist.linear, source_frame=source_frame, stamp=stamp
        )
        angular = self._transform_vector_to_base_link(
            odom.twist.twist.angular, source_frame=source_frame, stamp=stamp
        )
        if linear is None or angular is None:
            return None
        return np.concatenate([linear, angular]).astype(float)

    def _get_position(self, odom: Odometry) -> np.ndarray:
        p = odom.pose.pose.position
        return np.array([p.x, p.y, p.z], dtype=float)

    def _command_in_base_link(self, msg: TwistStamped) -> np.ndarray | None:
        source_frame = normalize_tf_frame_id(msg.header.frame_id)
        linear = self._transform_vector_to_base_link(
            msg.twist.linear, source_frame=source_frame, stamp=msg.header.stamp
        )
        angular = self._transform_vector_to_base_link(
            msg.twist.angular, source_frame=source_frame, stamp=msg.header.stamp
        )
        if linear is None or angular is None:
            return None
        return np.concatenate([linear, angular]).astype(float)

    def _reset_pid(self) -> None:
        self._prev_meas = None
        self._prev_feedback_sec = None
        self._integral = np.zeros(4, dtype=float)

    def _publish_manual_control(self, x: float, y: float, z: float, r: float) -> None:
        msg = ManualControl()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.x = float(x)
        msg.y = float(y)
        msg.z = float(z)
        msg.r = float(r)
        msg.buttons = 0
        self._mc_pub.publish(msg)

    def _publish_neutral(self) -> None:
        self._publish_manual_control(0.0, 0.0, 500.0, 0.0)
        self._last_ctrl_cmd = np.array([0.0, 0.0, 500.0, 0.0], dtype=float)

    def _tick(self) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9

        if self._last_cmd is None or (now - self._last_cmd_rx_sec) > self._cmd_timeout_sec:
            self._reset_pid()
            self._publish_neutral()
            if self._last_cmd is not None:
                self.get_logger().warn(
                    f"Command timeout ({self._cmd_timeout_sec:.2f}s); forcing zero control.",
                    throttle_duration_sec=2.0,
                )
            return

        if self._last_odom is None or (now - self._last_odom_rx_sec) > self._odom_timeout_sec:
            self._reset_pid()
            self._publish_neutral()
            if self._last_odom is not None:
                self.get_logger().warn(
                    f"Odometry timeout ({self._odom_timeout_sec:.2f}s); forcing zero control.",
                    throttle_duration_sec=2.0,
                )
            return

        # Tank range safety check
        pos = self._get_position(self._last_odom)
        if np.any(pos < self._tank_range[0]) or np.any(pos > self._tank_range[1]):
            self._reset_pid()
            self._publish_neutral()
            self.get_logger().warn(
                "Out of tank range; forcing zero control.",
                throttle_duration_sec=1.0,
            )
            return

        nu_cmd = self._command_in_base_link(self._last_cmd)
        if nu_cmd is None:
            self._reset_pid()
            self._publish_neutral()
            return

        nu = self._get_body_velocity(self._last_odom)
        if nu is None:
            self._reset_pid()
            self._publish_neutral()
            return

        # Velocity error on [surge, sway, heave, yaw]
        ref = np.array([nu_cmd[0], nu_cmd[1], nu_cmd[2], nu_cmd[5]], dtype=float)
        meas = np.array([nu[0], nu[1], nu[2], nu[5]], dtype=float)
        error = ref - meas

        # Derivative on measurement (avoids derivative kick on reference changes)
        if self._prev_meas is not None and self._prev_feedback_sec is not None:
            dt = now - self._prev_feedback_sec
            if dt > 1e-6:
                deriv = -(meas - self._prev_meas) / dt
                self._integral += error * dt
                self._integral = np.clip(
                    self._integral, -self._integral_limit, self._integral_limit
                )
            else:
                deriv = np.zeros(4, dtype=float)
        else:
            deriv = np.zeros(4, dtype=float)

        self._prev_meas = meas.copy()
        self._prev_feedback_sec = now

        output = (
            self._kff * ref
            + self._kp * error
            + self._ki * self._integral
            + self._kd * deriv
        )
        output = np.clip(output, -1.0, 1.0)
        output_limit = 1000.0
        # FRD body frame -> ArduSub ManualControl:
        #   surge (+x = forward)  -> MC.x (+1000 = forward)   same sign
        #   sway  (+y = right)    -> MC.y (+1000 = right)     same sign
        #   heave (+z = down)     -> MC.z (>500 = ascend)     inverted
        #   yaw   (+rz = CW)     -> MC.r (+1000 = CW)        same sign
        x = float(output[0]) * output_limit
        y = float(output[1]) * output_limit
        z = float(-output[2]) * output_limit/2 + 500.0
        r = float(output[3]) * output_limit
        alpha = np.array([0.7, 0.7, 0.7, 0.7], dtype=float)
        self._last_ctrl_cmd = alpha * np.array([x, y, z, r]) + (1 - alpha) * self._last_ctrl_cmd
        
        if self._debug_log:
            self.get_logger().info(
                f"err=({error[0]:.3f},{error[1]:.3f},{error[2]:.3f},{error[3]:.3f}) "
                f"mc=(x={x:.0f},y={y:.0f},z={z:.0f},r={r:.0f})"
            )

        self._publish_manual_control(self._last_ctrl_cmd[0], self._last_ctrl_cmd[1], self._last_ctrl_cmd[2], self._last_ctrl_cmd[3])


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BrovTwistSimpleController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
