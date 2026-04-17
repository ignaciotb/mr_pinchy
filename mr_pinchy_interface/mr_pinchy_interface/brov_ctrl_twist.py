#!/usr/bin/env python3
from __future__ import annotations

import math
import time

import numpy as np
import rclpy
from geometry_msgs.msg import TwistStamped, WrenchStamped
from mavros_msgs.msg import OverrideRCIn
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from tf2_ros import Buffer, TransformException, TransformListener

from mr_pinchy.mr_pinchy_interface.mr_pinchy_interface.brov_wrench_to_pwm import wrench_to_rc_override
from mr_pinchy.mr_pinchy_interface.mr_pinchy_interface.brov_dynamics import bluerov2_model_terms


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _param_value(node: Node, name: str):
    param = node.get_parameter(name)
    if hasattr(param, "value"):
        return param.value
    if hasattr(param, "get_parameter_value"):
        msg = param.get_parameter_value()
        for attr in (
            "string_value",
            "double_value",
            "integer_value",
            "bool_value",
            "integer_array_value",
            "double_array_value",
            "string_array_value",
        ):
            value = getattr(msg, attr, None)
            if value not in (None, "", 0, 0.0, False, []):
                return value
        if getattr(msg, "type", None) == 0:
            return None
        return (
            getattr(msg, "string_value", None)
            or getattr(msg, "double_value", None)
            or getattr(msg, "integer_value", None)
        )
    return param


def _quat_to_rpy(x: float, y: float, z: float, w: float) -> tuple[float, float, float]:
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    sinp = _clamp(sinp, -1.0, 1.0)
    pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def _quat_to_rotation_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=float,
    )


def twist_to_body_velocity(msg: TwistStamped) -> np.ndarray:
    return np.array(
        [
            msg.twist.linear.x,
            msg.twist.linear.y,
            msg.twist.linear.z,
            msg.twist.angular.x,
            msg.twist.angular.y,
            msg.twist.angular.z,
        ],
        dtype=float,
    )


def odom_to_body_state(odom: Odometry) -> tuple[np.ndarray, np.ndarray]:
    pose = odom.pose.pose
    twist = odom.twist.twist
    roll, pitch, yaw = _quat_to_rpy(
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    )
    eta = np.array(
        [
            pose.position.x,
            pose.position.y,
            pose.position.z,
            roll,
            pitch,
            yaw,
        ],
        dtype=float,
    )
    nu = np.array(
        [
            twist.linear.x,
            twist.linear.y,
            twist.linear.z,
            twist.angular.x,
            twist.angular.y,
            twist.angular.z,
        ],
        dtype=float,
    )
    return eta, nu


def static_wrench_feedforward(eta: np.ndarray, nu_ref: np.ndarray) -> np.ndarray:
    terms = bluerov2_model_terms(eta, nu_ref)
    return np.asarray(
        terms["C"] @ terms["nu"] + terms["D"] @ terms["nu_rel"] + terms["g"],
        dtype=float,
    )


def velocity_feedback_wrench(
    nu: np.ndarray,
    nu_ref: np.ndarray,
    gains: np.ndarray,
) -> np.ndarray:
    return np.asarray(gains * (nu_ref - nu), dtype=float)


def clamp_body_wrench(
    tau: np.ndarray,
    *,
    force_limits: np.ndarray,
    torque_limits: np.ndarray,
) -> np.ndarray:
    tau = np.asarray(tau, dtype=float).copy()
    tau[:3] = np.clip(tau[:3], -force_limits, force_limits)
    tau[3:] = np.clip(tau[3:], -torque_limits, torque_limits)
    return tau


def body_wrench_message(
    tau: np.ndarray,
    *,
    frame_id: str = "mr_pinchy/base_link",
    stamp=None,
) -> WrenchStamped:
    tau = np.asarray(tau, dtype=float)
    msg = WrenchStamped()
    msg.header.frame_id = frame_id
    if stamp is not None:
        msg.header.stamp = stamp
    msg.wrench.force.x = float(tau[0])
    msg.wrench.force.y = float(tau[1])
    msg.wrench.force.z = float(tau[2])
    msg.wrench.torque.x = float(tau[3])
    msg.wrench.torque.y = float(tau[4])
    msg.wrench.torque.z = float(tau[5])
    return msg


def _neutral_rc_override(neutral_pwm: int) -> OverrideRCIn:
    msg = OverrideRCIn()
    channels = [OverrideRCIn.CHAN_NOCHANGE] * 18
    for idx in range(6):
        channels[idx] = neutral_pwm
    msg.channels = channels
    return msg


class BrovCtrlSIWrench(Node):
    def __init__(self) -> None:
        super().__init__("brov_ctrl_si_wrench")

        self.declare_parameter("cmd_topic", "/pinchy/cmd_vel")
        self.declare_parameter("odom_topic", "/mocap/mr_pinchy/odom")
        self.declare_parameter("wrench_topic", "/cmd_wrench")
        self.declare_parameter("rc_topic", "/mavros/rc/override")
        self.declare_parameter("base_link_frame", "mr_pinchy/base_link")
        self.declare_parameter("si_link_frame", "mr_pinchy/si_link")
        self.declare_parameter("publish_rate_hz", 20.0)
        self.declare_parameter("cmd_timeout_sec", 0.1)
        self.declare_parameter("odom_timeout_sec", 0.3)
        self.declare_parameter("k_u", 8.0)
        self.declare_parameter("k_v", 8.0)
        self.declare_parameter("k_w", 8.0)
        self.declare_parameter("k_p", 2.0)
        self.declare_parameter("k_q", 2.0)
        self.declare_parameter("k_r", 2.0)
        self.declare_parameter("fx_max", 60.0)
        self.declare_parameter("fy_max", 60.0)
        self.declare_parameter("fz_max", 60.0)
        self.declare_parameter("tx_max", 20.0)
        self.declare_parameter("ty_max", 20.0)
        self.declare_parameter("tz_max", 20.0)
        self.declare_parameter("neutral_pwm", 1500)
        self.declare_parameter("delta_pwm", 100)
        self.declare_parameter("geometry", "60002")
        self.declare_parameter(
            "motor_direction_signs",
            [-1, -1, 1, 1, -1, 1, 1, -1],
        )

        cmd_topic = str(_param_value(self, "cmd_topic"))
        odom_topic = str(_param_value(self, "odom_topic"))
        wrench_topic = str(_param_value(self, "wrench_topic"))
        rc_topic = str(_param_value(self, "rc_topic"))
        self._base_link_frame = str(_param_value(self, "base_link_frame"))
        self._si_link_frame = str(_param_value(self, "si_link_frame"))
        publish_rate_hz = float(_param_value(self, "publish_rate_hz"))
        self._cmd_timeout_sec = float(_param_value(self, "cmd_timeout_sec"))
        self._odom_timeout_sec = float(_param_value(self, "odom_timeout_sec"))

        self._publish_rate_hz = publish_rate_hz if publish_rate_hz > 0.0 else 20.0
        self._cmd_timeout_sec = self._cmd_timeout_sec if self._cmd_timeout_sec > 0.0 else 0.1
        self._odom_timeout_sec = (
            self._odom_timeout_sec if self._odom_timeout_sec > 0.0 else 0.3
        )
        self._neutral_pwm = int(_param_value(self, "neutral_pwm"))
        self._delta_pwm = int(_param_value(self, "delta_pwm"))
        self._geometry = str(_param_value(self, "geometry"))
        self._motor_direction_signs = tuple(
            int(v) for v in list(_param_value(self, "motor_direction_signs"))
        )

        self._gains = np.array(
            [
                float(_param_value(self, "k_u")),
                float(_param_value(self, "k_v")),
                float(_param_value(self, "k_w")),
                float(_param_value(self, "k_p")),
                float(_param_value(self, "k_q")),
                float(_param_value(self, "k_r")),
            ],
            dtype=float,
        )
        self._force_limits = np.array(
            [
                float(_param_value(self, "fx_max")),
                float(_param_value(self, "fy_max")),
                float(_param_value(self, "fz_max")),
            ],
            dtype=float,
        )
        self._torque_limits = np.array(
            [
                float(_param_value(self, "tx_max")),
                float(_param_value(self, "ty_max")),
                float(_param_value(self, "tz_max")),
            ],
            dtype=float,
        )

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self._wrench_pub = self.create_publisher(WrenchStamped, wrench_topic, qos)
        self._rc_pub = self.create_publisher(OverrideRCIn, rc_topic, qos)
        self.create_subscription(TwistStamped, cmd_topic, self._cmd_cb, 10)
        self.create_subscription(Odometry, odom_topic, self._odom_cb, 10)

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._last_cmd: TwistStamped | None = None
        self._last_cmd_rx_sec: float = 0.0
        self._last_odom: Odometry | None = None
        self._last_odom_rx_sec: float = 0.0
        self._nu_ref: np.ndarray | None = None

        self.create_timer(1.0 / self._publish_rate_hz, self._tick)
        self.get_logger().info(
            f"brov_ctrl_si_wrench: cmd='{cmd_topic}', odom='{odom_topic}', "
            f"wrench='{wrench_topic}', rc='{rc_topic}', base_link='{self._base_link_frame}', "
            f"si_link='{self._si_link_frame}', rate={self._publish_rate_hz:.1f}Hz"
        )

    def _cmd_cb(self, msg: TwistStamped) -> None:
        self._last_cmd = msg
        self._last_cmd_rx_sec = time.monotonic()

    def _odom_cb(self, msg: Odometry) -> None:
        self._last_odom = msg
        self._last_odom_rx_sec = time.monotonic()

    def _command_in_base_link(self, msg: TwistStamped) -> np.ndarray | None:
        source_frame = msg.header.frame_id or self._si_link_frame
        nu_cmd = twist_to_body_velocity(msg)

        if source_frame == self._base_link_frame:
            return nu_cmd

        try:
            tf = self._tf_buffer.lookup_transform(
                self._base_link_frame,
                source_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.05),
            )
        except TransformException as exc:
            self.get_logger().warn(
                f"TF {source_frame} -> {self._base_link_frame} unavailable: {exc}",
                throttle_duration_sec=2.0,
            )
            return None

        rotation = _quat_to_rotation_matrix(
            tf.transform.rotation.x,
            tf.transform.rotation.y,
            tf.transform.rotation.z,
            tf.transform.rotation.w,
        )
        nu_base = nu_cmd.copy()
        nu_base[:3] = rotation @ nu_cmd[:3]
        nu_base[3:] = rotation @ nu_cmd[3:]
        return nu_base

    def _compute_control(self) -> np.ndarray | None:
        if self._nu_ref is None or self._last_odom is None:
            return None

        eta, nu = odom_to_body_state(self._last_odom)
        tau_ff = static_wrench_feedforward(eta, self._nu_ref)
        tau_fb = velocity_feedback_wrench(nu, self._nu_ref, self._gains)
        tau = tau_ff + tau_fb
        return clamp_body_wrench(
            tau,
            force_limits=self._force_limits,
            torque_limits=self._torque_limits,
        )

    def _publish_wrench_and_rc(self, tau: np.ndarray) -> None:
        stamp = self.get_clock().now().to_msg()
        wrench_msg = body_wrench_message(
            tau,
            frame_id=self._base_link_frame,
            stamp=stamp,
        )
        rc_msg = wrench_to_rc_override(
            wrench_msg,
            neutral_pwm=self._neutral_pwm,
            delta_pwm=self._delta_pwm,
            geometry=self._geometry,
            motor_direction_signs=self._motor_direction_signs,
        )
        self._wrench_pub.publish(wrench_msg)
        self._rc_pub.publish(rc_msg)

    def _publish_zero_wrench_and_neutral_rc(self) -> None:
        stamp = self.get_clock().now().to_msg()
        wrench_msg = body_wrench_message(
            np.zeros(6, dtype=float),
            frame_id=self._base_link_frame,
            stamp=stamp,
        )
        rc_msg = _neutral_rc_override(neutral_pwm=self._neutral_pwm)
        self._wrench_pub.publish(wrench_msg)
        self._rc_pub.publish(rc_msg)

    def _tick(self) -> None:
        now = time.monotonic()

        if self._last_cmd is None or (now - self._last_cmd_rx_sec) > self._cmd_timeout_sec:
            self._nu_ref = None
            self._publish_zero_wrench_and_neutral_rc()
            if self._last_cmd is not None:
                self.get_logger().warn(
                    f"Command timeout ({self._cmd_timeout_sec:.2f}s); forcing zero control.",
                    throttle_duration_sec=2.0,
                )
            return

        if self._last_odom is None or (now - self._last_odom_rx_sec) > self._odom_timeout_sec:
            self._publish_zero_wrench_and_neutral_rc()
            if self._last_odom is not None:
                self.get_logger().warn(
                    f"Odometry timeout ({self._odom_timeout_sec:.2f}s); forcing zero control.",
                    throttle_duration_sec=2.0,
                )
            return

        self._nu_ref = self._command_in_base_link(self._last_cmd)
        if self._nu_ref is None:
            self._publish_zero_wrench_and_neutral_rc()
            return

        tau = self._compute_control()
        if tau is None:
            self._publish_zero_wrench_and_neutral_rc()
            return

        self._publish_wrench_and_rc(tau)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BrovCtrlSIWrench()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
