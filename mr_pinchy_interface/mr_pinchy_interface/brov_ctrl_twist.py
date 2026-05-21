#!/usr/bin/env python3
from __future__ import annotations

import copy

import numpy as np
import rclpy
from geometry_msgs.msg import (
    TwistStamped,
    WrenchStamped,
)
from mavros_msgs.msg import OverrideRCIn
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from tf2_geometry_msgs import do_transform_pose_stamped, do_transform_vector3
from tf2_ros import Buffer, TransformException, TransformListener

from mr_pinchy_interface.brov_dynamics import bluerov2_model_terms
from mr_pinchy_interface.brov_utils import (
    body_wrench_message,
    pose_rpy_in_ned,
    pose_stamped_from_pose,
    pose_stamped_from_transform,
    transform_from_pose,
    vector3_stamped,
    vector3_to_array,
    wrench_to_rc_override,
)


def static_wrench_feedforward(eta: np.ndarray, nu_ref: np.ndarray) -> np.ndarray:
    terms = bluerov2_model_terms(eta, nu_ref)
    return np.asarray(
        terms["C"] @ terms["nu"] + terms["D"] @ terms["nu_rel"] + terms["g"],
        dtype=float,
    )


def velocity_feedback_wrench(
    nu: np.ndarray,
    nu_ref: np.ndarray,
    kp_gains: np.ndarray,
    kd_gains: np.ndarray | None = None,
    error_derivative: np.ndarray | None = None,
) -> np.ndarray:
    error = nu_ref - nu
    if kd_gains is None:
        kd_gains = np.zeros_like(kp_gains, dtype=float)
    if error_derivative is None:
        error_derivative = np.zeros_like(error, dtype=float)
    return np.asarray(kp_gains * error + kd_gains * error_derivative, dtype=float)


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


def normalize_tf_frame_id(frame_id: str) -> str:
    return str(frame_id).lstrip("/")


class BrovTwistController(Node):
    def __init__(self) -> None:
        super().__init__("brov_ctrl_si_wrench")

        self.declare_parameter("cmd_topic", "/mr_pinchy/cmd_vel")
        self.declare_parameter("odom_topic", "/mocap/mr_pinchy/odom_ekf")
        self.declare_parameter("wrench_topic", "/mr_pinchy/cmd_wrench")
        self.declare_parameter("rc_topic", "/mavros/rc/override")
        self.declare_parameter("base_link_frame", "mr_pinchy/base_link_ekf")
        self.declare_parameter("base_link_flu_frame", "mr_pinchy/base_link_ekf_flu")

        cmd_topic = str(self.get_parameter("cmd_topic").value)
        odom_topic = str(self.get_parameter("odom_topic").value)
        wrench_topic = str(self.get_parameter("wrench_topic").value)
        rc_topic = str(self.get_parameter("rc_topic").value)
        self._base_link_frame = normalize_tf_frame_id(self.get_parameter("base_link_frame").value)
        self._debug_log = False
        self._publish_rate_hz = 30.0
        self._odom_timeout_sec = 0.3
        self._cmd_timeout_sec = 0.3
        self._k_u = 30.0
        self._k_v = 30.0
        self._k_w_pos = 10.0
        self._k_w_neg = 20.0
        self._k_p = 2.0
        self._k_q = 2.0
        self._k_r = 4.0
        self._kd_u = 1.0
        self._kd_v = 1.0
        self._kd_w = 0.0
        self._kd_p = 0.0
        self._kd_q = 0.0
        self._kd_r = 0.1
        self._fx_max = 60.0
        self._fy_max = 60.0
        self._fz_max = 60.0
        self._tx_max = 20.0
        self._ty_max = 20.0
        self._tz_max = 20.0
        self._neutral_pwm = 1500
        self._delta_pwm = 300
        self._tf_timeout = Duration(seconds=0.05)

        self._kp_gains = np.array(
            [
                self._k_u,
                self._k_v,
                self._k_w_pos,
                self._k_p,
                self._k_q,
                self._k_r,
            ],
            dtype=float,
        )
        self._kd_gains = np.array(
            [
                self._kd_u,
                self._kd_v,
                self._kd_w,
                self._kd_p,
                self._kd_q,
                self._kd_r,
            ],
            dtype=float,
        )
        self._force_limits = np.array(
            [
                self._fx_max,
                self._fy_max,
                self._fz_max,
            ],
            dtype=float,
        )
        self._torque_limits = np.array(
            [
                self._tx_max,
                self._ty_max,
                self._tz_max,
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
        self._last_odom: Odometry | None = None
        self._last_odom_rx_sec: float = 0.0
        self._nu_ref: np.ndarray | None = None
        self._prev_velocity_error: np.ndarray | None = None
        self._prev_feedback_sec: float | None = None
        self._last_cmd_rx_sec: float = 0.0
        
        self.create_timer(1.0 / self._publish_rate_hz, self._tick)
        self.get_logger().info(
            f"brov_ctrl_twist: cmd='{cmd_topic}', odom='{odom_topic}', "
            f"wrench='{wrench_topic}', rc='{rc_topic}', base_link_frd='{self._base_link_frame}', "
            f"rate={self._publish_rate_hz:.1f}Hz"
        )

    def _cmd_cb(self, msg: TwistStamped) -> None:
        self._last_cmd = msg
        self._last_cmd_rx_sec = self.get_clock().now().nanoseconds * 1e-9

    def _odom_cb(self, msg: Odometry) -> None:
        self._last_odom = msg
        self._last_odom_rx_sec = self.get_clock().now().nanoseconds * 1e-9

    def _lookup_transform(self, target_frame: str, source_frame: str):
        return self._tf_buffer.lookup_transform(
            normalize_tf_frame_id(target_frame),
            normalize_tf_frame_id(source_frame),
            rclpy.time.Time(),
            timeout=self._tf_timeout,
        )

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
            transform = self._lookup_transform(self._base_link_frame, source_frame)
        except TransformException as exc:
            self.get_logger().warn(
                f"TF {source_frame} -> {self._base_link_frame} unavailable: {exc}",
                throttle_duration_sec=2.0,
            )
            return None

        vector = vector3_stamped(msg, frame_id=source_frame, stamp=stamp)
        transformed = do_transform_vector3(vector, copy.deepcopy(transform))
        return vector3_to_array(transformed.vector)

    def _get_fossen_state(self, odom: Odometry) -> tuple[np.ndarray, np.ndarray] | None:
        source_frame = normalize_tf_frame_id(getattr(odom, "child_frame_id", ""))
        if not source_frame:
            raise ValueError("odometry child_frame_id is required")
        world_frame = normalize_tf_frame_id(getattr(getattr(odom, "header", None), "frame_id", ""))
        stamp = getattr(getattr(odom, "header", None), "stamp", None)

        if source_frame == self._base_link_frame:
            pose_in_world = pose_stamped_from_pose(
                odom.pose.pose,
                frame_id=world_frame,
                stamp=stamp,
            )
        else:
            try:
                base_in_source = pose_stamped_from_transform(
                    self._lookup_transform(source_frame, self._base_link_frame)
                )
            except TransformException as exc:
                self.get_logger().warn(
                    f"TF {self._base_link_frame} -> {source_frame} unavailable: {exc}",
                    throttle_duration_sec=2.0,
                )
                return None

            odom_transform = transform_from_pose(
                odom.pose.pose,
                target_frame=world_frame,
                source_frame=source_frame,
                stamp=stamp,
            )
            pose_in_world = do_transform_pose_stamped(base_in_source, odom_transform)

        linear = self._transform_vector_to_base_link(
            odom.twist.twist.linear,
            source_frame=source_frame,
            stamp=stamp,
        )
        angular = self._transform_vector_to_base_link(
            odom.twist.twist.angular,
            source_frame=source_frame,
            stamp=stamp,
        )
        if linear is None or angular is None:
            return None

        roll, pitch, yaw = pose_rpy_in_ned(pose_in_world.pose, world_frame=world_frame)
        # eta in NED world and nu in FRD body
        eta = np.array(
            [
                pose_in_world.pose.position.x,
                pose_in_world.pose.position.y,
                pose_in_world.pose.position.z,
                roll,
                pitch,
                yaw,
            ],
            dtype=float,
        )
        nu = np.concatenate([linear, angular]).astype(float)
        return eta, nu

    def _command_in_base_link(self, msg: TwistStamped) -> np.ndarray | None:
        source_frame = normalize_tf_frame_id(msg.header.frame_id)
        linear = self._transform_vector_to_base_link(
            msg.twist.linear,
            source_frame=source_frame,
            stamp=msg.header.stamp,
        )
        angular = self._transform_vector_to_base_link(
            msg.twist.angular,
            source_frame=source_frame,
            stamp=msg.header.stamp,
        )
        if linear is None or angular is None:
            return None
        return np.concatenate([linear, angular]).astype(float)

    def _reset_feedback_state(self) -> None:
        self._prev_velocity_error = None
        self._prev_feedback_sec = None

    def _feedback_error_derivative(self, error: np.ndarray, now_sec: float) -> np.ndarray:
        if self._prev_velocity_error is None or self._prev_feedback_sec is None:
            derivative = np.zeros_like(error, dtype=float)
        else:
            dt = now_sec - self._prev_feedback_sec
            if dt <= 1e-6:
                derivative = np.zeros_like(error, dtype=float)
            else:
                derivative = (error - self._prev_velocity_error) / dt

        self._prev_velocity_error = error.copy()
        self._prev_feedback_sec = now_sec
        return derivative

    def _compute_control(self, feedback_time_sec: float | None = None) -> np.ndarray | None:
        if self._nu_ref is None or self._last_odom is None:
            self._reset_feedback_state()
            return None

        state = self._get_fossen_state(self._last_odom)
        if state is None:
            self._reset_feedback_state()
            return None
        eta, nu = state
        tank_range = np.array([[2.0, -1.5, 0.5], [6.0, 1.5, 3.0]])
        eta_array = np.array([eta[0], eta[1], eta[2]])
        if np.any(eta_array < tank_range[0]) or np.any(eta_array > tank_range[1]):
            self._reset_feedback_state()
            self.get_logger().warn(
                f"Out of tank range; forcing zero control.",
                throttle_duration_sec=1.0,
            )
            return None
        tau_ff = static_wrench_feedforward(eta, self._nu_ref)
        error = self._nu_ref - nu
        if feedback_time_sec is None:
            feedback_time_sec = self.get_clock().now().nanoseconds * 1e-9
        error_derivative = self._feedback_error_derivative(error, feedback_time_sec)
        kp_gains = self._kp_gains.copy()
        kp_gains[2] = self._k_w_neg if error[2] < 0 else self._k_w_pos
        tau_fb = velocity_feedback_wrench(
            nu,
            self._nu_ref,
            kp_gains,
            self._kd_gains,
            error_derivative,
        )
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
        )
        if self._debug_log:
            self.get_logger().info(
                f"wrench=({wrench_msg.wrench.force.x:.3f}, {wrench_msg.wrench.force.y:.3f}, {wrench_msg.wrench.force.z:.3f}, {wrench_msg.wrench.torque.x:.3f}, {wrench_msg.wrench.torque.y:.3f}, {wrench_msg.wrench.torque.z:.3f})",
            )
            self.get_logger().info(
                f"rc=({rc_msg.channels[0]}, {rc_msg.channels[1]}, {rc_msg.channels[2]}, {rc_msg.channels[3]}, {rc_msg.channels[4]}, {rc_msg.channels[5]})",
            )
        self._wrench_pub.publish(wrench_msg)
        self._rc_pub.publish(rc_msg)

    def _publish_zero_wrench_and_rc(self) -> None:
        self._publish_wrench_and_rc(np.zeros(6, dtype=float))

    def _tick(self) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9

        if self._last_cmd is None:
            self._nu_ref = None
            self._reset_feedback_state()
            self._publish_zero_wrench_and_rc()
            # if self._last_cmd is not None:
            #     self.get_logger().warn(
            #         f"Command timeout ({self._cmd_timeout_sec:.2f}s); forcing zero control.",
            #         throttle_duration_sec=2.0,
            #     )
            return

        if self._last_odom is None or (now - self._last_odom_rx_sec) > self._odom_timeout_sec:
            self._reset_feedback_state()
            self._publish_zero_wrench_and_rc()
            if self._last_odom is not None:
                self.get_logger().warn(
                    f"Odometry timeout ({self._odom_timeout_sec:.2f}s); forcing zero control.",
                    throttle_duration_sec=2.0,
                )
            return

        nu_cmd = self._command_in_base_link(self._last_cmd)
        if nu_cmd is None:
            self._reset_feedback_state()
            self._publish_zero_wrench_and_rc()
            return
        
        alpha = 0.1
        if self._nu_ref is None:
            self._nu_ref = np.zeros_like(nu_cmd)
        else:
            self._nu_ref = alpha * nu_cmd + (1 - alpha) * self._nu_ref

        tau = self._compute_control(now)
        if tau is None:
            self._reset_feedback_state()
            self._publish_zero_wrench_and_rc()
            return

        self._publish_wrench_and_rc(tau)


BrovCtrlSIWrench = BrovTwistController


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BrovTwistController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
