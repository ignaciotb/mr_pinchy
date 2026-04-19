#!/usr/bin/env python3
from __future__ import annotations

import copy
import math
import time

import numpy as np
import rclpy
from geometry_msgs.msg import (
    PoseStamped,
    TransformStamped,
    TwistStamped,
    Vector3Stamped,
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
from mr_pinchy_interface.brov_wrench_to_pwm import wrench_to_rc_override

_ENU_TO_NED_ROTATION = np.array(
    [
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=float,
)


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


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


def _rpy_from_rotation_matrix(rotation: np.ndarray) -> tuple[float, float, float]:
    roll = math.atan2(rotation[2, 1], rotation[2, 2])
    pitch = math.asin(_clamp(-rotation[2, 0], -1.0, 1.0))
    yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    return roll, pitch, yaw


def _world_to_ned_rotation(frame_id: str) -> np.ndarray:
    normalized = frame_id.rsplit("/", 1)[-1].lower()
    if normalized == "ned" or normalized.endswith("_ned"):
        return np.eye(3, dtype=float)
    return _ENU_TO_NED_ROTATION


def _pose_rpy_in_ned(pose, *, world_frame: str) -> tuple[float, float, float]:
    rotation_world_body = _quat_to_rotation_matrix(
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    )
    rotation_ned_body = _world_to_ned_rotation(world_frame) @ rotation_world_body
    return _rpy_from_rotation_matrix(rotation_ned_body)


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


def odom_to_body_state(odom: Odometry) -> tuple[np.ndarray,  np.ndarray]:
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


def _vector3_to_array(msg) -> np.ndarray:
    return np.array([msg.x, msg.y, msg.z], dtype=float)


def _vector3_stamped(msg, *, frame_id: str, stamp=None) -> Vector3Stamped:
    vector = Vector3Stamped()
    vector.header.frame_id = frame_id
    if stamp is not None:
        vector.header.stamp = stamp
    vector.vector.x = msg.x
    vector.vector.y = msg.y
    vector.vector.z = msg.z
    return vector


def _pose_stamped_from_pose(pose, *, frame_id: str, stamp=None) -> PoseStamped:
    msg = PoseStamped()
    msg.header.frame_id = frame_id
    if stamp is not None:
        msg.header.stamp = stamp
    msg.pose = pose
    return msg


def _pose_stamped_from_transform(transform: TransformStamped) -> PoseStamped:
    msg = PoseStamped()
    msg.header.frame_id = transform.header.frame_id
    msg.header.stamp = transform.header.stamp
    msg.pose.position.x = transform.transform.translation.x
    msg.pose.position.y = transform.transform.translation.y
    msg.pose.position.z = transform.transform.translation.z
    msg.pose.orientation.x = transform.transform.rotation.x
    msg.pose.orientation.y = transform.transform.rotation.y
    msg.pose.orientation.z = transform.transform.rotation.z
    msg.pose.orientation.w = transform.transform.rotation.w
    return msg


def _transform_from_pose(
    pose,
    *,
    target_frame: str,
    source_frame: str,
    stamp=None,
) -> TransformStamped:
    transform = TransformStamped()
    transform.header.frame_id = target_frame
    transform.child_frame_id = source_frame
    if stamp is not None:
        transform.header.stamp = stamp
    transform.transform.translation.x = pose.position.x
    transform.transform.translation.y = pose.position.y
    transform.transform.translation.z = pose.position.z
    transform.transform.rotation.x = pose.orientation.x
    transform.transform.rotation.y = pose.orientation.y
    transform.transform.rotation.z = pose.orientation.z
    transform.transform.rotation.w = pose.orientation.w
    return transform


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


class BrovTwistController(Node):
    def __init__(self) -> None:
        super().__init__("brov_ctrl_si_wrench")

        self.declare_parameter("cmd_topic", "/pinchy/cmd_vel")
        self.declare_parameter("odom_topic", "/mocap/mr_pinchy/odom")
        self.declare_parameter("wrench_topic", "/cmd_wrench")
        self.declare_parameter("rc_topic", "/mavros/rc/override")
        self.declare_parameter("base_link_frame", "mr_pinchy/base_link")
        self.declare_parameter("base_link_flu_frame", "mr_pinchy/base_link_flu")

        cmd_topic = str(self.get_parameter("cmd_topic").value)
        odom_topic = str(self.get_parameter("odom_topic").value)
        wrench_topic = str(self.get_parameter("wrench_topic").value)
        rc_topic = str(self.get_parameter("rc_topic").value)
        self._base_link_frame = str(self.get_parameter("base_link_frame").value)
        self._base_link_flu_frame = str(self.get_parameter("base_link_flu_frame").value)

        self._publish_rate_hz = 20.0
        self._odom_timeout_sec = 0.3
        self._k_u = 8.0
        self._k_v = 8.0
        self._k_w = 8.0
        self._k_p = 2.0
        self._k_q = 2.0
        self._k_r = 2.0
        self._fx_max = 60.0
        self._fy_max = 60.0
        self._fz_max = 60.0
        self._tx_max = 20.0
        self._ty_max = 20.0
        self._tz_max = 20.0
        self._neutral_pwm = 1500
        self._delta_pwm = 100
        self._geometry = "60002"
        self._motor_direction_signs = (-1, -1, 1, 1, -1, 1, 1, -1)
        self._tf_timeout = Duration(seconds=0.05)

        self._gains = np.array(
            [
                self._k_u,
                self._k_v,
                self._k_w,
                self._k_p,
                self._k_q,
                self._k_r,
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

        self.create_timer(1.0 / self._publish_rate_hz, self._tick)
        self.get_logger().info(
            f"brov_ctrl_twist: cmd='{cmd_topic}', odom='{odom_topic}', "
            f"wrench='{wrench_topic}', rc='{rc_topic}', base_link='{self._base_link_frame}', "
            f"base_link_flu='{self._base_link_flu_frame}', rate={self._publish_rate_hz:.1f}Hz"
        )

    def _cmd_cb(self, msg: TwistStamped) -> None:
        self._last_cmd = msg

    def _odom_cb(self, msg: Odometry) -> None:
        self._last_odom = msg
        self._last_odom_rx_sec = time.monotonic()

    def _lookup_transform(self, target_frame: str, source_frame: str):
        return self._tf_buffer.lookup_transform(
            target_frame,
            source_frame,
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
        if source_frame == self._base_link_frame:
            return _vector3_to_array(msg)

        try:
            transform = self._lookup_transform(self._base_link_frame, source_frame)
        except TransformException as exc:
            self.get_logger().warn(
                f"TF {source_frame} -> {self._base_link_frame} unavailable: {exc}",
                throttle_duration_sec=2.0,
            )
            return None

        vector = _vector3_stamped(msg, frame_id=source_frame, stamp=stamp)
        transformed = do_transform_vector3(vector, copy.deepcopy(transform))
        return _vector3_to_array(transformed.vector)

    def _odom_in_base_link(self, odom: Odometry) -> tuple[np.ndarray, np.ndarray] | None:
        source_frame = getattr(odom, "child_frame_id", "")
        if not source_frame:
            raise ValueError("odometry child_frame_id is required")
        world_frame = getattr(getattr(odom, "header", None), "frame_id", "")
        stamp = getattr(getattr(odom, "header", None), "stamp", None)

        if source_frame == self._base_link_frame:
            pose_in_world = _pose_stamped_from_pose(
                odom.pose.pose,
                frame_id=world_frame,
                stamp=stamp,
            )
        else:
            try:
                base_in_source = _pose_stamped_from_transform(
                    self._lookup_transform(source_frame, self._base_link_frame)
                )
            except TransformException as exc:
                self.get_logger().warn(
                    f"TF {self._base_link_frame} -> {source_frame} unavailable: {exc}",
                    throttle_duration_sec=2.0,
                )
                return None

            odom_transform = _transform_from_pose(
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

        roll, pitch, yaw = _pose_rpy_in_ned(pose_in_world.pose, world_frame=world_frame)
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
        source_frame = msg.header.frame_id or self._base_link_flu_frame
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

    def _compute_control(self) -> np.ndarray | None:
        if self._nu_ref is None or self._last_odom is None:
            return None

        state = self._odom_in_base_link(self._last_odom)
        if state is None:
            return None
        eta, nu = state
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

    def _publish_zero_wrench_and_rc(self) -> None:
        self._publish_wrench_and_rc(np.zeros(6, dtype=float))

    def _tick(self) -> None:
        now = time.monotonic()

        if self._last_cmd is None:
            self._nu_ref = None
            self._publish_zero_wrench_and_rc()
            return

        if self._last_odom is None or (now - self._last_odom_rx_sec) > self._odom_timeout_sec:
            self._publish_zero_wrench_and_rc()
            if self._last_odom is not None:
                self.get_logger().warn(
                    f"Odometry timeout ({self._odom_timeout_sec:.2f}s); forcing zero control.",
                    throttle_duration_sec=2.0,
                )
            return

        self._nu_ref = self._command_in_base_link(self._last_cmd)
        if self._nu_ref is None:
            self._publish_zero_wrench_and_rc()
            return

        tau = self._compute_control()
        if tau is None:
            self._publish_zero_wrench_and_rc()
            return

        self._publish_wrench_and_rc(tau)


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
