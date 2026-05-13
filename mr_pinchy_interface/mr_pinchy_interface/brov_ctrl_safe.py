#!/usr/bin/env python3
from __future__ import annotations

import copy
import time

import cvxpy as cp
import flax
import jax
import jax.numpy as jnp
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from tf2_geometry_msgs import do_transform_pose_stamped, do_transform_vector3
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker

from mr_pinchy_interface.brov_utils import (
    vector3_stamped,
    vector3_to_array,
)

jax.config.update("jax_enable_x64", True)
device = "cuda" if jax.lib.xla_bridge.get_backend().platform == "cuda" else "cpu"
jax.config.update("jax_platform_name", device)


ROBOT_RADIUS = 0.4
OBS_RADIUS = 0.5
GAMMA_COLAV = 5.0
kappa = 100.0


def normalize_tf_frame_id(frame_id: str) -> str:
    return str(frame_id).lstrip("/")


def single_integrator_dynamics(x):
    # state: [pos_x, pos_y, pos_z], action: [vel_x, vel_y, vel_z]
    f = jnp.zeros(3)
    G = jnp.eye(3)
    return f, G


def h_func_colav(x, obs_pos, safety_radius=ROBOT_RADIUS + OBS_RADIUS):
    diff = x[:3] - obs_pos
    h = jnp.sum(diff**2) - safety_radius**2
    return h


def cbf_constr_single(x, obj_pos, obj_vel, safety_radius=ROBOT_RADIUS + OBS_RADIUS):
    # Relative degree 1; assuming the obstacle has single-integrator dynamics.
    f, G = single_integrator_dynamics(x)
    h = h_func_colav(x, obj_pos, safety_radius)
    dh_dx, dh_do = jax.grad(h_func_colav, argnums=(0, 1))(x, obj_pos, safety_radius)
    return (dh_dx @ G), (dh_dx @ f + dh_do @ obj_vel), h


@jax.jit
def cbf_constr_sdf(
    x,
    all_obj_pos,
    all_obj_vel,
    safety_radius=ROBOT_RADIUS + OBS_RADIUS,
):
    # Smooth-min approximation using log-sum-exp.
    all_term1i, all_term2i, all_hi = jax.vmap(
        cbf_constr_single,
        in_axes=(None, 0, 0, None),
    )(x, all_obj_pos, all_obj_vel, safety_radius)
    weights_unnormalized = jnp.exp(-kappa * all_hi)
    weight_sum = jnp.sum(weights_unnormalized)
    safe_weight_sum = jnp.where(weight_sum > 0, weight_sum, 1.0)
    all_lambdas = weights_unnormalized / safe_weight_sum
    h = jnp.where(weight_sum > 0, -1 / kappa * jnp.log(safe_weight_sum), 0.0)
    a_weighted = jnp.sum(all_lambdas[:, None] * all_term1i, axis=0)
    drift = jnp.sum(all_lambdas * all_term2i)
    a = jnp.where(weight_sum > 0, a_weighted, jnp.zeros_like(a_weighted))
    b = jnp.where(weight_sum > 0, -drift - GAMMA_COLAV * h, -1.0)
    return a, b


@flax.struct.dataclass
class State:
    robot_state: jnp.ndarray  # [x, y, z, vx, vy, vz]
    all_obs_states: jnp.ndarray  # shape (N, 6): [x, y, z, vx, vy, vz] for each obstacle


def filter_single_integrator_velocity(
    robot_position: np.ndarray,
    reference_velocity: np.ndarray,
    obstacle_positions: np.ndarray,
    *,
    obstacle_velocities: np.ndarray | None = None,
    safety_radius: float = ROBOT_RADIUS + OBS_RADIUS,
) -> np.ndarray | None:
    robot_position = np.asarray(robot_position, dtype=float).reshape(3)
    reference_velocity = np.asarray(reference_velocity, dtype=float).reshape(3)
    obstacle_positions = np.asarray(obstacle_positions, dtype=float).reshape(-1, 3)
    if obstacle_velocities is None:
        obstacle_velocities = np.zeros_like(obstacle_positions, dtype=float)
    else:
        obstacle_velocities = np.asarray(obstacle_velocities, dtype=float).reshape(-1, 3)

    a_jax, b_jax = cbf_constr_sdf(
        jnp.asarray(robot_position),
        jnp.asarray(obstacle_positions),
        jnp.asarray(obstacle_velocities),
        float(safety_radius),
    )
    a = np.asarray(a_jax, dtype=float).reshape(3)
    b = float(np.asarray(b_jax, dtype=float))
    u = cp.Variable(3)
    slack = cp.Variable(1, nonneg=True)
    cost = cp.Minimize(0.5 * cp.sum_squares(u - reference_velocity) + 1000.0 * slack)
    constraint = []
    input_limit = 0.2
    constraint.append(a @ u >= b + slack)
    constraint.append(u >= -input_limit)
    constraint.append(u <= input_limit)
    problem = cp.Problem(cost, constraint)
    problem.solve(solver=cp.OSQP, warm_start=True)
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or u.value is None:
        print(f"QP failed: {problem.status}")
        return None
    return np.asarray(u.value, dtype=float).reshape(3)


class BrovCtrlSafe(Node):
    def __init__(self) -> None:
        super().__init__("brov_ctrl_safe")

        self.declare_parameter("input_cmd_topic", "/mr_pinchy/cmd_vel_teleop")
        self.declare_parameter("output_cmd_topic", "/mr_pinchy/cmd_vel")
        self.declare_parameter("odom_topic", "/mocap/mr_pinchy/odom_ekf")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_link_flu_frame", "mr_pinchy/base_link_ekf_flu")
        self.declare_parameter("obstacle_position", [0.0, 4.0, -1.0])
        self.declare_parameter("obstacle_radius", OBS_RADIUS)
        self.declare_parameter("robot_radius", ROBOT_RADIUS)
        self.declare_parameter("publish_rate_hz", 30.0)
        self.declare_parameter("cmd_timeout_sec", 0.3)
        self.declare_parameter("odom_timeout_sec", 0.3)
        self.declare_parameter("obstacle_marker_topic", "/mr_pinchy/safety/obstacle_marker")

        input_cmd_topic = str(self.get_parameter("input_cmd_topic").value)
        output_cmd_topic = str(self.get_parameter("output_cmd_topic").value)
        odom_topic = str(self.get_parameter("odom_topic").value)
        marker_topic = str(self.get_parameter("obstacle_marker_topic").value)
        self._map_frame = normalize_tf_frame_id(self.get_parameter("map_frame").value)
        self._base_link_flu_frame = normalize_tf_frame_id(
            self.get_parameter("base_link_flu_frame").value
        )
        self._obstacle_position = np.asarray(
            self.get_parameter("obstacle_position").value,
            dtype=float,
        ).reshape(3)
        self._obstacle_radius = float(self.get_parameter("obstacle_radius").value)
        self._robot_radius = float(self.get_parameter("robot_radius").value)
        self._safety_radius = self._robot_radius + self._obstacle_radius
        self._publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self._cmd_timeout_sec = float(self.get_parameter("cmd_timeout_sec").value)
        self._odom_timeout_sec = float(self.get_parameter("odom_timeout_sec").value)
        self._tf_timeout = Duration(seconds=0.05)

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self._cmd_pub = self.create_publisher(TwistStamped, output_cmd_topic, qos)
        self._obstacle_marker_pub = self.create_publisher(Marker, marker_topic, qos)
        self.create_subscription(TwistStamped, input_cmd_topic, self._cmd_cb, qos)
        self.create_subscription(Odometry, odom_topic, self._odom_cb, qos)

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._last_cmd: TwistStamped | None = None
        self._last_odom: Odometry | None = None
        self._last_odom_rx_sec: float = 0.0

        self.create_timer(1.0 / self._publish_rate_hz, self._tick)
        self.create_timer(0.5, self._publish_obstacle_marker)
        self.get_logger().info(
            f"brov_ctrl_safe: input='{input_cmd_topic}', output='{output_cmd_topic}', "
            f"odom='{odom_topic}', map_frame='{self._map_frame}', "
            f"base_link_flu='{self._base_link_flu_frame}', "
            f"obstacle=({self._obstacle_position[0]:.3f}, "
            f"{self._obstacle_position[1]:.3f}, {self._obstacle_position[2]:.3f}), "
            f"obstacle_radius={self._obstacle_radius:.3f}m, "
            f"robot_radius={self._robot_radius:.3f}m"
        )

    def _cmd_cb(self, msg: TwistStamped) -> None:
        self._last_cmd = msg
        self.get_logger().info(
            f"Cmd velocity linear x: {msg.twist.linear.x}, linear y: {msg.twist.linear.y}, linear z: {msg.twist.linear.z}",
        )

    def _odom_cb(self, msg: Odometry) -> None:
        self._last_odom = msg
        self._last_odom_rx_sec = time.monotonic()

    def _lookup_transform(self, target_frame: str, source_frame: str):
        return self._tf_buffer.lookup_transform(
            normalize_tf_frame_id(target_frame),
            normalize_tf_frame_id(source_frame),
            rclpy.time.Time(),
            timeout=self._tf_timeout,
        )

    def _transform_vector(
        self,
        vector_msg,
        *,
        target_frame: str,
        source_frame: str,
        stamp=None,
    ) -> np.ndarray | None:
        target_frame = normalize_tf_frame_id(target_frame)
        source_frame = normalize_tf_frame_id(source_frame)
        if target_frame == source_frame:
            return vector3_to_array(vector_msg)

        try:
            transform = self._lookup_transform(target_frame, source_frame)
        except TransformException as exc:
            self.get_logger().warn(
                f"TF {source_frame} -> {target_frame} unavailable: {exc}",
                throttle_duration_sec=2.0,
            )
            return None

        vector = vector3_stamped(vector_msg, frame_id=source_frame, stamp=stamp)
        transformed = do_transform_vector3(vector, copy.deepcopy(transform))
        return vector3_to_array(transformed.vector)

    def _robot_position_in_map(self, odom: Odometry) -> np.ndarray | None:
        odom_frame = normalize_tf_frame_id(getattr(odom.header, "frame_id", ""))
        if not odom_frame:
            self.get_logger().warn(
                "Odometry header.frame_id is required for CBF safety filtering.",
                throttle_duration_sec=2.0,
            )
            return None

        pose = PoseStamped()
        pose.header.frame_id = odom_frame
        pose.header.stamp = odom.header.stamp
        pose.pose = odom.pose.pose
        if odom_frame != self._map_frame:
            try:
                transform = self._lookup_transform(self._map_frame, odom_frame)
            except TransformException as exc:
                self.get_logger().warn(
                    f"TF {odom_frame} -> {self._map_frame} unavailable: {exc}",
                    throttle_duration_sec=2.0,
                )
                return None
            pose = do_transform_pose_stamped(pose, copy.deepcopy(transform))

        return np.array(
            [
                pose.pose.position.x,
                pose.pose.position.y,
                pose.pose.position.z,
            ],
            dtype=float,
        )

    def _safe_linear_velocity_in_map(self, cmd: TwistStamped, odom: Odometry) -> np.ndarray | None:
        cmd_frame = normalize_tf_frame_id(cmd.header.frame_id)
        assert cmd_frame is not None

        reference_velocity = self._transform_vector(
            cmd.twist.linear,
            target_frame=self._map_frame,
            source_frame=cmd_frame,
            stamp=cmd.header.stamp,
        )
        if reference_velocity is None:
            self.get_logger().warn(
                f"Command velocity transformation failed; publishing zero command.",
                throttle_duration_sec=2.0,
            )
            return None

        robot_position = self._robot_position_in_map(odom)
        if robot_position is None:
            self.get_logger().warn(
                f"Robot position transformation failed; publishing zero command.",
                throttle_duration_sec=2.0,
            )
            return None

        return filter_single_integrator_velocity(
            robot_position,
            reference_velocity,
            self._obstacle_position.reshape(1, 3),
            obstacle_velocities=np.zeros((1, 3), dtype=float),
            safety_radius=self._safety_radius,
        )

    def _filtered_command(self, cmd: TwistStamped, odom: Odometry) -> TwistStamped | None:
        safe_map_velocity = self._safe_linear_velocity_in_map(cmd, odom)
        if safe_map_velocity is None:
            return None

        safe_body_velocity = self._transform_vector(
            type(
                "Vector3Like",
                (),
                {
                    "x": float(safe_map_velocity[0]),
                    "y": float(safe_map_velocity[1]),
                    "z": float(safe_map_velocity[2]),
                },
            )(),
            target_frame=self._base_link_flu_frame,
            source_frame=self._map_frame,
            stamp=cmd.header.stamp,
        )
        if safe_body_velocity is None:
            return None

        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._base_link_flu_frame
        msg.twist.linear.x = float(safe_body_velocity[0])
        msg.twist.linear.y = float(safe_body_velocity[1])
        msg.twist.linear.z = float(safe_body_velocity[2])
        msg.twist.angular = copy.deepcopy(cmd.twist.angular)
        return msg

    def _zero_command(self) -> TwistStamped:
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._base_link_flu_frame
        return msg

    def _publish_obstacle_marker(self) -> None:
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = self._map_frame
        marker.ns = "mr_pinchy_safety"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(self._obstacle_position[0])
        marker.pose.position.y = float(self._obstacle_position[1])
        marker.pose.position.z = float(self._obstacle_position[2])
        marker.pose.orientation.w = 1.0
        diameter = 2.0 * self._obstacle_radius
        marker.scale.x = diameter
        marker.scale.y = diameter
        marker.scale.z = diameter
        marker.color.r = 1.0
        marker.color.g = 0.15
        marker.color.b = 0.05
        marker.color.a = 0.45
        self._obstacle_marker_pub.publish(marker)

    def _tick(self) -> None:
        now = time.monotonic()
        self._publish_obstacle_marker()

        if self._last_cmd is None:
            self._cmd_pub.publish(self._zero_command())
            return

        if self._last_odom is None or (now - self._last_odom_rx_sec) > self._odom_timeout_sec:
            self._cmd_pub.publish(self._zero_command())
            if self._last_odom is not None:
                self.get_logger().warn(
                    f"Odometry timeout ({self._odom_timeout_sec:.2f}s); publishing zero command.",
                    throttle_duration_sec=2.0,
                )
            return

        filtered = self._filtered_command(self._last_cmd, self._last_odom)
        if filtered is None:
            self._cmd_pub.publish(self._zero_command())
            self.get_logger().warn(
                "Safety filter failed; publishing zero command.",
                throttle_duration_sec=2.0,
            )
            return

        self._cmd_pub.publish(filtered)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BrovCtrlSafe()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
