#!/usr/bin/env python3
"""Safety filter for joystick teleop without global localization.

Inputs (all relative / body-frame):
  /mr_pinchy/cmd_vel_ref     TwistStamped, FLU      joystick reference
  /dvl/velocity              body velocity (FRD)    robot velocity v_r
  /dvl/imu                   gyro (FRD)             yaw rate omega
  /sonar_filter/point_cloud  PointCloud2 (sonar)    obstacle points
Outputs:
  /mr_pinchy/cmd_vel         TwistStamped, FLU      filtered command
  /mr_pinchy/odom_dvl        Odometry (twist only)  DVL + gyro repacked for brov_ctrl_twist_simple
  static TF                  base_link_ekf -> base_link_ekf_flu / sonar_link_ekf / dvl_link

The environment is assumed static: every sonar point is an obstacle. A soft-min
barrier over all points (JAX) gives one CBF half-plane, and a CBF-QP keeps the robot
circle clear of the points. Heave and yaw rate pass through unchanged. No global pose
is used anywhere; the odometry message carries body velocity only (pose stays
identity). See doc/relative_safety_filter.md.
"""
from __future__ import annotations

import math

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped, TwistStamped, TwistWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from sensor_msgs.msg import Imu, PointCloud2
from std_msgs.msg import Float64MultiArray
from tf2_ros import Buffer, StaticTransformBroadcaster, TransformException, TransformListener
from visualization_msgs.msg import Marker

from mr_pinchy_interface.brov_safe_relative_core import CbfQp, SoftminCbf
from mr_pinchy_interface.brov_utils import quat_to_rotation_matrix


def normalize_tf_frame_id(frame_id: str) -> str:
    return str(frame_id).lstrip("/")


def cloud_to_xyz(msg: PointCloud2) -> np.ndarray:
    """(N, 3) float array from an xyz float32 PointCloud2 (any point_step)."""
    offsets = {f.name: f.offset for f in msg.fields}
    dtype = np.dtype(
        {
            "names": ["x", "y", "z"],
            "formats": ["<f4"] * 3,
            "offsets": [offsets["x"], offsets["y"], offsets["z"]],
            "itemsize": msg.point_step,
        }
    )
    raw = np.frombuffer(msg.data, dtype=dtype)
    pts = np.stack([raw["x"], raw["y"], raw["z"]], axis=1).astype(float)
    return pts[np.isfinite(pts).all(axis=1)]


class SafeRelativeNode(Node):
    def __init__(self) -> None:
        super().__init__("safety_filter")

        # Topics / frames
        self.declare_parameter("input_cmd_topic", "/mr_pinchy/cmd_vel_ref")
        self.declare_parameter("output_cmd_topic", "/mr_pinchy/cmd_vel")
        self.declare_parameter("cloud_topic", "/sonar_filter/point_cloud")
        self.declare_parameter("dvl_topic", "/dvl/velocity")
        self.declare_parameter("imu_topic", "/dvl/imu")
        self.declare_parameter("base_link_flu_frame", "mr_pinchy/base_link_ekf_flu")
        self.declare_parameter("base_link_frd_frame", "mr_pinchy/base_link_ekf")
        self.declare_parameter("odom_topic", "/mr_pinchy/odom_dvl")
        self.declare_parameter("publish_static_tf", True)
        # Geometry / CBF
        self.declare_parameter("robot_radius", 0.6)
        self.declare_parameter("margin", 0.1)
        self.declare_parameter("alpha", 1.0)
        self.declare_parameter("kappa", 50.0)  # soft-min sharpness; h is below min_i h_i by <= log(N)/kappa
        self.declare_parameter("max_points", 2048)  # nearest points kept (fixed JAX array size)
        self.declare_parameter("v_max", 0.3)
        self.declare_parameter("slack_weight", 1000.0)
        self.declare_parameter("smooth_weight", 1.0)  # penalty on |u - u_prev|^2, damps jitter
        # Timing
        self.declare_parameter("rate_hz", 30.0)
        self.declare_parameter("cmd_timeout_sec", 0.3)
        self.declare_parameter("sensor_timeout_sec", 0.5)
        self.declare_parameter("obstacle_lost_timeout_sec", 1.0)
        # Obstacle points
        self.declare_parameter("max_range", 3.0)
        self.declare_parameter("max_abs_z", 1.0)

        g = lambda name: self.get_parameter(name).value  # noqa: E731
        self._flu_frame = normalize_tf_frame_id(g("base_link_flu_frame"))
        self._frd_frame = normalize_tf_frame_id(g("base_link_frd_frame"))
        self._robot_radius = float(g("robot_radius"))
        self._d_safe = self._robot_radius + float(g("margin"))
        self._cmd_timeout = float(g("cmd_timeout_sec"))
        self._sensor_timeout = float(g("sensor_timeout_sec"))
        self._lost_timeout = float(g("obstacle_lost_timeout_sec"))
        self._max_range = float(g("max_range"))
        self._max_abs_z = float(g("max_abs_z"))

        self._cbf = SoftminCbf(self._d_safe, float(g("alpha")), float(g("kappa")), int(g("max_points")))
        self._qp = CbfQp(float(g("v_max")), float(g("slack_weight")), float(g("smooth_weight")))
        self._u_prev = np.zeros(2)  # last published xy command (QP smoothing term)

        # Latest inputs and their receive times
        self._cmd: TwistStamped | None = None
        self._cmd_sec = -math.inf
        self._dvl: TwistWithCovarianceStamped | None = None
        self._dvl_sec = -math.inf
        self._gyro = np.zeros(3)
        self._points = np.zeros((0, 2))  # obstacle points in body FLU xy
        self._meas_sec = -math.inf  # last non-empty cloud

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._sonar_to_flu: tuple[np.ndarray, np.ndarray] | None = None  # (R, t), cached

        self._cmd_pub = self.create_publisher(TwistStamped, str(g("output_cmd_topic")), 10)
        self._odom_pub = self.create_publisher(Odometry, str(g("odom_topic")), 10)
        if bool(g("publish_static_tf")):
            self._static_tf = StaticTransformBroadcaster(self)
            self._static_tf.sendTransform(self._static_body_frames())
        self._robot_marker_pub = self.create_publisher(Marker, "/safety_filter/robot_marker", 10)
        self._debug_pub = self.create_publisher(Float64MultiArray, "/safety_filter/debug", 10)

        self.create_subscription(TwistStamped, str(g("input_cmd_topic")), self._cmd_cb, 10)
        self.create_subscription(TwistWithCovarianceStamped, str(g("dvl_topic")), self._dvl_cb, 10)
        self.create_subscription(Imu, str(g("imu_topic")), self._imu_cb, 10)
        self.create_subscription(PointCloud2, str(g("cloud_topic")), self._cloud_cb, 10)
        self.create_timer(1.0 / float(g("rate_hz")), self._tick)

        self.get_logger().info(
            f"safety_filter ready: d_safe={self._d_safe:.2f} m, alpha={g('alpha')}, kappa={g('kappa')}, "
            f"v_max={float(g('v_max'))}, cloud='{g('cloud_topic')}'"
        )

    # ------------------------------------------------------------------ inputs
    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _cmd_cb(self, msg: TwistStamped) -> None:
        self._cmd = msg
        self._cmd_sec = self._now()

    def _dvl_cb(self, msg: TwistWithCovarianceStamped) -> None:
        self._dvl = msg
        self._dvl_sec = self._now()

    def _imu_cb(self, msg: Imu) -> None:
        w = msg.angular_velocity
        self._gyro = np.array([w.x, w.y, w.z])

    def _static_body_frames(self) -> list[TransformStamped]:
        """Body-fixed frames (same numbers as brov_odom_ekf): FRD base -> FLU base,
        sonar (0.22 m ahead, 0.16 m up, pitched 30 deg down) and DVL."""
        def tf(child: str, xyz: tuple, q: tuple) -> TransformStamped:
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = self._frd_frame
            t.child_frame_id = child
            t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = xyz
            t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = q
            return t

        return [
            tf(self._flu_frame, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
            tf("mr_pinchy/sonar_link_ekf", (0.220, 0.0, -0.160), (0.0, -0.259, 0.0, 0.966)),
            tf("mr_pinchy/dvl_link", (0.002, -0.106, 0.141), (0.0, 0.0, 0.0, 1.0)),
        ]

    def _publish_odom(self) -> None:
        """Body velocity (FRD) for brov_ctrl_twist_simple; pose is left at identity."""
        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = "odom"
        odom.child_frame_id = self._frd_frame
        odom.pose.pose.orientation.w = 1.0
        odom.twist.twist.linear = self._dvl.twist.twist.linear
        odom.twist.twist.angular.x, odom.twist.twist.angular.y, odom.twist.twist.angular.z = self._gyro
        self._odom_pub.publish(odom)

    def _lookup_sonar_to_flu(self, sonar_frame: str) -> tuple[np.ndarray, np.ndarray] | None:
        if self._sonar_to_flu is None:
            try:
                tf = self._tf_buffer.lookup_transform(
                    self._flu_frame, sonar_frame, rclpy.time.Time(), timeout=Duration(seconds=0.1)
                )
            except TransformException as exc:
                self.get_logger().warn(f"TF {sonar_frame} -> {self._flu_frame}: {exc}", throttle_duration_sec=2.0)
                return None
            q, t = tf.transform.rotation, tf.transform.translation
            self._sonar_to_flu = (quat_to_rotation_matrix(q.x, q.y, q.z, q.w), np.array([t.x, t.y, t.z]))
        return self._sonar_to_flu

    def _track_active(self, now: float) -> bool:
        return (now - self._meas_sec) < self._lost_timeout

    def _cloud_cb(self, msg: PointCloud2) -> None:
        if msg.width * msg.height == 0:
            return
        tf = self._lookup_sonar_to_flu(normalize_tf_frame_id(msg.header.frame_id))
        if tf is None:
            return
        R, t = tf
        pts = (R @ cloud_to_xyz(msg).T).T + t
        keep = (np.linalg.norm(pts, axis=1) <= self._max_range) & (np.abs(pts[:, 2]) <= self._max_abs_z)
        if not keep.any():
            return
        self._points = pts[keep][:, :2]
        self._meas_sec = self._now()

    # ------------------------------------------------------------------ loop
    def _tick(self) -> None:
        now = self._now()
        active = self._track_active(now)
        if now - self._dvl_sec <= self._sensor_timeout:
            self._publish_odom()

        self._publish_robot_marker()

        if self._cmd is None or now - self._cmd_sec > self._cmd_timeout:
            self._u_prev = np.zeros(2)
            return  # deadman released: publish nothing, the twist controller goes neutral

        u_ref = np.array([self._cmd.twist.linear.x, self._cmd.twist.linear.y])
        h, slack, n = math.nan, 0.0, np.zeros(2)
        if active:
            n, b, h = self._cbf.constraint(self._points)
            u, slack = self._qp.solve(u_ref, n, b, self._u_prev)
            if u is None:
                self.get_logger().warn("CBF-QP failed; commanding zero", throttle_duration_sec=1.0)
                u = np.zeros(2)
        else:
            u = u_ref
        self._u_prev = u

        out = TwistStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = self._flu_frame
        out.twist.linear.x = float(u[0])
        out.twist.linear.y = float(u[1])
        out.twist.linear.z = self._cmd.twist.linear.z
        out.twist.angular = self._cmd.twist.angular
        self._cmd_pub.publish(out)

        dbg = Float64MultiArray()
        dbg.data = [h, slack, *u_ref, *u, *n, float(self._points.shape[0]), float(active)]
        self._debug_pub.publish(dbg)

    # ------------------------------------------------------------------ viz
    def _sphere(self, ns: str, radius: float, xy: np.ndarray, rgba: tuple) -> Marker:
        m = Marker()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = self._flu_frame
        m.ns, m.id, m.type, m.action = ns, 0, Marker.SPHERE, Marker.ADD
        m.pose.position.x, m.pose.position.y = float(xy[0]), float(xy[1])
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 2.0 * radius
        m.color.r, m.color.g, m.color.b, m.color.a = rgba
        return m

    def _publish_robot_marker(self) -> None:
        self._robot_marker_pub.publish(self._sphere("robot", self._robot_radius, np.zeros(2), (0.2, 0.6, 1.0, 0.35)))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SafeRelativeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
