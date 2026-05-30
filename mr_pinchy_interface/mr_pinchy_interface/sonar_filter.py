#!/usr/bin/env python3
from __future__ import annotations

from collections import deque
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial import KDTree
from sensor_msgs.msg import PointCloud2, PointField
from tf2_ros import Buffer, TransformException, TransformListener
import time

from mr_pinchy_interface.brov_utils import quat_to_rotation_matrix

_PC2_FIELDS = [
    PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
]


class SonarFilter(Node):
    def __init__(self) -> None:
        super().__init__("sonar_filter")

        self.declare_parameter("map_frame", "map")
        self.declare_parameter("tank_x_min", -2.5)
        self.declare_parameter("tank_x_max", 2.5)
        self.declare_parameter("tank_y_min", 0.0)
        self.declare_parameter("tank_y_max", 9.0)
        self.declare_parameter("tank_z_min", -3.0)
        self.declare_parameter("tank_z_max", 0.0)
        self.declare_parameter("wall_margin", 0.4)
        self.declare_parameter("sor_k", 6)
        self.declare_parameter("sor_alpha", 0.5)
        self.declare_parameter("sphere_radius", 3.0)
        self.declare_parameter("cluster_dist", 0.05)
        self.declare_parameter("min_cluster_size", 20)
        self.declare_parameter("watchdog_min_hz", 4.0)
        self.declare_parameter("watchdog_max_jitter_hz", 3.0)
        self.declare_parameter("watchdog_window", 20)
        self.declare_parameter("watchdog_check_period", 2.0)

        self._map_frame: str = self.get_parameter("map_frame").value
        x_min = self.get_parameter("tank_x_min").value
        x_max = self.get_parameter("tank_x_max").value
        y_min = self.get_parameter("tank_y_min").value
        y_max = self.get_parameter("tank_y_max").value
        z_min = self.get_parameter("tank_z_min").value
        z_max = self.get_parameter("tank_z_max").value
        margin = self.get_parameter("wall_margin").value
        self._sor_k: int = self.get_parameter("sor_k").value
        self._sor_alpha: float = self.get_parameter("sor_alpha").value
        self._sphere_radius: float = self.get_parameter("sphere_radius").value
        self._cluster_dist: float = self.get_parameter("cluster_dist").value
        self._min_cluster_size: int = self.get_parameter("min_cluster_size").value
        self._watchdog_min_hz: float = self.get_parameter("watchdog_min_hz").value
        self._watchdog_max_jitter_hz: float = self.get_parameter("watchdog_max_jitter_hz").value
        watchdog_window: int = self.get_parameter("watchdog_window").value
        watchdog_check_period: float = self.get_parameter("watchdog_check_period").value

        self._msg_times: deque = deque(maxlen=watchdog_window)

        self._keep_min = np.array(
            [x_min + margin, y_min + margin, z_min + margin], dtype=np.float64
        )
        self._keep_max = np.array(
            [x_max - margin, y_max - margin, z_max - margin], dtype=np.float64
        )

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._tf_timeout = Duration(seconds=0.1)

        self._pc_pub = self.create_publisher(PointCloud2, "~/point_cloud", 10)
        self.create_subscription(
            PointCloud2, "/sonar_node/point_cloud", self._pc_callback, 10
        )
        self.create_timer(watchdog_check_period, self._watchdog_cb)

        self.get_logger().info(
            f"Sonar filter ready — map_frame={self._map_frame}, "
            f"margin={margin}, sor_k={self._sor_k}, sor_alpha={self._sor_alpha}, "
            f"sphere_radius={self._sphere_radius}, "
            f"cluster_dist={self._cluster_dist}, min_cluster_size={self._min_cluster_size}"
        )

    def _watchdog_cb(self) -> None:
        now = time.time()

        if len(self._msg_times) < 2:
            self.get_logger().warn(
                "Sonar watchdog: no messages received from sonar",
                throttle_duration_sec=2.0,
            )
            return

        age = now - self._msg_times[-1]
        stale_threshold = 3.0 / self._watchdog_min_hz
        if age > stale_threshold:
            self.get_logger().warn(
                f"Sonar watchdog: last message was {age:.2f}s ago "
                f"(expected < {stale_threshold:.2f}s at {self._watchdog_min_hz} Hz)",
                throttle_duration_sec=2.0,
            )
            return

        intervals = np.diff(np.array(self._msg_times))
        freqs = 1.0 / intervals
        mean_hz = float(freqs.mean())
        std_hz = float(freqs.std())

        if mean_hz < self._watchdog_min_hz:
            self.get_logger().warn(
                f"Sonar watchdog: low frequency {mean_hz:.2f} Hz "
                f"(min {self._watchdog_min_hz} Hz)",
                throttle_duration_sec=2.0,
            )
        if std_hz > self._watchdog_max_jitter_hz:
            self.get_logger().warn(
                f"Sonar watchdog: unstable frequency — "
                f"std {std_hz:.2f} Hz, mean {mean_hz:.2f} Hz "
                f"(max jitter {self._watchdog_max_jitter_hz} Hz)",
                throttle_duration_sec=2.0,
            )

    def _pc_callback(self, msg: PointCloud2) -> None:
        if msg.width == 0:
            return
        time_start = time.time()
        self._msg_times.append(time_start)
        points = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, 3).copy()

        finite_mask = np.isfinite(points).all(axis=1)
        if not finite_mask.all():
            points = points[finite_mask]
        if points.shape[0] == 0:
            self._publish_empty(msg.header)
            return

        dists_from_sonar = np.linalg.norm(points, axis=1)
        sphere_mask = dists_from_sonar <= self._sphere_radius
        points = points[sphere_mask]
        if points.shape[0] == 0:
            self._publish_empty(msg.header)
            return

        try:
            tf = self._tf_buffer.lookup_transform(
                self._map_frame,
                msg.header.frame_id,
                rclpy.time.Time(),
                timeout=self._tf_timeout,
            )
        except TransformException as exc:
            self.get_logger().warn(
                f"TF lookup failed: {exc}", throttle_duration_sec=2.0
            )
            return

        t = tf.transform.translation
        q = tf.transform.rotation
        R = quat_to_rotation_matrix(q.x, q.y, q.z, q.w)
        translation = np.array([t.x, t.y, t.z], dtype=np.float64)

        pts_map = (R @ points.astype(np.float64).T).T + translation

        inside = np.all(pts_map >= self._keep_min, axis=1) & np.all(
            pts_map <= self._keep_max, axis=1
        )
        pts_map = pts_map[inside]
        if pts_map.shape[0] == 0:
            self._publish_empty(msg.header)
            return

        if pts_map.shape[0] > self._sor_k + 1:
            tree = KDTree(pts_map)
            dists, _ = tree.query(pts_map, k=self._sor_k + 1)
            mean_dists = dists[:, 1:].mean(axis=1)
            global_mean = mean_dists.mean()
            global_std = mean_dists.std()
            sor_mask = mean_dists <= global_mean + self._sor_alpha * global_std
            pts_map = pts_map[sor_mask]

        if pts_map.shape[0] == 0:
            self._publish_empty(msg.header)
            return

        if pts_map.shape[0] >= self._min_cluster_size:
            Z = linkage(pts_map, method="single")
            labels = fcluster(Z, t=self._cluster_dist, criterion="distance")
            unique, counts = np.unique(labels, return_counts=True)
            big = unique[counts >= self._min_cluster_size]
            if big.size > 0:
                pts_map = pts_map[np.isin(labels, big)]
            else:
                self._publish_empty(msg.header)
                return

        if pts_map.shape[0] == 0:
            self._publish_empty(msg.header)
            return

        pts_sonar = (R.T @ (pts_map - translation).T).T.astype(np.float32)

        cloud = PointCloud2()
        cloud.header = msg.header
        cloud.height = 1
        cloud.width = pts_sonar.shape[0]
        cloud.fields = _PC2_FIELDS
        cloud.is_bigendian = False
        cloud.point_step = 12
        cloud.row_step = 12 * pts_sonar.shape[0]
        cloud.data = pts_sonar.tobytes()
        cloud.is_dense = True
        self._pc_pub.publish(cloud)
        time_end = time.time()
        if time_end - time_start > 0.05:
            self.get_logger().warn(
                f"Sonar filter process took too long: {time_end - time_start:.3f} seconds",
                throttle_duration_sec=2.0,
            )

    def _publish_empty(self, header) -> None:
        cloud = PointCloud2()
        cloud.header = header
        cloud.height = 1
        cloud.width = 0
        cloud.fields = _PC2_FIELDS
        cloud.is_bigendian = False
        cloud.point_step = 12
        cloud.row_step = 0
        cloud.data = b""
        cloud.is_dense = True
        self._pc_pub.publish(cloud)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SonarFilter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
