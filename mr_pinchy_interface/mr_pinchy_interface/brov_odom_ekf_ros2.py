#!/usr/bin/env python3

import numpy as np
import rclpy
from geometry_msgs.msg import (
    PoseStamped,
    PoseWithCovarianceStamped,
    TransformStamped,
    TwistWithCovarianceStamped,
)
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu
from tf2_ros import StaticTransformBroadcaster

from mr_pinchy_interface.brov_odom_ekf import (
    base_link_z_axis_angle_rad,
    orientation_residual,
    quat_normalize,
)


class BrovOdomEkfRos2(Node):
    """Rule-based measurement gate that feeds robot_localization's ekf_node.

    The actual fusion is done by the stock robot_localization ekf_node
    (configured via YAML). This node keeps the original sanity checks: it
    gates raw mocap/IMU/DVL measurements and republishes only the accepted
    ones on dedicated topics that ekf_node subscribes to. Innovation gates
    compare each measurement against the latest /odometry/filtered estimate;
    the base_link z-axis tilt gate is absolute and needs no estimate.
    """

    def __init__(self):
        super().__init__("brov_odom_ekf_ros2")

        self.declare_parameter("pose_topic", "/mocap/mr_pinchy/pose")
        self.declare_parameter("imu_topic", "/dvl/imu")
        self.declare_parameter("dvl_topic", "/dvl/velocity")
        self.declare_parameter("filtered_odom_topic", "/odometry/filtered")
        self.declare_parameter("gated_pose_topic", "/gated/mocap/pose")
        self.declare_parameter("gated_imu_topic", "/gated/imu")
        self.declare_parameter("gated_dvl_topic", "/gated/dvl")
        self.declare_parameter("parent_frame", "mocap")
        self.declare_parameter("child_frame", "mr_pinchy/base_link_ekf")
        self.declare_parameter("child_frame_flu", "mr_pinchy/base_link_ekf_flu")
        self.declare_parameter("use_imu_gyro", True)
        self.declare_parameter("use_dvl", True)

        self.declare_parameter("position_std", 0.2)
        self.declare_parameter("orientation_std", 0.2)
        self.declare_parameter("gyro_std", 0.05)
        self.declare_parameter("dvl_std", 0.1)
        self.declare_parameter("max_position_innovation_m", 0.25)
        self.declare_parameter("max_orientation_innovation_rad", 0.75)
        self.declare_parameter("max_base_link_z_axis_angle_rad", 0.4)
        self.declare_parameter("max_gyro_innovation_rad_s", 2.0)
        self.declare_parameter("max_dvl_innovation_m_s", 1.0)

        self._pose_topic = str(self.get_parameter("pose_topic").value)
        self._imu_topic = str(self.get_parameter("imu_topic").value)
        self._dvl_topic = str(self.get_parameter("dvl_topic").value)
        self._filtered_odom_topic = str(self.get_parameter("filtered_odom_topic").value)
        self._gated_pose_topic = str(self.get_parameter("gated_pose_topic").value)
        self._gated_imu_topic = str(self.get_parameter("gated_imu_topic").value)
        self._gated_dvl_topic = str(self.get_parameter("gated_dvl_topic").value)
        self._parent_frame = str(self.get_parameter("parent_frame").value)
        self._child_frame = str(self.get_parameter("child_frame").value)
        self._child_frame_flu = str(self.get_parameter("child_frame_flu").value)
        self._use_imu_gyro = bool(self.get_parameter("use_imu_gyro").value)
        self._use_dvl = bool(self.get_parameter("use_dvl").value)

        self._position_std = float(self.get_parameter("position_std").value)
        self._orientation_std = float(self.get_parameter("orientation_std").value)
        self._gyro_std = float(self.get_parameter("gyro_std").value)
        self._dvl_std = float(self.get_parameter("dvl_std").value)
        self._max_position_innovation_m = float(
            self.get_parameter("max_position_innovation_m").value
        )
        self._max_orientation_innovation_rad = float(
            self.get_parameter("max_orientation_innovation_rad").value
        )
        self._max_base_link_z_axis_angle_rad = float(
            self.get_parameter("max_base_link_z_axis_angle_rad").value
        )
        self._max_gyro_innovation_rad_s = float(
            self.get_parameter("max_gyro_innovation_rad_s").value
        )
        self._max_dvl_innovation_m_s = float(
            self.get_parameter("max_dvl_innovation_m_s").value
        )

        # Latest fused estimate, used as the innovation reference. Pose is in
        # the world (parent) frame; twist is in the body (child) frame.
        self._have_estimate = False
        self._est_position = np.zeros(3, dtype=float)
        self._est_orientation = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        self._est_linear_velocity_body = np.zeros(3, dtype=float)
        self._est_angular_velocity_body = np.zeros(3, dtype=float)

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self._gated_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, self._gated_pose_topic, qos
        )
        if self._use_imu_gyro:
            self._gated_imu_pub = self.create_publisher(Imu, self._gated_imu_topic, qos)
        if self._use_dvl:
            self._gated_dvl_pub = self.create_publisher(
                TwistWithCovarianceStamped, self._gated_dvl_topic, qos
            )

        self.create_subscription(PoseStamped, self._pose_topic, self._pose_cb, qos)
        self.create_subscription(
            Odometry, self._filtered_odom_topic, self._filtered_cb, qos
        )
        if self._use_imu_gyro:
            self.create_subscription(Imu, self._imu_topic, self._imu_cb, qos)
        if self._use_dvl:
            self.create_subscription(
                TwistWithCovarianceStamped, self._dvl_topic, self._dvl_cb, qos
            )

        self._static_tf_broadcaster = StaticTransformBroadcaster(self)
        self._static_tf_timer = self.create_timer(
            1.0, self._publish_static_body_frames_once
        )

        self.get_logger().info(
            f"brov_odom_ekf_ros2: pose='{self._pose_topic}', "
            f"imu='{self._imu_topic}' (use={self._use_imu_gyro}), "
            f"dvl='{self._dvl_topic}' (use={self._use_dvl}), "
            f"filtered='{self._filtered_odom_topic}', "
            f"gated_pose='{self._gated_pose_topic}', "
            f"gated_imu='{self._gated_imu_topic}', "
            f"gated_dvl='{self._gated_dvl_topic}', "
            f"max_position_innovation={self._max_position_innovation_m:.3f}m, "
            f"max_orientation_innovation={self._max_orientation_innovation_rad:.3f}rad, "
            f"max_base_link_z_axis_angle={self._max_base_link_z_axis_angle_rad:.3f}rad, "
            f"max_gyro_innovation={self._max_gyro_innovation_rad_s:.3f}rad/s, "
            f"max_dvl_innovation={self._max_dvl_innovation_m_s:.3f}m/s"
        )

    @staticmethod
    def _within_gate(value, limit):
        return limit is None or limit <= 0.0 or value <= limit

    def _filtered_cb(self, msg):
        self._est_position = np.array(
            [
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z,
            ],
            dtype=float,
        )
        self._est_orientation = quat_normalize(
            np.array(
                [
                    msg.pose.pose.orientation.x,
                    msg.pose.pose.orientation.y,
                    msg.pose.pose.orientation.z,
                    msg.pose.pose.orientation.w,
                ],
                dtype=float,
            )
        )
        self._est_linear_velocity_body = np.array(
            [
                msg.twist.twist.linear.x,
                msg.twist.twist.linear.y,
                msg.twist.twist.linear.z,
            ],
            dtype=float,
        )
        self._est_angular_velocity_body = np.array(
            [
                msg.twist.twist.angular.x,
                msg.twist.twist.angular.y,
                msg.twist.twist.angular.z,
            ],
            dtype=float,
        )
        self._have_estimate = True

    def _pose_cb(self, msg):
        position = np.array(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
            dtype=float,
        )
        orientation = quat_normalize(
            np.array(
                [
                    msg.pose.orientation.x,
                    msg.pose.orientation.y,
                    msg.pose.orientation.z,
                    msg.pose.orientation.w,
                ],
                dtype=float,
            )
        )

        z_axis_angle_rad = base_link_z_axis_angle_rad(orientation)
        z_axis_ok = self._within_gate(
            z_axis_angle_rad, self._max_base_link_z_axis_angle_rad
        )

        if not self._have_estimate:
            # Bootstrap: only the absolute tilt gate applies until ekf_node
            # produces a first estimate.
            if z_axis_ok:
                self._publish_gated_pose(msg, position, orientation)
            else:
                self.get_logger().warn(
                    "Mocap rejected (z-axis tilt) before first estimate.",
                    throttle_duration_sec=1.0,
                )
            return

        position_residual = position - self._est_position
        orientation_residual_vec = orientation_residual(
            orientation, self._est_orientation
        )
        position_ok = self._within_gate(
            float(np.linalg.norm(position_residual)),
            self._max_position_innovation_m,
        )
        orientation_ok = self._within_gate(
            float(np.linalg.norm(orientation_residual_vec)),
            self._max_orientation_innovation_rad,
        ) and z_axis_ok

        if position_ok and orientation_ok:
            self._publish_gated_pose(msg, position, orientation)
        else:
            self.get_logger().warn(
                "Mocap measurement rejected.",
                throttle_duration_sec=1.0,
            )

    def _publish_gated_pose(self, msg, position, orientation):
        out = PoseWithCovarianceStamped()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self._parent_frame
        out.pose.pose.position.x = float(position[0])
        out.pose.pose.position.y = float(position[1])
        out.pose.pose.position.z = float(position[2])
        out.pose.pose.orientation.x = float(orientation[0])
        out.pose.pose.orientation.y = float(orientation[1])
        out.pose.pose.orientation.z = float(orientation[2])
        out.pose.pose.orientation.w = float(orientation[3])
        cov = np.zeros((6, 6), dtype=float)
        cov[0, 0] = cov[1, 1] = cov[2, 2] = self._position_std**2
        cov[3, 3] = cov[4, 4] = cov[5, 5] = self._orientation_std**2
        out.pose.covariance = cov.reshape(-1).tolist()
        self._gated_pose_pub.publish(out)

    def _imu_cb(self, msg):
        raw_angular_velocity = np.array(
            [
                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z,
            ],
            dtype=float,
        )
        alpha = 0.8
        previous = getattr(self, "_imu_gyro_lpf", None)
        if previous is None:
            filtered_angular_velocity = raw_angular_velocity
        else:
            filtered_angular_velocity = (
                alpha * raw_angular_velocity + (1.0 - alpha) * previous
            )
        self._imu_gyro_lpf = filtered_angular_velocity

        if self._have_estimate:
            residual = filtered_angular_velocity - self._est_angular_velocity_body
            accepted = self._within_gate(
                float(np.linalg.norm(residual)),
                self._max_gyro_innovation_rad_s,
            )
            if not accepted:
                self.get_logger().warn(
                    "Rejected IMU gyro outlier: "
                    f"max_gyro_innovation={self._max_gyro_innovation_rad_s:.3f}rad/s",
                    throttle_duration_sec=1.0,
                )
                return

        out = Imu()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = msg.header.frame_id or self._child_frame
        out.angular_velocity.x = float(filtered_angular_velocity[0])
        out.angular_velocity.y = float(filtered_angular_velocity[1])
        out.angular_velocity.z = float(filtered_angular_velocity[2])
        gyro_var = self._gyro_std**2
        out.angular_velocity_covariance = [
            gyro_var, 0.0, 0.0,
            0.0, gyro_var, 0.0,
            0.0, 0.0, gyro_var,
        ]
        # Mark orientation and linear acceleration as unavailable so ekf_node
        # does not fuse them (-1 on the first covariance element per sensor_msgs).
        out.orientation_covariance = [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        out.linear_acceleration_covariance = [
            -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ]
        self._gated_imu_pub.publish(out)

    def _dvl_cb(self, msg):
        linear_velocity_body = np.array(
            [
                msg.twist.twist.linear.x,
                msg.twist.twist.linear.y,
                msg.twist.twist.linear.z,
            ],
            dtype=float,
        )

        if self._have_estimate:
            residual = linear_velocity_body - self._est_linear_velocity_body
            accepted = self._within_gate(
                float(np.linalg.norm(residual)),
                self._max_dvl_innovation_m_s,
            )
            if not accepted:
                self.get_logger().warn(
                    "Rejected DVL velocity outlier: "
                    f"max_dvl_innovation={self._max_dvl_innovation_m_s:.3f}m/s",
                    throttle_duration_sec=1.0,
                )
                return

        out = TwistWithCovarianceStamped()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = msg.header.frame_id or self._child_frame
        out.twist.twist.linear.x = float(linear_velocity_body[0])
        out.twist.twist.linear.y = float(linear_velocity_body[1])
        out.twist.twist.linear.z = float(linear_velocity_body[2])
        cov = np.zeros((6, 6), dtype=float)
        cov[0, 0] = cov[1, 1] = cov[2, 2] = self._dvl_std**2
        cov[3, 3] = cov[4, 4] = cov[5, 5] = 1e6
        out.twist.covariance = cov.reshape(-1).tolist()
        self._gated_dvl_pub.publish(out)

    def _publish_static_body_frames_once(self) -> None:
        stamp = self.get_clock().now().to_msg()

        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = self._child_frame
        transform.child_frame_id = self._child_frame_flu
        transform.transform.translation.x = 0.0
        transform.transform.translation.y = 0.0
        transform.transform.translation.z = 0.0
        # FRD -> FLU is a 180 degree rotation about +X with coincident origins.
        transform.transform.rotation.x = 1.0
        transform.transform.rotation.y = 0.0
        transform.transform.rotation.z = 0.0
        transform.transform.rotation.w = 0.0
        sonar_transform = TransformStamped()
        sonar_transform.header.stamp = stamp
        sonar_transform.header.frame_id = self._child_frame
        sonar_transform.child_frame_id = "mr_pinchy/sonar_link"
        sonar_transform.transform.translation.x = 0.220
        sonar_transform.transform.translation.y = 0.0
        sonar_transform.transform.translation.z = -0.160
        sonar_transform.transform.rotation.x = 0.0
        sonar_transform.transform.rotation.y = -0.259
        sonar_transform.transform.rotation.z = 0.0
        sonar_transform.transform.rotation.w = 0.966
        dvl_transform = TransformStamped()
        dvl_transform.header.stamp = stamp
        dvl_transform.header.frame_id = self._child_frame
        dvl_transform.child_frame_id = "mr_pinchy/dvl_link"
        dvl_transform.transform.translation.x = 0.002
        dvl_transform.transform.translation.y = -0.106
        dvl_transform.transform.translation.z = 0.141
        dvl_transform.transform.rotation.x = 0.0
        dvl_transform.transform.rotation.y = 0.0
        dvl_transform.transform.rotation.z = 0.0
        dvl_transform.transform.rotation.w = 1.0
        self._static_tf_broadcaster.sendTransform(
            [transform, sonar_transform, dvl_transform]
        )


def main(args=None):
    rclpy.init(args=args)
    node = BrovOdomEkfRos2()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
