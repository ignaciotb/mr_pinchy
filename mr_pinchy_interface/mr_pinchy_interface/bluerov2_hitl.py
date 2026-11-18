#!/usr/bin/env python3
"""TF from the FRD base link to its FLU twin.

The BlueROV2's base_link is FRD (x forward, y right, z down), while the
controllers and teleop nodes work in the ROS-native FLU convention and expect
mr_pinchy/base_link_ekf_flu. On the real robot brov_odom_ekf publishes that
frame off its own EKF child frame; in sim nothing does, so this node fills in
the same relationship straight off base_link.
"""
from __future__ import annotations

import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from tf2_ros import TransformBroadcaster


class Bluerov2Hitl(Node):
    def __init__(self) -> None:
        super().__init__("bluerov2_hitl")

        self.declare_parameter("parent_frame", "mr_pinchy/base_link")
        self.declare_parameter("child_frame", "mr_pinchy/base_link_ekf_flu")
        self.declare_parameter("publish_rate_hz", 20.0)

        self._parent_frame = str(self.get_parameter("parent_frame").value)
        self._child_frame = str(self.get_parameter("child_frame").value)
        self._publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)

        self._tf_broadcaster = TransformBroadcaster(self)
        self.create_timer(1.0 / self._publish_rate_hz, self._publish_transform)

        self.get_logger().info(
            f"bluerov2_hitl: FRD '{self._parent_frame}' -> "
            f"FLU '{self._child_frame}' at {self._publish_rate_hz:.1f}Hz",
        )

    def _publish_transform(self) -> None:
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = self._parent_frame
        transform.child_frame_id = self._child_frame
        transform.transform.translation.x = 0.0
        transform.transform.translation.y = 0.0
        transform.transform.translation.z = 0.0
        # FRD -> FLU is a 180 degree rotation about +X with coincident origins.
        transform.transform.rotation.x = 1.0
        transform.transform.rotation.y = 0.0
        transform.transform.rotation.z = 0.0
        transform.transform.rotation.w = 0.0
        self._tf_broadcaster.sendTransform(transform)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = Bluerov2Hitl()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
