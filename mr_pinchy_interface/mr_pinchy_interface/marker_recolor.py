#!/usr/bin/env python3
"""Republish a Marker topic with a different colour.

Marker colours are baked into the message by whoever published them, and RViz
offers no override, so recolouring a recorded marker (e.g. while replaying a
rosbag) means relaying it. Set `alpha` to a negative value to keep whatever
alpha the incoming marker carries.
"""
from __future__ import annotations

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray


class MarkerRecolor(Node):
    def __init__(self) -> None:
        super().__init__("marker_recolor")

        self.declare_parameter("input_topic", "/safety_filter/robot_marker")
        self.declare_parameter("output_topic", "/safety_filter/robot_marker_recolored")
        self.declare_parameter("marker_array", False)
        self.declare_parameter("r", 1.0)
        self.declare_parameter("g", 0.85)
        self.declare_parameter("b", 0.1)
        self.declare_parameter("alpha", -1.0)

        input_topic: str = self.get_parameter("input_topic").value
        output_topic: str = self.get_parameter("output_topic").value
        msg_type = (
            MarkerArray if self.get_parameter("marker_array").value else Marker
        )
        self._r: float = self.get_parameter("r").value
        self._g: float = self.get_parameter("g").value
        self._b: float = self.get_parameter("b").value
        self._alpha: float = self.get_parameter("alpha").value

        self._pub = self.create_publisher(msg_type, output_topic, 10)
        self._sub = self.create_subscription(
            msg_type, input_topic, self._on_msg, 10
        )

        self.get_logger().info(
            f"recolouring '{input_topic}' -> '{output_topic}' as "
            f"rgb=({self._r}, {self._g}, {self._b}), "
            f"alpha={'unchanged' if self._alpha < 0.0 else self._alpha}"
        )

    def _recolor(self, marker: Marker) -> None:
        marker.color.r = self._r
        marker.color.g = self._g
        marker.color.b = self._b
        if self._alpha >= 0.0:
            marker.color.a = self._alpha
        # Per-point colours, when present, win over marker.color in RViz.
        for color in marker.colors:
            color.r = self._r
            color.g = self._g
            color.b = self._b
            if self._alpha >= 0.0:
                color.a = self._alpha

    def _on_msg(self, msg) -> None:
        if isinstance(msg, MarkerArray):
            for marker in msg.markers:
                self._recolor(marker)
        else:
            self._recolor(msg)
        self._pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MarkerRecolor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
