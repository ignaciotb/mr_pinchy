import rclpy
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from tf2_geometry_msgs import do_transform_pose_stamped
from rclpy.duration import Duration
from tf2_ros import Buffer, TransformException, TransformListener

from mr_pinchy_interface.brov_utils import (
    PIDAxis,
    quat_to_rpy,
)


class BrovWayPointTwistController(Node):
    def __init__(self) -> None:
        super().__init__("brov_ctrl_waypoint_twist")

        self.declare_parameter("goal_topic", "/goal_pose")
        self.declare_parameter("odom_topic", "/mocap/mr_pinchy/odom_ekf")
        self.declare_parameter("base_link_frame", "mr_pinchy/base_link_ekf_flu")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("cmd_topic", "/mr_pinchy/cmd_vel_ref")
        
        goal_topic = self.get_parameter("goal_topic").get_parameter_value().string_value
        odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value
        cmd_topic = self.get_parameter("cmd_topic").get_parameter_value().string_value
        self._base_link_frame = self.get_parameter("base_link_frame").get_parameter_value().string_value
        self._map_frame = self.get_parameter("map_frame").get_parameter_value().string_value
        self._publish_rate_hz = 30.0
        self._odom_timeout = 0.3

        linear_vel_limit = 0.15
        self._pid_x = PIDAxis(
            kp=0.5, ki=0.0, kd=0., integral_limit=1.0, output_limit=linear_vel_limit
        )
        self._pid_y = PIDAxis(
            kp=0.5, ki=0.0, kd=0., integral_limit=1.0, output_limit=linear_vel_limit
        )
        self._pid_z = PIDAxis(
            kp=0.5, ki=0.0, kd=0, integral_limit=1.0, output_limit=linear_vel_limit
        )
        
        angular_vel_limit = 0.3
        self._pid_yaw = PIDAxis(
            kp=0.5,
            ki=0.0,
            kd=0.0,
            integral_limit=1.0,
            output_limit=angular_vel_limit,
        )
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self._cmd_pub = self.create_publisher(TwistStamped, cmd_topic, qos)
        self.create_subscription(PoseStamped, goal_topic, self._goal_cb, 10)
        self.create_subscription(Odometry, odom_topic, self._odom_cb, 10)

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._goal_pose: PoseStamped | None = None
        self._goal_in_odom: PoseStamped | None = None
        self._last_odom: Odometry | None = None
        self._last_odom_rx_sec: float = 0.0
        self._last_tick_sec: float = 0.0

        self.create_timer(1.0 / self._publish_rate_hz, self._tick)
        
    def _goal_cb(self, msg: PoseStamped):
        self._goal_pose = msg
        self._goal_pose.pose.position.z = -1.2
        self._goal_in_odom = None
        self._reset_pid()
        self.get_logger().info(
            f"Received new goal pose: frame_id='{msg.header.frame_id}', "
            f"position=({msg.pose.position.x:.3f}, {msg.pose.position.y:.3f}, {msg.pose.position.z:.3f}), "
            f"orientation=({msg.pose.orientation.x:.3f}, {msg.pose.orientation.y:.3f}, {msg.pose.orientation.z:.3f}, {msg.pose.orientation.w:.3f})"
        )

    def _odom_cb(self, msg: Odometry) -> None:
        self._last_odom = msg
        self._last_odom_rx_sec = self.get_clock().now().nanoseconds / 1e9
    
    def _resolve_goal_in_odom(self) -> bool:
        if self._goal_in_odom is not None:
            return True
        if self._goal_pose is None or self._last_odom is None:
            return False

        goal_frame = self._goal_pose.header.frame_id or self._map_frame
        odom_frame = self._last_odom.header.frame_id
        if not odom_frame:
            return False

        src = PoseStamped()
        src.header.frame_id = goal_frame
        src.pose = self._goal_pose.pose

        if goal_frame == odom_frame:
            self._goal_in_odom = src
            return True

        try:
            tf = self._tf_buffer.lookup_transform(
                odom_frame, goal_frame, rclpy.time.Time(),
                timeout=Duration(seconds=0.05),
            )
            self._goal_in_odom = do_transform_pose_stamped(src, tf)
            return True
        except TransformException as exc:
            self._goal_in_odom = None
            self.get_logger().warn(
                f"TF {goal_frame} -> {odom_frame} unavailable: {exc}",
                throttle_duration_sec=2.0,
            )
            return False
    
    def _reset_pid(self) -> None:
        self._pid_x.reset()
        self._pid_y.reset()
        self._pid_z.reset()
        self._pid_yaw.reset()
        
    def _publish_zero_twist(self) -> None:
        twist = TwistStamped()
        twist.header.stamp = self.get_clock().now().to_msg()
        twist.header.frame_id = self._base_link_frame
        twist.twist.linear.x = 0.0
        twist.twist.linear.y = 0.0
        twist.twist.linear.z = 0.0
        twist.twist.angular.x = 0.0
        twist.twist.angular.y = 0.0
        twist.twist.angular.z = 0.0
        self._cmd_pub.publish(twist)

    def _tick(self) -> None:
        now_ns = self.get_clock().now().nanoseconds
        now_sec = now_ns / 1e9
        dt = max(now_sec - self._last_tick_sec, 1e-3)
        self._last_tick_sec = now_sec

        if self._goal_pose is None:
            self._publish_zero_twist()
            return

        odom_age = now_sec - self._last_odom_rx_sec
        if odom_age > self._odom_timeout:
            self._publish_zero_twist()
            self.get_logger().warn(
                f"Odometry timeout ({self._odom_timeout:.2f}s); forcing zero control.",
                throttle_duration_sec=2.0,
            )
            return
        
        if not self._resolve_goal_in_odom():
            self._publish_zero_twist()
            return

        goal = self._goal_in_odom.pose
        odom_pose = self._last_odom.pose.pose

        ex = goal.position.x - odom_pose.position.x
        ey = goal.position.y - odom_pose.position.y
        ez = goal.position.z - odom_pose.position.z

        _, _, goal_yaw = quat_to_rpy(
            goal.orientation.x, goal.orientation.y,
            goal.orientation.z, goal.orientation.w,
        )
        _, _, current_yaw = quat_to_rpy(
            odom_pose.orientation.x, odom_pose.orientation.y,
            odom_pose.orientation.z, odom_pose.orientation.w,
        )
        eyaw = goal_yaw - current_yaw

        surge_pid = self._pid_x.step(ex, dt)
        sway_pid = self._pid_y.step(ey, dt)
        heave_pid = self._pid_z.step(ez, dt)
        yaw_pid = self._pid_yaw.step(eyaw, dt, wrap_error=True)

        odom_frame = self._last_odom.header.frame_id
        stamp = self.get_clock().now().to_msg()
        twist = TwistStamped()
        twist.header.stamp = stamp
        twist.header.frame_id = odom_frame
        twist.twist.linear.x = surge_pid
        twist.twist.linear.y = sway_pid
        twist.twist.linear.z = heave_pid
        twist.twist.angular.z = yaw_pid
        self._cmd_pub.publish(twist)
    
def main(args=None) -> None:
    rclpy.init(args=args)
    node = BrovWayPointTwistController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()