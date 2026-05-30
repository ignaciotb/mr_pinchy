from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="false",
        description="Use simulation clock",
    )
    autotune_arg = DeclareLaunchArgument(
        "autotune",
        default_value="true",
        description="Run the FF+PI autotune sequence at startup",
    )
    cmd_topic_arg = DeclareLaunchArgument(
        "cmd_topic",
        default_value="/mr_pinchy/cmd_vel",
        description="TwistStamped command input topic",
    )
    odom_topic_arg = DeclareLaunchArgument(
        "odom_topic",
        default_value="/mocap/mr_pinchy/odom_ekf",
        description="Odometry feedback topic",
    )
    manual_control_topic_arg = DeclareLaunchArgument(
        "manual_control_topic",
        default_value="/mavros/manual_control/send",
        description="ArduSub ManualControl output topic",
    )
    base_link_frame_arg = DeclareLaunchArgument(
        "base_link_frame",
        default_value="mr_pinchy/base_link_ekf",
        description="Body (FRD) frame used for velocity control",
    )

    autotune_node = Node(
        package="mr_pinchy_interface",
        executable="brov_ctrl_twist_simple_autotune",
        name="brov_ctrl_twist_simple_autotune",
        output="screen",
        emulate_tty=True,
        parameters=[{
            "use_sim_time": LaunchConfiguration("use_sim_time"),
            "autotune": LaunchConfiguration("autotune"),
            "cmd_topic": LaunchConfiguration("cmd_topic"),
            "odom_topic": LaunchConfiguration("odom_topic"),
            "manual_control_topic": LaunchConfiguration("manual_control_topic"),
            "base_link_frame": LaunchConfiguration("base_link_frame"),
        }],
    )

    return LaunchDescription([
        use_sim_time_arg,
        autotune_arg,
        cmd_topic_arg,
        odom_topic_arg,
        manual_control_topic_arg,
        base_link_frame_arg,
        autotune_node,
    ])
