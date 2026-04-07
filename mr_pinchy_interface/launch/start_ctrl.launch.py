from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    default_rviz = PathJoinSubstitution(
        [FindPackageShare("mr_pinchy_interface"), "rviz", "watertank.rviz"]
    )

    rviz_config_arg = DeclareLaunchArgument(
        "rviz_config",
        default_value=default_rviz,
        description="Absolute path to RViz config file",
    )

    pid_node = Node(
        package="mr_pinchy_interface",
        executable="brov_ctrl_pid_wrench",
        name="brov_ctrl_pid",
        output="screen",
        parameters=[
            {
                "goal_topic": "/goal_pose",
                "odom_topic": "/mocap/mr_pinchy/odom",
                "rc_topic": "/mavros/rc/override",
                "map_frame": "map",
                "base_link_frame": "mr_pinchy/base_link",
            }
        ],
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", LaunchConfiguration("rviz_config")],
    )

    return LaunchDescription(
        [
            rviz_config_arg,
            pid_node,
            rviz_node,
        ]
    )
