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

    waypoint_controller = Node(
        package="mr_pinchy_interface",
        executable="brov_ctrl_waypoint",
        name="brov_ctrl_waypoint",
        output="screen",
    )

    twist_controller = Node(
        package="mr_pinchy_interface",
        executable="brov_ctrl_twist",
        name="brov_ctrl_twist",
        output="screen",
    )

    kf_odom = Node(
        package="mr_pinchy_interface",
        executable="brov_kf_odom",
        name="brov_kf_odom",
        output="screen",
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
            kf_odom,
            # waypoint_controller,
            # twist_controller,
            rviz_node,
        ]
    )
