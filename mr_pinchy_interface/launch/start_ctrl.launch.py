from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import os
import xacro

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
    
    safe_controller = Node(
        package="mr_pinchy_interface",
        executable="brov_ctrl_safe",
        name="brov_ctrl_safe",
        output="screen",
    )

    ekf_odom = Node(
        package="mr_pinchy_interface",
        executable="brov_odom_ekf",
        name="brov_odom_ekf",
        output="screen",
    )
    
    urdf = os.path.join(get_package_share_directory("mr_pinchy_interface"), "urdf", "bluerov2_heavy.urdf.xacro")
    robot_description = xacro.process_file(urdf).toxml()
    robot_state_pub = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        parameters=[{"robot_description": robot_description}],
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
            ekf_odom,
            # waypoint_controller,
            twist_controller,
            safe_controller,
            robot_state_pub,
            rviz_node,
        ]
    )
