from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import math
import os
import xacro


def generate_launch_description() -> LaunchDescription:
    default_rviz = PathJoinSubstitution(
        [FindPackageShare("mr_pinchy_interface"), "rviz", "overlay.rviz"]
    )

    rviz_config_arg = DeclareLaunchArgument(
        "rviz_config",
        default_value=default_rviz,
        description="Absolute path to RViz config file",
    )

    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="false",
        description="Use simulation clock",
    )
    use_sim_time = {"use_sim_time": LaunchConfiguration("use_sim_time")}

    overlay_node = Node(
        package="mr_pinchy_interface",
        executable="overlay_rviz",
        name="overlay_calib",
        output="screen",
        parameters=[use_sim_time],
    )

    # map -> mocap: identity translation, RPY [-180, 0, 90] deg
    # (quaternion xyzw [0.707, 0.707, 0, 0]).
    map_to_mocap = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="map_to_mocap",
        arguments=[
            "--frame-id", "map", "--child-frame-id", "mocap",
            "--roll", str(-math.pi), "--pitch", "0.0", "--yaw", str(math.pi / 2),
        ],
        parameters=[use_sim_time],
    )

    watertank_description = os.path.join(
        get_package_share_directory("mr_pinchy_interface"),
        "urdf", "watertank_description", "robots", "watertank_default.urdf.xacro",
    )
    watertank_description = xacro.process_file(watertank_description).toxml()
    watertank_state_pub = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="watertank_state_publisher",
        parameters=[{"robot_description": watertank_description}, use_sim_time],
        remappings=[("robot_description", "tank_description")],
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", LaunchConfiguration("rviz_config")],
        parameters=[use_sim_time],
    )

    return LaunchDescription(
        [
            rviz_config_arg,
            use_sim_time_arg,
            map_to_mocap,
            overlay_node,
            watertank_state_pub,
            rviz_node,
        ]
    )
