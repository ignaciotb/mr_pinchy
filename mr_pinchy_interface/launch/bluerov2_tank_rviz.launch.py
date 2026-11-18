#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import os
import xacro
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import IncludeLaunchDescription

def generate_launch_description() -> LaunchDescription:
    default_rviz = PathJoinSubstitution(
        [FindPackageShare("mr_pinchy_interface"), "rviz", "watertank.rviz"]
    )

    # Set up rviz
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

    # The 6-DOF waypoint tool (bluerov_waypoint_rviz_plugin/WaypointNav) and its
    # InteractiveMarkers display are carried by the rviz config above, not by
    # this launch file -- rviz2 discovers tool plugins through pluginlib, so
    # there is nothing to pass here. The package does have to be built and
    # sourced in this workspace, otherwise rviz2 starts with a broken tool
    # button and logs "Failed to load library".
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", LaunchConfiguration("rviz_config")],
        parameters=[use_sim_time],
    )

    watertank_description = os.path.join(get_package_share_directory("mr_pinchy_interface"), "urdf", "watertank_description", "robots", "watertank_default.urdf.xacro")
    watertank_description = xacro.process_file(watertank_description).toxml()
    watertank_state_pub = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="watertank_state_publisher",
        parameters=[{"robot_description": watertank_description}, use_sim_time],
        remappings=[("robot_description", "tank_description")],
    )

    # Both pinchy models come from the same xacro; the base link name is the
    # only thing that differs, so it is expanded at launch time (not here) and
    # comes from a launch argument.
    pinchy_xacro = os.path.join(get_package_share_directory("mr_pinchy_interface"), "urdf", "bluerov2_heavy.urdf.xacro")

    def pinchy_description(base_link_name_arg_name):
        return ParameterValue(
            Command(
                [
                    "xacro ",
                    pinchy_xacro,
                    " base_link_name:=",
                    LaunchConfiguration(base_link_name_arg_name),
                ]
            ),
            value_type=str,
        )

    # Sim pinchy
    sim_base_link_name_arg = DeclareLaunchArgument(
        "sim_base_link_name",
        default_value="BlueROV2_Heavy/base_link",
        description="Base link / TF frame of the simulated BlueROV2 Heavy",
    )

    pinchy_state_pub = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="pinchy_state_publisher",
        parameters=[{"robot_description": pinchy_description("sim_base_link_name")}, use_sim_time],
        remappings=[("robot_description", "pinchy_description")],
    )

    # Real pinchy
    real_base_link_name_arg = DeclareLaunchArgument(
        "real_base_link_name",
        default_value="mr_pinchy/base_link_ekf_flu",
        description="Base link / TF frame of the real BlueROV2 Heavy",
    )

    real_pinchy_state_pub = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="real_pinchy_state_publisher",
        parameters=[{"robot_description": pinchy_description("real_base_link_name")}, use_sim_time],
        remappings=[("robot_description", "real_pinchy_description")],
    )

    return LaunchDescription(
        [
            rviz_config_arg,
            use_sim_time_arg,
            watertank_state_pub,
            sim_base_link_name_arg,
            pinchy_state_pub,
            real_base_link_name_arg,
            real_pinchy_state_pub,
            rviz_node,
        ]
    )

