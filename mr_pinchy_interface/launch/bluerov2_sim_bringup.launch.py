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

    # ROS <-> Unity bridge
    ros_tcp_endpoint_node = Node(
        package="ros_tcp_endpoint",
        executable="default_server_endpoint",
        name="ros_tcp_endpoint",
        output="screen",
        parameters=[{"ROS_IP": "127.0.0.1"}],
    )
 
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use simulation clock",
    ) 
    
    use_sim_time = {"use_sim_time": LaunchConfiguration("use_sim_time")}

    # Set up bluerov sitl
    pkg_share = get_package_share_directory("bluerov_sitl")
    default_config = os.path.join(pkg_share, "config", "bluerov2_heavy.yaml")
    config_arg_sitl = DeclareLaunchArgument(
        "config_file_sitl",
        default_value=default_config,
        description="YAML file with bluerov_sitl node parameters",
    )

    sitl_node = Node(
        package="bluerov_sitl",
        executable="sitl_node",
        name="bluerov_sitl",
        output="screen",
        parameters=[LaunchConfiguration("config_file_sitl")],
    )


    # Controller nodes

    # keyboard_teleop_node = Node(
    #     package="mr_pinchy_interface",
    #     executable="bluerov2_keyboard_teleop",
    #     name="bluerov2_keyboard_teleop",
    #     output="screen",
    #     parameters=[use_sim_time],
    # )

    # Which mavros the velocity backend writes its RC override to. The SITL and
    # the real vehicle each have their own mavros namespace, so rc_topic is the knob
    # that decides which of the two the twist on cmd_topic actually drives.
    bluerov2_vel_rc_node = Node(
        package="mr_pinchy_interface",
        executable="bluerov2_vel_rc",
        name="bluerov2_vel_rc_node_sim",
        output="screen",
        parameters=[{"rc_topic": "/BlueROV2_Heavy/mavros/rc/override",
                     "cmd_topic": "/BlueROV2_Heavy/mr_pinchy/cmd_vel",
                     "input_is_flu": True},   # True for sim, false for real vehicle
                    # Must share the wp_ctrl clock: cmd_timeout_sec is measured
                    # against whatever clock this node runs on, and the twists it
                    # waits for are produced by a sim-time timer.
                    use_sim_time],
    )
   
    joy_node_sim = Node(package='joy', executable='joy_node', name='joy_node_sim',
            parameters=[{
                'deadzone': 0.05,
                'autorepeat_rate': 20.0}]
    )
    
    # NOTE — the sign convention here is FLU, matching input_is_flu:True above.
    # bluerov2_real_bringup.launch.py carries the SAME axes and magnitudes with
    # sway/heave/yaw NEGATED, because the real vehicle's cmd_topic is FRD. Keep
    # the two in sync by hand; do not "fix" one to match the other.
    teleop_node_sim = Node(package='teleop_twist_joy', executable='teleop_node',
        name='teleop_twist_joy_node_sim',
        parameters=[{
            # Surge — left stick Y (axis 1: up = +1 → forward)
            'axis_linear.x':  1,  'scale_linear.x':  0.2,
            'scale_linear_turbo.x': 0.6,

            # Sway — left stick X (axis 0: left = +1 → +y in FLU ✓)
            'axis_linear.y':  0,  'scale_linear.y':  0.2,

            # Heave — right stick Y (axis 4: up = +1 → ascend = +z ✓)
            'axis_linear.z':  4,  'scale_linear.z':  0.2,

            # Yaw — right stick X (axis 3: left = +1 → CCW = +z ang ✓ in REP-103)
            'axis_angular.yaw': 3, 'scale_angular.yaw': 0.3,

            # Deadman on RB
            'enable_button': 5,
            'require_enable_button': True,
            'publish_stamped_twist': True,
            'frame': 'mr_pinchy/base_link_ekf_flu',
        }],
        remappings=[('/cmd_vel', '/BlueROV2_Heavy/mr_pinchy/cmd_vel')]
    )

    default_config_waypoint = os.path.join(
        get_package_share_directory("mr_pinchy_interface"),
        "config",
        "bluerov2_wp_ctrl.yaml",
    )
    config_arg_waypoint = DeclareLaunchArgument(
        "config_file_waypoint",
        default_value=default_config_waypoint,
        description="YAML file with bluerov2_wp_ctrl node parameters",
    )

    waypoint_controller_node = Node(
        package="mr_pinchy_interface",
        executable="bluerov2_wp_ctrl",
        name="bluerov2_wp_ctrl_sim",
        output="screen",
        parameters=[LaunchConfiguration("config_file_waypoint"), 
                    {"odom_topic": "/BlueROV2_Heavy/odom_gt",
                     "cmd_topic": "/BlueROV2_Heavy/mr_pinchy/cmd_vel",
                     "base_link_frame": "BlueROV2_Heavy/base_link"},
                    use_sim_time],
    )

    # Tank TF tree (static): KTHTank World/tank_base -> tank_base -> mocap -> map,
    # plus mocap -> miqus_camera, and the tank's NavSatFix on /mocap/tank/gps.
    watertank_tf_utils_node = Node(
        package="watertank_utils",
        executable="watertank_tf_utils",
        name="watertank_tf_utils_node",
        output="screen",
        parameters=[use_sim_time],
        arguments=["--ros-args", "--log-level", "info"],
    )

    return LaunchDescription(
        [
            use_sim_time_arg,
            bluerov2_vel_rc_node,
            joy_node_sim,
            teleop_node_sim,
            config_arg_waypoint,
            waypoint_controller_node,
            config_arg_sitl,
            sitl_node,
            ros_tcp_endpoint_node,
            watertank_tf_utils_node,
            # keyboard_teleop_node,
        ]
    )

