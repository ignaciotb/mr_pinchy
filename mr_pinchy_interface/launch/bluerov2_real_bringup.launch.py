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
from launch.launch_description_sources import (
    AnyLaunchDescriptionSource,
    PythonLaunchDescriptionSource,
)
from launch.actions import IncludeLaunchDescription

def generate_launch_description() -> LaunchDescription:

    # mavros talking to the real ArduSub
    mavros_launch = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("mavros"), "launch", "apm.launch"
            )
        ),
        launch_arguments={"fcu_url": "udp-b://0.0.0.0:14550@14550"}.items(),
    )

    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="false",
        description="Use simulation clock",
    ) 
    
    use_sim_time = {"use_sim_time": LaunchConfiguration("use_sim_time")}

    # static FRD -> FLU twin of the real pinchy base link.
    hitl_node = Node(
        package="mr_pinchy_interface",
        executable="bluerov2_hitl",
        name="bluerov2_hitl",
        output="screen",
        parameters=[
            {
                "parent_frame": "mr_pinchy/base_link",
                "child_frame": "mr_pinchy/base_link_ekf_flu",
            },
            use_sim_time,
        ],
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
        name="bluerov2_vel_rc_node_real",
        output="screen",
        parameters=[{"rc_topic": "/mavros/rc/override", # Real vehicle
                     "cmd_topic": "/real/mr_pinchy/cmd_vel",
                     # FRD, because bluerov2_wp_ctrl publishes in
                     # mr_pinchy/base_link here. This is the whole reason the
                     # teleop scales below are negated -- flip one, flip both.
                     "input_is_flu": False}],
    )
   
    joy_node_real = Node(package='joy', executable='joy_node', name='joy_node_real',
            parameters=[{
                'deadzone': 0.05,
                'autorepeat_rate': 20.0}]
    )
    
    # NOTE — THE SIGNS BELOW DIFFER FROM bluerov2_sim_bringup.launch.py ON PURPOSE.
    # Keep the two in sync by hand: same axes, same magnitudes, inverted signs.
    #
    # teleop_twist_joy always emits REP-103 FLU (+y left, +z up, +yaw CCW); it has
    # no sim/real switch. bluerov2_vel_rc, though, interprets the whole cmd_topic
    # with ONE convention (input_is_flu), and on the real vehicle that has to be
    # FRD, because bluerov2_wp_ctrl publishes in mr_pinchy/base_link (FRD) here.
    # So teleop is made to speak FRD instead, by negating the three axes where
    # FLU and FRD disagree -- sway, heave and yaw. Surge is the same in both.
    #
    # Verified equivalent to the sim path: for every stick, negated scales through
    # vel_rc's FRD branch give byte-identical RC output to the sim scales through
    # its FLU branch. If you change input_is_flu on the node above, these signs
    # must flip back with it.
    #
    # Turbo: only scale_linear_turbo.x is set, and surge is not negated, so no
    # turbo scale needs inverting today. Add turbo on y/z/yaw and it will.
    teleop_node_real = Node(package='teleop_twist_joy', executable='teleop_node',
        name='teleop_twist_joy_node_real',
        parameters=[{
            # Surge — left stick Y (axis 1: up = +1 → forward). FLU == FRD here.
            'axis_linear.x':  1,  'scale_linear.x':  0.2,
            'scale_linear_turbo.x': 0.6,

            # Sway — left stick X (axis 0: left = +1). NEGATED: +y is left in FLU,
            # right in FRD.
            'axis_linear.y':  0,  'scale_linear.y':  -0.2,

            # Heave — right stick Y (axis 4: up = +1). NEGATED: +z is up in FLU,
            # down in FRD.
            'axis_linear.z':  4,  'scale_linear.z':  -0.2,

            # Yaw — right stick X (axis 3: left = +1). NEGATED: +yaw is CCW in
            # FLU (REP-103), CW in FRD.
            'axis_angular.yaw': 3, 'scale_angular.yaw': -0.3,

            # Deadman on RB. Must match teleop_enable_button in the wp_ctrl
            # config, which yields cmd_topic (and cancels its goal) while held.
            'enable_button': 5,
            'require_enable_button': True,
            'publish_stamped_twist': True,
            # FRD frame, matching the negated scales above.
            'frame': 'mr_pinchy/base_link',
        }],
        remappings=[('/cmd_vel', '/real/mr_pinchy/cmd_vel')]
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
        name="bluerov2_wp_ctrl_real",
        output="screen",
        parameters=[LaunchConfiguration("config_file_waypoint"), 
                    {"odom_topic": "/mocap/mr_pinchy/odom",
                     "cmd_topic": "/real/mr_pinchy/cmd_vel",
                     "base_link_frame": "mr_pinchy/base_link"},
                    use_sim_time],
    )
    return LaunchDescription(
        [
            use_sim_time_arg,
            mavros_launch,
            hitl_node,
            bluerov2_vel_rc_node,
            joy_node_real,
            teleop_node_real,
            config_arg_waypoint,
            waypoint_controller_node,
            # keyboard_teleop_node,
        ]
    )

