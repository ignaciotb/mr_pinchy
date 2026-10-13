"""Teleop stack with the relative-state safety filter, no mocap / no EKF.

    joystick --> /mr_pinchy/cmd_vel_ref --> safety_filter --> /mr_pinchy/cmd_vel --> twist controller

`safety_filter` (brov_safe_relative) uses only DVL, gyro and the sonar cloud. It also
republishes the DVL/gyro body velocity as /mr_pinchy/odom_dvl for the twist controller
and publishes the static body frames (FLU, sonar, DVL). Nothing here knows a global
pose: sonar_filter runs as a denoiser only (no tank-wall pruning, so walls within range
count as obstacles) and the twist controller's mocap-frame tank check is disabled.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

SONAR_FRAME = "mr_pinchy/sonar_link_ekf"


def generate_launch_description() -> LaunchDescription:
    rviz_config_arg = DeclareLaunchArgument(
        "rviz_config",
        default_value=PathJoinSubstitution(
            [FindPackageShare("mr_pinchy_interface"), "rviz", "relative.rviz"]
        ),
        description="Absolute path to RViz config file",
    )
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time", default_value="false", description="Use simulation clock"
    )
    use_sim_time = {"use_sim_time": LaunchConfiguration("use_sim_time")}

    def pkg_node(executable: str, name: str | None = None, params: dict | None = None) -> Node:
        return Node(
            package="mr_pinchy_interface",
            executable=executable,
            name=name or executable,
            output="screen",
            parameters=[use_sim_time, params or {}],
        )

    safety_filter = pkg_node("brov_safe_relative", name="safety_filter")

    twist_simple_controller = pkg_node(
        "brov_ctrl_twist_simple",
        params={"odom_topic": "/mr_pinchy/odom_dvl", "enable_tank_check": False},
    )

    # Denoiser only: "map" is the sonar frame itself and the box is effectively infinite.
    sonar_filter = pkg_node(
        "sonar_filter",
        params={
            "map_frame": SONAR_FRAME,
            "tank_x_min": -100.0, "tank_x_max": 100.0,
            "tank_y_min": -100.0, "tank_y_max": 100.0,
            "tank_z_min": -100.0, "tank_z_max": 100.0,
            "wall_margin": 0.0,
        },
    )

    sonar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("waterlinked_sonar_3d15"), "launch", "sonar_3d15.launch.py"
            )
        )
    )

    joy_node = Node(
        package="joy", executable="joy_node", name="joy_node",
        parameters=[{"deadzone": 0.05, "autorepeat_rate": 20.0}],
    )
    teleop_node = Node(
        package="teleop_twist_joy", executable="teleop_node", name="teleop_twist_joy",
        parameters=[{
            "axis_linear.x": 1, "scale_linear.x": 0.2, "scale_linear_turbo.x": 0.6,   # surge, left stick Y
            "axis_linear.y": 0, "scale_linear.y": 0.2,                                 # sway, left stick X
            "axis_linear.z": 4, "scale_linear.z": 0.2,                                 # heave, right stick Y
            "axis_angular.yaw": 3, "scale_angular.yaw": 0.3,                           # yaw, right stick X
            "enable_button": 5, "require_enable_button": True,                         # deadman on RB
            "publish_stamped_twist": True,
            "frame": "mr_pinchy/base_link_ekf_flu",
        }],
        remappings=[("/cmd_vel", "/mr_pinchy/cmd_vel_ref")],
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", LaunchConfiguration("rviz_config")],
        parameters=[use_sim_time],
    )

    return LaunchDescription([
        rviz_config_arg,
        use_sim_time_arg,
        safety_filter,
        twist_simple_controller,
        sonar_filter,
        sonar_launch,
        joy_node,
        teleop_node,
        rviz_node,
    ])
