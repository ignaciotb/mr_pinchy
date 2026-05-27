import os

from ament_index_python.packages import get_package_share_directory
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
    use_sim_time = {"use_sim_time": LaunchConfiguration("use_sim_time")}

    ekf_config = os.path.join(
        get_package_share_directory("mr_pinchy_interface"),
        "config",
        "ekf_ros2.yaml",
    )

    # Rule-based sanity-check / gating node. Republishes accepted mocap/IMU/DVL
    # measurements on /gated/* topics that ekf_node consumes.
    gating_node = Node(
        package="mr_pinchy_interface",
        executable="brov_odom_ekf_ros2",
        name="brov_odom_ekf_ros2",
        output="screen",
        parameters=[use_sim_time],
    )

    # Stock robot_localization EKF. Fuses the gated measurements and publishes
    # /odometry/filtered plus the mocap -> base_link_ekf transform.
    ekf_node = Node(
        package="robot_localization",
        executable="ekf_node",
        name="ekf_node",
        output="screen",
        parameters=[ekf_config, use_sim_time],
    )

    return LaunchDescription(
        [
            use_sim_time_arg,
            gating_node,
            ekf_node,
        ]
    )
