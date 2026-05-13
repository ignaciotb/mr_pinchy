from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    default_bag_path = "/home/shaohang/workspace/underwater_ws/rosbag2_2026_05_11-13_04_52"
    default_rviz_config = PathJoinSubstitution(
        [FindPackageShare("mr_pinchy_interface"), "rviz", "watertank.rviz"]
    )

    bag_path_arg = DeclareLaunchArgument(
        "bag_path",
        default_value=default_bag_path,
        description="Path to the rosbag2 directory to replay",
    )

    rviz_config_arg = DeclareLaunchArgument(
        "rviz_config",
        default_value=default_rviz_config,
        description="Absolute path to RViz config file",
    )

    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use the clock published by rosbag playback",
    )

    bag_play = ExecuteProcess(
        cmd=[
            "ros2", "bag", "play",
            LaunchConfiguration("bag_path"),
            "--clock", "100",
            "--loop",
        ],
        output="screen",
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", LaunchConfiguration("rviz_config")],
        parameters=[{"use_sim_time": LaunchConfiguration("use_sim_time")}],
    )

    static_sonar_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="sonar_static_tf",
        arguments=[
            "--x", "0.220",
            "--y", "0.0",
            "--z", "-0.160",
            "--qx", "0.0",
            "--qy", "-0.259",
            "--qz", "0.0",
            "--qw", "0.966",
            "--frame-id", "mr_pinchy/base_link_ekf",
            "--child-frame-id", "mr_pinchy/sonar_link",
        ],
        parameters=[{"use_sim_time": LaunchConfiguration("use_sim_time")}],
    )

    return LaunchDescription(
        [
            bag_path_arg,
            rviz_config_arg,
            use_sim_time_arg,
            bag_play,
            rviz_node,
            static_sonar_tf,
        ]
    )
