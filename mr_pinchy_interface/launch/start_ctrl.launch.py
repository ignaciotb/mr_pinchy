from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
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

    waypoint_wrench_controller = Node(
        package="mr_pinchy_interface",
        executable="brov_ctrl_waypoint_wrench",
        name="brov_ctrl_waypoint_wrench",
        output="screen",
        parameters=[use_sim_time],
    )

    waypoint_twist_controller = Node(
        package="mr_pinchy_interface",
        executable="brov_ctrl_waypoint_twist",
        name="brov_ctrl_waypoint_twist",
        output="screen",
        parameters=[use_sim_time],
    )

    twist_controller = Node(
        package="mr_pinchy_interface",
        executable="brov_ctrl_twist",
        name="brov_ctrl_twist",
        output="screen",
        parameters=[use_sim_time],
    )

    safe_controller = Node(
        package="mr_pinchy_interface",
        executable="brov_ctrl_safe",
        name="brov_ctrl_safe",
        output="screen",
        parameters=[use_sim_time],
    )

    ekf_odom = Node(
        package="mr_pinchy_interface",
        executable="brov_odom_ekf",
        name="brov_odom_ekf",
        output="screen",
        parameters=[use_sim_time],
    )
    
    urdf = os.path.join(get_package_share_directory("mr_pinchy_interface"), "urdf", "bluerov2_heavy.urdf.xacro")
    robot_description = xacro.process_file(urdf).toxml()
    robot_state_pub = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        parameters=[{"robot_description": robot_description}, use_sim_time],
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

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", LaunchConfiguration("rviz_config")],
        parameters=[use_sim_time],
    )
    
    sonar_pkg = get_package_share_directory('waterlinked_sonar_3d15')
    sonar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(sonar_pkg, 'launch', 'sonar_3d15.launch.py')
        )
    )
    
    twist_simple_controller = Node(
        package="mr_pinchy_interface",
        executable="brov_ctrl_twist_simple",
        name="brov_ctrl_twist_simple",
        output="screen",
        parameters=[use_sim_time],
    )
    
    sonar_filter = Node(
        package="mr_pinchy_interface",
        executable="sonar_filter",
        name="sonar_filter",
        output="screen",
        parameters=[use_sim_time],
    )
    

    return LaunchDescription(
        [
            rviz_config_arg,
            use_sim_time_arg,
            ekf_odom,
            waypoint_twist_controller,
            twist_simple_controller,
            sonar_filter,
            sonar_launch,
            watertank_state_pub,
            rviz_node,
        ]
    )

