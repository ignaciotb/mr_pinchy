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

    joy_node = Node(package='joy', executable='joy_node', name='joy_node',
            parameters=[{
                'deadzone': 0.05,
                'autorepeat_rate': 20.0}]
    )
    
    teleop_node = Node(package='teleop_twist_joy', executable='teleop_node',
        name='teleop_twist_joy',
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
        remappings=[('/cmd_vel', '/mr_pinchy/cmd_vel_ref')]
    )

    return LaunchDescription(
        [
            rviz_config_arg,
            use_sim_time_arg,
            ekf_odom,
            twist_simple_controller,
            sonar_filter,
            sonar_launch,
            joy_node,
            teleop_node,
            # waypoint_twist_controller,
            watertank_state_pub,
            rviz_node,
        ]
    )

