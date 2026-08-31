#!/usr/bin/env python3
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.launch_description_sources import AnyLaunchDescriptionSource, PythonLaunchDescriptionSource

def generate_launch_description():
    fcu_url_arg = DeclareLaunchArgument(
        "fcu_url",
        default_value="udp-b://0.0.0.0:14550@14550",
        description="MAVROS FCU connection URL",
    )

    sonar_pkg = get_package_share_directory('waterlinked_sonar_3d15')
    sonar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(sonar_pkg, 'launch', 'sonar_3d15.launch.py')
        )
    )

    mavros_pkg = get_package_share_directory('mavros')
    mavros_launch = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(
            os.path.join(mavros_pkg, 'launch', 'apm.launch')
        ),
        launch_arguments={
            "fcu_url": LaunchConfiguration("fcu_url"),
        }.items(),
    )

    # --- Nucleus nodes ---
    nucleus_node = Node(
        package='nucleus_driver_ros2',
        executable='nucleus_node',
        name='nucleus_node',
        output='screen',
        emulate_tty=True,
    )

    connect_tcp = TimerAction(
        period=3.0,
        actions=[
            Node(
                package='nucleus_driver_ros2',
                executable='connect_tcp',
                name='nucleus_connect',
                arguments=['192.168.32.23', 'nortek'],
                output='screen',
                emulate_tty=True,
            )
        ]
    )

    nucleus_start = TimerAction(
        period=6.0,
        actions=[
            Node(
                package='nucleus_driver_ros2',
                executable='start',
                name='nucleus_start',
                output='screen',
                emulate_tty=True,
            )
        ]
    )

    brov_dvl_bridge = Node(
        package='nucleus_driver_ros2',
        executable='brov_dvl_bridge',
        name='brov_dvl_bridge',
        output='screen',
        emulate_tty=True,
    )
    
    brov_imu_bridge = Node(
        package='nucleus_driver_ros2',
        executable='brov_imu_bridge',
        name='brov_imu_bridge',
        output='screen',
        emulate_tty=True,
    )

    return LaunchDescription([
        fcu_url_arg,
        mavros_launch,
        # sonar_launch,
        nucleus_node,
        connect_tcp,
        nucleus_start,
        brov_dvl_bridge,
        brov_imu_bridge,
    ])