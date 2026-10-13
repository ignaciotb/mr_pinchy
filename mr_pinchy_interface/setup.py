import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'mr_pinchy_interface'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (
            os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py'),
        ),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*.urdf.xacro')),
        (os.path.join('share', package_name, 'urdf', 'bluerov2_heavy', 'meshes'),
            glob('urdf/bluerov2_heavy/meshes/*.obj') +
            glob('urdf/bluerov2_heavy/meshes/*.mtl') +
            glob('urdf/bluerov2_heavy/meshes/*.dae')),
        (os.path.join('share', package_name, 'urdf', 'watertank_description', 'urdf'), glob('urdf/watertank_description/urdf/*.urdf.xacro')),
        (os.path.join('share', package_name, 'urdf', 'watertank_description', 'mesh'), glob('urdf/watertank_description/mesh/*.dae')),
        (os.path.join('share', package_name, 'urdf', 'watertank_description', 'robots'), glob('urdf/watertank_description/robots/*.urdf.xacro')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nacho',
    maintainer_email='ignaciotorroba@hotmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'brov_ctrl_waypoint_wrench = mr_pinchy_interface.brov_ctrl_waypoint_wrench:main',
            'brov_ctrl_waypoint_twist = mr_pinchy_interface.brov_ctrl_waypoint_twist:main',
            'brov_ctrl_safe = mr_pinchy_interface.brov_ctrl_safe:main',
            'brov_ctrl_twist = mr_pinchy_interface.brov_ctrl_twist:main',
            'brov_ctrl_twist_simple = mr_pinchy_interface.brov_ctrl_twist_simple:main',
            'brov_ctrl_twist_simple_autotune_helper = mr_pinchy_interface.brov_ctrl_twist_simple_autotune_helper:main',
            'brov_odom_ekf = mr_pinchy_interface.brov_odom_ekf:main',
            'brov_odom_ekf_ros2 = mr_pinchy_interface.brov_odom_ekf_ros2:main',
            'sonar_filter = mr_pinchy_interface.sonar_filter:main',
            'brov_safe_relative = mr_pinchy_interface.brov_safe_relative:main',
            'overlay_rviz = mr_pinchy_interface.overlay_rviz:main',
            'marker_recolor = mr_pinchy_interface.marker_recolor:main',
        ],
    },
)
