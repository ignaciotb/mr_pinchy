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
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*.urdf.xacro')),
        (os.path.join('share', package_name, 'urdf', 'bluerov2_heavy', 'meshes'),
            glob('urdf/bluerov2_heavy/meshes/*.obj') +
            glob('urdf/bluerov2_heavy/meshes/*.mtl') +
            glob('urdf/bluerov2_heavy/meshes/*.dae')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nacho',
    maintainer_email='ignaciotorroba@hotmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'brov_ctrl_waypoint_wrench = mr_pinchy_interface.brov_ctrl_waypoint_wrench:main',
            'brov_ctrl_waypoint_twist = mr_pinchy_interface.brov_ctrl_waypoint_twist:main',
            'brov_ctrl_safe = mr_pinchy_interface.brov_ctrl_safe:main',
            'brov_ctrl_twist = mr_pinchy_interface.brov_ctrl_twist:main',
            'brov_odom_ekf = mr_pinchy_interface.brov_odom_ekf:main',
        ],
    },
)
