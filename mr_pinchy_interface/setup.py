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
            'bluerov2_keyboard_teleop = mr_pinchy_interface.bluerov2_keyboard_teleop:main',
            'brov_ctrl_interface = mr_pinchy_interface.brov_ctrl_interface:main',
        ],
    },
)
