# MR Pinchy Twist Control Suite

ROS 2 Humble nodes for BlueROV velocity control, waypoint tracking, state
estimation, sonar filtering, and RViz visualization.

## Requirements

- ROS 2 Humble with MAVROS, `joy`, `teleop_twist_joy`, and
  `robot_state_publisher`
- `nucleus_driver_ros2` and `waterlinked_sonar_3d15` in the same workspace
- Python 3 packages: NumPy, SciPy, CVXPY, and OSQP

From the workspace root:

```bash
source /opt/ros/humble/setup.zsh
rosdep install --from-paths src --ignore-src -r -y \
  --skip-keys="python3-cvxpy python3-osqp"
python -m pip install cvxpy osqp
colcon build --symlink-install --packages-select mr_pinchy_interface
source install/setup.zsh
```

## Run

Start the hardware interfaces:

```bash
ros2 launch mr_pinchy_interface brov_bringup.launch.py
```

In another terminal, start the EKF, velocity controller, sonar, joystick
teleoperation, and RViz:

```bash
source install/setup.zsh
ros2 launch mr_pinchy_interface start_ctrl.launch.py
```

The launch file above already starts the EKF and velocity controller. To use
keyboard control, keep it running and load the helper in another terminal from
the workspace root:

```bash
source src/mr_pinchy/setup_env.sh
vel_ctrl_keyboard
```

For waypoint control, replace the keyboard command with:

```bash
ros2 run mr_pinchy_interface brov_ctrl_waypoint_twist
```

Remember to comment out the joystick or keyboard command in the launch file if you want to use waypoint control.

Then send `/goal_pose` from RViz. Use only one command source at a time.

## Conventions

Frame IDs do not use a leading `/`.

| Frame | Convention and use |
| --- | --- |
| `map` | Fixed world frame used by RViz goals and sonar filtering. |
| `mocap` | EKF world/odometry frame; the EKF publishes `mocap -> mr_pinchy/base_link_ekf`. |
| `mr_pinchy/base_link_ekf` | Vehicle body frame in FRD convention: x forward, y right, z down. Odometry twist and low-level control use this frame. |
| `mr_pinchy/base_link_ekf_flu` | ROS body frame in FLU convention: x forward, y left, z up. Keyboard, joystick, and waypoint velocity commands use this frame. |

The EKF publishes a static 180-degree rotation about x between the FRD and FLU
body frames. If `map` and `mocap` are distinct, another node must provide the
`map -> mocap` transform.

## Main topics

| Topic | Purpose |
| --- | --- |
| `/mr_pinchy/cmd_vel` | Stamped velocity command |
| `/mocap/mr_pinchy/odom_ekf` | Filtered odometry |
| `/mavros/manual_control/send` | ArduSub manual-control output |
| `/goal_pose` | Waypoint goal |
| `/sonar_filter/point_cloud` | Filtered sonar cloud |

The controller publishes neutral output when commands or odometry time out, or
when the estimated position leaves its configured tank bounds. Verify frames,
bounds, and gains before arming the vehicle.
