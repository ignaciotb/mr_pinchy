# Relative-state safety filter (`brov_safe_relative`)

A CBF-QP safety filter for joystick teleop that needs **no global localization**
(no mocap, no EKF). It tracks one obstacle relative to the robot with a small Kalman
filter and clips the joystick velocity so the robot circle never enters the obstacle circle.

```
joystick --> /mr_pinchy/cmd_vel_ref --> safety_filter --> /mr_pinchy/cmd_vel --> brov_ctrl_twist_simple
                                            ^  DVL, gyro, /sonar_filter/point_cloud
                                            +--> /mr_pinchy/odom_dvl (body velocity for the twist controller)
                                            +--> static TF: base_link_ekf -> base_link_ekf_flu / sonar_link_ekf / dvl_link
```

Files:
- `mr_pinchy_interface/brov_safe_relative_core.py` — the math (KF, clustering, CBF-QP), no ROS
- `mr_pinchy_interface/brov_safe_relative.py` — the ROS node
- `test/test_safe_relative.py` — pure-python checks
- `launch/start_ctrl_relative.launch.py`, `rviz/relative.rviz`

## Frames and inputs

Everything in the filter is 2D in the robot body frame **FLU** (`mr_pinchy/base_link_ekf_flu`,
x forward, y left). The sensors are FRD, so:

| input | topic | conversion |
|---|---|---|
| robot velocity `v_r` | `/dvl/velocity` (`dvl_link`, FRD) | `v_r = (vx, -vy)` |
| yaw rate `ω` | `/dvl/imu` (FRD gyro) | `ω = -gyro_z` |
| obstacle points | `/sonar_filter/point_cloud` (sonar frame) | rotated into FLU with the static `sonar_link_ekf → base_link_ekf_flu` TF, points with `‖xyz‖ > max_range` or `|z| > max_abs_z` dropped |
| joystick | `/mr_pinchy/cmd_vel_ref` (FLU) | `u_ref = (vx, vy)` |

The node never reads odometry or any global pose. Because the EKF is not running in this
stack, the node also provides what the rest of the stack used to get from it:

- `/mr_pinchy/odom_dvl`: an `Odometry` whose twist is the raw DVL velocity and gyro (FRD,
  child frame `mr_pinchy/base_link_ekf`); the pose stays identity. `brov_ctrl_twist_simple`
  uses it for velocity feedback with `enable_tank_check:=false` (its tank box is in the mocap frame).
- the static body frames (same numbers as `brov_odom_ekf`), so the twist controller can rotate
  the FLU command into FRD and the filter can rotate sonar points into FLU.

`sonar_filter` runs as a **denoiser only** (`map_frame` = sonar frame, huge box): without a
pose there is no tank-wall pruning, so a wall within `max_range` is an obstacle like any other.
The nearest-cluster rule then keeps the robot off walls too, at the cost of clipped commands
when driving toward one.

## Obstacle measurement

The cloud is projected onto the body xy-plane and clustered with single-linkage
(`cluster_eps`, same recipe as `sonar_filter`). The centroid of the **nearest** cluster with
at least `min_cluster_pts` points is the measurement `z`. The obstacle radius is a fixed
parameter: the sonar only sees the near face, so the cluster extent would under-estimate it.

## Relative-state KF

State `x = [p, v_o]`: obstacle position relative to the robot and obstacle velocity, both in
the body frame. With `J = [[0,-1],[1,0]]`, a vector that is constant in the world rotates in
the body frame as `ẋ = -ω J x`. So

```
ṗ   = v_o - v_r - ω J p
v̇_o = -ω J v_o              (obstacle velocity constant in the world, plus process noise)
z   = p + noise
```

Discretized with `R = R(-ω dt)` (a plain 2D rotation):

```
p⁺   = R p   + dt (v_o - v_r)
v_o⁺ = R v_o
A = [[R, dt I], [0, R]],   H = [I 0]
```

Noise: `Q = diag(q_pos, q_pos, q_vel, q_vel)·dt`, `R_meas = r_meas² I`. First detection resets
the filter (`p = z`, `v_o = 0`, large covariance on `v_o`). Prediction runs every tick from the
latest DVL/gyro; the update runs whenever a cluster arrives. If no measurement arrives for
`obstacle_lost_timeout_sec` the track is dropped and the command passes through.

## CBF and QP

Robot radius `r_r`, obstacle radius `r_o`, `d_safe = r_r + r_o + margin`.

```
h(p) = ‖p‖ - d_safe,     n = p / ‖p‖
ḣ = n·ṗ = n·(v_o - u) - ω n·(J p) = n·(v_o - u)        (n·Jp = 0, the yaw rate drops out)
```

The CBF condition `ḣ ≥ -α h` is one half-plane in the command `u`:

```
n·u ≤ n·v̂_o + α h =: b
```

The filter solves, every tick (cvxpy, OSQP, built once, parameters updated):

```
min_u,s  ½‖u - u_ref‖² + w_s s
s.t.     n·u ≤ b + s,   s ≥ 0,   |u_i| ≤ v_max
```

The KF speed `‖v̂_o‖` is clipped to `v_obs_max` before forming `b`, so a cluster jump
(for example the nearest cluster switching between a wall and the diver) cannot inject a
large transient velocity into the constraint.

Far from the obstacle `b` is large and `u = u_ref`. Near it, only the velocity component
toward the obstacle is removed. The slack `s` keeps the QP feasible when `h < 0` and the box
limit cannot satisfy the constraint (its value is published for diagnosis). Heave and yaw
rate are copied from the reference. If the joystick reference is older than `cmd_timeout_sec`
(deadman released), nothing is published and the twist controller goes neutral.

## Outputs

- `/mr_pinchy/cmd_vel` — `TwistStamped` in FLU
- `/mr_pinchy/odom_dvl` — `Odometry`, twist only (FRD), for the twist controller
- `/safety_filter/robot_marker`, `/safety_filter/obstacle_marker` — spheres (+ velocity arrow) in FLU
- `/safety_filter/debug` — `Float64MultiArray`
  `[h, slack, u_ref_x, u_ref_y, u_x, u_y, p_x, p_y, v_o_x, v_o_y, track_active]`

## Parameters (defaults)

`robot_radius 0.5`, `obstacle_radius 0.3`, `margin 0.1`, `alpha 1.0`, `v_obs_max 0.5`, `v_max 0.3`,
`slack_weight 1000`, `rate_hz 30`, `cmd_timeout_sec 0.3`, `sensor_timeout_sec 0.5`,
`obstacle_lost_timeout_sec 1.0`, `cluster_eps 0.1`, `min_cluster_pts 10`, `max_range 3.0`,
`max_abs_z 1.0`, `kf_q_pos 0.01`, `kf_q_vel 0.1`, `kf_r_meas 0.1`, `publish_static_tf true`,
plus the topic/frame names (`odom_topic /mr_pinchy/odom_dvl`, `base_link_frd_frame mr_pinchy/base_link_ekf`).

## Running

Live (tank):
```
ros2 launch mr_pinchy_interface brov_bringup.launch.py
ros2 launch mr_pinchy_interface start_ctrl_relative.launch.py
```

Build (the venv must provide the interpreter so the installed scripts can import cvxpy;
the venv's setuptools is pinned to 59.6.0 because newer versions ignore `setup.cfg`'s
script directory and `ros2 run` then cannot find the executables):
```
source setup_env.sh
venv/bin/python -m colcon build --base-paths src --packages-select mr_pinchy_interface --symlink-install
```

Tests (plugin autoload is disabled because a broken user-level pytest plugin is installed):
```
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src/mr_pinchy/mr_pinchy_interface \
    venv/bin/python -m pytest src/mr_pinchy/mr_pinchy_interface/test/test_safe_relative.py
```

Bag replay (`data_rosbag/diver_3` has DVL, IMU, raw sonar, `cmd_vel_ref`; its sonar frame is the
older `mr_pinchy/sonar_link`, carried by the bag's `/tf_static`). Play only the sensor topics so the
bag's recorded `/tf` and `cmd_vel` do not collide with the live nodes:
```
ros2 bag play data_rosbag/diver_3 --clock \
    --topics /dvl/velocity /dvl/imu /sonar_node/point_cloud /mr_pinchy/cmd_vel_ref /tf_static
ros2 run mr_pinchy_interface sonar_filter --ros-args -p use_sim_time:=true -p map_frame:=mr_pinchy/sonar_link \
    -p tank_x_min:=-100.0 -p tank_x_max:=100.0 -p tank_y_min:=-100.0 -p tank_y_max:=100.0 \
    -p tank_z_min:=-100.0 -p tank_z_max:=100.0 -p wall_margin:=0.0
ros2 run mr_pinchy_interface brov_safe_relative --ros-args -p use_sim_time:=true
ros2 topic echo /safety_filter/debug
rviz2 -d <pkg>/rviz/relative.rviz
```
