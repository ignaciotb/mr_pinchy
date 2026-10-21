# Relative-state safety filter (`brov_safe_relative`)

A CBF-QP safety filter for joystick teleop that needs **no global localization**
(no mocap, no EKF). The environment is assumed **static**: every sonar point is an obstacle.
A soft-min barrier over all points gives one CBF constraint that clips the joystick velocity
so the robot circle never reaches a point.

```
joystick --> /mr_pinchy/cmd_vel_ref --> safety_filter --> /mr_pinchy/cmd_vel --> brov_ctrl_twist_simple
                                            ^  DVL, gyro, /sonar_filter/point_cloud
                                            +--> /mr_pinchy/odom_dvl (body velocity for the twist controller)
                                            +--> static TF: base_link_ekf -> base_link_ekf_flu / sonar_link_ekf / dvl_link
```

Files:
- `mr_pinchy_interface/brov_safe_relative_core.py` — the math (soft-min CBF in JAX, CBF-QP), no ROS
- `mr_pinchy_interface/brov_safe_relative.py` — the ROS node
- `test/test_safe_relative.py` — pure-python checks
- `launch/start_ctrl_relative.launch.py`, `rviz/relative.rviz`

## Frames and inputs

Everything in the filter is 2D in the robot body frame **FLU** (`mr_pinchy/base_link_ekf_flu`,
x forward, y left). The sensors are FRD, so:

| input | topic | use |
|---|---|---|
| obstacle points | `/sonar_filter/point_cloud` (sonar frame) | rotated into FLU with the static `sonar_link_ekf → base_link_ekf_flu` TF, points with `‖xyz‖ > max_range` or `|z| > max_abs_z` dropped, then projected onto the body xy-plane |
| joystick | `/mr_pinchy/cmd_vel_ref` (FLU) | `u_ref = (vx, vy)` |
| DVL, gyro | `/dvl/velocity`, `/dvl/imu` (FRD) | only repacked into `/mr_pinchy/odom_dvl` for the twist controller |

The node never reads odometry or any global pose. Because the EKF is not running in this
stack, the node also provides what the rest of the stack used to get from it:

- `/mr_pinchy/odom_dvl`: an `Odometry` whose twist is the raw DVL velocity and gyro (FRD,
  child frame `mr_pinchy/base_link_ekf`); the pose stays identity. `brov_ctrl_twist_simple`
  uses it for velocity feedback with `enable_tank_check:=false` (its tank box is in the mocap frame).
- the static body frames (same numbers as `brov_odom_ekf`), so the twist controller can rotate
  the FLU command into FRD and the filter can rotate sonar points into FLU.

`sonar_filter` runs as a **denoiser only** (`map_frame` = sonar frame, huge box): without a
pose there is no tank-wall pruning, so a wall within `max_range` is an obstacle like any other.
The filter then keeps the robot off walls too, at the cost of clipped commands when driving
toward one. Driving *along* a wall is not clipped (the constraint only removes the velocity
component toward the obstacle).

## Soft-min CBF

Robot radius `r_r`, `d_safe = r_r + margin`. With `q` the robot position (origin of the body
frame) and points `p_i`, each point has the barrier `h_i = ‖p_i - q‖ - d_safe`. They are
combined with a soft-min (log-sum-exp), which is smooth and lies below the true minimum:

```
h = -(1/κ) log Σ_i exp(-κ h_i),        min_i h_i - log(N)/κ  ≤  h  ≤  min_i h_i
∇_q h = -Σ_i λ_i n_i,   λ_i = softmax(-κ h)_i,   n_i = (p_i - q)/‖p_i - q‖
```

`h` and `∇_q h` come from one `jax.value_and_grad` (JIT-compiled once; ~0.2 ms for 2048
points on the CPU). The points are static in the world, so in the body frame they move as
`ṗ_i = -u - ω J p_i`; the yaw-rate term is tangential (`n_i · J p_i = 0`) and drops out:

```
ḣ = ∇_q h · u
```

The CBF condition `ḣ ≥ -α h` is one half-plane in the command `u`:

```
n · u ≤ α h =: b,      n = -∇_q h = Σ_i λ_i n_i      (a convex combination of the radial directions, ‖n‖ ≤ 1)
```

`κ` trades smoothness for conservatism: `h` is below the nearest-point distance by at most
`log(N)/κ` (0.08 m for 2048 points at κ = 50), which acts as extra margin; a larger `κ` makes
`n` switch faster between points. To keep the JAX array size fixed the node keeps the
`max_points` nearest points and pads with far-away points (their `exp(-κ h_i)` is exactly 0).
Between sonar frames the last cloud is reused as is (at 0.3 m/s and 10 Hz that is 3 cm of
drift, covered by `margin`). If no non-empty cloud arrives for `obstacle_lost_timeout_sec`
the command passes through.

## QP

The filter solves, every tick (cvxpy, OSQP, built once, parameters updated):

```
min_u,s  ½‖u - u_ref‖² + ½ w_u ‖u - u_prev‖² + w_s s
s.t.     n·u ≤ b + s,   s ≥ 0,   |u_i| ≤ v_max
```

Far from the obstacles `b` is large and `u ≈ u_ref`. Near them, only the velocity component
toward the obstacles is removed. `u_prev` is the command published on the previous tick
(reset to zero when the deadman is released); the `w_u` term (`smooth_weight`) damps the
tick-to-tick jitter caused by `n` switching between sonar points, at the price of a
first-order lag of about `w_u / (1 + w_u)` per tick on the reference. The slack `s` keeps the QP feasible when `h < 0` and the box
limit cannot satisfy the constraint (its value is published for diagnosis). Heave and yaw
rate are copied from the reference. If the joystick reference is older than `cmd_timeout_sec`
(deadman released), nothing is published and the twist controller goes neutral.

## Outputs

- `/mr_pinchy/cmd_vel` — `TwistStamped` in FLU
- `/mr_pinchy/odom_dvl` — `Odometry`, twist only (FRD), for the twist controller
- `/safety_filter/robot_marker` — robot sphere (FLU)
- `/safety_filter/debug` — `Float64MultiArray`
  `[h, slack, u_ref_x, u_ref_y, u_x, u_y, n_x, n_y, n_points, track_active]`

## Parameters (defaults)

`robot_radius 0.5`, `margin 0.1`, `alpha 1.0`, `kappa 50`, `max_points 2048`, `v_max 0.3`,
`slack_weight 1000`, `smooth_weight 1.0`, `rate_hz 30`, `cmd_timeout_sec 0.3`, `sensor_timeout_sec 0.5`,
`obstacle_lost_timeout_sec 1.0`, `max_range 3.0`, `max_abs_z 1.0`, `publish_static_tf true`,
plus the topic/frame names (`odom_topic /mr_pinchy/odom_dvl`, `base_link_frd_frame mr_pinchy/base_link_ekf`).

## Running

Live (tank):
```
ros2 launch mr_pinchy_interface brov_bringup.launch.py
ros2 launch mr_pinchy_interface start_ctrl_relative.launch.py
```

Build (the venv must provide the interpreter so the installed scripts can import cvxpy and
jax; the venv's setuptools is pinned to 59.6.0 because newer versions ignore `setup.cfg`'s
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
