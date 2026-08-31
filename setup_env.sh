source venv/bin/activate
source /opt/ros/humble/setup.zsh
source install/setup.zsh
export ROS_DOMAIN_ID=0
alias start_mav_ros="ros2 launch mavros apm.launch fcu_url:=udp-b://0.0.0.0:14550@14550"
alias arm_mr_pinchy="ros2 service call /mavros/cmd/arming mavros_msgs/srv/CommandBool '{value: true}'"
alias disarm_mr_pinchy="ros2 service call /mavros/cmd/arming mavros_msgs/srv/CommandBool '{value: false}'"
alias auto_mode="ros2 param set /mavros/param SYSID_MYGCS 1"
alias manual_mode="ros2 param set /mavros/param SYSID_MYGCS 255"
# alias vel_ctrl_keyboard="ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args --remap /cmd_vel:=/mr_pinchy/cmd_vel -p stamped:=true -p frame_id:=mr_pinchy/base_link_flu"
vel_ctrl_keyboard() {
  ros2 run teleop_twist_keyboard teleop_twist_keyboard \
    --ros-args \
    --remap /cmd_vel:=/mr_pinchy/cmd_vel \
    -p stamped:=true \
    -p frame_id:=/mr_pinchy/base_link_ekf_flu \
    -p speed:=0.15 \
    -p turn:=0.2
}
export RCUTILS_COLORIZED_OUTPUT=1
export RCUTILS_CONSOLE_OUTPUT_FORMAT="[{severity}] [{name}]: {message}"
# export PYTHONPATH=$PYTHONPATH:/home/shaohang/workspace/underwater_ws/venv/lib/python3.10/site-packages 

record_bag_traj() {
  ros2 bag record -s mcap\
    /mocap/mr_pinchy/odom_ekf \
    /safety_filter/robot_marker \
    /mr_pinchy/cmd_vel \
    /tf \
    /tf_static
}

record_bag_sensor() {
  ros2 bag record -s mcap \
    --max-cache-size 536870912 \
    --qos-profile-overrides-path /home/shaohang/workspace/underwater_ws/bag_qos.yaml \
    --storage-config-file /home/shaohang/workspace/underwater_ws/mcap_writer.yaml \
    /mocap/mr_pinchy/odom_ekf \
    /mr_pinchy/cmd_vel \
    /dsp/particle_pc_resampled \
    /safety_filter/robot_marker \
    /safety_filter/obstacle_marker \
    /sonar_filter/point_cloud \
    /sonar_node/point_cloud \
    /tf \
    /tf_static
}

# Record trajectory + particles as TWO separate bags at the same time.
# Splitting the heavy point clouds into their own recorder gives them a
# dedicated writer thread, so they can't stall the light trajectory data.
# tf is recorded in both so each bag is independently playable/visualizable.
# Ctrl-C stops both and finalizes each bag.
record_bag_both() {
  local base=/home/shaohang/workspace/underwater_ws
  local stamp=$(date +%Y%m%d_%H%M%S)

  # light / trajectory bag
  ros2 bag record -s mcap -o traj_${stamp} \
    --max-cache-size 268435456 \
    --storage-config-file ${base}/mcap_writer.yaml \
    /mocap/mr_pinchy/odom_ekf \
    /safety_filter/robot_marker \
    /mr_pinchy/cmd_vel \
    /tf \
    /tf_static &
  local traj_pid=$!

  # heavy / particles + point clouds bag
  ros2 bag record -s mcap -o particles_${stamp} \
    --max-cache-size 536870912 \
    --qos-profile-overrides-path ${base}/bag_qos.yaml \
    --storage-config-file ${base}/mcap_writer.yaml \
    /dsp/particle_pc_resampled \
    /sonar_filter/point_cloud \
    /sonar_node/point_cloud \
    /safety_filter/obstacle_marker \
    /tf \
    /tf_static &
  local part_pid=$!

  echo "Recording -> traj_${stamp}/  and  particles_${stamp}/   (Ctrl-C to stop both)"
  trap "kill -INT ${traj_pid} ${part_pid} 2>/dev/null" INT TERM
  wait ${traj_pid} ${part_pid}
  trap - INT TERM
}