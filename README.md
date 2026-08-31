# Basics
Once the vehicle is on, you can access Blueos here http://192.168.32.110/
Then start cockpit and you're good to drive. 

# Mavros interface for uplink control
Steps to setup the control of MrPinchy from ROS2.
Launch mavros and set the GCS to ROS:
```
ros2 launch mavros apm.launch fcu_url:=udp-b://0.0.0.0:14550@14550
ros2 param set /mavros/param SYSID_MYGCS 1
```
If you want to drive again from the joystick, set it to 255.
Arm the vehicle:
```
ros2 service call /mavros/cmd/arming mavros_msgs/srv/CommandBool "{value: true}"
```
Run one of the interfaces in mr_pinchy_interface to map a controller output to mavros RC commands. I.e:
```
ros2 run mr_pinchy_interface brov_ctrl_interface
```

# Video
## To stream the onboard USB cam into ROS2 in your uplink PC, run:
```
ros2 run gscam gscam_node \
  --ros-args \
  -p gscam_config:="udpsrc port=5600 ! application/x-rtp,payload=96 ! rtph264depay ! avdec_h264 ! videoconvert" \
  -p frame_id:=camera \
  -r __ns:=/mr_pinchy/main
```
## To stream the HD payload camera, run in Pinchy:
```
./mavlink-camera-manager-aarch64 --mavlink=udpout:0.0.0.0:14000 --verbose
```
You can then check the stream and camera controls in http://192.168.32.116:6020/
