# Mavros interface
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
