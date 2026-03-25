# ROS2 Workspace

This workspace packages the Piper bridge as a standard ROS2 Python package so it can be built with `colcon` and launched with `ros2 run` or `ros2 launch`.

## Build

```bash
cd /home/lbj/VLA/Isaac-GR00T/ros2_ws
source /opt/ros/<distro>/setup.bash
colcon build
```

## Run

```bash
cd /home/lbj/VLA/Isaac-GR00T/ros2_ws
source /opt/ros/<distro>/setup.bash
source install/setup.bash
ros2 run gr00t_piper_bridge piper_bridge
```

## Useful Parameters

```bash
ros2 run gr00t_piper_bridge piper_bridge --ros-args \
  -p server_host:=192.168.0.10 \
  -p server_port:=5555 \
  -p prompt_text:='pick and place the target object' \
  -p topic_js_in:=/ec_robot_1/joint_states_single \
  -p topic_js_out:=/ec_robot_1/joint_states
```

## Launch

```bash
ros2 launch gr00t_piper_bridge piper_bridge.launch.py
```
