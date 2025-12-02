# PX100 Puppet Control

Real-time vision-based teleoperation for the PincherX100 robotic arm. Control your robot like a puppet using just your hand and a webcam.

## Demo

[![PX100 Puppet Control Demo](https://img.youtube.com/vi/WEzc5oIadew/maxresdefault.jpg)](https://www.youtube.com/watch?v=WEzc5oIadew)

*Click to watch full demo on YouTube*

---

**Built as proof of concept for intuitive control - this isn't a proper way to teleoperate a robot manipulator**

## Overview

Puppet Control transforms hand movements into precise robotic manipulation without VR headsets or physical controllers. This system demonstrates rapid prototyping of intuitive human-robot interfaces using computer vision and ROS2.

## Features

- **4-DOF Arm Control**: Hand position directly maps to waist, shoulder, elbow, and wrist joints
- **Pinch-to-Grasp**: Thumb-index pinch gesture controls gripper via MoveIt2
- **Real-time Tracking**: MediaPipe hand detection at 30 FPS
- **Smooth Motion**: Configurable exponential smoothing filter (default 70%)
- **Sub-second Response**: Average 200ms latency from hand movement to robot motion
- **Adjustable Parameters**: Live tuning of smoothing and grip strength

## Hardware Requirements

- Trossen Robotics PincherX100 robotic arm
- USB webcam (built-in or external)
- Ubuntu 22.04 with ROS2 Humble 

## Software Dependencies

### Python Packages
```bash
pip3 install mediapipe opencv-python numpy scipy
```

### ROS2 Packages
```bash
sudo apt install ros-humble-moveit ros-humble-interbotix-xsarm-control ros-humble-interbotix-xsarm-moveit
```

## Installation
```bash
git clone https://github.com/abdu7rahman/px100-puppet-control.git
cd px100-puppet-control
```

## Quick Start
```bash
# Terminal 1: Launch robot with MoveIt
ros2 launch interbotix_xsarm_moveit xsarm_moveit.launch.py robot_model:=px100 hardware_type:=actual

# Terminal 2: Run puppet control
python3 puppet_control.py
```

## Controls

### Hand Gestures

| Gesture | Action |
|---------|--------|
| Move hand left/right | Waist rotation |
| Move hand up/down | Shoulder angle |
| Move hand forward/back | Elbow angle |
| Pinch thumb + index | Close gripper |
| Separate fingers | Open gripper |

### Keyboard Commands

| Key | Function |
|-----|----------|
| `+` | Decrease smoothing (faster response) |
| `-` | Increase smoothing (slower response) |
| `t` | Tighter grip |
| `g` | Gentler grip |
| `r` | Reset to home position |
| `q` | Quit application |

## How It Works

1. **Hand Detection**: MediaPipe processes webcam frames to extract 21 hand landmarks
2. **Coordinate Mapping**: Index finger position maps to 3D joint space via interpolation
3. **Motion Smoothing**: Exponential filter reduces jitter (configurable 10-90%)
4. **Arm Control**: Joint commands published to `/px100/commands/joint_group` topic
5. **Gripper Control**: Pinch distance triggers MoveIt2 trajectory execution

## Configuration

Default parameters in `puppet_control.py`:
```python
self.smoothing = 0.7              # Arm motion smoothing (0.1 = fast, 0.9 = slow)
self.pinch_threshold = 0.06       # Pinch detection sensitivity
self.gripper_grasping = -0.037    # Gripper closed position (radians)
self.gripper_released = 0.037     # Gripper open position (radians)
```

## Troubleshooting

### Camera not detected
```bash
ls /dev/video*
```

### Robot not moving
```bash
ros2 topic list | grep px100
ros2 topic echo /px100/joint_states --once
```

### Gripper not responding
```bash
ros2 action list | grep execute_trajectory
```

## Author

**Mohammed Abdul Rahman**  
MS Robotics, Northeastern University  
[LinkedIn](https://linkedin.com/in/abdu7rahman) | mohammedabdulr.1@northeastern.edu

## License

MIT License - see LICENSE file

## Acknowledgments

- [MediaPipe](https://google.github.io/mediapipe/) by Google
- [Interbotix ROS Packages](https://github.com/Interbotix/interbotix_ros_manipulators)
- [MoveIt2](https://moveit.ros.org/)

---

*Proof of concept - not a proper teleoperation method but it works*
