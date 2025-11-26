## Built as proof of concept as intuitive control this isnt a proper way to teleoperate a robot manipulator

# PX100 Puppet Control

Real-time vision-based teleoperation for the PincherX100 robotic arm. Control your robot like a puppet using just your hand and a webcam.

**Demo video coming soon**

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

### Launch
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

## Architecture
```
Webcam → MediaPipe → Hand Landmarks → Coordinate Transform → Smoothing Filter
                                                                      ↓
                                                            Joint Commands
                                                                      ↓
                                              ┌─────────────────────────────────┐
                                              │                                 │
                                         Arm Control                    Gripper Control
                                       (Direct Publish)                  (MoveIt2 Action)
                                              │                                 │
                                              └─────────────────────────────────┘
                                                            ↓
                                                    PincherX100 Robot
```

## Technical Details

- **Framework**: ROS2 Humble
- **Hand Tracking**: MediaPipe Hands solution
- **Arm Interface**: Interbotix XS SDK via JointGroupCommand messages
- **Gripper Interface**: MoveIt2 ExecuteTrajectory action server
- **Control Rate**: 30 Hz camera feed with smoothed command output
- **Joint Limits**: Safety-constrained per manufacturer specifications
- **Coordinate System**: Modified Denavit-Hartenberg parameters

## Configuration

Default parameters in `puppet_control.py`:
```python
self.smoothing = 0.7              # Arm motion smoothing (0.1 = fast, 0.9 = slow)
self.pinch_threshold = 0.06       # Pinch detection sensitivity (lower = harder to trigger)
self.gripper_grasping = -0.037    # Gripper closed position (radians)
self.gripper_released = 0.037     # Gripper open position (radians)
self.wrist_angle = 1.23           # Fixed wrist orientation (radians)
```

### Joint Limits
```python
Waist:    -3.14 to  3.14 rad
Shoulder: -1.8  to  1.5  rad
Elbow:    -1.8  to  1.5  rad
Wrist:    -1.8  to  1.8  rad
```

## Workspace

The system maps hand movements to the following robot workspace:

- **X-axis (reach)**: 0.08m to 0.25m
- **Y-axis (lateral)**: -0.15m to 0.15m  
- **Z-axis (height)**: 0.05m to 0.25m

## Troubleshooting

### Camera not detected
```bash
ls /dev/video*
# If camera is not /dev/video0, update cap = cv2.VideoCapture(X) in code
```

### Robot not moving
```bash
# Check if robot driver is running
ros2 topic list | grep px100

# Verify joint states are publishing
ros2 topic echo /px100/joint_states --once

# Check for errors
ros2 node info /px100/xs_sdk
```

### Gripper not responding
```bash
# Verify MoveIt action server is available
ros2 action list | grep execute_trajectory

# Check gripper joint states
ros2 topic echo /px100/joint_states | grep left_finger
```

### Hand detection issues
- Ensure adequate lighting
- Keep hand within camera frame
- Avoid cluttered backgrounds
- Check MediaPipe version compatibility

### Slow or jerky motion
- Decrease smoothing factor with `-` key
- Reduce camera resolution if CPU usage is high
- Close other applications using camera/CPU

## Development

Built and tested on:
- Ubuntu 22.04 LTS
- ROS2 Humble Hawksbill
- Python 3.10
- MediaPipe 0.10.9
- OpenCV 4.8.1

## Performance

- Hand detection: ~30 FPS
- Command latency: ~200ms average
- Joint smoothing: 70% default (adjustable)
- Gripper response: 200ms trajectory execution

## Known Limitations

- Single hand tracking only
- Fixed wrist orientation (no roll control)
- Requires well-lit environment
- Limited to PincherX100 workspace constraints
- No collision avoidance

## Future Work

- Multi-hand support for dual-arm coordination
- Object detection integration for autonomous grasping
- Gesture library for complex manipulation sequences
- Spatial anchoring for improved precision
- Collision detection and avoidance
- Recording and playback of manipulation sequences

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

## Citation

If you use this work in your research, please cite or don't,  Not like its an invention but its just some teleoperation code that just helps give people insight that this is possible like ofc it is possible 
```
@misc{px100_puppet_control,
  author = {Abdul Rahman, Mohammed},
  title = {PX100 Puppet Control: Vision-Based Robotic Teleoperation},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/abdu7rahman/px100-puppet-control}
}
```

## Author

**Mohammed Abdul Rahman**  
MS Robotics, Northeastern University  
[LinkedIn](https://linkedin.com/in/abdu7rahman) | mohammedabdulr.1@northeastern.edu

## License

MIT License

Copyright (c) 2025 Mohammed Abdul Rahman

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Acknowledgments

Built using:
- [MediaPipe](https://google.github.io/mediapipe/) by Google
- [Interbotix ROS Packages](https://github.com/Interbotix/interbotix_ros_manipulators)
- [MoveIt2](https://moveit.ros.org/)

---

*Making robotic teleoperation accessible through computer vision for no absolute reason W unnecessary code*

