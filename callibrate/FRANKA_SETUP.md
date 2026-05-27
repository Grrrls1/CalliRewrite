# Franka Robot Setup Guide for CalliRewrite

This guide explains how to use the Franka robot with CalliRewrite for robotic calligraphy.

## 📋 Prerequisites

### Hardware
- Franka Emika robot (Panda, FR3, or compatible)
- Network connection to robot
- Pen/brush holder end-effector (or modify gripper)
- Writing surface (paper mounted on flat table)

### Software
- Python 3.10+
- `lerobot_robot_franka` package (already installed)
- Either `franky` or `frankx` library

## 🔧 Installation

### 1. Install Franka Control Library

**Option A: Install frankx (easier, works on macOS/Linux)**
```bash
pip install frankx
```

**Option B: Install franky (newer, recommended for Ubuntu)**
```bash
# Requires libfranka installed
pip install franky
```

### 2. Verify Installation

Test connection to your robot:
```bash
cd /Users/seer/CalliRewrite/callibrate
python RoboControl.py --test 172.16.0.2
```

Replace `172.16.0.2` with your robot's IP address.

## ⚙️ Configuration

### 1. Find Your Robot's IP Address

- Check the Franka Desk web interface
- Usually something like `172.16.0.2`
- Ensure your computer is on the same network

### 2. Calibrate Workspace

You need to define where your writing surface is in the robot's coordinate frame.

**Method A: Manual Teaching**
```python
# Use Franka Desk to manually move robot to paper surface center
# Record the X, Y, Z coordinates shown in the interface
# Update franka_config.yaml with these values
```

**Method B: Programmatic**
```python
from RoboControl import FrankaCalligraphyController

controller = FrankaCalligraphyController(robot_ip="172.16.0.2")
controller.connect()

# Move robot manually using Franka Desk to desired position
# Then read current position
state = controller._robot.current_pose()
print(f"Current position: X={state.x}, Y={state.y}, Z={state.z}")

controller.disconnect()
```

### 3. Update Configuration

Edit `franka_config.yaml` with your workspace coordinates:
```yaml
workspace_center:
  x: 0.4    # Your measured X
  y: 0.0    # Your measured Y
  z: 0.2    # Your measured Z (paper surface height)
```

## 🖊️ Usage

### Basic Workflow

```bash
# 1. Generate calibration data
python calibrate.py --mode generate --tool brush

# 2. Execute calibration on robot
python RoboControl.py ./test/test.npz

# 3. Measure stroke widths and fit calibration function
# (Use calibrate.py fit workflow)

# 4. Convert RL output to robot commands
python calibrate.py --mode convert --tool brush \
  --input data/永.npy --output data/永.npz --alpha 0.04

# 5. Execute calligraphy
python RoboControl.py ./data/永.npz
```

### Using in Python

```python
from RoboControl import Control

# Simple execution
success = Control(
    npz_path='./data/永.npz',
    robot_ip='172.16.0.2',
    speed=0.05
)

# Advanced control
from RoboControl import FrankaCalligraphyController

controller = FrankaCalligraphyController(
    robot_ip='172.16.0.2',
    default_speed=0.05,
    workspace_center=(0.4, 0.0, 0.2),
)

controller.connect()
controller.execute_trajectory(x, y, z, speed=0.05)
controller.disconnect()
```

### Visualization

Before executing on real robot, visualize the trajectory:

```python
from RoboControl import visualize_trajectory

visualize_trajectory('./data/永.npz')
```

## 🔍 Important Notes

### Coordinate System

```
CalliRewrite coordinates:
- X, Y, Z in meters
- Origin at workspace_center
- Z: negative = pressing down, positive = lifted up

Franka robot coordinates:
- X: forward from robot base
- Y: left (positive) / right (negative)
- Z: up (positive) / down (negative)
```

### Safety

1. **Always test in simulation first!**
   - Use `visualize_trajectory()` before real execution
   - Start with very slow speed (0.01-0.05)

2. **Enable User Stop**
   - Keep hand on emergency stop button
   - Be ready to press stop on Franka Desk

3. **Check workspace limits**
   - RoboControl automatically clamps positions
   - But verify your workspace_center is safe!

4. **Test incrementally**
   - First test: single line (calibration)
   - Then: simple character
   - Finally: complex calligraphy

### Calibration Tips

1. **Z-offset is critical**
   - In `convert_rl_to_npz()`, there's a `h = h - 0.09` offset
   - This must match your actual paper height!
   - Adjust in `calibrate.py` line 350

2. **Pen pressure**
   - Start with light pressure (small negative Z)
   - Gradually increase for darker strokes
   - Monitor pen wear

3. **Speed tuning**
   - Slower = better quality, longer time
   - Faster = risk of skipping, but efficient
   - Recommended: 0.03-0.08 for calligraphy

## 🐛 Troubleshooting

### "frankx library not found"
```bash
pip install frankx
# Or on Ubuntu:
sudo apt-get install libfranka-dev
pip install franky
```

### "Robot not connected"
```bash
# Check robot is unlocked (Franka Desk)
# Check network connectivity
ping 172.16.0.2

# Verify no other app is connected
# Close Franka Desk or other control software
```

### "Motion failed"
- Check joint limits not exceeded
- Verify workspace limits are reasonable
- Ensure robot is not in collision
- Check emergency stop is released

### Trajectory looks wrong
```python
# Visualize before executing
from RoboControl import visualize_trajectory
visualize_trajectory('./data/永.npz')

# Check coordinate ranges
import numpy as np
data = np.load('./data/永.npz')
print(f"X range: {data['pos_3d_x'].min()} to {data['pos_3d_x'].max()}")
print(f"Y range: {data['pos_3d_y'].min()} to {data['pos_3d_y'].max()}")
print(f"Z range: {data['pos_3d_z'].min()} to {data['pos_3d_z'].max()}")
```

## 📚 API Reference

### Main Functions

#### `Control(npz_path, robot_ip, speed)`
Execute trajectory from .npz file (CalliRewrite compatibility).

**Args:**
- `npz_path` (str): Path to trajectory file
- `robot_ip` (str): Robot IP address (default: "172.16.0.2")
- `speed` (float): Speed factor 0.0-1.0 (default: 0.05)

**Returns:**
- `bool`: True if successful

#### `FrankaCalligraphyController`
Low-level controller for custom trajectories.

**Methods:**
- `connect()`: Connect to robot
- `disconnect()`: Disconnect from robot
- `move_cartesian(x, y, z, speed)`: Move to position
- `execute_trajectory(x, y, z, speed, wait_time)`: Execute trajectory
- `move_to_home()`: Move to safe home position

### Utility Functions

#### `test_connection(robot_ip)`
Test connection to robot.

#### `visualize_trajectory(npz_path)`
Visualize trajectory using matplotlib.

## 📝 Examples

### Example 1: Test Connection
```bash
python RoboControl.py --test 172.16.0.2
```

### Example 2: Execute Calibration
```bash
python RoboControl.py ./test/test.npz 172.16.0.2 0.05
```

### Example 3: Execute Calligraphy
```bash
python RoboControl.py ./data/永.npz 172.16.0.2 0.08
```

### Example 4: Custom Python Script
```python
#!/usr/bin/env python3
import numpy as np
from RoboControl import FrankaCalligraphyController

# Initialize
controller = FrankaCalligraphyController(
    robot_ip="172.16.0.2",
    workspace_center=(0.4, 0.0, 0.2),
    default_speed=0.05,
)

# Connect
controller.connect()

# Load trajectory
data = np.load("./data/永.npz")
x = data['pos_3d_x']
y = data['pos_3d_y']
z = data['pos_3d_z']

# Execute
controller.execute_trajectory(x, y, z, speed=0.05)

# Cleanup
controller.disconnect()
```

## 🎯 Next Steps

1. ✅ Install franky/frankx library
2. ✅ Test connection with `--test`
3. ✅ Calibrate workspace center
4. ✅ Generate and execute calibration trajectory
5. ✅ Measure widths and fit calibration function
6. ✅ Convert RL output and execute calligraphy!

## 📞 Support

For issues specific to:
- **Franka robot**: See [Franka Documentation](https://frankaemika.github.io/)
- **frankx library**: See [frankx GitHub](https://github.com/pantor/frankx)
- **CalliRewrite**: See [CalliRewrite Paper](https://arxiv.org/abs/2405.15776)

---

**Created for CalliRewrite × Franka Integration**
