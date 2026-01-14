# KINOVA Gen3 PI0.5 Deployment Examples (ROS-based)

Example scripts for deploying PI0.5 policy on KINOVA Gen3 robot using your existing ROS controller.

**NO KORTEX API REQUIRED** - Uses your existing `kinova_ros_control` ROS topics!

## Prerequisites

1. **Your ROS Controller Must Be Running**
   Your `kinova_ros_control` package from `/home/peiqi/projects-2026/kinova_two_realsense/kinova_ros_control`

2. **Install LeRobot with PI0.5 support**:
   ```bash
   cd /home/peiqi/projects-2026/lerobot
   pip install -e ".[pi]"
   ```

3. **Find your RealSense camera serial numbers**:
   ```bash
   rs-enumerate-devices | grep Serial
   ```

## Setup

### Step 1: Start Your ROS Controller

In Terminal 1:
```bash
cd /home/peiqi/projects-2026/kinova_two_realsense/kinova_ros_control
source devel/setup.bash
# Start your KINOVA ROS controller
roslaunch your_launch_file.launch
```

### Step 2: Verify ROS Topics

In Terminal 2:
```bash
source /home/peiqi/projects-2026/kinova_two_realsense/kinova_ros_control/devel/setup.bash

# Check topics are publishing
rostopic list | grep kinova_ros_control

# Should show:
#   /kinova_ros_control/joint_target
#   /kinova_ros_control/feedback_joint_states
#   /kinova_ros_control/gripper_target
```

## Quick Start

### 1. Test Robot Connection

```bash
cd /home/peiqi/projects-2026/lerobot
source /home/peiqi/projects-2026/kinova_two_realsense/kinova_ros_control/devel/setup.bash

python examples/kinova_gen3/test_kinova_connection.py
```

### 2. Deploy PI0.5

```bash
source /home/peiqi/projects-2026/kinova_two_realsense/kinova_ros_control/devel/setup.bash

python examples/kinova_gen3/deploy_pi05_kinova.py \
    --policy_path lerobot/pi05_base \
    --wrist_camera_serial YOUR_SERIAL_1 \
    --front_camera_serial YOUR_SERIAL_2 \
    --task "pick up the red block" \
    --duration 30 \
    --fps 10
```

## Configuration

### ROS Topics (from your ros_joint.py)

Default topics match your `ros_joint.py`:
- Joint commands: `/kinova_ros_control/joint_target` (Float64MultiArray, degrees)
- Joint feedback: `/kinova_ros_control/feedback_joint_states` (JointState, radians)
- Gripper commands: `/kinova_ros_control/gripper_target` (JointState)

To use different topics:
```bash
python deploy_pi05_kinova.py \
    --joint_target_topic /your/custom/topic \
    --joint_feedback_topic /your/feedback/topic \
    --task "your task"
```

### Camera Settings

Find serial numbers:
```bash
rs-enumerate-devices
```

Then use in deployment:
```bash
--wrist_camera_serial YOUR_SERIAL_HERE
--front_camera_serial YOUR_SERIAL_HERE
```

### Policy Settings

- `--policy_path`: Path to PI0.5 model (default: lerobot/pi05_base)
- `--device`: cuda or cpu
- `--fps`: Control frequency (default: 10 Hz, matching your ros_joint.py LOOP_HZ)

### Task Settings

- `--task`: Natural language task description
- `--duration`: How long to run (seconds)

## How It Works

### Your ROS Controller (ros_joint.py pattern):
```python
# Publishes joint targets in DEGREES
pub = rospy.Publisher('/kinova_ros_control/joint_target', Float64MultiArray)
msg.data = [deg1, deg2, deg3, deg4, deg5, deg6, deg7]

# Receives joint feedback in RADIANS
rospy.Subscriber('/kinova_ros_control/feedback_joint_states', JointState, callback)
current_positions = msg.position[:7]  # radians
```

### LeRobot Integration:
```python
# Automatically converts between radians (LeRobot) and degrees (your controller)
robot.get_observation()  # Returns radians from feedback
robot.send_action(action)  # Converts radians to degrees for command
```

## Troubleshooting

### "No joint feedback received"

1. Check ROS controller is running:
   ```bash
   rostopic list | grep kinova
   ```

2. Echo feedback topic:
   ```bash
   rostopic echo /kinova_ros_control/feedback_joint_states
   ```

3. Source ROS workspace:
   ```bash
   source /home/peiqi/projects-2026/kinova_two_realsense/kinova_ros_control/devel/setup.bash
   ```

### Camera not found

1. List cameras: `rs-enumerate-devices`
2. Check USB connection
3. Install librealsense: `pip install pyrealsense2`

### Policy errors

1. Check if policy supports your camera setup
2. Verify model was trained on similar robot (7 DOF + gripper)
3. Ensure action dimensions match (7 joints for Gen3)

## What's Different from Other Robots

| Feature | Other LeRobot Robots | KINOVA Gen3 (Your Setup) |
|---------|---------------------|--------------------------|
| Control | Direct motor bus | ROS topics |
| Connection | USB/Serial | ROS publish/subscribe |
| Units | Varies | Radians (internal) ↔ Degrees (ROS) |
| API | Dynamixel/Feetech | Your kinova_ros_control |
| Calibration | Manual | Not needed (ROS handles it) |

## Example Workflow

```bash
# Terminal 1: Start your ROS controller
cd /home/peiqi/projects-2026/kinova_two_realsense/kinova_ros_control
source devel/setup.bash
roslaunch your_kinova.launch

# Terminal 2: Test connection
cd /home/peiqi/projects-2026/lerobot
source /home/peiqi/projects-2026/kinova_two_realsense/kinova_ros_control/devel/setup.bash
python examples/kinova_gen3/test_kinova_connection.py

# Terminal 2: Deploy PI0.5
python examples/kinova_gen3/deploy_pi05_kinova.py \
    --policy_path lerobot/pi05_base \
    --task "pick up the block" \
    --duration 30
```

## Next Steps

1. **Test basic connection**: Run `test_kinova_connection.py`
2. **Add cameras**: Find serial numbers and add to deployment
3. **Test PI0.5**: Deploy pretrained model
4. **Collect data**: Record your own demonstrations
5. **Fine-tune**: Train PI0.5 on your KINOVA data
