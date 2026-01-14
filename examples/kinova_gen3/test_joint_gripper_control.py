#!/usr/bin/env python3
"""
Test script for KINOVA Gen3 joint and gripper control via ROS.
Tests both joint positions and gripper open/close commands.
"""

import time
import numpy as np
from lerobot.robots.kinova_gen3 import KinovaGen3, KinovaGen3Config


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_observation(obs, label="Observation"):
    """Print current observation nicely."""
    print(f"\n{label}:")
    
    # Joint positions
    joints = [obs.get(f"joint_{i}.pos", 0.0) for i in range(1, 8)]
    joints_deg = np.rad2deg(joints)
    print(f"  Joint positions (rad): {[f'{j:.3f}' for j in joints]}")
    print(f"  Joint positions (deg): {[f'{j:.1f}' for j in joints_deg]}")
    
    # Gripper
    gripper = obs.get("gripper.pos", None)
    if gripper is not None:
        print(f"  Gripper position: {gripper:.3f} (0=open, 1=close)")
    
    # Camera shapes
    for cam in ["wrist_cam", "fixed_cam"]:
        if cam in obs:
            shape = obs[cam].shape
            print(f"  {cam} shape: {shape}")


def test_joint_movement(robot):
    """Test basic joint movement."""
    print_section("Test 1: Joint Position Reading")
    
    obs = robot.get_observation()
    print_observation(obs, "Initial observation")
    
    current_pos = np.array([obs.get(f"joint_{i}.pos", 0.0) for i in range(1, 8)])
    
    # Small movement on joint 1
    print("\nMoving joint 1 by +0.1 radians...")
    target_pos = current_pos.copy()
    target_pos[0] += 0.1
    
    action_dict = {f"joint_{i}.pos": float(target_pos[i-1]) for i in range(1, 8)}
    
    print(f"Sending action dict with {len(action_dict)} joints")
    robot.send_action(action_dict, wait=False, loop_hz=10.0)
    
    print("Waiting 2 seconds...")
    time.sleep(2.0)
    
    obs = robot.get_observation()
    print_observation(obs, "After joint movement")
    
    return True


def test_gripper_control(robot):
    """Test gripper open/close."""
    print_section("Test 2: Gripper Control")
    
    obs = robot.get_observation()
    gripper_init = obs.get("gripper.pos", None)
    print(f"\nInitial gripper position: {gripper_init}")
    
    # Close gripper
    print("\n[Step 1] Closing gripper (sending gripper.pos = 1.0)...")
    print("Watch the gripper_target topic in another terminal for messages")
    
    robot.send_action({"gripper.pos": 1.0}, wait=False, loop_hz=10.0)
    print("Gripper close command sent!")
    time.sleep(2.0)
    
    obs = robot.get_observation()
    gripper_closed = obs.get("gripper.pos", None)
    print(f"Gripper position after close attempt: {gripper_closed}")
    
    # Open gripper
    print("\n[Step 2] Opening gripper (sending gripper.pos = 0.0)...")
    robot.send_action({"gripper.pos": 0.0}, wait=False, loop_hz=10.0)
    print("Gripper open command sent!")
    time.sleep(2.0)
    
    obs = robot.get_observation()
    gripper_open = obs.get("gripper.pos", None)
    print(f"Gripper position after open attempt: {gripper_open}")
    
    return True


def test_combined_control(robot):
    """Test joint and gripper control together."""
    print_section("Test 3: Combined Joint + Gripper Control")
    
    obs = robot.get_observation()
    current_pos = np.array([obs.get(f"joint_{i}.pos", 0.0) for i in range(1, 8)])
    
    print("\nSending combined action (move joint 2 + close gripper)...")
    
    target_pos = current_pos.copy()
    target_pos[1] += 0.05  # Move joint 2 slightly
    
    action_dict = {f"joint_{i}.pos": float(target_pos[i-1]) for i in range(1, 8)}
    action_dict["gripper.pos"] = 1.0  # Close gripper
    
    print(f"Action keys: {list(action_dict.keys())}")
    robot.send_action(action_dict, wait=False, loop_hz=10.0)

    action_dict["gripper.pos"] = 0.0 # Open gripper after closing
    robot.send_action(action_dict, wait=False, loop_hz=10.0)
    
    print("Combined action sent!")
    time.sleep(2.0)
    
    obs = robot.get_observation()
    print_observation(obs, "After combined action")
    
    return True


def test_monitoring(robot):
    """Test continuous monitoring of robot state."""
    print_section("Test 4: Continuous Monitoring")
    
    print("\nMonitoring robot state for 5 seconds...\n")
    
    start_time = time.time()
    count = 0
    
    while (time.time() - start_time) < 5.0:
        obs = robot.get_observation()
        
        joints = [obs.get(f"joint_{i}.pos", 0.0) for i in range(1, 8)]
        joints_deg = np.rad2deg(joints)
        gripper = obs.get("gripper.pos", None)
        
        gripper_str = f"gripper={gripper:.2f}" if gripper is not None else "gripper=N/A"
        
        print(f"[{count:2d}] Joints (deg): [{', '.join(f'{j:6.1f}' for j in joints_deg)}] | {gripper_str}")
        
        time.sleep(0.5)
        count += 1
    
    print("\nMonitoring complete!")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("  KINOVA Gen3 - Joint and Gripper Control Test Suite")
    print("=" * 80)
    
    # Create config and robot
    cfg = KinovaGen3Config()
    cfg.has_gripper = True
    
    print(f"\nConfiguration:")
    print(f"  Robot ID: {cfg.id}")
    print(f"  Has gripper: {cfg.has_gripper}")
    print(f"  Joint target topic: {cfg.joint_target_topic}")
    print(f"  Gripper target topic: {cfg.gripper_target_topic}")
    print(f"  Joint feedback topic: {cfg.joint_feedback_topic}")
    print(f"  Wrist camera topic: {cfg.wrist_cam_topic}")
    print(f"  Fixed camera topic: {cfg.fixed_cam_topic}")
    
    # Connect to robot
    print_section("Connecting to Robot")
    try:
        robot = KinovaGen3(cfg)
        robot.connect()
        print("✓ Connected successfully!")
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return False
    
    try:
        # Run all tests
        test_joint_movement(robot)
        test_gripper_control(robot)
        test_combined_control(robot)
        test_monitoring(robot)
        
        print_section("All Tests Complete")
        print("\n✓ Test suite completed!")
        print("\nNote: If gripper didn't respond, check:")
        print("  1. ROS controller is running (roslaunch kinova_ros_control ...)")
        print("  2. Monitor /kinova_ros_control/gripper_target topic for messages")
        print("  3. Check ROS logs for errors")
        
    except Exception as e:
        print_section("Test Failed")
        print(f"✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Disconnect
        print_section("Disconnecting")
        try:
            robot.disconnect()
            print("✓ Disconnected successfully!")
        except Exception as e:
            print(f"✗ Disconnect error: {e}")
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
