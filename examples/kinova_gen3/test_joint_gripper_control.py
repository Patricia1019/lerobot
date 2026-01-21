#!/usr/bin/env python3
"""
Test script for KINOVA Gen3 joint and gripper control via ROS.
Tests trajectory execution similar to ros_joint.py: HOME -> SUBGOAL -> TARGET -> SUBGOAL -> HOME
"""

import time
import numpy as np
from scipy.interpolate import CubicSpline
from lerobot.robots.kinova_gen3 import KinovaGen3, KinovaGen3Config

# Define trajectory waypoints (same as ros_joint.py)
HOME_ANGLE_RAD = np.array([5.537776947021484, 0.36067965626716614, 2.6596689224243164, 
                           4.500516891479492, 0.29103657603263855, 5.156563758850098, 
                           0.20305243134498596], dtype=float)

SUBGOAL_ANGLE_RAD = np.array([5.8246331214904785, 1.1113364696502686, 2.85727858543396, 
                               4.989912509918213, 0.505342960357666, 5.460690021514893, 
                               0.5903300046920776], dtype=float)

TARGET_ANGLE_RAD = np.array([5.817646503448486, 1.1553007364273071, 2.8702352046966553, 
                              4.999231815338135, 0.5153821110725403, 5.49638557434082, 
                              0.5797866582870483], dtype=float)

NUM_WAYPOINTS = 5


def generate_spline_trajectory(start_joint_rad, end_joint_rad, num_points=NUM_WAYPOINTS):
    """Generate smooth joint trajectory using cubic spline interpolation (from ros_joint.py)."""
    trajectory = []
    
    for joint_idx in range(7):
        start_angle = start_joint_rad[joint_idx]
        end_angle = end_joint_rad[joint_idx]
        
        # Calculate shortest path (handle wrap-around)
        diff = end_angle - start_angle
        if diff > np.pi:
            diff -= 2 * np.pi
        elif diff < -np.pi:
            diff += 2 * np.pi
        
        # Create interpolation points
        t = np.linspace(0, 1, num_points)
        angles = start_angle + diff * t
        angles = np.mod(angles, 2 * np.pi)
        trajectory.append(angles)
    
    # Transpose to get waypoints
    waypoints = np.array(trajectory).T
    
    # Apply cubic spline smoothing
    t_points = np.linspace(0, 1, num_points)
    smoothed_waypoints = []
    
    for joint_idx in range(7):
        cs = CubicSpline(t_points, waypoints[:, joint_idx], bc_type='natural')
        smoothed = cs(t_points)
        smoothed = np.mod(smoothed, 2 * np.pi)
        smoothed_waypoints.append(smoothed)
    
    return np.array(smoothed_waypoints).T


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


def test_home_position(robot):
    """Step 1: Move to HOME position."""
    print_section("Step 1: Open Gripper & Move to HOME")
    
    obs = robot.get_observation()
    print_observation(obs, "Initial observation")
    
    print("\nOpening gripper...")
    action_dict = {f"joint_{i}.pos": float(HOME_ANGLE_RAD[i-1]) for i in range(1, 8)}
    action_dict["gripper.pos"] = 0.0  # Open
    robot.send_action(action_dict, loop_hz=10.0)
    time.sleep(0.3)
    
    print("\nMoving to HOME position...")
    robot.move_to_joint_position(HOME_ANGLE_RAD, position_name="HOME", tolerance_deg=0.8, timeout=30.0, loop_hz=10.0)
    
    obs = robot.get_observation()
    print_observation(obs, "At HOME position")
    return True


def test_home_to_subgoal_trajectory(robot):
    """Step 2: Generate and execute trajectory from HOME to SUBGOAL."""
    print_section("Step 2: Trajectory from HOME to SUBGOAL")
    
    num_intermediate_points = NUM_WAYPOINTS - 3
    home_to_subgoal_sequence = generate_spline_trajectory(HOME_ANGLE_RAD, SUBGOAL_ANGLE_RAD, num_intermediate_points + 2)
    
    print(f"\nGenerated {len(home_to_subgoal_sequence)} waypoints using cubic spline")
    print(f"First waypoint (deg): {np.rad2deg(home_to_subgoal_sequence[0]).tolist()}")
    print(f"Last waypoint (deg): {np.rad2deg(home_to_subgoal_sequence[-1]).tolist()}")
    
    print("\nExecuting trajectory (HOME -> SUBGOAL)...")
    for i, waypoint in enumerate(home_to_subgoal_sequence):
        action_dict = {f"joint_{j}.pos": float(waypoint[j-1]) for j in range(1, 8)}
        robot.send_action(action_dict, wait=False, loop_hz=10.0)
        if (i + 1) % 2 == 0:
            print(f"  Waypoint {i+1}/{len(home_to_subgoal_sequence)}")
        time.sleep(0.1)
    
    print("\nMoving to SUBGOAL position...")
    robot.move_to_joint_position(SUBGOAL_ANGLE_RAD, position_name="SUBGOAL", tolerance_deg=0.8, timeout=30.0, loop_hz=10.0)
    
    obs = robot.get_observation()
    print_observation(obs, "At SUBGOAL position")
    return True


def test_subgoal_to_target_trajectory(robot):
    """Step 3: Generate and execute trajectory from SUBGOAL to TARGET."""
    print_section("Step 3: Trajectory from SUBGOAL to TARGET")
    
    num_intermediate_points = NUM_WAYPOINTS - 3
    subgoal_to_target_sequence = generate_spline_trajectory(SUBGOAL_ANGLE_RAD, TARGET_ANGLE_RAD, num_intermediate_points + 2)
    
    print(f"\nGenerated {len(subgoal_to_target_sequence)} waypoints")
    
    print("Executing trajectory (SUBGOAL -> TARGET)...")
    for i, waypoint in enumerate(subgoal_to_target_sequence):
        action_dict = {f"joint_{j}.pos": float(waypoint[j-1]) for j in range(1, 8)}
        robot.send_action(action_dict, wait=False, loop_hz=10.0)
        if (i + 1) % 2 == 0:
            print(f"  Waypoint {i+1}/{len(subgoal_to_target_sequence)}")
        time.sleep(0.1)
    
    print("\nMoving to TARGET position (strict tolerance)...")
    robot.move_to_joint_position_strict(TARGET_ANGLE_RAD, position_name="TARGET", tolerance_deg=0.5, timeout=30.0, loop_hz=10.0)
    
    print("\nClosing gripper at TARGET...")
    action_dict = {f"joint_{i}.pos": float(TARGET_ANGLE_RAD[i-1]) for i in range(1, 8)}
    action_dict["gripper.pos"] = 1.0  # Close
    robot.send_action(action_dict, wait=False, loop_hz=10.0)
    time.sleep(1.0)
    
    obs = robot.get_observation()
    print_observation(obs, "At TARGET with gripper closed")
    return True


def test_return_to_home(robot):
    """Step 4: Return from TARGET to SUBGOAL to HOME using reverse paths."""
    print_section("Step 4: Return to HOME (reverse paths)")
    
    num_intermediate_points = NUM_WAYPOINTS - 3
    subgoal_to_target_sequence = generate_spline_trajectory(SUBGOAL_ANGLE_RAD, TARGET_ANGLE_RAD, num_intermediate_points + 2)
    target_to_subgoal_sequence = subgoal_to_target_sequence[::-1]
    
    print("\nExecuting reverse trajectory (TARGET -> SUBGOAL)...")
    for i, waypoint in enumerate(target_to_subgoal_sequence):
        action_dict = {f"joint_{j}.pos": float(waypoint[j-1]) for j in range(1, 8)}
        action_dict["gripper.pos"] = 1.0  # Keep closed
        robot.send_action(action_dict, wait=False, loop_hz=10.0)
        if (i + 1) % 2 == 0:
            print(f"  Waypoint {i+1}/{len(target_to_subgoal_sequence)}")
        time.sleep(0.1)
    
    print("\nMoving to SUBGOAL position...")
    robot.move_to_joint_position(SUBGOAL_ANGLE_RAD, position_name="SUBGOAL", tolerance_deg=0.8, timeout=30.0, loop_hz=10.0)
    
    home_to_subgoal_sequence = generate_spline_trajectory(HOME_ANGLE_RAD, SUBGOAL_ANGLE_RAD, num_intermediate_points + 2)
    subgoal_to_home_sequence = home_to_subgoal_sequence[::-1]
    
    print("\nExecuting reverse trajectory (SUBGOAL -> HOME)...")
    for i, waypoint in enumerate(subgoal_to_home_sequence):
        action_dict = {f"joint_{j}.pos": float(waypoint[j-1]) for j in range(1, 8)}
        action_dict["gripper.pos"] = 1.0  # Keep closed
        robot.send_action(action_dict, wait=False, loop_hz=10.0)
        if (i + 1) % 2 == 0:
            print(f"  Waypoint {i+1}/{len(subgoal_to_home_sequence)}")
        time.sleep(0.1)
    
    print("\nMoving to HOME position...")
    robot.move_to_joint_position(HOME_ANGLE_RAD, position_name="HOME", tolerance_deg=0.8, timeout=30.0, loop_hz=10.0)
    
    print("\nOpening gripper at HOME...")
    action_dict = {f"joint_{i}.pos": float(HOME_ANGLE_RAD[i-1]) for i in range(1, 8)}
    action_dict["gripper.pos"] = 0.0  # Open
    robot.send_action(action_dict, wait=False, loop_hz=10.0)
    time.sleep(1.0)
    
    obs = robot.get_observation()
    print_observation(obs, "Back at HOME with gripper open")
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
        # Run trajectory tests
        test_home_position(robot)
        test_home_to_subgoal_trajectory(robot)
        test_subgoal_to_target_trajectory(robot)
        test_return_to_home(robot)  # Steps 4-9: Commented out - return path
        
        print_section("Tests Complete")
        print("\n✓ Test suite completed!")
        print("\nTrajectory executed:")
        print("  1. Opened gripper, moved to HOME")
        print("  2. Executed smooth trajectory HOME -> SUBGOAL")
        print("  3. Executed smooth trajectory SUBGOAL -> TARGET")
        print("  4. Closed gripper at TARGET")
        print("\n(Steps 4-9: Return paths commented out)")
        
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
