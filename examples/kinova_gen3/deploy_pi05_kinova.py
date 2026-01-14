#!/usr/bin/env python

"""
Deploy PI0.5 policy on KINOVA Gen3 robot via ROS.

This script runs PI0.5 inference on your KINOVA Gen3 robot using:
- Your existing kinova_ros_control ROS controller
- ROS image topics for cameras (wrist_cam and fixed_cam)
- LeRobot policy framework

NO KORTEX API REQUIRED - pure ROS-based!

Prerequisites:
    1. kinova_ros_control running (roslaunch your_launch.launch)
    2. ROS workspace sourced
    3. ROS image topics publishing (/wrist_cam/camera/color/image_raw, /fixed_cam/camera/color/image_raw)
    4. LeRobot installed with PI0.5 support

Usage:
    # Terminal 1: Start ROS controller + cameras
    cd kinova_two_realsense/kinova_ros_control
    source devel/setup.bash
    roslaunch kinova_ros_control two_realsense.launch
    
    # Terminal 2: Run deployment
    cd lerobot
    source ../kinova_two_realsense/kinova_ros_control/devel/setup.bash
    python examples/kinova_gen3/deploy_pi05_kinova.py \\
        --task "pick up the red cube" \\
        --pretrained_name_or_path lerobot/pi05_base \\
        --duration 30 \\
        --fps 10
"""

import argparse
import time
import numpy as np
import torch
from typing import Dict, Any

from lerobot.robots.kinova_gen3 import KinovaGen3, KinovaGen3Config
from lerobot.policies.factory import get_policy_class
from lerobot.async_inference.helpers import (
    map_robot_keys_to_lerobot_features,
    prepare_raw_observation,
)

print(f"[INFO] Torch version: {torch.__version__}")
print(f"[INFO] CUDA available: {torch.cuda.is_available()}")


def main():
    parser = argparse.ArgumentParser(description="Deploy PI0.5 on KINOVA Gen3 via ROS")
    
    # Policy settings
    parser.add_argument("--pretrained_name_or_path", type=str, 
                        default="lerobot/pi05_base",
                        help="Path or HuggingFace repo ID of PI0.5 model")
    parser.add_argument("--policy_device", type=str, 
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run policy on (cuda/cpu)")
    
    # Task settings
    parser.add_argument("--task", type=str, required=True,
                        help="Task description (e.g., 'pick up the red block')")
    parser.add_argument("--duration", type=float, default=30.0,
                        help="How long to run (seconds)")
    parser.add_argument("--fps", type=float, default=10.0,
                        help="Control frequency (Hz)")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("KINOVA Gen3 + PI0.5 Deployment (ROS-based)")
    print("=" * 80)
    print(f"Task: '{args.task}'")
    print(f"Model: {args.pretrained_name_or_path}")
    print(f"Device: {args.policy_device}")
    
    # ============================================================================
    # 1. Initialize Robot with default ROS topics
    # ============================================================================
    print("\n[1/4] Initializing KINOVA Gen3 robot with ROS cameras...")
    
    # Use KinovaGen3Config with default topic settings
    robot_config = KinovaGen3Config(
        id="kinova_gen3",
        has_gripper=True,
        cameras={},  # Will auto-populate from ROS topics
        max_relative_target=10.0,  # Safety limit: max 10 degrees per step
    )
    
    # Print the topic configuration being used
    print(f"  Joint target topic: {robot_config.joint_target_topic}")
    print(f"  Joint feedback topic: {robot_config.joint_feedback_topic}")
    print(f"  Gripper target topic: {robot_config.gripper_target_topic}")
    print(f"  Wrist camera topic: {robot_config.wrist_cam_topic}")
    print(f"  Fixed camera topic: {robot_config.fixed_cam_topic}")
    
    robot = KinovaGen3(robot_config)
    
    try:
        robot.connect()
        print(f"  ✓ Robot connected via ROS topics")
    except Exception as e:
        print(f"  ✗ Connection failed: {e}")
        print("\n  Troubleshooting:")
        print("    1. Start ROS: roslaunch kinova_ros_control two_realsense.launch")
        print("    2. Check topics: rostopic list | grep -E 'kinova_ros_control|camera'")
        print("    3. Source ROS: source /path/to/devel/setup.bash")
        print("    4. Verify feedback: rostopic echo /kinova_ros_control/feedback_joint_states")
        return
    
    # Wait for first camera images
    print("\n  Waiting for camera images...")
    time.sleep(2.0)
    
    # Get observation to build feature mapping
    print("\n  Testing observation capture...")
    try:
        raw_obs = robot.get_observation()
        raw_obs["task"] = args.task
        
        print(f"  ✓ Observation captured:")
        print(f"    - Joint positions: {sum(1 for k in raw_obs if '.pos' in k)} joints")
        print(f"    - wrist_cam shape: {raw_obs.get('wrist_cam', np.array([])).shape}")
        print(f"    - fixed_cam shape: {raw_obs.get('fixed_cam', np.array([])).shape}")
        
        # Build lerobot feature mapping
        lerobot_features = map_robot_keys_to_lerobot_features(robot)
        
    except Exception as e:
        print(f"  ✗ Observation failed: {e}")
        robot.disconnect()
        return
    
    # ============================================================================
    # 2. Load PI0.5 Policy
    # ============================================================================
    print(f"\n[2/4] Loading PI0.5 policy from {args.pretrained_name_or_path}...")
    
    try:
        # Get policy class
        policy_class = get_policy_class("pi05")
        
        # Load pretrained policy
        policy = policy_class.from_pretrained(args.pretrained_name_or_path)
        policy.to(args.policy_device)
        policy.eval()
        
        print(f"  ✓ Policy loaded on {args.policy_device}")
        print(f"    - Action dim: {policy.config.max_action_dim}")
        print(f"    - State dim: {policy.config.max_state_dim}")
        print(f"    - Chunk size: {policy.config.chunk_size}")
        
    except Exception as e:
        print(f"  ✗ Failed to load policy: {e}")
        import traceback
        traceback.print_exc()
        robot.disconnect()
        return
    
    # ============================================================================
    # 3. Run Inference Loop
    # ============================================================================
    print(f"\n[3/4] Starting control loop for {args.duration}s at {args.fps} Hz...")
    print("  Press Ctrl+C to stop early\n")
    
    dt = 1.0 / args.fps
    start_time = time.time()
    step_count = 0
    action_buffer = []
    
    try:
        while (time.time() - start_time) < args.duration:
            loop_start = time.perf_counter()
            
            # ================================================================
            # Get observation
            # ================================================================
            raw_obs = robot.get_observation()
            raw_obs["task"] = args.task
            
            # ================================================================
            # Preprocess observation
            # ================================================================
            processed_obs = prepare_raw_observation(
                raw_obs,
                lerobot_features,
                policy.config.image_features
            )
            
            # Add batch dimension for images and prepare for policy
            for key in processed_obs:
                if isinstance(processed_obs[key], torch.Tensor):
                    if 'image' in key:
                        # Convert uint8 [0, 255] to float32 [0, 1]
                        processed_obs[key] = processed_obs[key].float() / 255.0
                        # Add batch dimension: (C, H, W) -> (1, C, H, W)
                        processed_obs[key] = processed_obs[key].unsqueeze(0).to(args.policy_device)
                    else:
                        # State already has batch dim from prepare_raw_observation
                        processed_obs[key] = processed_obs[key].to(args.policy_device)
            
            # Add task (already a string, policy expects it)
            if "task" in raw_obs:
                processed_obs["task"] = raw_obs["task"]
            
            # ================================================================
            # Generate action chunk if buffer empty
            # ================================================================
            if len(action_buffer) == 0:
                with torch.no_grad():
                    # Get action from policy (returns chunk)
                    action_chunk = policy(processed_obs)  # (1, chunk_size, action_dim)
                
                # Split chunk into individual actions
                if isinstance(action_chunk, dict):
                    # Handle dict output - take first action key
                    action_chunk = list(action_chunk.values())[0]
                
                action_buffer = list(action_chunk[0])  # Remove batch dim, split chunk
            
            # ================================================================
            # Execute next action
            # ================================================================
            if action_buffer:
                next_action_tensor = action_buffer.pop(0)
                
                # Convert tensor to action dict
                action_dict = {
                    key: float(next_action_tensor[i].cpu().item())
                    for i, key in enumerate(robot.action_features)
                }
                
                # Send to robot
                robot.send_action(action_dict, wait=False)
                step_count += 1
                
                # Progress logging
                if step_count % 10 == 0:
                    elapsed = time.time() - start_time
                    loop_time = time.perf_counter() - loop_start
                    print(f"  Step {step_count:4d} | Elapsed: {elapsed:5.1f}s | "
                          f"Loop: {loop_time*1000:5.1f}ms | Buffer: {len(action_buffer):2d} actions")
            
            # ================================================================
            # Maintain loop frequency
            # ================================================================
            loop_time = time.perf_counter() - loop_start
            sleep_time = max(0, dt - loop_time)
            time.sleep(sleep_time)
        
        elapsed = time.time() - start_time
        avg_fps = step_count / elapsed if elapsed > 0 else 0
        print(f"\n  ✓ Completed {step_count} steps in {elapsed:.1f}s (avg FPS: {avg_fps:.2f})")
        
    except KeyboardInterrupt:
        print("\n  ! Interrupted by user")
    
    except Exception as e:
        print(f"\n  ✗ Control loop error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        print("\n[4/4] Disconnecting...")
        try:
            robot.disconnect()
            print("  ✓ Robot disconnected")
        except Exception as e:
            print(f"  ✗ Disconnect error: {e}")


if __name__ == "__main__":
    main()
