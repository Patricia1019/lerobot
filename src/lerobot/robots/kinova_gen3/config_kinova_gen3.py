#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Configuration for KINOVA Gen3 robot."""

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("kinova_gen3")
@dataclass
class KinovaGen3Config(RobotConfig):
    """
    Configuration for KINOVA Gen3 robot using ROS topics.
    
    Based on your ros_joint.py control setup:
    - Uses ROS topics for control (no Kortex API needed)
    - RealSense cameras from kinova_two_realsense
    """
    
    # ROS topic names (from your ros_joint.py)
    joint_target_topic: str = "/kinova_ros_control/joint_target"
    joint_feedback_topic: str = "/kinova_ros_control/feedback_joint_states"
    gripper_target_topic: str = "/kinova_ros_control/gripper_target"
    
    # Robot configuration
    num_joints: int = 7  # Gen3 has 7 DOF
    has_gripper: bool = True  # Set to False if no gripper
    
    # Safety limits
    # Maximum relative joint movement per action (in degrees)
    # Based on your ros_joint.py POSITION_TOLERANCE_DEG
    # Set lower for safer operation (e.g., 5-10 degrees)
    max_relative_target: float | None = 10.0
    
    # Camera configuration
    # Example: RealSense cameras from your kinova_two_realsense setup
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    # ROS image topics (color)
    wrist_cam_topic: str = "/wrist_cam/camera/color/image_raw"
    fixed_cam_topic: str = "/fixed_cam/camera/color/image_raw"
    
    # Control loop frequency (from your ros_joint.py LOOP_HZ)
    control_frequency: float = 10.0  # Hz
