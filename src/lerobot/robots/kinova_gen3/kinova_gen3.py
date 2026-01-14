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

"""
KINOVA Gen3 Robot Implementation for LeRobot
Uses ROS topics to interface with your existing kinova_ros_control controller
Based on your ros_joint.py control pattern

Notes:
- Compatible with NumPy 2.x by avoiding cv_bridge (which is often compiled against NumPy 1.x on ROS Noetic).
- Converts sensor_msgs/Image (and optional CompressedImage) to RGB numpy arrays using pure NumPy (+ cv2 only for compressed).
- Adds image lock for thread-safety, and returns fixed-shape zero images until first frames arrive.
"""

import logging
import time
import threading
from functools import cached_property
from typing import Any, Optional

import numpy as np

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from .config_kinova_gen3 import KinovaGen3Config

# Import ROS (using your existing ROS setup)
import rospy
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
import pdb

logger = logging.getLogger(__name__)


class KinovaGen3(Robot):
    """
    KINOVA Gen3 (6 or 7 DOF) robot arm integration for LeRobot.

    - Uses ROS topics for joint control and feedback
    - Publishes to /kinova_ros_control/joint_target (Float64MultiArray)
    - Subscribes to /kinova_ros_control/feedback_joint_states (JointState)
    - Publishes to /kinova_ros_control/gripper_target (JointState)
    - Camera integration via ROS Image topics

    No Kortex API required - uses your existing ROS controller!
    """

    config_class = KinovaGen3Config
    name = "kinova_gen3"

    def __init__(self, config: KinovaGen3Config):
        super().__init__(config)
        self.config = config

        # ROS publishers and subscribers
        self.pub_joint_target = None
        self.pub_gripper_target = None
        self.sub_joint_feedback = None
        

        # Current joint positions (thread-safe)
        self.current_joint_positions: Optional[float] = None
        self.current_gripper_position: Optional[float] = None
        self.joint_lock = threading.Lock()

        # Connection status
        self._ros_connected = False

        # Camera subscribers and latest frames for ROS image topics
        self.sub_wrist_image = None
        self.sub_fixed_image = None
        self._latest_wrist_image: Optional[np.ndarray] = None
        self._latest_fixed_image: Optional[np.ndarray] = None
        self._image_lock = threading.Lock()

        # Default image shapes (height, width, channels) until first frame arrives
        self._camera_shapes: dict[str, tuple[int, int, int]] = {
            "wrist_cam": (480, 640, 3),
            "fixed_cam": (480, 640, 3),
        }

        # Define joint names (7 DOF for Gen3)
        self.joint_names = [
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            "joint_6",
            "joint_7",
        ]

        # Add gripper if configured
        if config.has_gripper:
            self.joint_names.append("gripper")

    @property
    def _motors_ft(self) -> dict[str, type]:
        """Motor features - one position value per joint."""
        return {f"{joint}.pos": float for joint in self.joint_names}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        """Camera features - image dimensions for each camera."""
        return {
            "wrist_cam": self._camera_shapes["wrist_cam"],
            "fixed_cam": self._camera_shapes["fixed_cam"],
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """Observation features returned by get_observation()."""
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        """Action features expected by send_action()."""
        return self._motors_ft

    def _joint_state_callback(self, msg: JointState):
        with self.joint_lock:
            if len(msg.position) >= 7:
                self.current_joint_positions = np.array(msg.position[:7], dtype=float)

            # Optional: parse gripper from the same message if present
            if self.config.has_gripper:
                gripper_val = None

                # Prefer name-based lookup if names exist
                if msg.name:
                    # Adjust these candidate names to match your system
                    candidates = {"gripper", "gripper_joint", "finger_joint", "left_finger_joint", "right_finger_joint"}
                    for i, n in enumerate(msg.name):
                        if n in candidates and i < len(msg.position):
                            gripper_val = float(msg.position[i])
                            break

                # Fallback: 8th position
                if gripper_val is None and len(msg.position) >= 8:
                    gripper_val = float(msg.position[7])

                if gripper_val is not None:
                    self.current_gripper_position = gripper_val


    @staticmethod
    def _imgmsg_to_rgb_numpy(msg) -> np.ndarray:
        """
        Convert sensor_msgs/Image to an RGB numpy array without cv_bridge.

        Supports common encodings:
        - rgb8, bgr8, rgba8, bgra8, mono8
        - 16UC1, 32FC1 (returned as 3-channel repeated image for consistency)
        """
        encoding = (getattr(msg, "encoding", "") or "").lower()

        if encoding in ("rgb8", "bgr8", "rgba8", "bgra8", "mono8"):
            dtype = np.uint8
        elif encoding == "16uc1":
            dtype = np.uint16
        elif encoding == "32fc1":
            dtype = np.float32
        else:
            raise ValueError(f"Unsupported Image encoding: {getattr(msg, 'encoding', None)}")

        data = np.frombuffer(msg.data, dtype=dtype)

        # msg.step is bytes per row
        row_step_elems = msg.step // np.dtype(dtype).itemsize
        if row_step_elems <= 0:
            raise ValueError(f"Invalid msg.step={msg.step} for dtype={dtype}")

        # reshape to handle potential row padding
        img2d = data.reshape((msg.height, row_step_elems))

        if encoding == "mono8":
            img = img2d[:, : msg.width].copy()
            return np.repeat(img[:, :, None], 3, axis=2)

        if encoding in ("16uc1", "32fc1"):
            img = img2d[:, : msg.width].copy()
            return np.repeat(img[:, :, None], 3, axis=2)

        # color images
        channels = 3 if encoding in ("rgb8", "bgr8") else 4
        expected_row_elems = msg.width * channels

        img2d = img2d[:, :expected_row_elems]
        img = img2d.reshape((msg.height, msg.width, channels))

        if encoding == "bgr8":
            img = img[:, :, ::-1]  # BGR -> RGB
        elif encoding == "bgra8":
            img = img[:, :, [2, 1, 0, 3]]  # BGRA -> RGBA

        if channels == 4:
            img = img[:, :, :3]  # drop alpha

        return img.copy()

    @staticmethod
    def _compressed_to_rgb_numpy(msg) -> np.ndarray:
        """
        Convert sensor_msgs/CompressedImage to RGB numpy array using cv2.imdecode.
        Requires opencv (cv2), but not cv_bridge.
        """
        import cv2  # type: ignore

        arr = np.frombuffer(msg.data, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("cv2.imdecode failed")
        rgb = bgr[:, :, ::-1]
        return rgb

    @property
    def is_connected(self) -> bool:
        """Check if ROS topics are connected."""
        ros_ok = (
            self._ros_connected
            and self.pub_joint_target is not None
            and self.current_joint_positions is not None
        )
        return ros_ok

    def connect(self, calibrate: bool = True) -> None:
        """
        Connect to KINOVA robot via ROS topics.
        Uses your existing kinova_ros_control ROS controller.
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        try:
            # Initialize ROS node if not already initialized
            if not rospy.core.is_initialized():
                rospy.init_node("lerobot_kinova_gen3", anonymous=True)
                logger.info("Initialized ROS node")

            # Setup ROS publisher for joint target
            self.pub_joint_target = rospy.Publisher(
                self.config.joint_target_topic,
                Float64MultiArray,
                queue_size=10,
            )

            # Setup gripper publisher (optional)
            if self.config.has_gripper:
                self.pub_gripper_target = rospy.Publisher(
                    self.config.gripper_target_topic,
                    JointState,
                    queue_size=10,
                )
                logger.info(f"Gripper publisher created: {self.config.gripper_target_topic}")
                time.sleep(0.5)  # Give publisher time to establish
            else:
                logger.info("Gripper disabled (has_gripper=False)")

            # Setup ROS subscriber for joint feedback
            self.sub_joint_feedback = rospy.Subscriber(
                self.config.joint_feedback_topic,
                JointState,
                self._joint_state_callback,
                queue_size=1,
            )

            logger.info(f"Subscribed to: {self.config.joint_feedback_topic}")
            logger.info(f"Publishing to: {self.config.joint_target_topic}")
            if self.config.has_gripper:
                logger.info(f"Gripper target: {self.config.gripper_target_topic}")

            # Wait for joint feedback
            logger.info("Waiting for joint feedback...")
            rate = rospy.Rate(10.0)
            timeout = 10.0
            start_time = time.time()

            while self.current_joint_positions is None and not rospy.is_shutdown():
                if (time.time() - start_time) > timeout:
                    raise TimeoutError("Timeout waiting for joint feedback")
                rate.sleep()

            self._ros_connected = True
            logger.info("KINOVA ROS topics connected successfully")

        except Exception as e:
            logger.error(f"Failed to connect to KINOVA via ROS: {e}")
            self._cleanup_connection()
            raise

        # Subscribe to ROS image topics for wrist and fixed cameras
        try:
            from sensor_msgs.msg import Image, CompressedImage  # type: ignore

            def wrist_cb(msg):
                try:
                    if isinstance(msg, CompressedImage):
                        img = self._compressed_to_rgb_numpy(msg)
                    else:
                        img = self._imgmsg_to_rgb_numpy(msg)

                    with self._image_lock:
                        self._latest_wrist_image = img

                    h, w = img.shape[:2]
                    self._camera_shapes["wrist_cam"] = (h, w, 3)
                except Exception as e:
                    logger.warning(f"Wrist image callback error: {e}")

            def fixed_cb(msg):
                try:
                    if isinstance(msg, CompressedImage):
                        img = self._compressed_to_rgb_numpy(msg)
                    else:
                        img = self._imgmsg_to_rgb_numpy(msg)

                    with self._image_lock:
                        self._latest_fixed_image = img

                    h, w = img.shape[:2]
                    self._camera_shapes["fixed_cam"] = (h, w, 3)
                except Exception as e:
                    logger.warning(f"Fixed image callback error: {e}")

            # Heuristic: if topic endswith "/compressed", subscribe as CompressedImage
            wrist_type = CompressedImage if self.config.wrist_cam_topic.endswith("/compressed") else Image
            fixed_type = CompressedImage if self.config.fixed_cam_topic.endswith("/compressed") else Image

            self.sub_wrist_image = rospy.Subscriber(
                self.config.wrist_cam_topic,
                wrist_type,
                wrist_cb,
                queue_size=1,
            )
            self.sub_fixed_image = rospy.Subscriber(
                self.config.fixed_cam_topic,
                fixed_type,
                fixed_cb,
                queue_size=1,
            )

            logger.info(
                f"Subscribed to wrist cam: {self.config.wrist_cam_topic} ({wrist_type.__name__}) "
                f"and fixed cam: {self.config.fixed_cam_topic} ({fixed_type.__name__})"
            )
        except Exception as e:
            logger.error(f"Failed to subscribe to ROS image topics: {e}")
            # Continue without cameras

        logger.info(f"{self} connected.")

    def get_observation(self) -> dict[str, Any]:
        """
        Get current robot state and camera images.

        Returns:
            Dictionary with joint positions (radians) and camera images (RGB uint8).
            If no frame has arrived yet, returns a zero-image with the default/last-known shape.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        obs_dict: dict[str, Any] = {}

        # Read joint positions
        start = time.perf_counter()
        with self.joint_lock:
            if self.current_joint_positions is None:
                raise RuntimeError("No joint feedback received")
            joint_pos = self.current_joint_positions.copy()

        for i, joint_name in enumerate(self.joint_names[:7]):
            obs_dict[f"{joint_name}.pos"] = float(joint_pos[i])

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read joint state: {dt_ms:.1f}ms")

        # Read images (thread-safe)
        with self._image_lock:
            wrist = None if self._latest_wrist_image is None else self._latest_wrist_image.copy()
            fixed = None if self._latest_fixed_image is None else self._latest_fixed_image.copy()

        if wrist is None:
            h, w, c = self._camera_shapes["wrist_cam"]
            wrist = np.zeros((h, w, c), dtype=np.uint8)
        if fixed is None:
            h, w, c = self._camera_shapes["fixed_cam"]
            fixed = np.zeros((h, w, c), dtype=np.uint8)

        obs_dict["wrist_cam"] = wrist
        obs_dict["fixed_cam"] = fixed

        if self.config.has_gripper:
            obs_dict["gripper.pos"] = float(self.current_gripper_position) if self.current_gripper_position is not None else 0.0


        return obs_dict

    def send_action(
        self,
        action: dict[str, Any],
        wait: bool = True,
        position_name: str = "target",
        tolerance_deg: float | None = None,
        timeout: float = 30.0,
        loop_hz: float = 10.0,
    ) -> dict[str, Any]:
        """
        Send joint position commands via ROS.

        Key behavior (matching your working joint_control script):
        - Build a full 7D target (use current for unspecified joints)
        - Wrap target to the nearest equivalent angle to avoid 0↔2π discontinuities
        - Publish continuously at loop_hz until reached (or timeout) when wait=True
        - Publish a short burst when wait=False to ensure controller receives it
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Default tolerance: strict like your example for final target
        tol = 0.5 if tolerance_deg is None else float(tolerance_deg)

        # Read current joint positions
        with self.joint_lock:
            if self.current_joint_positions is None:
                raise RuntimeError("No joint feedback received")
            current = self.current_joint_positions.copy()

        # Build goal (radians): use current for unspecified joints
        goal = current.copy()

        for i, joint_name in enumerate(self.joint_names[:7]):
            k = f"{joint_name}.pos"
            if k in action:
                val = action[k]
                # Handle torch.Tensor or numpy scalars
                if hasattr(val, 'item'):
                    val = val.item()
                else:
                    val = float(val)
                goal[i] = float(val)

        if len(action)==0:
            logger.warning("No valid joint position in action dictionary")
            return {}

        # Handle gripper command if present
        gripper_cmd = None
        if "gripper.pos" in action:
            val = action["gripper.pos"]
            if hasattr(val, 'item'):
                val = val.item()
            else:
                val = float(val)
            gripper_cmd = float(val)
            
            logger.info(f"Gripper command received: {gripper_cmd}")
            
            # Send gripper command (matching ros_joint.py pattern)
            if self.pub_gripper_target is not None:
                gripper_msg = JointState()
                gripper_msg.position = [gripper_cmd]
                
                # Publish gripper command multiple times to ensure receipt
                logger.info(f"Publishing gripper command to {self.config.gripper_target_topic}")
                for i in range(5):
                    self.pub_gripper_target.publish(gripper_msg)
                    time.sleep(0.05)  # Small delay between publishes
                
                logger.info(f"Gripper command sent: {gripper_cmd}")
            else:
                logger.warning("Gripper publisher not initialized!")

        # Wrap each joint target to be near current (shortest angular move)
        for i in range(7):
            goal[i] = self._wrap_to_near(current[i], goal[i])

        # Optional safety: clip relative movement (degrees in config)
        if self.config.max_relative_target is not None:
            max_diff_rad = np.deg2rad(float(self.config.max_relative_target))
            diff = np.clip(goal - current, -max_diff_rad, max_diff_rad)
            goal = current + diff

        # Normalize to [0, 2*pi) before sending (this matches your script’s convention)
        goal_norm = self._normalize_0_2pi(goal)

        # Controller expects degrees (your working script publishes degrees)
        target_deg = np.rad2deg(goal_norm)

        msg = Float64MultiArray()
        msg.data = target_deg.tolist()

        rate = rospy.Rate(loop_hz)

        # If not waiting, still publish a short burst (important!)
        if not wait:
            for _ in range(max(3, int(loop_hz * 0.3))):  # ~0.3s burst
                self.pub_joint_target.publish(msg)
                rate.sleep()
            return {f"{self.joint_names[i]}.pos": float(goal[i]) for i in range(7)}

        # wait=True: publish continuously until reached or timeout
        logger.info(
            f"Moving to {position_name} (tolerance: {tol:.2f} deg, timeout: {timeout:.1f}s, hz: {loop_hz:.1f})"
        )

        start_time = time.time()
        while not rospy.is_shutdown():
            # Publish target
            self.pub_joint_target.publish(msg)

            # Check error
            with self.joint_lock:
                if self.current_joint_positions is not None:
                    max_error = self._compute_max_error_deg(self.current_joint_positions, goal)
                    if max_error < tol:
                        elapsed = time.time() - start_time
                        logger.info(
                            f"Reached {position_name} in {elapsed:.2f}s (error: {max_error:.3f} deg)"
                        )
                        break

            if (time.time() - start_time) > timeout:
                logger.warning(f"Timeout while moving to {position_name}")
                break

            rate.sleep()

        return {f"{self.joint_names[i]}.pos": float(goal[i]) for i in range(7)}

    def _compute_max_error_deg(self, current_rad: np.ndarray, target_rad: np.ndarray) -> float:
        """Compute max absolute joint error in degrees with wrap-around handling."""
        current_deg = np.rad2deg(current_rad)
        target_deg = np.rad2deg(target_rad)
        errors = np.abs(current_deg - target_deg)
        errors = np.minimum(errors, 360.0 - errors)
        return float(np.max(errors))
    
    @staticmethod
    def _wrap_to_near(current_rad: float, target_rad: float) -> float:
        """Adjust target angle by +/- 2*pi so it's the closest equivalent to current."""
        diff = target_rad - current_rad
        if diff > np.pi:
            diff -= 2 * np.pi
        elif diff < -np.pi:
            diff += 2 * np.pi
        return current_rad + diff

    @staticmethod
    def _normalize_0_2pi(rad: np.ndarray) -> np.ndarray:
        """Normalize angles to [0, 2*pi)."""
        return np.mod(rad, 2 * np.pi)


    def move_to_joint_position(
        self,
        target_joint_rad: np.ndarray,
        position_name: str = "target",
        tolerance_deg: float | None = None,
        timeout: float = 30.0,
        loop_hz: float = 10.0,
    ) -> bool:
        """Move to a specific joint position with standard tolerance (default ~0.8°)."""
        tol = 0.8 if tolerance_deg is None else tolerance_deg
        return self.move_to_joint_position_strict(
            target_joint_rad,
            position_name=position_name,
            tolerance_deg=tol,
            timeout=timeout,
            loop_hz=loop_hz,
        )

    def move_to_joint_position_strict(
        self,
        target_joint_rad: np.ndarray,
        position_name: str = "target",
        tolerance_deg: float = 0.5,
        timeout: float = 30.0,
        loop_hz: float = 10.0,
    ) -> bool:
        """Move to a specific joint position with strict tolerance (default ~0.5°)."""
        target_joint_deg = np.rad2deg(target_joint_rad)
        logger.info(f"Moving to {position_name} joint position (tolerance: {tolerance_deg:.2f} deg)...")

        rate = rospy.Rate(loop_hz)
        start_time = time.time()

        while not rospy.is_shutdown():
            with self.joint_lock:
                if self.current_joint_positions is not None:
                    max_error = self._compute_max_error_deg(self.current_joint_positions, target_joint_rad)
                    if max_error < tolerance_deg:
                        elapsed = time.time() - start_time
                        logger.info(
                            f"Reached {position_name} position in {elapsed:.2f}s (error: {max_error:.3f} deg)"
                        )
                        return True

            if (time.time() - start_time) > timeout:
                logger.warning(f"Timeout while moving to {position_name}")
                return False

            # Publish target again while waiting
            msg = Float64MultiArray()
            msg.data = target_joint_deg.tolist()
            self.pub_joint_target.publish(msg)
            rate.sleep()

        return False

    def disconnect(self):
        """Disconnect from ROS topics and cameras."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # (In this implementation, cameras are ROS topics; no camera objects to disconnect.)
        self._cleanup_connection()
        logger.info(f"{self} disconnected.")

    def _cleanup_connection(self):
        """Clean up ROS publishers and subscribers."""
        try:
            if self.sub_joint_feedback is not None:
                self.sub_joint_feedback.unregister()
                self.sub_joint_feedback = None

            if self.sub_wrist_image is not None:
                self.sub_wrist_image.unregister()
                self.sub_wrist_image = None

            if self.sub_fixed_image is not None:
                self.sub_fixed_image.unregister()
                self.sub_fixed_image = None

            if self.pub_joint_target is not None:
                self.pub_joint_target.unregister()
                self.pub_joint_target = None

            if self.pub_gripper_target is not None:
                self.pub_gripper_target.unregister()
                self.pub_gripper_target = None

            self._ros_connected = False

        except Exception as e:
            logger.warning(f"Error during ROS cleanup: {e}")

    @property
    def is_calibrated(self) -> bool:
        """KINOVA robots are factory calibrated."""
        return True

    def calibrate(self) -> None:
        """No calibration needed for KINOVA Gen3."""
        logger.info("KINOVA Gen3 is factory calibrated, no calibration needed.")

    def configure(self) -> None:
        """Optional: Configure robot parameters if needed."""
        pass

    def setup_motors(self) -> None:
        """Not applicable for KINOVA (no motor ID setup needed)."""
        pass

