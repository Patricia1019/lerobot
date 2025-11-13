#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import threading
import time
from functools import cached_property
from typing import Any, Dict, Optional

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, JointState

from lerobot.utils.errors import DeviceNotConnectedError

from ..robot import Robot
from .config import R1LiteLeftArmRobotConfig

logger = logging.getLogger(__name__)


class R1LiteLeftArmRobot(Robot):
    """
    R1 Lite robot integration restricted to the left arm + gripper.

    Observation (7-D, positions only):
        [ L_arm(6), L_gripper(1) ]

    Action (7-D, target positions):
        [ L_arm(6), L_gripper(1) ]

    Cameras:
        head_left_rgb, left_wrist_rgb (compressed RGB only)
    """

    config_class = R1LiteLeftArmRobotConfig
    name = "r1lite_left_arm"

    def __init__(self, config: R1LiteLeftArmRobotConfig):
        super().__init__(config)
        self.config = config

        rospy.init_node("r1lite_left_arm_robot", anonymous=True)

        self.left_joint_state_pub = rospy.Publisher(
            "/motion_target/target_joint_state_arm_left", JointState, queue_size=10
        )
        self.left_gripper_joint_state_pub = rospy.Publisher(
            "/motion_target/target_position_gripper_left", JointState, queue_size=10
        )

        self._lock = threading.Lock()
        self._bridge = CvBridge()
        self._is_connected = False
        self._is_calibrated = False

        self._qpos_left = np.zeros(6, dtype=np.float32)
        self._qvel_left = np.zeros(6, dtype=np.float32)
        self._gripper_left_pos = 0.0
        self._gripper_left_vel = 0.0

        self._head_left_rgb_bgr: Optional[np.ndarray] = None
        self._left_wrist_rgb_bgr: Optional[np.ndarray] = None

        self._img_counter = {"HL": 0, "WL": 0}
        self._img_last_ts = {"HL": 0.0, "WL": 0.0}
        self._debug_thread_stop = threading.Event()
        self._debug_thread: Optional[threading.Thread] = None

        rospy.Subscriber(
            "/hdas/camera_head/left_raw/image_raw_color/compressed",
            CompressedImage,
            self._cb_head_left_rgb_compressed,
            queue_size=1,
        )
        rospy.Subscriber(
            "/hdas/camera_wrist_left/color/image_raw/compressed",
            CompressedImage,
            self._cb_left_wrist_rgb_compressed,
            queue_size=1,
        )

        rospy.Subscriber("/hdas/feedback_arm_left", JointState, self._cb_left, queue_size=50)
        rospy.Subscriber("/hdas/feedback_gripper_left", JointState, self._cb_gripper_left, queue_size=50)

        rospy.loginfo("[R1LiteLeftArmRobot] Publishers/Subscribers ready (7D obs&action; compressed RGB only).")

    # ============================ Feature Specs ============================

    @cached_property
    def observation_features(self) -> Dict[str, Any]:
        cam = self.config.cameras
        feats = {f"joint_{i}": float for i in range(7)}
        feats.update(
            {
                "head_left_rgb": (cam["head_left_rgb"].height, cam["head_left_rgb"].width, 3),
                "left_wrist_rgb": (cam["left_wrist_rgb"].height, cam["left_wrist_rgb"].width, 3),
            }
        )
        return feats

    @cached_property
    def action_features(self) -> Dict[str, type]:
        return {
            "left_shoulder_pitch.pos": float,
            "left_shoulder_roll.pos": float,
            "left_elbow_pitch.pos": float,
            "left_elbow_roll.pos": float,
            "left_wrist_pitch.pos": float,
            "left_wrist_roll.pos": float,
            "left_gripper.pos": float,
        }

    # ============================ Lifecycle ============================

    def configure(self) -> None:
        return

    def calibrate(self) -> None:
        self._is_calibrated = True

    def connect(self, calibrate: bool = True) -> bool:
        required_topics = [
            "/hdas/feedback_arm_left",
            "/hdas/feedback_gripper_left",
            "/hdas/camera_head/left_raw/image_raw_color/compressed",
            "/hdas/camera_wrist_left/color/image_raw/compressed",
        ]
        active = [name for (name, _type) in rospy.get_published_topics()]
        ok = all(topic in active for topic in required_topics)

        if not ok:
            rospy.logwarn("[R1LiteLeftArmRobot] Not all required topics are active.")
            self._is_connected = False
            return False

        if calibrate:
            self.calibrate()

        rospy.loginfo("[R1LiteLeftArmRobot] All required topics are active.")
        self._is_connected = True
        self._start_debug_thread()
        return True

    def disconnect(self) -> None:
        try:
            self._stop_debug_thread()
        finally:
            self._is_connected = False

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated

    # ============================ Observation ============================

    def get_observation(self) -> Dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        obs: Dict[str, Any] = {}
        with self._lock:
            state7 = np.zeros(7, dtype=np.float32)
            state7[0:6] = self._qpos_left
            state7[6] = float(self._gripper_left_pos)

            for i in range(7):
                obs[f"joint_{i}"] = float(state7[i])
            obs["observation.state"] = state7
            obs["state"] = state7

            if self._head_left_rgb_bgr is not None:
                obs["head_left_rgb"] = cv2.cvtColor(self._head_left_rgb_bgr, cv2.COLOR_BGR2RGB).astype(
                    np.uint8, copy=False
                )
            if self._left_wrist_rgb_bgr is not None:
                obs["left_wrist_rgb"] = cv2.cvtColor(self._left_wrist_rgb_bgr, cv2.COLOR_BGR2RGB).astype(
                    np.uint8, copy=False
                )

        try:
            rospy.loginfo(f"[R1LiteLeftArmRobot] obs keys={list(obs.keys())[:6]}..., state7.shape={state7.shape}")
        except Exception as e:
            rospy.logwarn(f"[R1LiteLeftArmRobot] obs logging failed: {e}")

        return obs

    # ============================ Actions ============================

    def action_tensor_to_dict(self, action_tensor) -> Dict[str, float]:
        keys = list(self.action_features.keys())
        arr = np.asarray(action_tensor, dtype=np.float32).reshape(-1)
        if arr.shape[0] != len(keys):
            raise ValueError(f"Expected {len(keys)} dims, got {arr.shape[0]}")
        return {k: float(arr[i]) for i, k in enumerate(keys)}

    def _quantize_gripper_target(self, value: Any) -> float:
        try:
            target = float(value)
        except (TypeError, ValueError):
            return 0.0
        if not np.isfinite(target):
            return 0.0
        target = max(0.0, min(100.0, target))
        return 100.0 if target >= 50.0 else 0.0

    def send_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        left_msg = JointState()
        left_gripper_msg = JointState()

        left_msg.position = [
            float(action.get("left_shoulder_pitch.pos", 0.0)),
            float(action.get("left_shoulder_roll.pos", 0.0)),
            float(action.get("left_elbow_pitch.pos", 0.0)),
            float(action.get("left_elbow_roll.pos", 0.0)),
            float(action.get("left_wrist_pitch.pos", 0.0)),
            float(action.get("left_wrist_roll.pos", 0.0)),
        ]

        left_gripper_msg.position = [self._quantize_gripper_target(action.get("left_gripper.pos", 0.0))]

        self._publish_actions(left_msg, left_gripper_msg)
        return {"ok": True}

    def _publish_actions(self, left_msg: JointState, left_gripper_msg: JointState):
        self.left_joint_state_pub.publish(left_msg)
        self.left_gripper_joint_state_pub.publish(left_gripper_msg)

    # ============================ Debug monitor ============================

    def _start_debug_thread(self, period: float = 1.0):
        if self._debug_thread is not None:
            return
        self._debug_thread_stop.clear()
        self._debug_thread = threading.Thread(target=self._debug_loop, args=(period,), daemon=True)
        self._debug_thread.start()

    def _stop_debug_thread(self):
        if self._debug_thread is None:
            return
        self._debug_thread_stop.set()
        self._debug_thread.join(timeout=1.0)
        self._debug_thread = None

    def _debug_loop(self, period: float):
        while not self._debug_thread_stop.is_set():
            now = time.time()
            with self._lock:
                hl_c = self._img_counter["HL"]
                wl_c = self._img_counter["WL"]
                hl_age = now - self._img_last_ts["HL"] if self._img_last_ts["HL"] > 0 else float("inf")
                wl_age = now - self._img_last_ts["WL"] if self._img_last_ts["WL"] > 0 else float("inf")
            rospy.loginfo(
                f"[R1LiteLeftArmRobot] IMG HL={hl_c} (age {hl_age:.02f}s) "
                f"WL={wl_c} (age {wl_age:.02f}s)"
            )
            self._debug_thread_stop.wait(period)

    # ============================ Callbacks ============================

    def _cb_left(self, msg: JointState):
        with self._lock:
            n_arm = min(6, len(msg.position))
            if n_arm > 0:
                self._qpos_left[:n_arm] = np.array(msg.position[:n_arm], dtype=np.float32)
            if len(msg.velocity) >= n_arm and n_arm > 0:
                self._qvel_left[:n_arm] = np.array(msg.velocity[:n_arm], dtype=np.float32)
            if len(msg.position) >= 7:
                self._gripper_left_pos = float(msg.position[6])
            if len(msg.velocity) >= 7:
                self._gripper_left_vel = float(msg.velocity[6])

    def _cb_gripper_left(self, msg: JointState):
        with self._lock:
            if msg.position:
                self._gripper_left_pos = float(msg.position[0])
            if msg.velocity:
                self._gripper_left_vel = float(msg.velocity[0])

    def _cb_head_left_rgb_compressed(self, msg: CompressedImage):
        img_bgr = self._bridge.compressed_imgmsg_to_cv2(msg)
        with self._lock:
            self._head_left_rgb_bgr = img_bgr
            self._img_counter["HL"] += 1
            self._img_last_ts["HL"] = time.time()

    def _cb_left_wrist_rgb_compressed(self, msg: CompressedImage):
        img_bgr = self._bridge.compressed_imgmsg_to_cv2(msg)
        with self._lock:
            self._left_wrist_rgb_bgr = img_bgr
            self._img_counter["WL"] += 1
            self._img_last_ts["WL"] = time.time()
