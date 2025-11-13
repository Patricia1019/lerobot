#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import time
from functools import cached_property
from typing import Any, Dict, Optional
import numpy as np
import cv2
import threading

# ROS
import rospy
from sensor_msgs.msg import JointState, Imu, CompressedImage
from geometry_msgs.msg import TwistStamped, PoseStamped
from std_msgs.msg import Bool
from cv_bridge import CvBridge

# LeRobot
from lerobot.utils.errors import DeviceNotConnectedError

# Your project imports
from ..robot import Robot
from .config_r1lite import R1LiteRobotConfig

logger = logging.getLogger(__name__)


class R1LiteRobot(Robot):
    """
    R1 Lite robot integration (14D-only: arms + grippers).

    Observation (14-D, positions only):
        [ L_arm(6), R_arm(6), L_gripper(1), R_gripper(1) ]

    Action (14-D, target positions):
        [ L_arm(6), R_arm(6), L_gripper(1), R_gripper(1) ]

    Notes:
      - Cameras kept (compressed RGB only).
      - Torso/IMU/EE/velocities are ignored for obs/action 14D packing.
    """

    config_class = R1LiteRobotConfig
    name = "r1lite"

    # ============================ Lifecycle & ROS IO ============================

    def __init__(self, config: R1LiteRobotConfig):
        super().__init__(config)
        self.config = config

        rospy.init_node("r1lite_robot", anonymous=True)

        # ---------------- Publishers (only what we actually use) ----------------
        self.left_joint_state_pub = rospy.Publisher(
            "/motion_target/target_joint_state_arm_left", JointState, queue_size=10
        )
        self.right_joint_state_pub = rospy.Publisher(
            "/motion_target/target_joint_state_arm_right", JointState, queue_size=10
        )
        self.left_gripper_joint_state_pub = rospy.Publisher(
            "/motion_target/target_position_gripper_left", JointState, queue_size=10
        )
        self.right_gripper_joint_state_pub = rospy.Publisher(
            "/motion_target/target_position_gripper_right", JointState, queue_size=10
        )

        # ---------------- Internal state caches ----------------
        self._lock = threading.Lock()
        self._bridge = CvBridge()
        self._is_connected = False
        self._is_calibrated = False

        # 内部关节布局：torso(最多4) + right(6) + left(6) = 16 槽位（仍保留读取，但不进 14D 打包）
        self._qpos_real = np.zeros(16, dtype=np.float32)
        self._qvel_real = np.zeros(16, dtype=np.float32)
        self._torso_len = 0  # 最近一次 torso 实际读到的维度（<=4）

        # Grippers：位置+速度（爪作为各臂的第7维；14D只用位置）
        self._gripper_left_pos = 0.0
        self._gripper_right_pos = 0.0
        self._gripper_left_vel = 0.0
        self._gripper_right_vel = 0.0

        # 相机缓存（仍保留）
        self._head_left_rgb_bgr: Optional[np.ndarray] = None
        self._head_right_rgb_bgr: Optional[np.ndarray] = None
        self._left_wrist_rgb_bgr: Optional[np.ndarray] = None
        self._right_wrist_rgb_bgr: Optional[np.ndarray] = None

        # Debug counters
        self._img_counter = {"HL": 0, "HR": 0, "WL": 0, "WR": 0}
        self._img_last_ts = {"HL": 0.0, "HR": 0.0, "WL": 0.0, "WR": 0.0}
        self._debug_thread_stop = threading.Event()
        self._debug_thread: Optional[threading.Thread] = None

        # ---------------- Subscribers (compressed RGB + joints + grippers) ----------------
        rospy.Subscriber(
            "/hdas/camera_head/left_raw/image_raw_color/compressed",
            CompressedImage, self._cb_head_left_rgb_compressed, queue_size=1
        )
        rospy.Subscriber(
            "/hdas/camera_head/right_raw/image_raw_color/compressed",
            CompressedImage, self._cb_head_right_rgb_compressed, queue_size=1
        )
        rospy.Subscriber(
            "/hdas/camera_wrist_left/color/image_raw/compressed",
            CompressedImage, self._cb_left_wrist_rgb_compressed, queue_size=1
        )
        rospy.Subscriber(
            "/hdas/camera_wrist_right/color/image_raw/compressed",
            CompressedImage, self._cb_right_wrist_rgb_compressed, queue_size=1
        )

        # Joint feedback
        rospy.Subscriber("/hdas/feedback_torso", JointState, self._cb_torso, queue_size=50)
        rospy.Subscriber("/hdas/feedback_arm_right", JointState, self._cb_right, queue_size=50)
        rospy.Subscriber("/hdas/feedback_arm_left", JointState, self._cb_left, queue_size=50)

        # Gripper feedback（如 arm 话题第7维已含爪，这两条保底；存在就覆盖）
        rospy.Subscriber("/hdas/feedback_gripper_left", JointState, self._cb_gripper_left, queue_size=50)
        rospy.Subscriber("/hdas/feedback_gripper_right", JointState, self._cb_gripper_right, queue_size=50)

        rospy.loginfo("[R1LiteRobot] Publishers/Subscribers ready (14D obs&action; compressed RGB only).")

    # ============================ Feature Specs ============================

    @cached_property
    def observation_features(self) -> Dict[str, Any]:
        """
        只导出 14 个标量（joint_0..joint_13）+ 4 路 RGB。
        """
        cam = self.config.cameras
        feats = {f"joint_{i}": float for i in range(14)}
        feats.update({
            "head_left_rgb":  (cam["head_left_rgb"].height,  cam["head_left_rgb"].width,  3),
            "head_right_rgb": (cam["head_right_rgb"].height, cam["head_right_rgb"].width, 3),
            "left_wrist_rgb": (cam["left_wrist_rgb"].height, cam["left_wrist_rgb"].width, 3),
            "right_wrist_rgb":(cam["right_wrist_rgb"].height,cam["right_wrist_rgb"].width,3),
        })
        return feats

    @cached_property
    def action_features(self) -> Dict[str, type]:
        """
        14D target positions: 左臂6 + 右臂6 + 左爪 + 右爪
        """
        return {
            # Left arm (6)
            "left_shoulder_pitch.pos": float,
            "left_shoulder_roll.pos": float,
            "left_elbow_pitch.pos": float,
            "left_elbow_roll.pos": float,
            "left_wrist_pitch.pos": float,
            "left_wrist_roll.pos": float,
            # Right arm (6)
            "right_shoulder_pitch.pos": float,
            "right_shoulder_roll.pos": float,
            "right_elbow_pitch.pos": float,
            "right_elbow_roll.pos": float,
            "right_wrist_pitch.pos": float,
            "right_wrist_roll.pos": float,
            # Grippers (2)
            "left_gripper.pos": float,
            "right_gripper.pos": float,
        }

    # ============================ Lifecycle ============================

    def configure(self) -> None:
        return

    def calibrate(self) -> None:
        self._is_calibrated = True

    def connect(self, calibrate: bool = True) -> bool:
        required_topics = [
            "/hdas/feedback_torso",
            "/hdas/feedback_arm_right",
            "/hdas/feedback_arm_left",
            "/hdas/feedback_gripper_left",
            "/hdas/feedback_gripper_right",
            "/hdas/camera_head/left_raw/image_raw_color/compressed",
            "/hdas/camera_head/right_raw/image_raw_color/compressed",
            "/hdas/camera_wrist_left/color/image_raw/compressed",
            "/hdas/camera_wrist_right/color/image_raw/compressed",
        ]
        active = [name for (name, _type) in rospy.get_published_topics()]
        ok = all(topic in active for topic in required_topics)

        if not ok:
            rospy.logwarn("[R1LiteRobot] Not all required topics are active.")
            self._is_connected = False
            return False

        if calibrate:
            self.calibrate()

        rospy.loginfo("[R1LiteRobot] All required topics are active.")
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
            # 14D：L(6) + R(6) + Lg(1) + Rg(1)
            state14 = np.zeros(14, dtype=np.float32)

            # 左臂6关节（内部布局：left在 qpos[10:16]）
            state14[0:6] = self._qpos_real[10:16]

            # 右臂6关节（内部布局：right在 qpos[4:10]）
            state14[6:12] = self._qpos_real[4:10]

            # 爪位姿
            state14[12] = float(self._gripper_left_pos)
            state14[13] = float(self._gripper_right_pos)

            # 展开为 joint_0..joint_13
            for i in range(14):
                obs[f"joint_{i}"] = float(state14[i])

            # 也保留向量（若下游需要）
            obs["observation.state"] = state14
            obs["state"] = state14

            # 相机：依旧转换 BGR->RGB
            if self._head_left_rgb_bgr is not None:
                obs["head_left_rgb"] = cv2.cvtColor(self._head_left_rgb_bgr, cv2.COLOR_BGR2RGB).astype(np.uint8, copy=False)
            if self._head_right_rgb_bgr is not None:
                obs["head_right_rgb"] = cv2.cvtColor(self._head_right_rgb_bgr, cv2.COLOR_BGR2RGB).astype(np.uint8, copy=False)
            if self._left_wrist_rgb_bgr is not None:
                obs["left_wrist_rgb"] = cv2.cvtColor(self._left_wrist_rgb_bgr, cv2.COLOR_BGR2RGB).astype(np.uint8, copy=False)
            if self._right_wrist_rgb_bgr is not None:
                obs["right_wrist_rgb"] = cv2.cvtColor(self._right_wrist_rgb_bgr, cv2.COLOR_BGR2RGB).astype(np.uint8, copy=False)

        try:
            rospy.loginfo(f"[R1LiteRobot] obs keys={list(obs.keys())[:10]}..., state14.shape={state14.shape}")
        except Exception as e:
            rospy.logwarn(f"[R1LiteRobot] obs logging failed: {e}")

        return obs

    # ============================ Actions ============================

    def action_tensor_to_dict(self, action_tensor) -> Dict[str, float]:
        keys = list(self.action_features.keys())  # len == 14
        arr = np.asarray(action_tensor, dtype=np.float32).reshape(-1)
        if arr.shape[0] != len(keys):
            raise ValueError(f"Expected {len(keys)} dims, got {arr.shape[0]}")
        return {k: float(arr[i]) for i, k in enumerate(keys)}

    def _quantize_gripper_target(self, value: Any) -> float:
        """
        Force gripper commands to a binary {0, 100} target in the hardware range.
        """
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
        print(action)
        left_msg = JointState()
        right_msg = JointState()
        left_gripper_msg = JointState()
        right_gripper_msg = JointState()

        # 左臂6
        left_msg.position = [
            float(action.get("left_shoulder_pitch.pos", 0.0)),
            float(action.get("left_shoulder_roll.pos", 0.0)),
            float(action.get("left_elbow_pitch.pos", 0.0)),
            float(action.get("left_elbow_roll.pos", 0.0)),
            float(action.get("left_wrist_pitch.pos", 0.0)),
            float(action.get("left_wrist_roll.pos", 0.0)),
        ]
        # 右臂6
        right_msg.position = [
            float(action.get("right_shoulder_pitch.pos", 0.0)),
            float(action.get("right_shoulder_roll.pos", 0.0)),
            float(action.get("right_elbow_pitch.pos", 0.0)),
            float(action.get("right_elbow_roll.pos", 0.0)),
            float(action.get("right_wrist_pitch.pos", 0.0)),
            float(action.get("right_wrist_roll.pos", 0.0)),
        ]
        # 爪
        left_gripper_msg.position = [
            self._quantize_gripper_target(action.get("left_gripper.pos", 0.0))
        ]
        right_gripper_msg.position = [
            self._quantize_gripper_target(action.get("right_gripper.pos", 0.0))
        ]

        self._publish_actions(
            left_msg,
            right_msg,
            left_gripper_msg,
            right_gripper_msg,
        )
        return {"ok": True}

    def _publish_actions(
        self,
        left_msg: JointState,
        right_msg: JointState,
        left_gripper_msg: JointState,
        right_gripper_msg: JointState,
    ):
        self.left_joint_state_pub.publish(left_msg)
        self.right_joint_state_pub.publish(right_msg)
        self.left_gripper_joint_state_pub.publish(left_gripper_msg)
        self.right_gripper_joint_state_pub.publish(right_gripper_msg)

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
                hl_c, hr_c = self._img_counter["HL"], self._img_counter["HR"]
                wl_c, wr_c = self._img_counter["WL"], self._img_counter["WR"]
                hl_age = now - self._img_last_ts["HL"] if self._img_last_ts["HL"] > 0 else float("inf")
                hr_age = now - self._img_last_ts["HR"] if self._img_last_ts["HR"] > 0 else float("inf")
                wl_age = now - self._img_last_ts["WL"] if self._img_last_ts["WL"] > 0 else float("inf")
                wr_age = now - self._img_last_ts["WR"] if self._img_last_ts["WR"] > 0 else float("inf")
            rospy.loginfo(
                f"[R1LiteRobot] IMG HL={hl_c} (age {hl_age:.02f}s) "
                f"HR={hr_c} (age {hr_age:.02f}s) "
                f"WL={wl_c} (age {wl_age:.02f}s) "
                f"WR={wr_c} (age {wr_age:.02f}s)"
            )
            self._debug_thread_stop.wait(period)

    # ============================ Callbacks ============================

    def _cb_torso(self, msg: JointState):
        # 仍然读入以保持内部状态完整；14D 打包时不会使用
        with self._lock:
            n = min(4, len(msg.position))
            if n > 0:
                self._qpos_real[0:n] = np.array(msg.position[:n], dtype=np.float32)
            if len(msg.velocity) >= n and n > 0:
                self._qvel_real[0:n] = np.array(msg.velocity[:n], dtype=np.float32)
            self._torso_len = n

    def _cb_right(self, msg: JointState):
        with self._lock:
            n_arm = min(6, len(msg.position))
            if n_arm > 0:
                self._qpos_real[4:4 + n_arm] = np.array(msg.position[:n_arm], dtype=np.float32)
            if len(msg.velocity) >= n_arm and n_arm > 0:
                self._qvel_real[4:4 + n_arm] = np.array(msg.velocity[:n_arm], dtype=np.float32)
            if len(msg.position) >= 7:
                self._gripper_right_pos = float(msg.position[6])
            if len(msg.velocity) >= 7:
                self._gripper_right_vel = float(msg.velocity[6])

    def _cb_left(self, msg: JointState):
        with self._lock:
            n_arm = min(6, len(msg.position))
            if n_arm > 0:
                self._qpos_real[10:10 + n_arm] = np.array(msg.position[:n_arm], dtype=np.float32)
            if len(msg.velocity) >= n_arm and n_arm > 0:
                self._qvel_real[10:10 + n_arm] = np.array(msg.velocity[:n_arm], dtype=np.float32)
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

    def _cb_gripper_right(self, msg: JointState):
        with self._lock:
            if msg.position:
                self._gripper_right_pos = float(msg.position[0])
            if msg.velocity:
                self._gripper_right_vel = float(msg.velocity[0])

    # Cameras (compressed)
    def _cb_head_left_rgb_compressed(self, msg: CompressedImage):
        img_bgr = self._bridge.compressed_imgmsg_to_cv2(msg)
        with self._lock:
            self._head_left_rgb_bgr = img_bgr
            self._img_counter["HL"] += 1
            self._img_last_ts["HL"] = time.time()

    def _cb_head_right_rgb_compressed(self, msg: CompressedImage):
        img_bgr = self._bridge.compressed_imgmsg_to_cv2(msg)
        with self._lock:
            self._head_right_rgb_bgr = img_bgr
            self._img_counter["HR"] += 1
            self._img_last_ts["HR"] = time.time()

    def _cb_left_wrist_rgb_compressed(self, msg: CompressedImage):
        img_bgr = self._bridge.compressed_imgmsg_to_cv2(msg)
        with self._lock:
            self._left_wrist_rgb_bgr = img_bgr
            self._img_counter["WL"] += 1
            self._img_last_ts["WL"] = time.time()

    def _cb_right_wrist_rgb_compressed(self, msg: CompressedImage):
        img_bgr = self._bridge.compressed_imgmsg_to_cv2(msg)
        with self._lock:
            self._right_wrist_rgb_bgr = img_bgr
            self._img_counter["WR"] += 1
            self._img_last_ts["WR"] = time.time()
