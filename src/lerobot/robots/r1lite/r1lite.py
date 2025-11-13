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
    R1 Lite robot integration.

    Observation:
      - 'observation.state': float32, shape (62,) with exact ordering:
          0-6   : left arm pos(7)  -> 6关节 + 左爪
          7-13  : left arm vel(7)  -> 6关节速度 + 左爪速度
          14-20 : right arm pos(7) -> 6关节 + 右爪
          21-27 : right arm vel(7) -> 6关节速度 + 右爪速度
          28-31 : torso pos(<=4) 读到几维就写几维，其余补0
          32-35 : torso vel(<=4) 读到几维就写几维，其余补0
          36    : left gripper pos(1)
          37    : right gripper pos(1)
          38-41 : IMU quat (x,y,z,w)
          42-44 : IMU gyro (wx,wy,wz)
          45-47 : IMU acc  (ax,ay,az)
          48-54 : left EE  (x,y,z,qx,qy,qz,qw)
          55-61 : right EE (x,y,z,qx,qy,qz,qw)
      - Four RGB images (compressed sources only):
          'observation.images.head_left_rgb'
          'observation.images.head_right_rgb'
          'observation.images.left_wrist_rgb'
          'observation.images.right_wrist_rgb'

    Action (20-D order):
      1) Left arm target joints (6)
      2) Right arm target joints (6)
      3) Left gripper target (1)
      4) Right gripper target (1)
      5) Chassis target speed (vx, vy, wz) (3)
      6) Torso target speed (vx, vy, wz) (3)
    """

    config_class = R1LiteRobotConfig
    name = "r1lite"

    # ============================ Lifecycle & ROS IO ============================

    def __init__(self, config: R1LiteRobotConfig):
        super().__init__(config)
        self.config = config

        rospy.init_node("r1lite_robot", anonymous=True)

        # ---------------- Publishers ----------------
        self.left_joint_state_pub = rospy.Publisher(
            "/motion_target/target_joint_state_arm_left", JointState, queue_size=10
        )
        self.right_joint_state_pub = rospy.Publisher(
            "/motion_target/target_joint_state_arm_right", JointState, queue_size=10
        )
        self.torso_joint_state_pub = rospy.Publisher(
            "/motion_target/target_joint_state_torso", JointState, queue_size=10
        )
        self.left_gripper_joint_state_pub = rospy.Publisher(
            "/motion_target/target_position_gripper_left", JointState, queue_size=10
        )
        self.right_gripper_joint_state_pub = rospy.Publisher(
            "/motion_target/target_position_gripper_right", JointState, queue_size=10
        )
        self.chassis_speed_pub = rospy.Publisher(
            "/motion_target/target_speed_chassis", TwistStamped, queue_size=10
        )
        self.torso_speed_pub = rospy.Publisher(
            "/motion_target/target_speed_torso", TwistStamped, queue_size=10
        )
        # optional (kept for completeness)
        self.acc_limit_pub = rospy.Publisher(
            "/motion_target/chassis_acc_limit", TwistStamped, queue_size=10
        )
        self.braking_mode_pub = rospy.Publisher(
            "/motion_target/brake_mode", Bool, queue_size=10
        )

        # ---------------- Internal state caches ----------------
        self._lock = threading.Lock()
        self._bridge = CvBridge()
        self._is_connected = False
        self._is_calibrated = False

        # 内部关节布局：torso(最多4) + right(6) + left(6) = 16 槽位
        self._qpos_real = np.zeros(16, dtype=np.float32)
        self._qvel_real = np.zeros(16, dtype=np.float32)
        self._torso_len = 0  # 最近一次 torso 实际读到的维度（<=4）

        # Grippers：位置+速度（爪作为各臂的第7维）
        self._gripper_left_pos = 0.0
        self._gripper_right_pos = 0.0
        self._gripper_left_vel = 0.0
        self._gripper_right_vel = 0.0

        # IMU
        self._imu_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        self._imu_gyro = np.zeros(3, dtype=np.float32)
        self._imu_acc = np.zeros(3, dtype=np.float32)

        # End-effectors (7 each: x,y,z,qx,qy,qz,qw)
        self._ee_left = np.zeros(7, dtype=np.float32)
        self._ee_right = np.zeros(7, dtype=np.float32)

        # Cameras (BGR in cache; convert to RGB when exporting)
        self._head_left_rgb_bgr: Optional[np.ndarray] = None
        self._head_right_rgb_bgr: Optional[np.ndarray] = None
        self._left_wrist_rgb_bgr: Optional[np.ndarray] = None
        self._right_wrist_rgb_bgr: Optional[np.ndarray] = None

        # Debug counters
        self._img_counter = {"HL": 0, "HR": 0, "WL": 0, "WR": 0}
        self._img_last_ts = {"HL": 0.0, "HR": 0.0, "WL": 0.0, "WR": 0.0}
        self._debug_thread_stop = threading.Event()
        self._debug_thread: Optional[threading.Thread] = None

        # ---------------- Subscribers (compressed RGB only) ----------------
        # Head cameras (compressed)
        rospy.Subscriber(
            "/hdas/camera_head/left_raw/image_raw_color/compressed",
            CompressedImage, self._cb_head_left_rgb_compressed, queue_size=1
        )
        rospy.Subscriber(
            "/hdas/camera_head/right_raw/image_raw_color/compressed",
            CompressedImage, self._cb_head_right_rgb_compressed, queue_size=1
        )

        # Wrist cameras (compressed)
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

        # IMU + EE pose
        rospy.Subscriber("/hdas/imu_torso", Imu, self._cb_imu, queue_size=50)  # 如需改成 /hdas/imu_chassis 可替换
        rospy.Subscriber("/motion_control/pose_ee_arm_left", PoseStamped, self._cb_ee_left, queue_size=50)
        rospy.Subscriber("/motion_control/pose_ee_arm_right", PoseStamped, self._cb_ee_right, queue_size=50)

        rospy.loginfo("[R1LiteRobot] Publishers and Subscribers ready (compressed RGB only).")

    # ============================ Feature Specs ============================

    @cached_property
    def observation_features(self) -> Dict[str, Any]:
        cam = self.config.cameras
        return {
            # 关节状态：只需要告诉函数这些是 float 值
            **{f"joint_{i}": float for i in range(62)},  # 或者用真实关节名

            # 相机特征：值为 tuple 形状
            "head_left_rgb":  (cam["head_left_rgb"].height,  cam["head_left_rgb"].width,  3),
            "head_right_rgb": (cam["head_right_rgb"].height, cam["head_right_rgb"].width, 3),
            "left_wrist_rgb": (cam["left_wrist_rgb"].height, cam["left_wrist_rgb"].width, 3),
            "right_wrist_rgb":(cam["right_wrist_rgb"].height,cam["right_wrist_rgb"].width,3),
        }



    @cached_property
    def action_features(self) -> Dict[str, type]:
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
            # Chassis speed (3)
            "chassis.vx": float,
            "chassis.vy": float,
            "chassis.wz": float,
            # Torso speed (3)
            "torso.vx": float,
            "torso.vy": float,
            "torso.wz": float,
        }

    # ============================ Lifecycle ============================

    def configure(self) -> None:
        """ROS 版可为空实现。"""
        return

    def calibrate(self) -> None:
        """可选标定步骤。若无标定流程，标记为已校准即可。"""
        self._is_calibrated = True

    def connect(self, calibrate: bool = True) -> bool:
        """
        Check required topics (compressed RGB endpoints).
        """
        required_topics = [
            "/hdas/feedback_torso",
            "/hdas/feedback_arm_right",
            "/hdas/feedback_arm_left",
            "/hdas/feedback_gripper_left",
            "/hdas/feedback_gripper_right",
            "/hdas/imu_torso",  # or /hdas/imu_chassis
            "/motion_control/pose_ee_arm_left",
            "/motion_control/pose_ee_arm_right",
            "/hdas/camera_head/left_raw/image_raw_color/compressed",
            "/hdas/camera_head/right_raw/image_raw_color/compressed",
            "/hdas/camera_wrist_left/color/image_raw/compressed",
            "/hdas/camera_wrist_right/color/image_raw/compressed",
        ]
        active = [name for (name, _type) in rospy.get_published_topics()]
        ok = all(topic in active for topic in required_topics)

        if not ok:
            rospy.logwarn("[R1LiteRobot] Not all required topics are active (compressed RGB).")
            self._is_connected = False
            return False

        if calibrate:
            rospy.loginfo("[R1LiteRobot] (Optional) Calibration step...")
            self.calibrate()

        rospy.loginfo("[R1LiteRobot] All required topics are active.")
        self._is_connected = True

        # 启动调试线程，1Hz 打印一次计数与帧龄
        self._start_debug_thread()
        return True

    def disconnect(self) -> None:
        """断开与机器人/ROS 的连接，释放资源。"""
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
            state = np.zeros(62, dtype=np.float32)
            qpos, qvel = self._qpos_real, self._qvel_real

            # 左/右臂：6关节 + 爪第7维
            left_pos7 = np.empty(7, dtype=np.float32)
            left_pos7[:6] = qpos[10:16]
            left_pos7[6]  = float(self._gripper_left_pos)

            left_vel7 = np.empty(7, dtype=np.float32)
            left_vel7[:6] = qvel[10:16]
            left_vel7[6]  = float(self._gripper_left_vel)

            right_pos7 = np.empty(7, dtype=np.float32)
            right_pos7[:6] = qpos[4:10]
            right_pos7[6]  = float(self._gripper_right_pos)

            right_vel7 = np.empty(7, dtype=np.float32)
            right_vel7[:6] = qvel[4:10]
            right_vel7[6]  = float(self._gripper_right_vel)

            # 写入 62D
            state[0:7]   = left_pos7
            state[7:14]  = left_vel7
            state[14:21] = right_pos7
            state[21:28] = right_vel7

            n = int(self._torso_len) if (0 <= self._torso_len <= 4) else 0
            if n > 0:
                state[28:28+n] = qpos[0:n]
                state[32:32+n] = qvel[0:n]

            state[36] = float(self._gripper_left_pos)
            state[37] = float(self._gripper_right_pos)

            state[38:42] = self._imu_quat
            state[42:45] = self._imu_gyro
            state[45:48] = self._imu_acc
            state[48:55] = self._ee_left
            state[55:62] = self._ee_right

            # 确保 float32
            state = state.astype(np.float32, copy=False)

            # ✅ 关键改动 1：展开为 joint_0..joint_61
            for i in range(62):
                obs[f"joint_{i}"] = float(state[i])

            # （可选）兼容其它用法：同时放整向量，若别处需要
            obs["observation.state"] = state
            obs["state"] = state

            # ✅ 关键改动 2：相机用“裸键名”，不要前缀
            if self._head_left_rgb_bgr is not None:
                img = cv2.cvtColor(self._head_left_rgb_bgr, cv2.COLOR_BGR2RGB)
                obs["head_left_rgb"] = img.astype(np.uint8, copy=False)
            if self._head_right_rgb_bgr is not None:
                img = cv2.cvtColor(self._head_right_rgb_bgr, cv2.COLOR_BGR2RGB)
                obs["head_right_rgb"] = img.astype(np.uint8, copy=False)
            if self._left_wrist_rgb_bgr is not None:
                img = cv2.cvtColor(self._left_wrist_rgb_bgr, cv2.COLOR_BGR2RGB)
                obs["left_wrist_rgb"] = img.astype(np.uint8, copy=False)
            if self._right_wrist_rgb_bgr is not None:
                img = cv2.cvtColor(self._right_wrist_rgb_bgr, cv2.COLOR_BGR2RGB)
                obs["right_wrist_rgb"] = img.astype(np.uint8, copy=False)

        # 日志
        try:
            rospy.loginfo(
                f"[R1LiteRobot] obs keys={list(obs.keys())[:10]}..., "
                f"state.shape={state.shape}, state.dtype={state.dtype}"
            )
        except Exception as e:
            rospy.logwarn(f"[R1LiteRobot] obs logging failed: {e}")

        return obs

    # ============================ Actions ============================

    def action_tensor_to_dict(self, action_tensor) -> Dict[str, float]:
        keys = list(self.action_features.keys())  # len == 20
        arr = np.asarray(action_tensor, dtype=np.float32).reshape(-1)
        if arr.shape[0] != len(keys):
            raise ValueError(f"Expected {len(keys)} dims, got {arr.shape[0]}")
        return {k: float(arr[i]) for i, k in enumerate(keys)}

    def send_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        left_msg = JointState()
        right_msg = JointState()
        torso_msg = JointState()
        left_gripper_msg = JointState()
        right_gripper_msg = JointState()
        chassis_twist = TwistStamped()
        torso_twist = TwistStamped()
        print(action)
        # 1) Left arm (6)
        left_msg.position = [
            float(action.get("left_shoulder_pitch.pos", 0.0)),
            float(action.get("left_shoulder_roll.pos", 0.0)),
            float(action.get("left_elbow_pitch.pos", 0.0)),
            float(action.get("left_elbow_roll.pos", 0.0)),
            float(action.get("left_wrist_pitch.pos", 0.0)),
            float(action.get("left_wrist_roll.pos", 0.0)),
        ]
        # 2) Right arm (6)
        right_msg.position = [
            float(action.get("right_shoulder_pitch.pos", 0.0)),
            float(action.get("right_shoulder_roll.pos", 0.0)),
            float(action.get("right_elbow_pitch.pos", 0.0)),
            float(action.get("right_elbow_roll.pos", 0.0)),
            float(action.get("right_wrist_pitch.pos", 0.0)),
            float(action.get("right_wrist_roll.pos", 0.0)),
        ]
        # 3) Left gripper (1)
        left_gripper_msg.position = [float(action.get("left_gripper.pos", 0.0))]
        # 4) Right gripper (1)
        right_gripper_msg.position = [float(action.get("right_gripper.pos", 0.0))]

        # 5) Chassis twist (3)
        chassis_twist.twist.linear.x = float(action.get("chassis.vx", 0.0))
        chassis_twist.twist.linear.y = float(action.get("chassis.vy", 0.0))
        chassis_twist.twist.angular.z = float(action.get("chassis.wz", 0.0))
        chassis_twist.header.stamp = rospy.Time.now()


        # 6) Torso twist (3)
        torso_twist.twist.linear.x = float(action.get("torso.vx", 0.0))
        torso_twist.twist.linear.y = float(action.get("torso.vy", 0.0))
        torso_twist.twist.angular.z = float(action.get("torso.wz", 0.0))
        torso_twist.header.stamp = chassis_twist.header.stamp

        # （可选）躯干位置目标 via JointState（如果你使用它们）
        torso_msg.position = [
            float(action.get("torso_1.pos", 0.0)),
            float(action.get("torso_2.pos", 0.0)),
            float(action.get("torso_3.pos", 0.0)),
        ]

        self._publish_actions(
            left_msg,
            right_msg,
            torso_msg,
            left_gripper_msg,
            right_gripper_msg,
            chassis_twist,
            torso_twist,
        )
        return {"ok": True}

    def _publish_actions(
        self,
        left_msg: JointState,
        right_msg: JointState,
        torso_msg: JointState,
        left_gripper_msg: JointState,
        right_gripper_msg: JointState,
        chassis_twist: TwistStamped,
        torso_twist: TwistStamped,
    ):
        self.left_joint_state_pub.publish(left_msg)
        self.right_joint_state_pub.publish(right_msg)
        # self.torso_joint_state_pub.publish(torso_msg)
        self.left_gripper_joint_state_pub.publish(left_gripper_msg)
        self.right_gripper_joint_state_pub.publish(right_gripper_msg)
        # self.chassis_speed_pub.publish(chassis_twist)
        # self.torso_speed_pub.publish(torso_twist)

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
                f"[R1LiteRobot] IMG HL={hl_c}(age {hl_age:.02f}s) HR={hr_age:.02f}s) "
                f"WL={wl_c}(age {wl_age:.02f}s) WR={wr_c}(age {wr_age:.02f}s)"
            )
            self._debug_thread_stop.wait(period)

    # ============================ Callbacks ============================

    # Torso（读到几维就写几维，最多4）
    def _cb_torso(self, msg: JointState):
        with self._lock:
            n = min(4, len(msg.position))
            if n > 0:
                self._qpos_real[0:n] = np.array(msg.position[:n], dtype=np.float32)
            if len(msg.velocity) >= n and n > 0:
                self._qvel_real[0:n] = np.array(msg.velocity[:n], dtype=np.float32)
            self._torso_len = n

    # Right arm（前6维为关节；第7维（如存在）视为右爪）
    def _cb_right(self, msg: JointState):
        with self._lock:
            n_arm = min(6, len(msg.position))
            if n_arm > 0:
                self._qpos_real[4:4 + n_arm] = np.array(msg.position[:n_arm], dtype=np.float32)
            if len(msg.velocity) >= n_arm and n_arm > 0:
                self._qvel_real[4:4 + n_arm] = np.array(msg.velocity[:n_arm], dtype=np.float32)
            # 第7维当作“右爪”
            if len(msg.position) >= 7:
                self._gripper_right_pos = float(msg.position[6])
            if len(msg.velocity) >= 7:
                self._gripper_right_vel = float(msg.velocity[6])

    # Left arm（前6维为关节；第7维（如存在）视为左爪）
    def _cb_left(self, msg: JointState):
        with self._lock:
            n_arm = min(6, len(msg.position))
            if n_arm > 0:
                self._qpos_real[10:10 + n_arm] = np.array(msg.position[:n_arm], dtype=np.float32)
            if len(msg.velocity) >= n_arm and n_arm > 0:
                self._qvel_real[10:10 + n_arm] = np.array(msg.velocity[:n_arm], dtype=np.float32)
            # 第7维当作“左爪”
            if len(msg.position) >= 7:
                self._gripper_left_pos = float(msg.position[6])
            if len(msg.velocity) >= 7:
                self._gripper_left_vel = float(msg.velocity[6])

    # Grippers（可选保底：若单独有爪反馈，则以此为准覆盖）
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

    # IMU & EE poses
    def _cb_imu(self, msg: Imu):
        with self._lock:
            q = msg.orientation
            self._imu_quat[:] = [q.x, q.y, q.z, q.w]
            self._imu_gyro[:] = [
                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z,
            ]
            self._imu_acc[:] = [
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z,
            ]

    def _cb_ee_left(self, msg: PoseStamped):
        with self._lock:
            p = msg.pose.position
            q = msg.pose.orientation
            self._ee_left[:] = [p.x, p.y, p.z, q.x, q.y, q.z, q.w]

    def _cb_ee_right(self, msg: PoseStamped):
        with self._lock:
            p = msg.pose.position
            q = msg.pose.orientation
            self._ee_right[:] = [p.x, p.y, p.z, q.x, q.y, q.z, q.w]
