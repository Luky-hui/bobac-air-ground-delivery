#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
task2_grasp_tag79_curobo.py

基于 cuRobo MotionGen 的 AR 码物块抓取脚本。
替代 task2_grasp_tag79_v5.py 中的数值雅可比方案，实现更稳定、无碰撞的运动规划。

架构：
- 作为独立 ROS2 节点运行
- 通过 TF 和话题与 Isaac Sim 通信
- 使用 cuRobo MotionGen 进行运动规划

运行方式：
cd ~/ros2_ws/src/isaacsim2026/isaac_sim_2026/curobo
~/isaac-sim-4.5.0/python.sh ../launch/task2_grasp_tag79_curobo.py --ros-args -p use_sim_time:=true

常用参数：
-p robot_cfg_path:=/home/u-zhuang/ros2_ws/src/isaac/isaac_sim_2026/config/turingstack_x1_2_right_arm.yml
-p pregrasp_z_offset:=0.18
-p grasp_z_offset:=0.06
-p lift_dz:=0.15
-p interp_dt:=0.02
-p target_offset_xyz:=[0.0,0.0,0.0]  # extra XYZ offset applied to target
-p use_gripper_center_as_tcp:=True  # compute TCP offset from gripper left/right frames
-p gripper_left_frame:=gripper01_left1
-p gripper_right_frame:=gripper01_right1

直接坐标模式 (绕过 TF 查找):
-p use_direct_coords:=True
-p direct_grasp_position:=[X,Y,Z]  # 在 base_link 坐标系中的目标位置
-p direct_mode_auto_adjust:=True   # 自动旋转底盘使目标进入右臂工作空间
-p direct_mode_target_y:=-0.15     # 目标 Y 位置（右臂工作空间内）

示例 - 直接坐标抓取:
~/isaac-sim-4.5.0/python.sh ../launch/task2_grasp_tag79_curobo.py --ros-args \
  -p use_sim_time:=true \
  -p use_direct_coords:=True \
  -p direct_grasp_position:=[0.455,0.401,0.866] \
  -p distance_correction_x:=0
"""

import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---- ROS2 imports ----
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool

from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

# ---- cuRobo imports ----
try:
    import torch
    from curobo.types.base import TensorDeviceType
    from curobo.types.math import Pose
    from curobo.types.robot import JointState as CuJointState
    from curobo.geom.sdf.world import CollisionCheckerType
    from curobo.geom.types import WorldConfig
    from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
    from curobo.util_file import load_yaml
    CUROBO_AVAILABLE = True
    CUROBO_IMPORT_ERROR = None
except Exception as e:
    CUROBO_AVAILABLE = False
    CUROBO_IMPORT_ERROR = e
    torch = None


def _is_arm_joint(name: str) -> bool:
    """检查是否为机械臂关节"""
    return name.startswith("joint") and (name.endswith("_R") or name.endswith("_L"))


def _joint_index(name: str) -> int:
    """从关节名称提取索引号"""
    m = ""
    for ch in name:
        if ch.isdigit():
            m += ch
    return int(m) if m else 999


def _xyzw_to_wxyz(q_xyzw: np.ndarray) -> np.ndarray:
    """将 ROS 四元数 (x,y,z,w) 转换为 cuRobo 格式 (w,x,y,z)"""
    x, y, z, w = q_xyzw.tolist()
    return np.array([w, x, y, z], dtype=np.float32)

def _rotate_vec_by_quat_wxyz(v: np.ndarray, q_wxyz: np.ndarray) -> np.ndarray:
    """Rotate vector v by quaternion q (w,x,y,z)."""
    w, x, y, z = q_wxyz.tolist()
    r00 = 1.0 - 2.0 * (y * y + z * z)
    r01 = 2.0 * (x * y - z * w)
    r02 = 2.0 * (x * z + y * w)
    r10 = 2.0 * (x * y + z * w)
    r11 = 1.0 - 2.0 * (x * x + z * z)
    r12 = 2.0 * (y * z - x * w)
    r20 = 2.0 * (x * z - y * w)
    r21 = 2.0 * (y * z + x * w)
    r22 = 1.0 - 2.0 * (x * x + y * y)
    rx = r00 * v[0] + r01 * v[1] + r02 * v[2]
    ry = r10 * v[0] + r11 * v[1] + r12 * v[2]
    rz = r20 * v[0] + r21 * v[1] + r22 * v[2]
    return np.array([rx, ry, rz], dtype=np.float32)

def _quat_wxyz_to_rot(q_wxyz: np.ndarray) -> np.ndarray:
    """Convert quaternion (w,x,y,z) to rotation matrix."""
    w, x, y, z = q_wxyz.tolist()
    r00 = 1.0 - 2.0 * (y * y + z * z)
    r01 = 2.0 * (x * y - z * w)
    r02 = 2.0 * (x * z + y * w)
    r10 = 2.0 * (x * y + z * w)
    r11 = 1.0 - 2.0 * (x * x + z * z)
    r12 = 2.0 * (y * z - x * w)
    r20 = 2.0 * (x * z - y * w)
    r21 = 2.0 * (y * z + x * w)
    r22 = 1.0 - 2.0 * (x * x + y * y)
    return np.array([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]], dtype=np.float32)


def _rot_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to quaternion (w,x,y,z)."""
    tr = float(R[0, 0] + R[1, 1] + R[2, 2])
    if tr > 0.0:
        S = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    q = np.array([w, x, y, z], dtype=np.float32)
    q /= np.linalg.norm(q) + 1e-9
    return q


def _normalize_vec(v: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm < eps:
        return v
    return v / norm




class MovingAverageFilter:
    """滑动窗口平均滤波器，用于平滑 AprilTag 坐标"""

    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.buffer: List[np.ndarray] = []

    def update(self, value: np.ndarray) -> np.ndarray:
        """更新滤波器并返回平均值"""
        self.buffer.append(value.copy())
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        return np.mean(self.buffer, axis=0).astype(np.float32)

    def reset(self):
        """重置滤波器"""
        self.buffer = []

    def is_stable(self, threshold: float = 0.01) -> bool:
        """检查数据是否稳定（标准差小于阈值）"""
        if len(self.buffer) < 3:
            return False
        std = np.std(self.buffer, axis=0)
        return float(np.max(std)) < threshold


class Task2GraspTag79CuRobo(Node):
    """基于 cuRobo MotionGen 的 AR 码物块抓取控制器"""

    # ---- 关节限位（弧度）- 与 URDF 精确同步 ----
    # Source: mercury_x1_turing_stack.urdf lines 474-533
    # Updated to exact URDF values for maximum IK accuracy
    JOINT_LIMITS = {
        "joint1_R": (-2.879, 2.879),     # -165° to 165° (URDF exact)
        "joint2_R": (-0.9599, 1.6581),   # -55° to 95° (URDF exact)
        "joint3_R": (-3.0194, 0.0873),   # -173° to 5° (URDF exact)
        "joint4_R": (-2.879, 2.879),     # -165° to 165° (URDF exact)
        "joint5_R": (-0.349, 4.625),     # -20° to 265° (URDF exact)
        "joint6_R": (-3.14, 3.14),       # -180° to 180° (URDF exact)
    }

    # 关节限位边距（弧度）- 约 5.7°
    JOINT_LIMIT_MARGIN = 0.1

    def __init__(self):
        super().__init__("task2_grasp_tag79_curobo")

        # ---- 参数声明 ----
        # 注意：use_sim_time 是 ROS2 内置参数，通过命令行传递，不需要声明

        # 坐标系
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("ee_frame", "link7_R")
        self.declare_parameter("tag_frame", "tag36h11:79")
        self.declare_parameter("tag_source", "fused")  # fused|left|right|direct
        self.declare_parameter("tag_fusion_warn_dist", 0.05)
        self.declare_parameter("tag_fusion_warn_period", 1.0)

        # cuRobo 配置
        self.declare_parameter(
            "robot_cfg_path",
            "/home/u-zhuang/ros2_ws/src/isaac/isaac_sim_2026/config/turingstack_x1_2_right_arm.yml"
        )
        self.declare_parameter(
            "curobo_joint_names",
            ["joint1_R", "joint2_R", "joint3_R", "joint4_R", "joint5_R", "joint6_R"]
        )

        # 抓取几何参数（米）
        self.declare_parameter("pregrasp_z_offset", 0.12)  # 预抓取高度（标签上方，降低以减少倾倒风险）
        self.declare_parameter("grasp_z_offset", 0.06)     # 抓取高度（接近标签）
        self.declare_parameter("lift_dz", 0.10)            # 抬起高度（降低以减少倾倒风险）

        # 可选 XY 偏移（米）
        self.declare_parameter("grasp_x_offset", 0.0)
        self.declare_parameter("grasp_y_offset", 0.0)

        # 距离修正参数（米）- 用于修正视觉测距偏差
        # 如果视觉测距总是比实际距离大，使用负值将目标拉近
        self.declare_parameter("distance_correction_x", 0.0)  # X方向修正 cm（增加以减少倾倒风险）
        self.declare_parameter("distance_correction_y", 0.0)    # Y方向修正
        self.declare_parameter("distance_correction_z", 0.0)    # Z方向修正
        self.declare_parameter("target_offset_xyz", [0.0, 0.0, 0.0])
        self.declare_parameter("use_gripper_center_as_tcp", False)
        self.declare_parameter("gripper_left_frame", "gripper01_left1")
        self.declare_parameter("gripper_right_frame", "gripper01_right1")
        self.declare_parameter("gripper_center_samples", 20)
        self.declare_parameter("gripper_center_timeout", 1.5)

        # 时间参数
        self.declare_parameter("interp_dt", 0.02)          # 轨迹插值时间步
        self.declare_parameter("exec_dt", 0.1)             # 执行时间步（10Hz，防止机器人倾倒）
        # safety limits
        self.declare_parameter("max_joint_speed", 1.0)
        self.declare_parameter("max_joint_step", 0.12)
        self.declare_parameter("max_target_distance", 0.7)
        self.declare_parameter("max_pose_tf_age", 0.5)
        self.declare_parameter("abort_on_workspace_violation", False)
        # self.declare_parameter("traj_speed_factor", 0.3)   # 不再使用，已在 _exec_traj 中固定
        self.declare_parameter("max_attempts", 20)         # 最大规划尝试次数

        # TF 参数
        self.declare_parameter("max_tf_age", 0.5)
        self.declare_parameter("tag_filter_alpha", 0.4)
        self.declare_parameter("stable_need", 15)  # 从 30 降到 15
        self.declare_parameter("tf_wait_timeout", 30.0)  # TF 等待超时（秒）

        # 夹爪参数（与 v5 保持一致）
        self.declare_parameter("grip_open", 0.0)
        self.declare_parameter("grip_close", 0.25)  # 修正：从 0.4 改为 0.25
        self.declare_parameter("grip_steps", 25)

        # 世界碰撞配置（简单桌面）
        self.declare_parameter("table_dims", [2.0, 2.0, 0.1])
        self.declare_parameter("table_pose", [0.0, 0.0, -0.05, 1.0, 0.0, 0.0, 0.0])
        self.declare_parameter("collision_checker_type", "PRIMITIVE")
        self.declare_parameter("sync_world_from_stage", False)
        self.declare_parameter("sync_world_use_live_stage", True)
        self.declare_parameter("sync_world_stage_path", "")
        self.declare_parameter("sync_world_reference_prim", "/World")
        self.declare_parameter("sync_world_only_paths", ["/World"])
        self.declare_parameter("sync_world_ignore_paths", [])
        self.declare_parameter(
            "sync_world_ignore_substrings",
            [
                "/curobo",
                "/World/defaultGroundPlane",
                "/World/target",
                "/World/TuringStack_X1",
                "/World/mercury_x1",
            ],
        )
        self.declare_parameter("sync_world_update_interval", 2.0)
        self.declare_parameter("sync_world_fallback_to_table", True)
        self.declare_parameter("sync_world_mesh_process", False)

        # 安全姿态（与 v5 保持一致）
        self.declare_parameter("safe_pose_R", [0.9, 0.0, -0.05, -0.5, 0.0, 0.0])
        self.declare_parameter("safe_pose_L", [0.9, -0.3, 0.0, -0.5, 0.0, 0.0])
        self.declare_parameter("safe_pose_duration", 3.0)

        # 调试模式
        self.declare_parameter("debug_mode", False)
        self.declare_parameter("reachability_diagnostic_mode", False)

        # pose match mode (task2)
        self.declare_parameter("pose_match_mode", False)
        self.declare_parameter("pose_match_count", 3)
        self.declare_parameter("pose_match_z_offset", 0.08)
        self.declare_parameter("prefer_downward_grasp", True)
        self.declare_parameter("adaptive_downward_grasp", True)
        self.declare_parameter("downward_grasp_mode", "ee")  # ee|tag
        self.declare_parameter("pose_match_hold_sec", 2.0)
        self.declare_parameter("pose_match_min_delta", 0.05)
        self.declare_parameter("pose_match_wait_timeout", 60.0)
        self.declare_parameter("pose_log_rate_hz", 10.0)

        # alignment gate
        self.declare_parameter("wait_for_alignment", True)
        self.declare_parameter("alignment_done_topic", "/tag_alignment_done")
        self.declare_parameter("alignment_wait_timeout", 120.0)

        # 直接坐标模式（绕过 TF 查找）
        self.declare_parameter("use_direct_coords", False)
        self.declare_parameter("direct_grasp_position", [0.0, 0.0, 0.0])  # XYZ in base_link frame
        self.declare_parameter("direct_mode_auto_adjust", True)  # 直接坐标模式下自动调整底盘
        self.declare_parameter("direct_mode_target_y", -0.15)    # 目标 Y 位置（右臂工作空间内）

        # 底盘移动参数（已禁用 - 移动会导致目标丢失）
        self.declare_parameter("enable_base_movement", False)  # 禁用底盘移动
        self.declare_parameter("base_linear_speed", 0.15)  # m/s
        self.declare_parameter("base_angular_speed", 0.3)  # rad/s
        self.declare_parameter("base_move_timeout", 10.0)  # seconds
        self.declare_parameter("target_x_in_workspace", 0.40)  # 目标 X 位置（工作空间内）
        self.declare_parameter("target_y_in_workspace", -0.10)  # 目标 Y 位置（右侧）
        self.declare_parameter("workspace_x_margin", 0.08)  # X 方向容差
        self.declare_parameter("workspace_y_margin", 0.15)  # Y 方向容差

        # ---- ROS I/O ----
        qos = QoSProfile(depth=10)
        self.sub = self.create_subscription(
            JointState, "/joint_states", self._on_joint_states, qos
        )
        self.pub = self.create_publisher(JointState, "/joint_command", qos)
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        alignment_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self._alignment_done = False
        self._alignment_sub = self.create_subscription(
            Bool,
            self.get_parameter("alignment_done_topic").value,
            self._on_alignment_done,
            alignment_qos,
        )

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ---- 状态变量 ----
        self.pose: Dict[str, float] = {}
        self.data_ready = False

        self.R_JOINTS: List[str] = []
        self.L_JOINTS: List[str] = []
        self.GRIP_R: Optional[str] = None

        self.cmd: Dict[str, float] = {}
        self._tag_filt: Optional[np.ndarray] = None
        self._pose_log_period = None
        self._last_pose_log_time = 0.0
        self._pose_match_total = 0
        self._tcp_offset_ee = None
        self._last_tag_warn_time = 0.0
        self._ee_tf_warned = False
        self._usd_helper = None
        self._last_world_sync_time = 0.0
        self._world_sync_failed = False
        self._collision_checker_type = None

        # ---- cuRobo ----
        self.motion_gen = None
        self.tensor_args = None

        self.get_logger().info("[Init] Task2GraspTag79CuRobo 节点已创建")

    # ---------------------- ROS 回调 ----------------------
    def _on_joint_states(self, msg: JointState):
        """处理关节状态消息"""
        for n, p in zip(msg.name, msg.position):
            self.pose[n] = float(p)

        if not self.R_JOINTS:
            allj = [n for n in msg.name if _is_arm_joint(n)]
            allj.sort(key=_joint_index)
            self.R_JOINTS = [j for j in allj if j.endswith("_R")]
            self.L_JOINTS = [j for j in allj if j.endswith("_L")]
            self.GRIP_R = "gripper_controller" if "gripper_controller" in msg.name else None
            self.get_logger().info(
                f"[Discover] R={self.R_JOINTS} L={self.L_JOINTS} grip={self.GRIP_R}"
            )

        self.data_ready = True

    # ---------------------- TF 辅助 ----------------------
    def _is_finite(self, arr: np.ndarray) -> bool:
        return bool(np.all(np.isfinite(arr)))

    def _lookup_tf(
        self, parent: str, child: str
    ) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
        """
        查找 TF 变换
        返回: (pos_xyz, quat_xyzw, age_sec) 或 None
        """
        try:
            try:
                zero = Time(clock_type=self.get_clock().clock_type)
            except TypeError:
                zero = Time()
            tf = self.tf_buffer.lookup_transform(parent, child, zero)

            t = tf.transform.translation
            q = tf.transform.rotation
            p = np.array([t.x, t.y, t.z], dtype=np.float32)
            quat = np.array([q.x, q.y, q.z, q.w], dtype=np.float32)
            if not self._is_finite(p) or not self._is_finite(quat):
                return None

            try:
                stamp = Time.from_msg(tf.header.stamp, clock_type=self.get_clock().clock_type)
            except TypeError:
                stamp = Time.from_msg(tf.header.stamp)
            age = (self.get_clock().now() - stamp).nanoseconds * 1e-9
            return p, quat, float(age)
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None

    def _tag_filtered(self, p: np.ndarray) -> np.ndarray:
        """低通滤波标签位置"""
        a = float(self.get_parameter("tag_filter_alpha").value)
        if self._tag_filt is None:
            self._tag_filt = p.copy()
        else:
            self._tag_filt = a * p + (1.0 - a) * self._tag_filt
        return self._tag_filt.copy()

    def _lookup_stereo_tf(
        self, parent: str, tag_id: str = "tag36h11:79", allow_stale: bool = False
    ) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
        """
        Query tag TF with optional stereo fusion.

        left_tag36h11:79 / right_tag36h11:79 are published by stereo cameras.
        Returns: (pos_xyz, quat_xyzw, age_sec) or None.
        """
        left_child = f"left_{tag_id}"
        right_child = f"right_{tag_id}"
        tag_source = str(self.get_parameter("tag_source").value).lower()
        max_age = float(self.get_parameter("max_tf_age").value)

        def _age_ok(tf_msg: Optional[Tuple[np.ndarray, np.ndarray, float]]) -> bool:
            return tf_msg is not None and (allow_stale or tf_msg[2] < max_age)

        left_tf = self._lookup_tf(parent, left_child)
        right_tf = self._lookup_tf(parent, right_child)

        if tag_source == "left":
            if _age_ok(left_tf):
                return left_tf
            if _age_ok(right_tf):
                return right_tf
            return self._lookup_tf(parent, tag_id)
        if tag_source == "right":
            if _age_ok(right_tf):
                return right_tf
            if _age_ok(left_tf):
                return left_tf
            return self._lookup_tf(parent, tag_id)
        if tag_source == "direct":
            return self._lookup_tf(parent, tag_id)

        # fused (default)
        if not _age_ok(left_tf):
            left_tf = None
        if not _age_ok(right_tf):
            right_tf = None

        if left_tf is None and right_tf is None:
            return self._lookup_tf(parent, tag_id)
        if left_tf is None:
            return right_tf
        if right_tf is None:
            return left_tf

        p_left, q_left, age_left = left_tf
        p_right, q_right, age_right = right_tf

        warn_dist = float(self.get_parameter("tag_fusion_warn_dist").value)
        warn_period = float(self.get_parameter("tag_fusion_warn_period").value)
        diff = float(np.linalg.norm(p_left - p_right))
        now = time.time()
        if diff > warn_dist and (now - self._last_tag_warn_time) > warn_period:
            self._last_tag_warn_time = now
            self.get_logger().warn(
                f"[tag] left/right diff {diff:.3f}m; consider tag_source=left/right"
            )

        p_fused = (p_left + p_right) * 0.5
        q_fused = q_left if age_left <= age_right else q_right
        age_fused = min(age_left, age_right)
        return p_fused, q_fused, age_fused

    # ---------------------- 底盘移动 ----------------------

    def _compute_gripper_center_offset_ee(self, ee_frame: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        # Compute gripper center offset in ee frame.
        left_frame = str(self.get_parameter("gripper_left_frame").value)
        right_frame = str(self.get_parameter("gripper_right_frame").value)
        samples = int(self.get_parameter("gripper_center_samples").value)
        timeout = float(self.get_parameter("gripper_center_timeout").value)
        max_tf_age = float(self.get_parameter("max_tf_age").value)

        offsets = []
        start = time.time()
        while len(offsets) < samples and (time.time() - start) < timeout:
            left_tf = self._lookup_tf(ee_frame, left_frame)
            right_tf = self._lookup_tf(ee_frame, right_frame)
            if left_tf and right_tf:
                p_left, _, age_l = left_tf
                p_right, _, age_r = right_tf
                if age_l < max_tf_age and age_r < max_tf_age:
                    p_center = (p_left + p_right) * 0.5
                    offsets.append(p_center)
            rclpy.spin_once(self, timeout_sec=0.02)
            time.sleep(0.02)

        if not offsets:
            self.get_logger().warn(
                "[tcp] failed to compute gripper center offset, using target_offset_xyz only"
            )
            return None

        offset_ee = np.mean(np.stack(offsets, axis=0), axis=0).astype(np.float32)
        self.get_logger().info(
            f"[tcp] gripper_center_offset_ee=({offset_ee[0]:+.3f},{offset_ee[1]:+.3f},{offset_ee[2]:+.3f})"
        )
        return offset_ee

    def _get_effective_target_offset(self, base_frame: str, ee_frame: str) -> np.ndarray:
        target_offset = list(self.get_parameter("target_offset_xyz").value)
        if len(target_offset) != 3:
            self.get_logger().warn("[target] invalid target_offset_xyz, using [0,0,0]")
            target_offset = [0.0, 0.0, 0.0]
        return np.array(target_offset, dtype=np.float32)


    def _apply_tcp_offset(self, goal_pos: np.ndarray, q_wxyz: np.ndarray, tcp_offset_ee: Optional[np.ndarray]) -> np.ndarray:
        if tcp_offset_ee is None:
            return goal_pos
        offset_base = _rotate_vec_by_quat_wxyz(tcp_offset_ee, q_wxyz)
        return (goal_pos - offset_base).astype(np.float32)

    def _compute_downward_quat(
        self, ee_q_wxyz: np.ndarray, tag_q_xyzw: Optional[np.ndarray]
    ) -> np.ndarray:
        mode = str(self.get_parameter("downward_grasp_mode").value).lower()
        z_world = np.array([0.0, 0.0, -1.0], dtype=np.float32)

        if mode == "tag" and tag_q_xyzw is not None:
            tag_q_wxyz = _xyzw_to_wxyz(tag_q_xyzw)
            R_tag = _quat_wxyz_to_rot(tag_q_wxyz)
            z_axis = -R_tag[:, 2]
            x_hint = R_tag[:, 0]
        else:
            R_ee = _quat_wxyz_to_rot(ee_q_wxyz)
            z_axis = z_world
            x_hint = R_ee[:, 0]

        z_axis = _normalize_vec(z_axis)
        if float(np.linalg.norm(z_axis)) < 1e-6:
            z_axis = z_world

        x_axis = x_hint - np.dot(x_hint, z_axis) * z_axis
        if float(np.linalg.norm(x_axis)) < 1e-6:
            x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis
        x_axis = _normalize_vec(x_axis)

        y_axis = np.cross(z_axis, x_axis)
        y_axis = _normalize_vec(y_axis)

        R = np.stack([x_axis, y_axis, z_axis], axis=1)
        return _rot_to_quat_wxyz(R)

    def _stop_base(self):
        """停止底盘"""
        msg = Twist()
        for _ in range(5):
            self.cmd_vel_pub.publish(msg)
            time.sleep(0.02)

    def _check_workspace_bounds(self, p_tag: np.ndarray, label: str = "目标") -> Tuple[bool, str]:
        """
        检查目标是否在工作空间内
        返回: (in_bounds, message)
        """
        target_x = float(self.get_parameter("target_x_in_workspace").value)
        target_y = float(self.get_parameter("target_y_in_workspace").value)
        x_margin = float(self.get_parameter("workspace_x_margin").value)
        y_margin = float(self.get_parameter("workspace_y_margin").value)

        # 工作空间边界（基于调试结果）
        WORKSPACE_BOUNDS = {
            'x': (-0.36, 0.54),
            'y': (-0.65, 0.24),
        }

        x, y = p_tag[0], p_tag[1]
        dx_needed = 0.0
        dy_needed = 0.0

        # 检查 X 方向
        if x > WORKSPACE_BOUNDS['x'][1] - x_margin:
            dx_needed = x - target_x  # 需要前进
        elif x < WORKSPACE_BOUNDS['x'][0] + x_margin:
            dx_needed = x - target_x  # 需要后退

        # 检查 Y 方向（右臂工作空间主要在 Y<0）
        if y > WORKSPACE_BOUNDS['y'][1] - y_margin:
            dy_needed = y - target_y  # 需要左转
        elif y < WORKSPACE_BOUNDS['y'][0] + y_margin:
            dy_needed = y - target_y  # 需要右转

        in_bounds = abs(dx_needed) < 0.03 and abs(dy_needed) < 0.03

        if in_bounds:
            msg = f"[{label}] 位置在工作空间内: X={x:.3f}, Y={y:.3f}"
        else:
            msg = (f"[{label}] 位置超出工作空间范围: X={x:.3f} (范围 {WORKSPACE_BOUNDS['x']}), "
                   f"Y={y:.3f} (范围 {WORKSPACE_BOUNDS['y']})")

        return in_bounds, msg

    def _rotate_base(self, dy: float, speed: float, timeout: float) -> bool:
        """
        旋转底盘对准 AR 标签（闭环比例控制）

        参考 task2_direct.py 的实现：
        - 使用 tag 在 base_link 中的 X 坐标来控制角速度
        - X > 0 表示 tag 在右侧，需要右转（angular.z < 0）
        - X < 0 表示 tag 在左侧，需要左转（angular.z > 0）

        注意：旋转底盘会改变 tag 在 base_link 中的 X 坐标（不是 Y）
        """
        if abs(dy) < 0.03:
            return True

        base = str(self.get_parameter("base_frame").value)
        tagf = str(self.get_parameter("tag_frame").value)
        target_x = float(self.get_parameter("target_x_in_workspace").value)

        # 比例控制参数（参考 task2_direct.py）
        k_yaw = 2.0  # rad/s per meter
        deadband = 0.03  # 死区 3cm
        alpha = 0.35  # 平滑因子

        w_last = 0.0
        start_time = time.time()
        last_log_time = 0.0

        self.get_logger().info(f"[底盘] 旋转对准（闭环）: 目标 X≈{target_x:.2f}m")

        while rclpy.ok() and time.time() - start_time < timeout:
            # 查询 TF
            tag_tf = self._lookup_stereo_tf(base, tagf)

            if tag_tf is None:
                # tag 丢失，停止并等待
                self._stop_base()
                rclpy.spin_once(self, timeout_sec=0.05)
                time.sleep(0.05)
                continue

            p_tag, _, age = tag_tf
            if age > 0.5:
                # TF 过期，停止并等待
                self._stop_base()
                rclpy.spin_once(self, timeout_sec=0.05)
                time.sleep(0.05)
                continue

            # 计算偏差
            # 使用 Y 坐标来判断是否对准（Y 接近目标值表示对准）
            # 但控制角速度时使用 X 坐标的变化趋势
            y_error = p_tag[1] - float(self.get_parameter("target_y_in_workspace").value)

            if abs(y_error) < deadband:
                self.get_logger().info(
                    f"[底盘] 对准完成: tag_pos=[{p_tag[0]:.3f}, {p_tag[1]:.3f}, {p_tag[2]:.3f}]"
                )
                self._stop_base()
                return True

            # 比例控制：y_error > 0 表示 tag 在左侧，需要左转（w > 0）
            w_cmd = k_yaw * y_error
            w_cmd = max(-speed, min(speed, w_cmd))  # 限幅

            # 平滑滤波
            w_cmd = alpha * w_cmd + (1 - alpha) * w_last
            w_last = w_cmd

            msg = Twist()
            msg.angular.z = w_cmd
            self.cmd_vel_pub.publish(msg)
            self._publish_cmd_full()

            # 定期日志
            elapsed = time.time() - start_time
            if elapsed - last_log_time > 1.0:
                self.get_logger().info(
                    f"[底盘] 旋转中: y_err={y_error:.3f}m, w={w_cmd:.2f}rad/s, "
                    f"tag_y={p_tag[1]:.3f}m"
                )
                last_log_time = elapsed

            rclpy.spin_once(self, timeout_sec=0.01)
            time.sleep(0.1)  # 10Hz 控制频率

        self._stop_base()
        self.get_logger().warn("[底盘] 旋转超时")
        return False

    def _move_base_linear(self, dx: float, speed: float, timeout: float) -> bool:
        """
        前进/后退调整 X 方向
        dx > 0: 目标太远，需要前进（linear.x > 0）
        dx < 0: 目标太近，需要后退（linear.x < 0）
        """
        if abs(dx) < 0.03:
            return True

        base = str(self.get_parameter("base_frame").value)
        tagf = str(self.get_parameter("tag_frame").value)
        target_x = float(self.get_parameter("target_x_in_workspace").value)

        msg = Twist()
        msg.linear.x = speed if dx > 0 else -speed

        self.get_logger().info(f"[底盘] 直线移动: dx={dx:.3f}m, linear.x={msg.linear.x:.2f} m/s")

        start_time = time.time()
        while rclpy.ok() and time.time() - start_time < timeout:
            # 检查是否已经到位
            tag_tf = self._lookup_stereo_tf(base, tagf)
            if tag_tf is not None:
                p_tag, q_tag_xyzw, age = tag_tf
                if age < 0.5:
                    current_dx = p_tag[0] - target_x
                    if abs(current_dx) < 0.05:
                        self.get_logger().info(f"[底盘] 直线移动完成: current_x={p_tag[0]:.3f}m")
                        self._stop_base()
                        return True

            self.cmd_vel_pub.publish(msg)
            self._publish_cmd_full()  # 保持机械臂姿态
            rclpy.spin_once(self, timeout_sec=0.01)
            time.sleep(0.05)

        self._stop_base()
        self.get_logger().warn("[底盘] 直线移动超时")
        return False

    def _move_base_to_target(self, p_tag: np.ndarray) -> bool:
        """
        移动底盘使目标进入工作空间
        返回: True 如果成功调整或不需要调整
        """
        enable_base = bool(self.get_parameter("enable_base_movement").value)
        if not enable_base:
            return True

        in_bounds, msg = self._check_workspace_bounds(p_tag)
        if in_bounds:
            self.get_logger().info("[底盘] 目标已在工作空间内，无需移动")
            return True

        self.get_logger().info(msg)

        # 重新计算偏移量（原有逻辑）
        target_x = float(self.get_parameter("target_x_in_workspace").value)
        target_y = float(self.get_parameter("target_y_in_workspace").value)
        x_margin = float(self.get_parameter("workspace_x_margin").value)
        y_margin = float(self.get_parameter("workspace_y_margin").value)

        WORKSPACE_BOUNDS = {
            'x': (-0.36, 0.54),
            'y': (-0.65, 0.24),
        }

        x, y = p_tag[0], p_tag[1]
        dx_needed = 0.0
        dy_needed = 0.0

        if x > WORKSPACE_BOUNDS['x'][1] - x_margin:
            dx_needed = x - target_x
        elif x < WORKSPACE_BOUNDS['x'][0] + x_margin:
            dx_needed = x - target_x

        if y > WORKSPACE_BOUNDS['y'][1] - y_margin:
            dy_needed = y - target_y
        elif y < WORKSPACE_BOUNDS['y'][0] + y_margin:
            dy_needed = y - target_y

        self.get_logger().info(
            f"[底盘] 需要调整: dx={dx_needed:.3f}m, dy={dy_needed:.3f}m"
        )

        linear_speed = float(self.get_parameter("base_linear_speed").value)
        angular_speed = float(self.get_parameter("base_angular_speed").value)
        timeout = float(self.get_parameter("base_move_timeout").value)

        # 阶段 1：旋转对准（调整 Y）
        if abs(dy_needed) > 0.03:
            self.get_logger().info("[底盘] 阶段1: 旋转对准")
            if not self._rotate_base(dy_needed, angular_speed, timeout / 2):
                self.get_logger().warn("[底盘] 旋转调整失败")

        # 阶段 2：前进/后退（调整 X）
        if abs(dx_needed) > 0.03:
            self.get_logger().info("[底盘] 阶段2: 直线移动")
            if not self._move_base_linear(dx_needed, linear_speed, timeout / 2):
                self.get_logger().warn("[底盘] 直线移动失败")

        # 停止底盘
        self._stop_base()

        # 等待 TF 稳定
        self.get_logger().info("[底盘] 等待 TF 稳定...")
        time.sleep(0.5)

        return True

    def _search_for_tag(self, timeout: float = 10.0) -> bool:
        """
        搜索 AR 标签（缓慢旋转直到找到）

        当 tag 丢失时，机器人会缓慢旋转搜索。
        找到 tag 后返回 True，超时返回 False。
        """
        base = str(self.get_parameter("base_frame").value)
        tagf = str(self.get_parameter("tag_frame").value)
        search_speed = 0.2  # rad/s，搜索时的旋转速度

        self.get_logger().info("[搜索] 开始搜索 AR 标签...")

        msg = Twist()
        msg.angular.z = search_speed

        start_time = time.time()
        last_log_time = 0.0

        while rclpy.ok() and time.time() - start_time < timeout:
            # 检查是否找到 tag
            tag_tf = self._lookup_stereo_tf(base, tagf)
            if tag_tf is not None:
                p_tag, _, age = tag_tf
                if age < 0.5:
                    self.get_logger().info(
                        f"[搜索] 找到标签: pos=[{p_tag[0]:.3f}, {p_tag[1]:.3f}, {p_tag[2]:.3f}]"
                    )
                    self._stop_base()
                    return True

            self.cmd_vel_pub.publish(msg)
            self._publish_cmd_full()

            # 定期日志
            elapsed = time.time() - start_time
            if elapsed - last_log_time > 2.0:
                self.get_logger().info(f"[搜索] 搜索中... 已旋转 {elapsed:.1f}s")
                last_log_time = elapsed

            rclpy.spin_once(self, timeout_sec=0.01)
            time.sleep(0.1)

        self._stop_base()
        self.get_logger().warn("[搜索] 搜索超时，未找到标签")
        return False

    def _adjust_base_for_direct_coords(self, p_tag: np.ndarray) -> np.ndarray:
        """
        直接坐标模式下自动调整底盘位姿

        当目标 Y > 0（在机器人左侧）时，旋转底盘使目标移到右侧（Y < 0）。
        这是因为右臂的工作空间主要在 Y < 0 区域。

        原理：
        - 机器人旋转角度 θ 后，目标在 base_link 中的新坐标为：
          x' = x*cos(θ) + y*sin(θ)
          y' = -x*sin(θ) + y*cos(θ)
        - 我们需要找到 θ 使得 y' = target_y（目标 Y 位置）

        参数:
            p_tag: 目标在 base_link 中的原始坐标 [x, y, z]

        返回:
            调整后的目标坐标 [x', y', z]
        """
        auto_adjust = bool(self.get_parameter("direct_mode_auto_adjust").value)
        if not auto_adjust:
            self.get_logger().info("[直接坐标模式] 自动调整已禁用")
            return p_tag.copy()

        target_y = float(self.get_parameter("direct_mode_target_y").value)
        angular_speed = float(self.get_parameter("base_angular_speed").value)

        x, y, z = p_tag[0], p_tag[1], p_tag[2]

        # 检查是否需要调整
        # 右臂工作空间 Y 范围: (-0.65, 0.24)，目标 Y 应该在 -0.15 左右
        if y <= 0.1:  # 已经在右侧或接近中心，不需要调整
            self.get_logger().info(f"[直接坐标模式] 目标 Y={y:.3f}m 已在右臂工作空间内，无需调整")
            return p_tag.copy()

        self.get_logger().info(f"[直接坐标模式] 目标 Y={y:.3f}m 在机器人左侧，需要旋转底盘")

        # 计算需要旋转的角度
        # 目标: y' = -x*sin(θ) + y*cos(θ) = target_y
        # 使用 atan2 计算: θ = atan2(y - target_y, x) - atan2(y, x) (近似)
        # 更精确的方法: 使用数值方法或解析解

        # 计算到原点的距离
        r = math.sqrt(x*x + y*y)
        if r < 0.1:
            self.get_logger().warn("[直接坐标模式] 目标距离太近，无法计算旋转角度")
            return p_tag.copy()

        # 当前角度 (从 X 轴正方向逆时针)
        current_angle = math.atan2(y, x)

        # 目标角度: 使得 y' = target_y
        # y' = r * sin(current_angle - θ) = target_y
        # sin(current_angle - θ) = target_y / r
        sin_target = target_y / r
        if abs(sin_target) > 1.0:
            self.get_logger().warn(f"[直接坐标模式] 无法达到目标 Y={target_y:.3f}m (距离 r={r:.3f}m)")
            sin_target = max(-1.0, min(1.0, sin_target))

        target_angle = math.asin(sin_target)
        rotation_needed = current_angle - target_angle

        self.get_logger().info(
            f"[直接坐标模式] 计算旋转: 当前角度={math.degrees(current_angle):.1f}°, "
            f"目标角度={math.degrees(target_angle):.1f}°, 需旋转={math.degrees(rotation_needed):.1f}°"
        )

        # 执行旋转
        if abs(rotation_needed) < 0.05:  # 小于 3 度，不需要旋转
            self.get_logger().info("[直接坐标模式] 旋转角度太小，跳过")
            return p_tag.copy()

        # 旋转方向: rotation_needed > 0 表示需要逆时针旋转 (angular.z > 0)
        rotation_time = abs(rotation_needed) / angular_speed
        rotation_direction = 1.0 if rotation_needed > 0 else -1.0

        self.get_logger().info(
            f"[直接坐标模式] 开始旋转: 方向={'逆时针' if rotation_direction > 0 else '顺时针'}, "
            f"预计时间={rotation_time:.2f}s"
        )

        msg = Twist()
        msg.angular.z = rotation_direction * angular_speed

        start_time = time.time()
        while rclpy.ok() and time.time() - start_time < rotation_time:
            self.cmd_vel_pub.publish(msg)
            self._publish_cmd_full()  # 保持机械臂姿态
            rclpy.spin_once(self, timeout_sec=0.01)
            time.sleep(0.05)

        self._stop_base()

        # 计算旋转后的新坐标
        actual_rotation = rotation_direction * angular_speed * (time.time() - start_time)
        cos_r = math.cos(actual_rotation)
        sin_r = math.sin(actual_rotation)

        new_x = x * cos_r + y * sin_r
        new_y = -x * sin_r + y * cos_r
        new_z = z

        new_pos = np.array([new_x, new_y, new_z], dtype=np.float32)

        self.get_logger().info(
            f"[直接坐标模式] 旋转完成: 原坐标=[{x:.3f}, {y:.3f}, {z:.3f}], "
            f"新坐标=[{new_x:.3f}, {new_y:.3f}, {new_z:.3f}]"
        )

        # 等待稳定
        time.sleep(0.3)

        return new_pos

    # ---------------------- 发布（全量格式）----------------------
    def _publish_cmd_full(self):
        """发布全量关节命令（右臂 + 左臂 + 夹爪）"""
        if not self.R_JOINTS or not self.L_JOINTS:
            return

        names: List[str] = []
        pos: List[float] = []

        # 右臂 6 关节
        for j in self.R_JOINTS:
            names.append(j)
            pos.append(float(self.cmd.get(j, self.pose.get(j, 0.0))))

        # 左臂 6 关节（保持当前位置）
        for j in self.L_JOINTS:
            names.append(j)
            pos.append(float(self.cmd.get(j, self.pose.get(j, 0.0))))

        # 右夹爪
        if self.GRIP_R is not None:
            names.append(self.GRIP_R)
            pos.append(float(self.cmd.get(self.GRIP_R, self.pose.get(self.GRIP_R, 0.0))))

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = names
        msg.position = pos
        self.pub.publish(msg)

    def _init_hold(self):
        """锁定所有关节到当前位置（避免突变）"""
        for j in self.R_JOINTS + self.L_JOINTS:
            if j in self.pose:
                self.cmd[j] = float(self.pose[j])
        if self.GRIP_R is not None and self.GRIP_R in self.pose:
            self.cmd[self.GRIP_R] = float(self.pose[self.GRIP_R])
        self._publish_cmd_full()

    # ---------------------- World Sync ----------------------
    def _get_collision_checker_type(self) -> CollisionCheckerType:
        raw = str(self.get_parameter("collision_checker_type").value).strip().upper()
        if raw in ("MESH", "MESHES"):
            return CollisionCheckerType.MESH
        if raw in ("PRIMITIVE", "OBB"):
            return CollisionCheckerType.PRIMITIVE
        if raw in ("BLOX", "NVBLOX"):
            return CollisionCheckerType.BLOX
        self.get_logger().warn(
            f"[cuRobo] unknown collision_checker_type '{raw}', defaulting to PRIMITIVE"
        )
        return CollisionCheckerType.PRIMITIVE

    def _build_table_world(self) -> WorldConfig:
        table_dims = list(self.get_parameter("table_dims").value)
        table_pose = list(self.get_parameter("table_pose").value)
        return WorldConfig.from_dict({
            "cuboid": {
                "table": {
                    "dims": table_dims,
                    "pose": table_pose,
                }
            }
        })

    def _load_stage_world_raw(self) -> Optional[WorldConfig]:
        if not bool(self.get_parameter("sync_world_from_stage").value):
            return None
        try:
            from curobo.util.usd_helper import UsdHelper
        except Exception as exc:
            if not self._world_sync_failed:
                self.get_logger().warn(f"[world-sync] UsdHelper unavailable: {exc}")
                self._world_sync_failed = True
            return None

        if self._usd_helper is None:
            self._usd_helper = UsdHelper()
        usd_help = self._usd_helper

        stage = None
        if bool(self.get_parameter("sync_world_use_live_stage").value):
            try:
                import omni.usd

                stage = omni.usd.get_context().get_stage()
            except Exception as exc:
                if not self._world_sync_failed:
                    self.get_logger().warn(f"[world-sync] live stage unavailable: {exc}")
                    self._world_sync_failed = True

        stage_path = str(self.get_parameter("sync_world_stage_path").value)
        if stage is not None:
            usd_help.load_stage(stage)
        elif stage_path:
            if not os.path.exists(stage_path):
                if not self._world_sync_failed:
                    self.get_logger().warn(f"[world-sync] stage file not found: {stage_path}")
                    self._world_sync_failed = True
                return None
            usd_help.load_stage_from_file(stage_path)
        else:
            if not self._world_sync_failed:
                self.get_logger().warn(
                    "[world-sync] no stage available and sync_world_stage_path is empty"
                )
                self._world_sync_failed = True
            return None

        only_paths = list(self.get_parameter("sync_world_only_paths").value)
        ignore_paths = list(self.get_parameter("sync_world_ignore_paths").value)
        ignore_substrings = list(self.get_parameter("sync_world_ignore_substrings").value)
        reference_prim = str(self.get_parameter("sync_world_reference_prim").value)
        if reference_prim == "":
            reference_prim = None

        obstacles = usd_help.get_obstacles_from_stage(
            only_paths=only_paths if only_paths else None,
            ignore_paths=ignore_paths if ignore_paths else None,
            ignore_substring=ignore_substrings if ignore_substrings else None,
            reference_prim_path=reference_prim,
        )

        if len(obstacles.objects) == 0 and bool(
            self.get_parameter("sync_world_fallback_to_table").value
        ):
            return self._build_table_world()
        return obstacles

    def _normalize_world_for_checker(
        self, world_cfg: WorldConfig, checker_type: CollisionCheckerType
    ) -> WorldConfig:
        mesh_process = bool(self.get_parameter("sync_world_mesh_process").value)
        if checker_type == CollisionCheckerType.PRIMITIVE:
            return world_cfg.get_obb_world()
        return world_cfg.get_collision_check_world(mesh_process=mesh_process)

    def _sync_world_from_stage(self, force: bool = False) -> None:
        if self.motion_gen is None:
            return
        if not bool(self.get_parameter("sync_world_from_stage").value):
            return

        now = time.time()
        min_dt = float(self.get_parameter("sync_world_update_interval").value)
        if not force and min_dt > 0.0 and now - self._last_world_sync_time < min_dt:
            return

        world_raw = self._load_stage_world_raw()
        if world_raw is None:
            return
        world_cfg = self._normalize_world_for_checker(world_raw, self._collision_checker_type)

        try:
            self.motion_gen.update_world(world_cfg)
        except Exception as exc:
            if not self._world_sync_failed:
                self.get_logger().warn(f"[world-sync] update failed: {exc}")
                self._world_sync_failed = True
            return

        self._last_world_sync_time = now
        self._world_sync_failed = False
        self.get_logger().info(f"[world-sync] updated obstacles={len(world_cfg.objects)}")

    # ---------------------- cuRobo Init ----------------------
    def _init_curobo(self):
        """Initialize cuRobo MotionGen."""
        if not CUROBO_AVAILABLE:
            raise RuntimeError(
                "cuRobo import failed. Use Isaac Sim Python. "
                f"Original error: {CUROBO_IMPORT_ERROR}"
            )

        robot_cfg_path = str(self.get_parameter("robot_cfg_path").value)
        interp_dt = float(self.get_parameter("interp_dt").value)

        self.get_logger().info(f"[cuRobo] load robot cfg: {robot_cfg_path}")

        robot_cfg = load_yaml(robot_cfg_path)["robot_cfg"]

        self._collision_checker_type = self._get_collision_checker_type()

        world_raw = None
        world_cfg = None
        if bool(self.get_parameter("sync_world_from_stage").value):
            world_raw = self._load_stage_world_raw()
            if world_raw is not None:
                world_cfg = self._normalize_world_for_checker(
                    world_raw, self._collision_checker_type
                )

        if world_cfg is None:
            world_cfg = self._build_table_world()
            if self._collision_checker_type != CollisionCheckerType.PRIMITIVE:
                world_cfg = self._normalize_world_for_checker(
                    world_cfg, self._collision_checker_type
                )

        table_dims = list(self.get_parameter("table_dims").value)
        if world_raw is None:
            self.get_logger().info(f"[cuRobo] world=table dims={table_dims}")
        else:
            self.get_logger().info(f"[cuRobo] world=stage obstacles={len(world_cfg.objects)}")
            if (
                self._collision_checker_type == CollisionCheckerType.PRIMITIVE
                and world_raw.mesh is not None
                and len(world_raw.mesh) > 0
            ):
                self.get_logger().warn("[world-sync] mesh obstacles detected; using OBB")

        self.tensor_args = TensorDeviceType()

        collision_cache = None
        if self._collision_checker_type == CollisionCheckerType.MESH:
            try:
                cache = world_cfg.get_cache_dict()
                collision_cache = {"obb": cache["obb"], "mesh": cache["mesh"]}
            except Exception:
                collision_cache = None

        motion_gen_config = MotionGenConfig.load_from_robot_config(
            robot_cfg,
            world_cfg,
            self.tensor_args,
            interpolation_dt=interp_dt,
            collision_checker_type=self._collision_checker_type,
            collision_cache=collision_cache,
            num_trajopt_seeds=12,
            num_graph_seeds=12,
            collision_activation_distance=0.025,
            maximum_trajectory_dt=0.5,
        )

        self.motion_gen = MotionGen(motion_gen_config)

        self.get_logger().info("[cuRobo] warmup() ... (first run compiles GPU kernels)")
        self.motion_gen.warmup(enable_graph=True, warmup_js_trajopt=False)
        self._sync_world_from_stage(force=True)
        self.get_logger().info("[cuRobo] ready.")

    def _get_current_cu_js(self) -> "CuJointState":
        """获取当前关节状态（cuRobo 格式）- 参考官方示例"""
        joint_names = list(self.get_parameter("curobo_joint_names").value)
        q = [float(self.cmd.get(j, self.pose.get(j, 0.0))) for j in joint_names]

        # 按照参考文件的方式创建 JointState（传入列表，内部转为 1D 张量）
        cu_js = CuJointState(
            position=self.tensor_args.to_device(q),
            velocity=self.tensor_args.to_device([0.0] * len(q)),
            acceleration=self.tensor_args.to_device([0.0] * len(q)),
            jerk=self.tensor_args.to_device([0.0] * len(q)),
            joint_names=joint_names,
        )
        # 重新排序关节以匹配 motion_gen 的关节顺序（关键步骤）
        cu_js = cu_js.get_ordered_joint_state(self.motion_gen.kinematics.joint_names)
        return cu_js


    def _get_current_ee_quat_wxyz(self):
        """Return current EE quaternion (wxyz) using cuRobo FK."""
        try:
            cu_js = self._get_current_cu_js()
            kin_state = self.motion_gen.kinematics.get_state(cu_js.position.view(1, -1))
            quat = kin_state.ee_pose.quaternion.cpu().numpy().flatten().astype(np.float32)
            if not self._is_finite(quat):
                return None
            return quat
        except Exception as exc:
            self.get_logger().warn(f"[tf] fallback ee quat failed: {exc}")
            return None

    def _plan_to_pose(
        self, goal_pos: np.ndarray, goal_quat_wxyz: np.ndarray
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        规划到目标位姿

        Args:
            goal_pos: 目标位置 [x, y, z]
            goal_quat_wxyz: 目标姿态四元数 [w, x, y, z]

        Returns:
            轨迹数组 [T, DOF] 或 None（规划失败）
        """
        # 工作空间边界检查（基于调试结果）
        if not self._is_finite(goal_pos) or not self._is_finite(goal_quat_wxyz):
            self.get_logger().error("[plan] invalid target pose (nan/inf)")
            return None
        self._sync_world_from_stage()
        WORKSPACE_BOUNDS = {
            'x': (-0.36, 0.54),
            'y': (-0.65, 0.24),
            'z': (0.42, 1.35),
        }

        x, y, z = goal_pos
        warnings = []
        if x < WORKSPACE_BOUNDS['x'][0] or x > WORKSPACE_BOUNDS['x'][1]:
            warnings.append(f"X={x:.3f} 超出范围 {WORKSPACE_BOUNDS['x']}")
        if y < WORKSPACE_BOUNDS['y'][0] or y > WORKSPACE_BOUNDS['y'][1]:
            warnings.append(f"Y={y:.3f} 超出范围 {WORKSPACE_BOUNDS['y']}")
        if z < WORKSPACE_BOUNDS['z'][0] or z > WORKSPACE_BOUNDS['z'][1]:
            warnings.append(f"Z={z:.3f} 超出范围 {WORKSPACE_BOUNDS['z']}")

        abort_on_oob = bool(self.get_parameter("abort_on_workspace_violation").value)
        if warnings and abort_on_oob:
            self.get_logger().error("[workspace] target out of bounds, aborting plan")
            return None
        if warnings:
            self.get_logger().warn(f"[工作空间警告] 目标位置可能超出可达范围:")
            for w in warnings:
                self.get_logger().warn(f"  - {w}")

        # 诊断日志
        self.get_logger().info(
            f"[规划] 目标: pos=[{x:.3f}, {y:.3f}, {z:.3f}], "
            f"quat={goal_quat_wxyz.tolist()}"
        )

        max_attempts = int(self.get_parameter("max_attempts").value)
        max_target_distance = float(self.get_parameter("max_target_distance").value)
        if max_target_distance > 0.0:
            base = str(self.get_parameter("base_frame").value)
            eef = str(self.get_parameter("ee_frame").value)
            cur_tf = self._lookup_tf(base, eef)
            if cur_tf is not None:
                dist = float(np.linalg.norm(goal_pos - cur_tf[0]))
                if dist > max_target_distance:
                    self.get_logger().error(f"[plan] target too far: {dist:.3f}m > {max_target_distance:.3f}m")
                    return None

        # 创建目标 Pose
        ik_goal = Pose(
            position=self.tensor_args.to_device(goal_pos.astype(np.float32)),
            quaternion=self.tensor_args.to_device(goal_quat_wxyz.astype(np.float32)),
        )

        # 获取当前关节状态
        cu_js = self._get_current_cu_js()

        # 规划配置
        plan_config = MotionGenPlanConfig(
            enable_graph=True,
            enable_graph_attempt=4,
            max_attempts=max_attempts,
            enable_finetune_trajopt=True,
            time_dilation_factor=1.0,  # 从 0.5 改为 1.0（更慢的轨迹）
        )

        # 执行规划
        result = self.motion_gen.plan_single(cu_js.unsqueeze(0), ik_goal, plan_config)

        if result.success.item():
            traj = result.get_interpolated_plan()
            pos = traj.position
            if isinstance(pos, torch.Tensor):
                pos = pos.detach().cpu().numpy()
            pos = np.array(pos, dtype=np.float32)
            if pos.ndim == 3:
                pos = pos[0]
            self.get_logger().info(f"[规划成功] 轨迹长度={pos.shape[0]}")
            return pos
        else:
            self.get_logger().warn(f"[规划失败] status={result.status}")
            return None

    # ---------------------- 轨迹安全检查方法 ----------------------
    def _clamp_joint(self, joint_name: str, value: float) -> float:
        """钳制关节角度到限位范围内"""
        limits = self.JOINT_LIMITS.get(joint_name)
        if limits is None:
            return float(value)
        lo, hi = limits
        margin = self.JOINT_LIMIT_MARGIN
        clamped = float(max(lo + margin, min(hi - margin, value)))
        return clamped

    def _check_trajectory_limits(self, joint_names: List[str], q_traj: np.ndarray) -> bool:
        """检查轨迹是否在关节限位内"""
        for i, q_point in enumerate(q_traj):
            for jn, q in zip(joint_names, q_point):
                limits = self.JOINT_LIMITS.get(jn)
                if limits is None:
                    continue
                lo, hi = limits
                if q < lo or q > hi:
                    self.get_logger().error(
                        f"[轨迹] 点 {i} 关节 {jn} 超出限位: {q:.3f} rad, 限位=[{lo:.3f}, {hi:.3f}]"
                    )
                    return False
        return True

    def _max_trajectory_speed(self, q_traj: np.ndarray, dt: float) -> float:
        if q_traj.shape[0] < 2:
            return 0.0
        step = np.abs(np.diff(q_traj, axis=0))
        return float(np.max(step) / max(dt, 1e-6))

    def _check_trajectory_speed(self, q_traj: np.ndarray, dt: float) -> bool:
        """Check trajectory speed limit."""
        max_speed_limit = float(self.get_parameter("max_joint_speed").value)
        if max_speed_limit <= 0.0:
            return True
        for i in range(1, len(q_traj)):
            dq = q_traj[i] - q_traj[i - 1]
            speed = np.abs(dq) / dt
            max_speed = np.max(speed)
            if max_speed > max_speed_limit:
                self.get_logger().warn(
                    f"[traj] step {i} speed high: {max_speed:.2f} rad/s > {max_speed_limit} rad/s"
                )
                return False
        return True
    def _exec_traj(self, joint_names: List[str], q_traj: np.ndarray):
        """
        执行轨迹（带完整安全检查）

        安全措施：
        1. 检查 NaN/inf
        2. 检查关节限位
        3. 检查速度限制
        4. 钳制每个执行点
        5. 使用可配置控制频率（默认20Hz，防止物理引擎不稳定）
        """
        # 参数
        exec_dt = float(self.get_parameter("exec_dt").value)  # 使用参数值（20Hz，防止物理引擎不稳定）
        interp_dt = float(self.get_parameter("interp_dt").value)

        # 1. 检查 NaN/inf
        if np.any(~np.isfinite(q_traj)):
            self.get_logger().error("[轨迹] 轨迹包含 NaN/inf，中止执行")
            return

        original_len = q_traj.shape[0]
        if original_len < 2:
            self.get_logger().warn("[轨迹] 轨迹点数不足，跳过执行")
            return

        # 2. 检查关节限位
        if not self._check_trajectory_limits(joint_names, q_traj):
            self.get_logger().error("[轨迹] 轨迹超出关节限位，中止执行")
            return

        # 3. 检查速度（使用原始轨迹的时间间隔）
        if not self._check_trajectory_speed(q_traj, interp_dt):
            self.get_logger().warn("[轨迹] 轨迹速度过高，将降速执行")

        # 4. 计算执行步数（保持原始时间，不拉长）
        original_time = (original_len - 1) * interp_dt
        max_speed = self._max_trajectory_speed(q_traj, interp_dt)
        max_speed_limit = float(self.get_parameter("max_joint_speed").value)
        if max_speed_limit <= 0.0:
            return True
        max_step_limit = float(self.get_parameter("max_joint_step").value)
        target_steps = max(original_len, int(math.ceil(original_time / max(exec_dt, 1e-6))))
        if max_speed_limit > 0.0 and max_speed > max_speed_limit:
            scale = max_speed / max_speed_limit
            target_steps = int(math.ceil(target_steps * scale))
        if max_step_limit > 0.0 and original_len > 1:
            max_step = float(np.max(np.abs(np.diff(q_traj, axis=0))))
            if max_step > max_step_limit:
                scale = max_step / max_step_limit
                target_steps = max(target_steps, int(math.ceil(original_len * scale)))

        self.get_logger().info(
            f"[轨迹] 原始 {original_len} 点，执行 {target_steps} 步，"
            f"总时间 {original_time:.2f}s，exec_dt={exec_dt}s"
        )

        # 5. 线性插值执行（带钳制）
        new_indices = np.linspace(0, original_len - 1, target_steps)

        for i, idx in enumerate(new_indices):
            idx_low = int(idx)
            idx_high = min(idx_low + 1, original_len - 1)
            alpha = idx - idx_low

            q_interp = (1.0 - alpha) * q_traj[idx_low] + alpha * q_traj[idx_high]

            # 检查 NaN
            if np.any(~np.isfinite(q_interp)):
                self.get_logger().error(f"[轨迹] 插值点 {i} 包含 NaN，中止执行")
                return

            # 钳制并发送
            for j, q in zip(joint_names, q_interp.tolist()):
                self.cmd[j] = self._clamp_joint(j, q)

            self._publish_cmd_full()
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(exec_dt)

    def _smooth_gripper(self, target: float):
        """平滑控制夹爪"""
        if self.GRIP_R is None:
            return
        steps = int(self.get_parameter("grip_steps").value)
        exec_dt = float(self.get_parameter("exec_dt").value)
        cur = float(self.cmd.get(self.GRIP_R, self.pose.get(self.GRIP_R, 0.0)))
        for k in range(steps):
            a = (k + 1) / steps
            self.cmd[self.GRIP_R] = (1 - a) * cur + a * float(target)
            self._publish_cmd_full()
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(exec_dt)

    def _smooth_move_joints(self, target_dict: Dict[str, float], duration: float = 2.0):
        """平滑移动关节到目标位置"""
        steps = max(1, int(duration * 50.0))
        start = {k: float(self.cmd.get(k, self.pose.get(k, 0.0))) for k in target_dict.keys()}
        end = {k: float(v) for k, v in target_dict.items()}

        for i in range(steps + 1):
            alpha = i / steps
            for k in end.keys():
                self.cmd[k] = float(start[k] + (end[k] - start[k]) * alpha)
            self._publish_cmd_full()
            rclpy.spin_once(self, timeout_sec=0.01)
            time.sleep(0.02)

    def _move_to_safe_pose(self):
        """移动到安全姿态"""
        safe_pose_R = list(self.get_parameter("safe_pose_R").value)
        safe_pose_L = list(self.get_parameter("safe_pose_L").value)
        duration = float(self.get_parameter("safe_pose_duration").value)

        if len(self.R_JOINTS) >= 6 and len(safe_pose_R) == 6:
            upd = {j: float(v) for j, v in zip(self.R_JOINTS[:6], safe_pose_R)}
            self.get_logger().info("[安全] 右臂 -> safe_pose_R")
            self._smooth_move_joints(upd, duration=duration)

        if len(self.L_JOINTS) >= 6 and len(safe_pose_L) == 6:
            upd = {j: float(v) for j, v in zip(self.L_JOINTS[:6], safe_pose_L)}
            self.get_logger().info("[安全] 左臂 -> safe_pose_L")
            self._smooth_move_joints(upd, duration=duration)

    # ---------------------- 调试方法 ----------------------
    def _debug_fk(self):
        """验证 FK 是否正确"""
        self.get_logger().info("=" * 60)
        self.get_logger().info("[FK Debug] 开始 FK 验证")

        # 获取 retract 配置
        retract = self.motion_gen.get_retract_config()
        self.get_logger().info(f"[FK Debug] retract_config: {retract.cpu().numpy()}")

        # 计算 FK
        kin_state = self.motion_gen.kinematics.get_state(retract.view(1, -1))
        ee_pos = kin_state.ee_pose.position.cpu().numpy()
        ee_quat = kin_state.ee_pose.quaternion.cpu().numpy()

        self.get_logger().info(f"[FK Debug] Retract EE position: {ee_pos}")
        self.get_logger().info(f"[FK Debug] Retract EE quaternion: {ee_quat}")

        # 验证当前关节状态的 FK
        cu_js = self._get_current_cu_js()
        kin_state_cur = self.motion_gen.kinematics.get_state(cu_js.position.view(1, -1))
        ee_pos_cur = kin_state_cur.ee_pose.position.cpu().numpy()
        ee_quat_cur = kin_state_cur.ee_pose.quaternion.cpu().numpy()

        self.get_logger().info(f"[FK Debug] Current joint: {cu_js.position.cpu().numpy()}")
        self.get_logger().info(f"[FK Debug] Current EE position: {ee_pos_cur}")
        self.get_logger().info(f"[FK Debug] Current EE quaternion: {ee_quat_cur}")

        # 打印关节限位
        joint_limits = self.motion_gen.kinematics.get_joint_limits()
        self.get_logger().info(f"[FK Debug] Joint limits shape: {joint_limits.position.shape}")
        # 处理不同的张量形状: [2, 6] 表示 [lower/upper, joints]
        pos_tensor = joint_limits.position.cpu().numpy()
        if pos_tensor.ndim == 2 and pos_tensor.shape[0] == 2:
            # Shape is [2, num_joints] - first row is lower, second is upper
            lower = pos_tensor[0, :]
            upper = pos_tensor[1, :]
        elif pos_tensor.ndim == 3:
            lower = pos_tensor[0, :, 0]
            upper = pos_tensor[0, :, 1]
        elif pos_tensor.ndim == 2:
            lower = pos_tensor[:, 0]
            upper = pos_tensor[:, 1]
        else:
            lower = pos_tensor
            upper = pos_tensor
        self.get_logger().info(f"[FK Debug] Joint lower limits: {lower}")
        self.get_logger().info(f"[FK Debug] Joint upper limits: {upper}")

        # 打印 cuRobo 解析的关节名称
        joint_names = self.motion_gen.kinematics.joint_names
        self.get_logger().info(f"[FK Debug] cuRobo joint names: {joint_names}")
        self.get_logger().info(f"[FK Debug] EE link: {self.motion_gen.kinematics.ee_link}")

        self.get_logger().info("=" * 60)

    def _debug_workspace(self):
        """估算机械臂工作空间边界"""
        self.get_logger().info("=" * 60)
        self.get_logger().info("[Workspace Debug] 开始工作空间估算")

        # 在关节空间采样
        n_samples = 1000
        joint_limits = self.motion_gen.kinematics.get_joint_limits()
        # 处理不同的张量形状: [2, 6] 表示 [lower/upper, joints]
        pos_tensor = joint_limits.position.cpu().numpy()
        if pos_tensor.ndim == 2 and pos_tensor.shape[0] == 2:
            # Shape is [2, num_joints] - first row is lower, second is upper
            lower = pos_tensor[0, :]
            upper = pos_tensor[1, :]
        elif pos_tensor.ndim == 3:
            lower = pos_tensor[0, :, 0]
            upper = pos_tensor[0, :, 1]
        elif pos_tensor.ndim == 2:
            lower = pos_tensor[:, 0]
            upper = pos_tensor[:, 1]
        else:
            # 使用默认限位
            lower = np.array([-2.879, -0.8726, -2.879, -2.879, -1.9198, -3.14])
            upper = np.array([2.879, 2.0943, 0.087, 2.879, 3.0543, 3.14])

        self.get_logger().info(f"[Workspace] Joint lower: {lower}")
        self.get_logger().info(f"[Workspace] Joint upper: {upper}")

        samples = np.random.uniform(lower, upper, (n_samples, 6))
        samples_t = torch.tensor(samples, device="cuda:0", dtype=torch.float32)

        # 计算 FK
        kin_states = self.motion_gen.kinematics.get_state(samples_t)
        ee_positions = kin_states.ee_pose.position.cpu().numpy()

        # 统计
        x_range = (float(ee_positions[:, 0].min()), float(ee_positions[:, 0].max()))
        y_range = (float(ee_positions[:, 1].min()), float(ee_positions[:, 1].max()))
        z_range = (float(ee_positions[:, 2].min()), float(ee_positions[:, 2].max()))

        self.get_logger().info(f"[Workspace] 采样数: {n_samples}")
        self.get_logger().info(f"[Workspace] X range: [{x_range[0]:.3f}, {x_range[1]:.3f}]")
        self.get_logger().info(f"[Workspace] Y range: [{y_range[0]:.3f}, {y_range[1]:.3f}]")
        self.get_logger().info(f"[Workspace] Z range: [{z_range[0]:.3f}, {z_range[1]:.3f}]")

        self.get_logger().info("=" * 60)
        return x_range, y_range, z_range

    def _debug_ik_reachability(self, goal_pos: np.ndarray, goal_quat: np.ndarray) -> bool:
        """测试目标是否在 IK 可达范围内（禁用碰撞）"""
        from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

        self.get_logger().info("-" * 40)
        self.get_logger().info(f"[IK Debug] 测试目标: pos={goal_pos.tolist()}, quat={goal_quat.tolist()}")

        robot_cfg_path = str(self.get_parameter("robot_cfg_path").value)
        robot_cfg = load_yaml(robot_cfg_path)["robot_cfg"]

        try:
            # 创建无碰撞的 IK 求解器
            ik_config = IKSolverConfig.load_from_robot_config(
                robot_cfg,
                None,  # 无世界碰撞
                rotation_threshold=0.1,
                position_threshold=0.01,
                num_seeds=100,  # 大量种子
                self_collision_check=False,  # 禁用自碰撞
                tensor_args=self.tensor_args,
            )
            ik_solver = IKSolver(ik_config)

            # 测试 IK
            ik_goal = Pose(
                position=self.tensor_args.to_device(goal_pos.astype(np.float32)),
                quaternion=self.tensor_args.to_device(goal_quat.astype(np.float32)),
            )

            retract = self.motion_gen.get_retract_config()
            result = ik_solver.solve_single(ik_goal, retract.view(1, -1), retract.view(1, 1, -1))

            success = result.success.item()
            self.get_logger().info(f"[IK Debug] IK success (no collision): {success}")

            if success:
                sol = result.solution.cpu().numpy()
                self.get_logger().info(f"[IK Debug] Solution shape: {sol.shape}, values: {sol}")

                # 验证解的 FK - 需要 squeeze 成 [batch, dof] 形状
                sol_tensor = result.solution.squeeze()  # 从 [1,1,6] 变成 [6]
                if sol_tensor.dim() == 1:
                    sol_tensor = sol_tensor.unsqueeze(0)  # 变成 [1, 6]
                kin_state = self.motion_gen.kinematics.get_state(sol_tensor)
                ee_pos_result = kin_state.ee_pose.position.cpu().numpy().flatten()
                self.get_logger().info(f"[IK Debug] FK of solution: {ee_pos_result.tolist()}")
                pos_error = np.linalg.norm(ee_pos_result - goal_pos)
                self.get_logger().info(f"[IK Debug] Position error: {pos_error:.6f} m")

            return success

        except Exception as e:
            self.get_logger().error(f"[IK Debug] 异常: {e}")
            return False

    def _run_target_reachability_diagnostics(
        self,
        p_tag: np.ndarray,
        last_ee_q_wxyz: Optional[np.ndarray],
        last_tag_q_xyzw: Optional[np.ndarray],
        tcp_offset_ee: Optional[np.ndarray],
    ) -> None:
        """Diagnose whether target failure comes from pose estimation, IK, or MotionGen."""
        x_off = float(self.get_parameter("grasp_x_offset").value)
        y_off = float(self.get_parameter("grasp_y_offset").value)
        pre_z = float(self.get_parameter("pregrasp_z_offset").value)
        grasp_z = float(self.get_parameter("grasp_z_offset").value)
        lift_dz = float(self.get_parameter("lift_dz").value)

        retract = self.motion_gen.get_retract_config()
        kin_state = self.motion_gen.kinematics.get_state(retract.view(1, -1))
        retract_quat = kin_state.ee_pose.quaternion.cpu().numpy().flatten().astype(np.float32)

        q_seed = last_ee_q_wxyz if last_ee_q_wxyz is not None else retract_quat
        q_down = np.array([0.707, 0.0, 0.707, 0.0], dtype=np.float32)
        if bool(self.get_parameter("adaptive_downward_grasp").value):
            q_down = self._compute_downward_quat(q_seed, last_tag_q_xyzw)
        q_slant = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)

        quat_candidates: List[Tuple[str, np.ndarray]] = [
            ("downward", q_down),
            ("retract", retract_quat),
            ("current_ee", last_ee_q_wxyz),
            ("slant", q_slant),
        ]
        quat_candidates = [
            (name, q.astype(np.float32))
            for name, q in quat_candidates
            if q is not None and self._is_finite(q)
        ]

        targets: List[Tuple[str, np.ndarray]] = [
            (
                "PREGRASP",
                np.array([p_tag[0] + x_off, p_tag[1] + y_off, p_tag[2] + pre_z], dtype=np.float32),
            ),
            (
                "APPROACH",
                np.array([p_tag[0] + x_off, p_tag[1] + y_off, p_tag[2] + grasp_z], dtype=np.float32),
            ),
            (
                "LIFT",
                np.array(
                    [p_tag[0] + x_off, p_tag[1] + y_off, p_tag[2] + grasp_z + lift_dz],
                    dtype=np.float32,
                ),
            ),
        ]

        self.get_logger().info("=" * 72)
        self.get_logger().info("[Reachability Diagnose] start")
        self.get_logger().info(
            f"[Reachability Diagnose] corrected tag=({p_tag[0]:+.3f},{p_tag[1]:+.3f},{p_tag[2]:+.3f})"
        )
        in_bounds, msg = self._check_workspace_bounds(p_tag, "修正后 tag")
        self.get_logger().info(msg)

        for stage, target in targets:
            self.get_logger().info("-" * 72)
            self.get_logger().info(
                f"[Reachability Diagnose] stage={stage} target=({target[0]:+.3f},{target[1]:+.3f},{target[2]:+.3f})"
            )
            for quat_name, quat in quat_candidates:
                goal_pos = self._apply_tcp_offset(target, quat, tcp_offset_ee)
                self.get_logger().info(
                    f"[Reachability Diagnose] test stage={stage} quat={quat_name} "
                    f"goal=({goal_pos[0]:+.3f},{goal_pos[1]:+.3f},{goal_pos[2]:+.3f})"
                )
                ik_ok = self._debug_ik_reachability(goal_pos, quat)
                plan_ok = False
                if ik_ok:
                    plan_ok = self._plan_to_pose(goal_pos, quat) is not None
                self.get_logger().info(
                    f"[Reachability Diagnose] result stage={stage} quat={quat_name} "
                    f"ik_no_collision={ik_ok} motion_gen={plan_ok}"
                )

        self.get_logger().info("[Reachability Diagnose] done")
        self.get_logger().info("=" * 72)

    def _run_debug_mode(self):
        """运行调试模式"""
        self.get_logger().info("=" * 60)
        self.get_logger().info("[DEBUG MODE] 开始调试")
        self.get_logger().info("=" * 60)

        # 1. FK 验证
        self._debug_fk()

        # 2. 工作空间估算
        x_range, y_range, z_range = self._debug_workspace()

        # 3. IK 可达性测试 - 使用 EE 位置微扰
        self.get_logger().info("=" * 60)
        self.get_logger().info("[IK Debug] 开始 IK 可达性测试（基于 EE 微扰）")

        # 获取当前 EE 位置作为基准
        retract = self.motion_gen.get_retract_config()
        kin_state = self.motion_gen.kinematics.get_state(retract.view(1, -1))
        ee_pos_base = kin_state.ee_pose.position.cpu().numpy().flatten()
        ee_quat_base = kin_state.ee_pose.quaternion.cpu().numpy().flatten()

        self.get_logger().info(f"[IK Debug] 基准 EE 位置: {ee_pos_base.tolist()}")
        self.get_logger().info(f"[IK Debug] 基准 EE 四元数: {ee_quat_base.tolist()}")

        # 从基准位置做小扰动（在工作空间内）
        perturbations = [
            ([0.0, 0.0, 0.0], "原位"),
            ([0.05, 0.0, 0.0], "+X 5cm"),
            ([-0.05, 0.0, 0.0], "-X 5cm"),
            ([0.0, 0.05, 0.0], "+Y 5cm"),
            ([0.0, -0.05, 0.0], "-Y 5cm"),
            ([0.0, 0.0, 0.05], "+Z 5cm"),
            ([0.0, 0.0, -0.05], "-Z 5cm"),
            ([0.1, 0.0, 0.0], "+X 10cm"),
            ([0.0, 0.0, -0.1], "-Z 10cm"),
            ([0.05, -0.05, -0.05], "对角线"),
        ]

        results = []
        for delta, name in perturbations:
            test_pos = ee_pos_base + np.array(delta, dtype=np.float32)
            self.get_logger().info(f"\n[IK Test] {name}: pos={test_pos.tolist()}")
            success = self._debug_ik_reachability(test_pos, ee_quat_base.astype(np.float32))
            results.append((name, success))

        # 汇总结果
        self.get_logger().info("=" * 60)
        self.get_logger().info("[DEBUG MODE] 测试结果汇总")
        self.get_logger().info("=" * 60)
        success_count = sum(1 for _, s in results if s)
        self.get_logger().info(f"成功: {success_count}/{len(results)}")
        for name, success in results:
            status = "✓ 成功" if success else "✗ 失败"
            self.get_logger().info(f"  {name}: {status}")

        self.get_logger().info("=" * 60)
        self.get_logger().info("[DEBUG MODE] 调试完成")
        self.get_logger().info("=" * 60)

    # ---------------------- 主流程 ----------------------
    def _on_alignment_done(self, msg: Bool) -> None:
        if bool(msg.data):
            self._alignment_done = True

    def _wait_for_alignment(self) -> bool:
        if not bool(self.get_parameter("wait_for_alignment").value):
            return True
        timeout = float(self.get_parameter("alignment_wait_timeout").value)
        start = time.time()
        self.get_logger().info("[sync] waiting for alignment_done...")
        while rclpy.ok() and not self._alignment_done:
            if time.time() - start > timeout:
                self.get_logger().error("[sync] alignment wait timeout")
                return False
            rclpy.spin_once(self, timeout_sec=0.1)
            time.sleep(0.05)
        self.get_logger().info("[sync] alignment_done received")
        return True

    def _log_pose_if_due(
        self,
        base_frame: str,
        ee_frame: str,
        target_pos: Optional[np.ndarray] = None,
        target_idx: Optional[int] = None,
        force: bool = False,
    ) -> None:
        if self._pose_log_period is None:
            return
        now = time.time()
        if not force and (now - self._last_pose_log_time) < self._pose_log_period:
            return
        self._last_pose_log_time = now
        ee_tf = self._lookup_tf(base_frame, ee_frame)
        if ee_tf is None:
            self.get_logger().info("[pose] ee tf not available")
            return
        p_ee, q_ee_xyzw, age = ee_tf
        max_pose_age = float(self.get_parameter("max_pose_tf_age").value)
        if max_pose_age > 0.0 and age > max_pose_age:
            self.get_logger().info(f"[pose] ee tf stale age={age:.2f}s")
            return
        if target_pos is not None:
            err = float(np.linalg.norm(p_ee - target_pos))
            tag = f"{target_idx}/{self._pose_match_total}" if target_idx else "-"
            self.get_logger().info(
                f"[pose {tag}] ee=({p_ee[0]:+.3f},{p_ee[1]:+.3f},{p_ee[2]:+.3f}) "
                f"target=({target_pos[0]:+.3f},{target_pos[1]:+.3f},{target_pos[2]:+.3f}) "
                f"err={err:.3f} age={age:.2f}s"
            )
        else:
            self.get_logger().info(
                f"[pose] ee=({p_ee[0]:+.3f},{p_ee[1]:+.3f},{p_ee[2]:+.3f}) age={age:.2f}s"
            )

    def _wait_for_stable_tag_pose(
        self,
        base_frame: str,
        tag_frame: str,
        max_tf_age: float,
        stable_need: int,
        timeout: float,
        min_delta: float,
        prev_pos: Optional[np.ndarray],
        target_idx: int,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        self._tag_filt = None
        stable = 0
        start = time.time()
        last_log = 0.0
        while rclpy.ok():
            elapsed = time.time() - start
            if elapsed > timeout:
                self.get_logger().error("[pose-match] tag wait timeout")
                return None
            tag_tf = self._lookup_stereo_tf(base_frame, tag_frame)
            if tag_tf is None:
                stable = 0
            else:
                p_tag, q_tag_xyzw, age = tag_tf
                if age > max_tf_age:
                    stable = 0
                else:
                    p_tag = self._tag_filtered(p_tag)
                    if not self._is_finite(p_tag):
                        stable = 0
                        continue
                    if prev_pos is not None and min_delta > 0.0:
                        if float(np.linalg.norm(p_tag - prev_pos)) < min_delta:
                            stable = 0
                        else:
                            stable += 1
                    else:
                        stable += 1
                    if stable >= stable_need:
                        return p_tag.copy(), q_tag_xyzw.copy(), q_tag_xyzw.copy()

            if elapsed - last_log > 2.0:
                self.get_logger().info(
                    f"[pose-match] waiting target {target_idx}/{self._pose_match_total} "
                    f"stable={stable}/{stable_need}"
                )
                last_log = elapsed

            self._publish_cmd_full()
            self._log_pose_if_due(base_frame, self.get_parameter("ee_frame").value)
            rclpy.spin_once(self, timeout_sec=0.05)
            time.sleep(0.05)

        return None

    def _run_pose_match_sequence(self) -> None:
        base = str(self.get_parameter("base_frame").value)
        tagf = str(self.get_parameter("tag_frame").value)
        eef = str(self.get_parameter("ee_frame").value)
        max_tf_age = float(self.get_parameter("max_tf_age").value)
        stable_need = int(self.get_parameter("stable_need").value)
        tf_wait_timeout = float(self.get_parameter("tf_wait_timeout").value)

        count = int(self.get_parameter("pose_match_count").value)
        hold_sec = float(self.get_parameter("pose_match_hold_sec").value)
        min_delta = float(self.get_parameter("pose_match_min_delta").value)
        wait_timeout = float(self.get_parameter("pose_match_wait_timeout").value)
        z_offset = float(self.get_parameter("pose_match_z_offset").value)
        log_rate = float(self.get_parameter("pose_log_rate_hz").value)

        self._pose_match_total = max(1, count)
        self._pose_log_period = 1.0 / max(log_rate, 0.1)
        self._last_pose_log_time = 0.0

        if not self._wait_for_alignment():
            return

        joint_names = list(self.get_parameter("curobo_joint_names").value)
        x_off = float(self.get_parameter("grasp_x_offset").value)
        y_off = float(self.get_parameter("grasp_y_offset").value)
        dist_corr_x = float(self.get_parameter("distance_correction_x").value)
        dist_corr_y = float(self.get_parameter("distance_correction_y").value)
        dist_corr_z = float(self.get_parameter("distance_correction_z").value)
        target_offset = self._get_effective_target_offset(base, eef)
        tcp_offset_ee = None
        if bool(self.get_parameter("use_gripper_center_as_tcp").value):
            if self._tcp_offset_ee is None:
                self._tcp_offset_ee = self._compute_gripper_center_offset_ee(eef)
            tcp_offset_ee = self._tcp_offset_ee

        retract = self.motion_gen.get_retract_config()
        kin_state = self.motion_gen.kinematics.get_state(retract.view(1, -1))
        retract_quat = kin_state.ee_pose.quaternion.cpu().numpy().flatten().astype(np.float32)

        last_tag = None
        last_target = None

        for idx in range(self._pose_match_total):
            self.get_logger().info(f"[pose-match] target {idx+1}/{self._pose_match_total} acquire")
            if idx == 0:
                tag_res = self._wait_for_stable_tag_pose(
                    base, tagf, max_tf_age, stable_need, tf_wait_timeout, 0.0, None, idx + 1
                )
            else:
                tag_res = self._wait_for_stable_tag_pose(
                    base, tagf, max_tf_age, stable_need, wait_timeout, min_delta, last_tag, idx + 1
                )
            if tag_res is None:
                self.get_logger().error("[pose-match] failed to get tag pose")
                return

            tag_pos, tag_q_xyzw = tag_res
            last_tag = tag_pos.copy()
            target = tag_pos.copy()
            target[0] += dist_corr_x + x_off + target_offset[0]
            target[1] += dist_corr_y + y_off + target_offset[1]
            target[2] += dist_corr_z + z_offset + target_offset[2]

            self.get_logger().info(
                f"[pose-match] tag=({tag_pos[0]:+.3f},{tag_pos[1]:+.3f},{tag_pos[2]:+.3f}) "
                f"target=({target[0]:+.3f},{target[1]:+.3f},{target[2]:+.3f})"
            )

            cur_tf = self._lookup_tf(base, eef)
            if cur_tf is not None:
                _, q_cur_xyzw, _ = cur_tf
                q_cur = _xyzw_to_wxyz(q_cur_xyzw)
            else:
                q_cur = retract_quat

            q_down = np.array([0.707, 0.0, 0.707, 0.0], dtype=np.float32)
            if bool(self.get_parameter("adaptive_downward_grasp").value):
                q_down = self._compute_downward_quat(q_cur, tag_q_xyzw)
            prefer_downward = bool(self.get_parameter("prefer_downward_grasp").value)
            if prefer_downward:
                q_candidates = [q_down, q_cur, retract_quat]
            else:
                q_candidates = [q_cur, retract_quat, q_down]

            q_traj = None
            for qi, q_wxyz in enumerate(q_candidates):
                q_use = q_wxyz.astype(np.float32)
                goal_pos = self._apply_tcp_offset(target, q_use, tcp_offset_ee)
                q_traj = self._plan_to_pose(goal_pos, q_use)
                if q_traj is not None:
                    self.get_logger().info(
                        f"[pose-match] plan ok target {idx+1}/{self._pose_match_total} quat_try={qi}"
                    )
                    break
                self.get_logger().warn(
                    f"[pose-match] plan failed target {idx+1}/{self._pose_match_total} quat_try={qi}"
                )

            if q_traj is None:
                self.get_logger().error("[pose-match] plan failed, aborting")
                return

            self._exec_traj(joint_names, q_traj)
            self._log_pose_if_due(base, eef, target_pos=target, target_idx=idx + 1, force=True)

            hold_start = time.time()
            while rclpy.ok() and time.time() - hold_start < hold_sec:
                self._publish_cmd_full()
                self._log_pose_if_due(base, eef, target_pos=target, target_idx=idx + 1)
                rclpy.spin_once(self, timeout_sec=0.05)
                time.sleep(0.05)

            last_target = target.copy()

        self.get_logger().info("[pose-match] completed all targets")
        if last_target is not None:
            self._log_pose_if_due(base, eef, target_pos=last_target, target_idx=self._pose_match_total, force=True)

    def run(self):
        """主运行流程"""
        # 1) 等待关节状态
        self.get_logger().info("[同步] 等待 /joint_states ...")
        while rclpy.ok() and not self.data_ready:
            rclpy.spin_once(self, timeout_sec=0.1)
            time.sleep(0.05)

        if not self.R_JOINTS or not self.L_JOINTS:
            self.get_logger().error("[失败] 未发现机械臂关节名")
            return

        # 2) 锁定当前姿态
        self._init_hold()
        self.get_logger().info("[同步] hold 初始化完成")

        # 3) 初始化 cuRobo
        self._init_curobo()

        # 3.5) 调试模式
        if self.get_parameter("debug_mode").value:
            self._run_debug_mode()
            self.get_logger().info("[DEBUG MODE] 调试完成，退出程序")
            return

        diagnostic_only = bool(self.get_parameter("reachability_diagnostic_mode").value)
        if diagnostic_only:
            self.get_logger().info("[Reachability Diagnose] skip safe pose and gripper motion")
        else:
            # 4) 移动到安全姿态
            self._move_to_safe_pose()

            # 5) 打开夹爪
            grip_open = float(self.get_parameter("grip_open").value)
            if self.GRIP_R is not None:
                self.get_logger().info("[动作] 打开夹爪")
                self._smooth_gripper(grip_open)

        if bool(self.get_parameter("pose_match_mode").value):
            self._run_pose_match_sequence()
            return

        # 6) 获取目标位置 - 支持直接坐标模式或 TF 模式
        base = str(self.get_parameter("base_frame").value)
        tagf = str(self.get_parameter("tag_frame").value)
        eef = str(self.get_parameter("ee_frame").value)
        max_tf_age = float(self.get_parameter("max_tf_age").value)
        stable_need = int(self.get_parameter("stable_need").value)
        tf_wait_timeout = float(self.get_parameter("tf_wait_timeout").value)
        use_direct_coords = bool(self.get_parameter("use_direct_coords").value)

        last_tag = None
        last_ee_q_wxyz = None
        last_tag_q_xyzw = None

        if use_direct_coords:
            # ==================== 直接坐标模式 ====================
            self.get_logger().info("[直接坐标模式] 跳过 TF 查找，使用指定坐标")

            direct_pos = list(self.get_parameter("direct_grasp_position").value)
            if len(direct_pos) != 3:
                self.get_logger().error(
                    f"[失败] direct_grasp_position 必须包含 3 个坐标值 (X, Y, Z)，当前: {direct_pos}"
                )
                return

            last_tag = np.array(direct_pos, dtype=np.float32)
            self.get_logger().info(
                f"[直接坐标模式] 目标坐标 (base_link): X={last_tag[0]:.3f}, "
                f"Y={last_tag[1]:.3f}, Z={last_tag[2]:.3f}"
            )

            # 工作空间边界验证
            in_bounds, msg = self._check_workspace_bounds(last_tag, "直接坐标")
            self.get_logger().info(msg)
            if not in_bounds:
                self.get_logger().warn(
                    "[直接坐标模式] 警告: 目标位置可能超出工作空间，尝试自动调整底盘"
                )
                # 自动调整底盘使目标进入右臂工作空间
                last_tag = self._adjust_base_for_direct_coords(last_tag)

                # 重新验证
                in_bounds_after, msg_after = self._check_workspace_bounds(last_tag, "调整后坐标")
                self.get_logger().info(msg_after)
                if not in_bounds_after:
                    self.get_logger().warn(
                        "[直接坐标模式] 调整后仍超出工作空间，规划可能失败"
                    )

            # 获取 EE 四元数 - 使用 retract 配置
            retract = self.motion_gen.get_retract_config()
            kin_state = self.motion_gen.kinematics.get_state(retract.view(1, -1))
            last_ee_q_wxyz = kin_state.ee_pose.quaternion.cpu().numpy().flatten().astype(np.float32)
            self.get_logger().info(f"[直接坐标模式] 使用 retract 四元数: {last_ee_q_wxyz.tolist()}")

        else:
            # ==================== 原始 TF 模式 ====================
            self.get_logger().info(f"[同步] 等待稳定 TF: {base}->left_{tagf} 或 right_{tagf} / {base}->{eef}")
            self.get_logger().info(f"[同步] 需要连续 {stable_need} 次成功，超时 {tf_wait_timeout}s")
            stable = 0
            tf_wait_start = time.time()
            last_log_time = 0.0

            while rclpy.ok() and stable < stable_need:
                # 超时检查
                elapsed = time.time() - tf_wait_start
                if elapsed > tf_wait_timeout:
                    self.get_logger().warn(f"[超时] TF 等待超过 {tf_wait_timeout}s，使用当前数据继续")
                    break

                # 进度日志（每 5 秒）
                if elapsed - last_log_time > 5.0:
                    self.get_logger().info(f"[等待] TF 稳定进度: {stable}/{stable_need}, 已等待 {elapsed:.1f}s")
                    last_log_time = elapsed

                tag_tf = self._lookup_stereo_tf(base, tagf)
                ee_tf = self._lookup_tf(base, eef)
                if tag_tf is None:
                    stable = 0
                    self._publish_cmd_full()
                    rclpy.spin_once(self, timeout_sec=0.05)
                    time.sleep(0.05)
                    continue

                p_tag, q_tag_xyzw, age_tag = tag_tf
                if age_tag > max_tf_age:
                    stable = 0
                    self._publish_cmd_full()
                    rclpy.spin_once(self, timeout_sec=0.05)
                    time.sleep(0.05)
                    continue

                last_tag = self._tag_filtered(p_tag)
                last_tag_q_xyzw = q_tag_xyzw

                if ee_tf is None:
                    q_fallback = self._get_current_ee_quat_wxyz()
                    if q_fallback is None:
                        stable = 0
                        self._publish_cmd_full()
                        rclpy.spin_once(self, timeout_sec=0.05)
                        time.sleep(0.05)
                        continue
                    if not self._ee_tf_warned:
                        self.get_logger().warn("[tf] ee tf missing/stale; using cuRobo FK quat")
                        self._ee_tf_warned = True
                    last_ee_q_wxyz = q_fallback
                else:
                    p_ee, q_ee_xyzw, age_ee = ee_tf
                    if age_ee > max_tf_age:
                        q_fallback = self._get_current_ee_quat_wxyz()
                        if q_fallback is None:
                            stable = 0
                            self._publish_cmd_full()
                            rclpy.spin_once(self, timeout_sec=0.05)
                            time.sleep(0.05)
                            continue
                        if not self._ee_tf_warned:
                            self.get_logger().warn("[tf] ee tf missing/stale; using cuRobo FK quat")
                            self._ee_tf_warned = True
                        last_ee_q_wxyz = q_fallback
                    else:
                        last_ee_q_wxyz = _xyzw_to_wxyz(q_ee_xyzw)

                stable += 1
                rclpy.spin_once(self, timeout_sec=0.0)
                time.sleep(0.02)
            if last_tag is None:
                stale_tf = self._lookup_stereo_tf(base, tagf, allow_stale=True)
                if stale_tf is not None:
                    last_tag, last_tag_q_xyzw, _ = stale_tf
                    self.get_logger().warn("[tf] tag tf stale; using last available tag")
                else:
                    self.get_logger().error("[error] TF unavailable; check apriltag.launch.py")
                    return
            if last_ee_q_wxyz is None:
                retract = self.motion_gen.get_retract_config()
                kin_state = self.motion_gen.kinematics.get_state(retract.view(1, -1))
                last_ee_q_wxyz = kin_state.ee_pose.quaternion.cpu().numpy().flatten().astype(np.float32)
                self.get_logger().warn("[tf] ee tf unavailable; using retract quat")

        # 7) 冻结目标位置
        x_off = float(self.get_parameter("grasp_x_offset").value)
        y_off = float(self.get_parameter("grasp_y_offset").value)
        pre_z = float(self.get_parameter("pregrasp_z_offset").value)
        grasp_z = float(self.get_parameter("grasp_z_offset").value)
        lift_dz = float(self.get_parameter("lift_dz").value)

        # 距离修正（修正视觉测距偏差）
        dist_corr_x = float(self.get_parameter("distance_correction_x").value)
        dist_corr_y = float(self.get_parameter("distance_correction_y").value)
        dist_corr_z = float(self.get_parameter("distance_correction_z").value)
        target_offset = self._get_effective_target_offset(base, eef)
        tcp_offset_ee = None
        if bool(self.get_parameter("use_gripper_center_as_tcp").value):
            if self._tcp_offset_ee is None:
                self._tcp_offset_ee = self._compute_gripper_center_offset_ee(eef)
            tcp_offset_ee = self._tcp_offset_ee

        p_tag = last_tag.copy()
        p_tag_original = p_tag.copy()  # 保存原始值用于日志

        # 应用距离修正
        p_tag[0] += dist_corr_x + target_offset[0]
        p_tag[1] += dist_corr_y + target_offset[1]
        p_tag[2] += dist_corr_z + target_offset[2]

        self.get_logger().info(
            f"[目标] tag原始位置={p_tag_original.tolist()}"
        )
        self.get_logger().info(
            f"[目标] 距离修正=({dist_corr_x:.3f}, {dist_corr_y:.3f}, {dist_corr_z:.3f})"
        )
        self.get_logger().info(
            f"[target] extra_offset=({target_offset[0]:.3f}, {target_offset[1]:.3f}, {target_offset[2]:.3f})"
        )
        self.get_logger().info(
            f"[目标] tag修正后位置={p_tag.tolist()} offsets(xy)=({x_off},{y_off})"
        )

        # 7.5) 底盘调整 - 如果目标超出工作空间（仅 TF 模式）
        if not use_direct_coords:
            self._move_base_to_target(p_tag)

        # 7.6) 底盘移动后重新获取稳定 TF（仅 TF 模式）
        if not use_direct_coords:
            self.get_logger().info("[同步] 底盘调整后重新获取稳定 TF...")
            self._tag_filt = None  # 重置滤波器
            stable = 0
            tf_wait_start = time.time()
            last_log_time = 0.0
            while rclpy.ok() and stable < stable_need:
                # 超时检查
                elapsed = time.time() - tf_wait_start
                if elapsed > tf_wait_timeout:
                    self.get_logger().warn(f"[超时] TF 等待超过 {tf_wait_timeout}s，使用当前数据继续")
                    break

                # 进度日志（每 5 秒）
                if elapsed - last_log_time > 5.0:
                    self.get_logger().info(f"[等待] TF 稳定进度: {stable}/{stable_need}, 已等待 {elapsed:.1f}s")
                    last_log_time = elapsed

                tag_tf = self._lookup_stereo_tf(base, tagf)
                ee_tf = self._lookup_tf(base, eef)
                if tag_tf is None or ee_tf is None:
                    stable = 0
                    self._publish_cmd_full()
                    rclpy.spin_once(self, timeout_sec=0.05)
                    time.sleep(0.05)
                    continue

                p_tag_new, q_tag_xyzw, age_tag = tag_tf
                p_ee, q_ee_xyzw, age_ee = ee_tf

                if age_tag > max_tf_age or age_ee > max_tf_age:
                    stable = 0
                    self._publish_cmd_full()
                    rclpy.spin_once(self, timeout_sec=0.05)
                    time.sleep(0.05)
                    continue

                last_tag = self._tag_filtered(p_tag_new)
                last_tag_q_xyzw = q_tag_xyzw
                last_ee_q_wxyz = _xyzw_to_wxyz(q_ee_xyzw)
                stable += 1
                rclpy.spin_once(self, timeout_sec=0.0)
                time.sleep(0.02)

            # 更新目标位置
            p_tag = last_tag.copy()

            # 应用距离修正（与第一次相同）
            p_tag[0] += dist_corr_x + target_offset[0]
            p_tag[1] += dist_corr_y + target_offset[1]
            p_tag[2] += dist_corr_z + target_offset[2]

            self.get_logger().info(f"[目标] 更新后 tag修正位置={p_tag.tolist()}")
        else:
            self.get_logger().info("[直接坐标模式] 跳过底盘调整和 TF 重新获取")

        # 检查目标是否在右臂工作空间内
        if p_tag[1] > 0.1:  # Y > 0.1m，在机器人左侧
            self.get_logger().warn(
                f"[警告] 目标 Y={p_tag[1]:.3f}m 在机器人左侧，右臂可能无法到达。"
                f"考虑使用左臂或调整机器人位置。"
            )

        # 8) 姿态候选列表 - 优先使用 retract 配置的四元数（经过验证可达）
        # 获取 retract 配置的 EE 四元数
        retract = self.motion_gen.get_retract_config()
        kin_state = self.motion_gen.kinematics.get_state(retract.view(1, -1))
        retract_quat = kin_state.ee_pose.quaternion.cpu().numpy().flatten().astype(np.float32)

        self.get_logger().info(f"[姿态] retract 四元数: {retract_quat.tolist()}")

        q_down = np.array([0.707, 0.0, 0.707, 0.0], dtype=np.float32)
        if bool(self.get_parameter("adaptive_downward_grasp").value):
            q_seed = last_ee_q_wxyz if last_ee_q_wxyz is not None else retract_quat
            q_down = self._compute_downward_quat(q_seed, last_tag_q_xyzw)
        q_slant = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        prefer_downward = bool(self.get_parameter("prefer_downward_grasp").value)
        if prefer_downward:
            q_candidates = [q_down, retract_quat, last_ee_q_wxyz, q_slant]
        else:
            q_candidates = [retract_quat, last_ee_q_wxyz, q_down, q_slant]

        joint_names = list(self.get_parameter("curobo_joint_names").value)

        if bool(self.get_parameter("reachability_diagnostic_mode").value):
            self._run_target_reachability_diagnostics(
                p_tag,
                last_ee_q_wxyz,
                last_tag_q_xyzw,
                tcp_offset_ee,
            )
            self.get_logger().info("[Reachability Diagnose] diagnostic mode enabled; exiting before motion.")
            return

        def plan_with_quat(goal_pos_xyz: np.ndarray, stage: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
            """尝试多个姿态进行规划"""
            for qi, q_wxyz in enumerate(q_candidates):
                if q_wxyz is None:
                    continue
                goal_pos = self._apply_tcp_offset(goal_pos_xyz, q_wxyz, tcp_offset_ee)
                q_traj = self._plan_to_pose(goal_pos, q_wxyz)
                if q_traj is not None:
                    self.get_logger().info(f"[规划成功] stage={stage} quat_try={qi}")
                    return q_traj
                self.get_logger().warn(f"[规划失败] stage={stage} quat_try={qi}")
            return None

        # 9) PREGRASP - 移动到物体上方
        self.get_logger().info("[阶段] PREGRASP")
        p_pre = np.array(
            [p_tag[0] + x_off, p_tag[1] + y_off, p_tag[2] + pre_z],
            dtype=np.float32
        )
        q_traj = plan_with_quat(p_pre, "PREGRASP")
        if q_traj is None:
            self.get_logger().error("[失败] PREGRASP 规划失败")
            return
        self._exec_traj(joint_names, q_traj)

        # 10) APPROACH - 下降到抓取高度
        self.get_logger().info("[阶段] APPROACH")
        p_grasp = np.array(
            [p_tag[0] + x_off, p_tag[1] + y_off, p_tag[2] + grasp_z],
            dtype=np.float32
        )
        q_traj = plan_with_quat(p_grasp, "APPROACH")
        if q_traj is None:
            self.get_logger().error("[失败] APPROACH 规划失败")
            return
        self._exec_traj(joint_names, q_traj)

        # 10.5) 视觉二次确认 - 检查 tag 位置是否有偏差（仅 TF 模式）
        if not use_direct_coords:
            self.get_logger().info("[阶段] 视觉二次确认")
            visual_stable = 0
            p_tag_recheck = None
            while rclpy.ok() and visual_stable < 10:
                tag_tf = self._lookup_stereo_tf(base, tagf)
                if tag_tf is not None:
                    p_tag_new, _, age = tag_tf
                    if age < max_tf_age:
                        p_tag_recheck = p_tag_new.copy()
                        visual_stable += 1
                self._publish_cmd_full()
                rclpy.spin_once(self, timeout_sec=0.02)
                time.sleep(0.02)

            if p_tag_recheck is not None:
                p_tag_recheck_corr = p_tag_recheck.copy()
                p_tag_recheck_corr[0] += dist_corr_x + target_offset[0]
                p_tag_recheck_corr[1] += dist_corr_y + target_offset[1]
                p_tag_recheck_corr[2] += dist_corr_z + target_offset[2]
                pos_error = np.linalg.norm(p_tag_recheck_corr - p_tag)
                self.get_logger().info(f"[二次确认] 位置偏差: {pos_error:.4f}m")
                if pos_error > 0.03:  # 偏差超过 3cm
                    self.get_logger().warn(f"[二次确认] 偏差较大，重新规划 APPROACH")
                    # 更新目标位置
                    p_tag = p_tag_recheck_corr.copy()
                    p_grasp = np.array(
                        [p_tag[0] + x_off, p_tag[1] + y_off, p_tag[2] + grasp_z],
                        dtype=np.float32
                    )
                    q_traj = plan_with_quat(p_grasp, "APPROACH_REFINE")
                    if q_traj is not None:
                        self._exec_traj(joint_names, q_traj)
                    else:
                        self.get_logger().warn("[二次确认] 重新规划失败，继续使用原位置")
        else:
            self.get_logger().info("[直接坐标模式] 跳过视觉二次确认")

        # 10.6) 预压动作 - 下探 2cm 确保接触
        self.get_logger().info("[阶段] 预压 (下探 2cm)")
        p_prepress = np.array(
            [p_grasp[0], p_grasp[1], p_grasp[2] - 0.02],
            dtype=np.float32
        )
        q_traj = plan_with_quat(p_prepress, "PREPRESS")
        if q_traj is not None:
            self._exec_traj(joint_names, q_traj)
        else:
            self.get_logger().warn("[预压] 规划失败，跳过预压")

        # 11) CLOSE - 闭合夹爪
        self.get_logger().info("[阶段] CLOSE")
        grip_close = float(self.get_parameter("grip_close").value)
        self._smooth_gripper(grip_close)
        time.sleep(0.3)

        # 12) LIFT - 抬起物体
        self.get_logger().info("[阶段] LIFT")
        p_lift = np.array(
            [p_grasp[0], p_grasp[1], p_grasp[2] + lift_dz],
            dtype=np.float32
        )
        q_traj = plan_with_quat(p_lift, "LIFT")
        if q_traj is None:
            self.get_logger().warn("[告警] LIFT 规划失败（但已夹住）")
        else:
            self._exec_traj(joint_names, q_traj)

        # 13) 完成 - 保持姿态
        self.get_logger().info("[完成] 抓取流程结束：保持姿态（Ctrl-C 退出）")
        while rclpy.ok():
            self._publish_cmd_full()
            rclpy.spin_once(self, timeout_sec=0.05)
            time.sleep(0.1)


def main():
    rclpy.init()
    node = Task2GraspTag79CuRobo()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
