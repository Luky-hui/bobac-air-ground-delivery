#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tag-based right-arm alignment + grasping controller.

Usage:
  python3 /home/u-zhuang/ros2_ws/src/isaac/isaac_sim_2026/launch/task2_grasp_tag79_v5.py \
    --ros-args -p use_sim_time:=true
"""

import math
import time
from collections import deque
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener, TransformException

try:
    from urdf_parser_py.urdf import URDF
    from kdl_parser_py.urdf import treeFromUrdfModel
    import PyKDL
except Exception:  # pragma: no cover - ROS environment provides these
    PyKDL = None


class GraspState(Enum):
    WAIT = 0
    RAISE = 1
    ALIGN_XY = 2
    PREGRASP = 3
    APPROACH = 4
    CLOSE = 5
    LIFT = 6
    DONE = 7


def _stamp_to_ns(stamp_msg) -> int:
    return int(stamp_msg.sec) * 1_000_000_000 + int(stamp_msg.nanosec)


class TagGraspController(Node):
    def __init__(self) -> None:
        super().__init__("task2_grasp_tag79_v5")

        # ---------------- Parameters ----------------
        self.declare_parameter(
            "urdf_path",
            "/home/u-zhuang/ros2_ws/src/isaac/isaac_sim_2026/config/mercury_x1_turing_stack.urdf",
        )
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("ee_link", "gripper01_base")

        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("joint_command_topic", "/joint_command")

        self.declare_parameter("tag_frame", "tag36h11:79")
        self.declare_parameter("target_frames", [])
        self.declare_parameter("max_tf_age_sec", 0.5)
        self.declare_parameter("tag_window", 10)

        self.declare_parameter("rate_hz", 20.0)
        self.declare_parameter("step_cart_limit", 0.01)
        self.declare_parameter("dls_lambda", 0.20)
        self.declare_parameter("max_joint_step", 0.05)
        self.declare_parameter("max_error_norm", 3.0)

        self.declare_parameter("joint_limit_margin_deg", 10.0)

        self.declare_parameter("align_xy_tol", 0.05)
        self.declare_parameter("pregrasp_tol", 0.05)
        self.declare_parameter("approach_tol", 0.03)
        self.declare_parameter("lift_tol", 0.05)
        self.declare_parameter("raise_tol", 0.03)

        self.declare_parameter("pregrasp_z_offset", 0.18)
        self.declare_parameter("approach_dz", 0.05)
        self.declare_parameter("lift_dz", 0.12)
        self.declare_parameter("raise_extra_z", 0.05)
        self.declare_parameter("min_ee_z", 0.15)

        self.declare_parameter("tag_offset_x", 0.0)
        self.declare_parameter("tag_offset_y", 0.0)
        self.declare_parameter("tag_offset_z", 0.0)

        self.declare_parameter("gripper_name", "gripper_controller")
        self.declare_parameter("gripper_open_pos", 1.5)
        self.declare_parameter("gripper_close_pos", -1.5)
        self.declare_parameter("gripper_close_hold_sec", 0.6)

        self.declare_parameter("align_only", False)
        self.declare_parameter("lost_timeout_sec", 1.0)

        # ---------------- Load params ----------------
        self.urdf_path = Path(self.get_parameter("urdf_path").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.ee_link = str(self.get_parameter("ee_link").value)

        self.joint_state_topic = str(self.get_parameter("joint_state_topic").value)
        self.joint_command_topic = str(self.get_parameter("joint_command_topic").value)

        self.tag_frame = str(self.get_parameter("tag_frame").value)
        raw_frames = self.get_parameter("target_frames").value
        if isinstance(raw_frames, list) and len(raw_frames) > 0:
            self.target_frames = [str(x) for x in raw_frames]
        else:
            self.target_frames = [
                self.tag_frame,
                f"left_{self.tag_frame}",
                f"right_{self.tag_frame}",
            ]

        self.max_tf_age = float(self.get_parameter("max_tf_age_sec").value)
        self.tag_window = int(self.get_parameter("tag_window").value)

        self.rate_hz = float(self.get_parameter("rate_hz").value)
        self.period = 1.0 / max(self.rate_hz, 1e-6)
        self.step_cart_limit = float(self.get_parameter("step_cart_limit").value)
        self.dls_lambda = float(self.get_parameter("dls_lambda").value)
        self.max_joint_step = float(self.get_parameter("max_joint_step").value)
        self.max_error_norm = float(self.get_parameter("max_error_norm").value)

        self.joint_limit_margin_deg = float(self.get_parameter("joint_limit_margin_deg").value)

        self.align_xy_tol = float(self.get_parameter("align_xy_tol").value)
        self.pregrasp_tol = float(self.get_parameter("pregrasp_tol").value)
        self.approach_tol = float(self.get_parameter("approach_tol").value)
        self.lift_tol = float(self.get_parameter("lift_tol").value)
        self.raise_tol = float(self.get_parameter("raise_tol").value)

        self.pregrasp_z_offset = float(self.get_parameter("pregrasp_z_offset").value)
        self.approach_dz = float(self.get_parameter("approach_dz").value)
        self.lift_dz = float(self.get_parameter("lift_dz").value)
        self.raise_extra_z = float(self.get_parameter("raise_extra_z").value)
        self.min_ee_z = float(self.get_parameter("min_ee_z").value)

        self.tag_offset = np.array(
            [
                float(self.get_parameter("tag_offset_x").value),
                float(self.get_parameter("tag_offset_y").value),
                float(self.get_parameter("tag_offset_z").value),
            ],
            dtype=float,
        )

        self.gripper_name = str(self.get_parameter("gripper_name").value)
        self.gripper_open_pos = float(self.get_parameter("gripper_open_pos").value)
        self.gripper_close_pos = float(self.get_parameter("gripper_close_pos").value)
        self.gripper_close_hold_sec = float(self.get_parameter("gripper_close_hold_sec").value)

        self.align_only = bool(self.get_parameter("align_only").value)
        self.lost_timeout_sec = float(self.get_parameter("lost_timeout_sec").value)

        # ---------------- Joint limits (rad) ----------------
        self.right_arm_joints = [
            "joint1_R",
            "joint2_R",
            "joint3_R",
            "joint4_R",
            "joint5_R",
            "joint6_R",
        ]
        self.joint_limits = {
            "joint1_R": (math.radians(-165.0), math.radians(165.0)),
            "joint2_R": (math.radians(-55.0), math.radians(95.0)),
            "joint3_R": (math.radians(-173.0), math.radians(5.0)),
            "joint4_R": (math.radians(-165.0), math.radians(165.0)),
            "joint5_R": (math.radians(-20.0), math.radians(265.0)),
            "joint6_R": (math.radians(-180.0), math.radians(180.0)),
        }

        # ---------------- TF and ROS I/O ----------------
        self.tf_buffer = Buffer(cache_time=Duration(seconds=30.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.pub_joint_cmd = self.create_publisher(JointState, self.joint_command_topic, 10)
        self.create_subscription(JointState, self.joint_state_topic, self._joint_state_cb, 10)

        self.current_joints: Dict[str, float] = {}
        self.joint_ready = False
        self.last_cmd_q: Optional[np.ndarray] = None

        self.tag_history = deque(maxlen=max(1, self.tag_window))
        self.last_tag_wall: Optional[float] = None

        # ---------------- KDL ----------------
        self.chain_joint_names: List[str] = []
        self.chain_index: Dict[str, int] = {}
        self.fk_solver = None
        self.jac_solver = None
        self._load_kdl_chain()

        # ---------------- State machine ----------------
        self.state = GraspState.WAIT
        self.close_start_wall: Optional[float] = None

        self.timer = self.create_timer(self.period, self._control_step)

        self.get_logger().info(
            f"TagGraspController ready | base={self.base_frame} ee={self.ee_link} "
            f"frames={self.target_frames}"
        )

    # ---------------- Setup helpers ----------------
    def _load_kdl_chain(self) -> None:
        if PyKDL is None:
            raise RuntimeError("PyKDL/kdl_parser_py not available in this ROS environment.")

        if not self.urdf_path.exists():
            raise RuntimeError(f"URDF not found: {self.urdf_path}")

        robot = URDF.from_xml_file(str(self.urdf_path))
        ok, tree = treeFromUrdfModel(robot)
        if not ok:
            raise RuntimeError("Failed to build KDL tree from URDF.")

        chain = tree.getChain(self.base_frame, self.ee_link)
        self.fk_solver = PyKDL.ChainFkSolverPos_recursive(chain)
        self.jac_solver = PyKDL.ChainJntToJacSolver(chain)

        self.chain_joint_names = []
        none_type = getattr(PyKDL.Joint, "None", None)
        if none_type is None:
            none_type = PyKDL.Joint.Fixed
        for i in range(chain.getNrOfSegments()):
            joint = chain.getSegment(i).getJoint()
            if joint.getType() != none_type:
                self.chain_joint_names.append(joint.getName())
        self.chain_index = {n: i for i, n in enumerate(self.chain_joint_names)}

        missing = [n for n in self.right_arm_joints if n not in self.chain_index]
        if missing:
            raise RuntimeError(f"KDL chain is missing joints: {missing}")

    # ---------------- ROS callbacks ----------------
    def _joint_state_cb(self, msg: JointState) -> None:
        for i, name in enumerate(msg.name):
            self.current_joints[name] = msg.position[i]
        self.joint_ready = all(j in self.current_joints for j in self.right_arm_joints)

    # ---------------- TF helpers ----------------
    def _pick_best_tag_tf(self) -> Optional[Tuple[str, object, float]]:
        now_ns = int(self.get_clock().now().nanoseconds)
        best = None  # (age, frame, tf_msg)
        for frame in self.target_frames:
            if not self.tf_buffer.can_transform(self.base_frame, frame, Time()):
                continue
            try:
                t = self.tf_buffer.lookup_transform(self.base_frame, frame, Time())
            except TransformException:
                continue
            age = max(0.0, (now_ns - _stamp_to_ns(t.header.stamp)) / 1e9)
            if best is None or age < best[0]:
                best = (age, frame, t)
        if best is None:
            return None
        age, frame, t = best
        if age > self.max_tf_age:
            return None
        return frame, t, age

    def _get_tag_position(self) -> Optional[np.ndarray]:
        picked = self._pick_best_tag_tf()
        if picked is None:
            return None
        _, _, t = picked
        p = np.array(
            [t.transform.translation.x, t.transform.translation.y, t.transform.translation.z],
            dtype=float,
        )
        self.tag_history.append(p)
        self.last_tag_wall = time.time()
        avg = np.mean(np.array(self.tag_history), axis=0)
        return avg + self.tag_offset

    # ---------------- Kinematics ----------------
    def _build_jnt_array(self) -> Optional["PyKDL.JntArray"]:
        q = PyKDL.JntArray(len(self.chain_joint_names))
        for name, idx in self.chain_index.items():
            if name not in self.current_joints:
                return None
            q[idx] = self.current_joints[name]
        return q

    def _fk(self, q: "PyKDL.JntArray") -> np.ndarray:
        frame = PyKDL.Frame()
        self.fk_solver.JntToCart(q, frame)
        return np.array([frame.p[0], frame.p[1], frame.p[2]], dtype=float)

    def _jacobian(self, q: "PyKDL.JntArray") -> np.ndarray:
        jac = PyKDL.Jacobian(len(self.chain_joint_names))
        self.jac_solver.JntToJac(q, jac)
        j = np.zeros((6, len(self.chain_joint_names)), dtype=float)
        for r in range(6):
            for c in range(len(self.chain_joint_names)):
                j[r, c] = jac[r, c]
        return j

    def _dls(self, j_pos: np.ndarray, dx: np.ndarray) -> np.ndarray:
        jj_t = j_pos @ j_pos.T
        damp = (self.dls_lambda ** 2) * np.eye(3)
        return j_pos.T @ np.linalg.solve(jj_t + damp, dx)

    def _clamp_joint(self, name: str, value: float) -> float:
        lo, hi = self.joint_limits.get(name, (-math.inf, math.inf))
        margin = math.radians(self.joint_limit_margin_deg)
        lo_m = lo + margin
        hi_m = hi - margin
        if lo_m > hi_m:
            return min(max(value, lo), hi)
        return min(max(value, lo_m), hi_m)

    # ---------------- State logic ----------------
    def _set_state(self, new_state: GraspState) -> None:
        if new_state == self.state:
            return
        self.state = new_state
        if new_state == GraspState.CLOSE:
            self.close_start_wall = time.time()
        self.get_logger().info(f"State -> {new_state.name}")

    def _publish_command(self, q_cmd: np.ndarray, gripper_cmd: float) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(self.right_arm_joints) + [self.gripper_name]
        msg.position = [
            float(q_cmd[self.chain_index[name]]) for name in self.right_arm_joints
        ] + [float(gripper_cmd)]
        self.pub_joint_cmd.publish(msg)
        self.last_cmd_q = q_cmd.copy()

    # ---------------- Control loop ----------------
    def _control_step(self) -> None:
        if not self.joint_ready:
            return

        tag_pos = self._get_tag_position()
        if tag_pos is None:
            if self.state != GraspState.WAIT and self.last_cmd_q is not None:
                if (
                    self.last_tag_wall is not None
                    and (time.time() - self.last_tag_wall) > self.lost_timeout_sec
                ):
                    self.get_logger().warn("Tag lost. Returning to WAIT.")
                    self._set_state(GraspState.WAIT)
                self._publish_command(self.last_cmd_q, self._desired_gripper())
            return

        q = self._build_jnt_array()
        if q is None:
            return

        q_np = np.array([q[i] for i in range(len(self.chain_joint_names))], dtype=float)
        ee_pos = self._fk(q)

        # -------- State transitions --------
        if self.state == GraspState.WAIT:
            self._set_state(GraspState.RAISE)
            return

        target, mask, tol = self._target_for_state(tag_pos, ee_pos)
        error = (target - ee_pos) * mask

        if np.linalg.norm(error) > self.max_error_norm:
            self.get_logger().warn("EE error too large. Holding position.")
            self._publish_command(q_np, self._desired_gripper())
            return

        if self._reached(error, tol, mask):
            self._advance_state()
            if self.state == GraspState.DONE:
                return
            target, mask, tol = self._target_for_state(tag_pos, ee_pos)
            error = (target - ee_pos) * mask

        # -------- Close stage: hold arm, close gripper --------
        if self.state == GraspState.CLOSE:
            self._publish_command(q_np, self._desired_gripper())
            return

        # -------- DLS position control --------
        if np.linalg.norm(error) < 1e-9:
            self._publish_command(q_np, self._desired_gripper())
            return

        dx = error.copy()
        norm = np.linalg.norm(dx)
        if norm > self.step_cart_limit:
            dx = dx / norm * self.step_cart_limit

        j = self._jacobian(q)
        j_pos = j[0:3, :]
        dq = self._dls(j_pos, dx)
        dq = np.clip(dq, -self.max_joint_step, self.max_joint_step)

        q_cmd = q_np + dq
        for name in self.right_arm_joints:
            idx = self.chain_index[name]
            q_cmd[idx] = self._clamp_joint(name, q_cmd[idx])

        self._publish_command(q_cmd, self._desired_gripper())

    def _desired_gripper(self) -> float:
        if self.state in (GraspState.CLOSE, GraspState.LIFT, GraspState.DONE):
            return self.gripper_close_pos
        return self.gripper_open_pos

    def _target_for_state(
        self, tag_pos: np.ndarray, ee_pos: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        if self.state == GraspState.RAISE:
            target = ee_pos.copy()
            safe_z = max(
                ee_pos[2],
                tag_pos[2] + self.pregrasp_z_offset + self.raise_extra_z,
                self.min_ee_z,
            )
            target[2] = safe_z
            return target, np.array([0.0, 0.0, 1.0]), self.raise_tol

        if self.state == GraspState.ALIGN_XY:
            target = ee_pos.copy()
            target[0] = tag_pos[0]
            target[1] = tag_pos[1]
            return target, np.array([1.0, 1.0, 0.0]), self.align_xy_tol

        if self.state == GraspState.PREGRASP:
            target = tag_pos.copy()
            target[2] = max(tag_pos[2] + self.pregrasp_z_offset, self.min_ee_z)
            return target, np.array([1.0, 1.0, 1.0]), self.pregrasp_tol

        if self.state == GraspState.APPROACH:
            target = tag_pos.copy()
            target[2] = max(tag_pos[2] + self.approach_dz, self.min_ee_z)
            return target, np.array([0.0, 0.0, 1.0]), self.approach_tol

        if self.state == GraspState.LIFT:
            target = tag_pos.copy()
            target[2] = max(tag_pos[2] + self.lift_dz, self.min_ee_z)
            return target, np.array([0.0, 0.0, 1.0]), self.lift_tol

        return ee_pos.copy(), np.array([0.0, 0.0, 0.0]), 0.0

    def _reached(self, error: np.ndarray, tol: float, mask: np.ndarray) -> bool:
        if mask.sum() == 0:
            return True
        if mask[2] == 1.0 and mask[0] == 0.0 and mask[1] == 0.0:
            return abs(error[2]) < tol
        return np.linalg.norm(error) < tol

    def _advance_state(self) -> None:
        if self.state == GraspState.RAISE:
            self._set_state(GraspState.ALIGN_XY)
        elif self.state == GraspState.ALIGN_XY:
            if self.align_only:
                self._set_state(GraspState.DONE)
            else:
                self._set_state(GraspState.PREGRASP)
        elif self.state == GraspState.PREGRASP:
            self._set_state(GraspState.APPROACH)
        elif self.state == GraspState.APPROACH:
            self._set_state(GraspState.CLOSE)
        elif self.state == GraspState.CLOSE:
            if self.close_start_wall is None:
                self.close_start_wall = time.time()
            if (time.time() - self.close_start_wall) >= self.gripper_close_hold_sec:
                self._set_state(GraspState.LIFT)
        elif self.state == GraspState.LIFT:
            self._set_state(GraspState.DONE)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = TagGraspController()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
