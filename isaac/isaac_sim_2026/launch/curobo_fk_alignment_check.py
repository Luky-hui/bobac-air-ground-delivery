#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Read-only FK alignment check between Isaac Sim TF and cuRobo/URDF FK.

This script does not publish any commands. It reads:
- /joint_states for the current right-arm joint values
- /tf for Isaac Sim link transforms

Then it computes FK from the latest TuringStack_X1_2.urdf through cuRobo and
writes a per-link error table.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformException, TransformListener

CUROBO_SRC = "/home/u-zhuang/ros2_ws/src/isaac/isaac_sim_2026/curobo/src"

RIGHT_ARM_JOINTS = ["joint1_R", "joint2_R", "joint3_R", "joint4_R", "joint5_R", "joint6_R"]

CHECK_LINKS = [
    "link1_R",
    "link2_R",
    "link3_R",
    "link4_R",
    "link5_R",
    "link6_R",
    "link7_R",
    "gripper01_base",
    "right_tcp",
]


def quat_xyzw_to_wxyz(q_xyzw: np.ndarray) -> np.ndarray:
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64)


def normalize_quat(q: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        raise ValueError("zero quaternion")
    return q / norm


def quat_angle_error_rad(a_wxyz: np.ndarray, b_wxyz: np.ndarray) -> float:
    a = normalize_quat(a_wxyz.astype(np.float64))
    b = normalize_quat(b_wxyz.astype(np.float64))
    dot = abs(float(np.dot(a, b)))
    dot = min(1.0, max(-1.0, dot))
    return 2.0 * math.acos(dot)


def transform_to_pose_wxyz(t) -> Tuple[np.ndarray, np.ndarray]:
    pos = np.array(
        [t.transform.translation.x, t.transform.translation.y, t.transform.translation.z],
        dtype=np.float64,
    )
    quat_xyzw = np.array(
        [
            t.transform.rotation.x,
            t.transform.rotation.y,
            t.transform.rotation.z,
            t.transform.rotation.w,
        ],
        dtype=np.float64,
    )
    return pos, quat_xyzw_to_wxyz(quat_xyzw)


class FKAlignmentReader(Node):
    def __init__(self, base_frame: str, check_links: List[str]):
        super().__init__("curobo_fk_alignment_check")
        self.base_frame = base_frame
        self.check_links = check_links
        self.joint_state: Dict[str, float] = {}
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)
        self.create_subscription(JointState, "/joint_states", self._joint_cb, 10)

    def _joint_cb(self, msg: JointState):
        self.joint_state = {name: float(pos) for name, pos in zip(msg.name, msg.position)}

    def wait_for_data(self, timeout_s: float) -> None:
        deadline = time.time() + timeout_s
        missing_tf = list(self.check_links)
        while rclpy.ok() and time.time() < deadline:
            time.sleep(0.05)
            have_joints = all(j in self.joint_state for j in RIGHT_ARM_JOINTS)
            if have_joints:
                missing_tf = []
                for link in sorted(set(self.check_links)):
                    try:
                        self.tf_buffer.lookup_transform(
                            self.base_frame,
                            link,
                            rclpy.time.Time(),
                            timeout=Duration(seconds=0.01),
                        )
                    except TransformException:
                        missing_tf.append(link)
                if not missing_tf:
                    return
        if not self.joint_state:
            raise RuntimeError("Timed out waiting for /joint_states")
        missing_joints = [j for j in RIGHT_ARM_JOINTS if j not in self.joint_state]
        if missing_joints:
            raise RuntimeError(f"/joint_states is missing joints: {missing_joints}")
        if missing_tf:
            raise RuntimeError(f"Timed out waiting for TF {self.base_frame} -> {missing_tf}")

    def get_isaac_poses(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for link in self.check_links:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame,
                link,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.2),
            )
            out[link] = transform_to_pose_wxyz(tf)
        return out


def compute_curobo_fk(args: argparse.Namespace, joint_state: Dict[str, float]) -> Tuple[List[str], Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    import torch

    if CUROBO_SRC not in os.sys.path:
        os.sys.path.insert(0, CUROBO_SRC)

    from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelConfig
    from curobo.types.base import TensorDeviceType
    from curobo.util_file import load_yaml

    tensor_args = TensorDeviceType()
    robot_yaml = load_yaml(args.robot_cfg_path)
    robot_cfg = robot_yaml["robot_cfg"]
    kin = robot_cfg["kinematics"]
    kin["urdf_path"] = args.urdf_path
    kin["asset_root_path"] = os.path.dirname(args.urdf_path)
    kin["base_link"] = args.base_frame
    kin["ee_link"] = args.ee_link
    kin["link_names"] = list(args.check_links)
    kin.setdefault("lock_joints", {})
    if args.lock_body_to_current:
        kin["lock_joints"]["body"] = float(joint_state.get("body", 0.0))

    model_cfg = CudaRobotModelConfig.from_data_dict(robot_cfg, tensor_args=tensor_args)
    model = CudaRobotModel(model_cfg)
    q_np = np.array([joint_state[j] for j in RIGHT_ARM_JOINTS], dtype=np.float32)
    q = tensor_args.to_device(q_np).view(1, -1)
    state = model.get_state(q)
    torch.cuda.synchronize()

    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    link_names = list(state.link_names)
    pos = state.links_position.detach().cpu().numpy()[0]
    quat = state.links_quaternion.detach().cpu().numpy()[0]
    for i, name in enumerate(link_names):
        out[name] = (pos[i].astype(np.float64), quat[i].astype(np.float64))
    return model.joint_names, out


def write_outputs(args: argparse.Namespace, joint_state: Dict[str, float], rows: List[dict]) -> Tuple[str, str]:
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "fk_alignment_check.csv")
    txt_path = os.path.join(args.out_dir, "fk_alignment_check.txt")

    fieldnames = [
        "link",
        "isaac_x",
        "isaac_y",
        "isaac_z",
        "curobo_x",
        "curobo_y",
        "curobo_z",
        "position_error_m",
        "rotation_error_deg",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("FK alignment check: Isaac Sim TF vs cuRobo/URDF FK\n")
        f.write(f"urdf_path: {args.urdf_path}\n")
        f.write(f"robot_cfg_path: {args.robot_cfg_path}\n")
        f.write(f"base_frame: {args.base_frame}\n")
        f.write(f"ee_link: {args.ee_link}\n")
        f.write(f"right_arm_joint_state: {{")
        f.write(", ".join(f"{j}: {joint_state[j]:.6f}" for j in RIGHT_ARM_JOINTS))
        f.write("}\n")
        f.write(f"body_lock_used: {joint_state.get('body', 0.0):.6f}\n\n")
        f.write(
            f"{'link':<16} {'isaac xyz':>31} {'curobo xyz':>31} "
            f"{'pos_err(m)':>11} {'rot_err(deg)':>13}\n"
        )
        for row in rows:
            f.write(
                f"{row['link']:<16} "
                f"({row['isaac_x']:+.4f},{row['isaac_y']:+.4f},{row['isaac_z']:+.4f}) "
                f"({row['curobo_x']:+.4f},{row['curobo_y']:+.4f},{row['curobo_z']:+.4f}) "
                f"{row['position_error_m']:>11.4f} {row['rotation_error_deg']:>13.2f}\n"
            )
    return csv_path, txt_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--urdf-path",
        default="/home/u-zhuang/ros2_ws/src/isaac/isaac_sim_2026/config/TuringStack_X1_2.urdf",
    )
    parser.add_argument(
        "--robot-cfg-path",
        default="/home/u-zhuang/ros2_ws/src/isaac/isaac_sim_2026/config/turingstack_x1_2_right_tcp.yml",
    )
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--ee-link", default="right_tcp")
    parser.add_argument("--out-dir", default="/home/u-zhuang/ros2_ws/src/isaac/isaac_sim_2026/alignment_results")
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--lock-body-to-current", action="store_true", default=True)
    args = parser.parse_args()
    args.check_links = CHECK_LINKS

    rclpy.init()
    node = FKAlignmentReader(args.base_frame, CHECK_LINKS)
    try:
        node.wait_for_data(args.timeout)
        joint_state = dict(node.joint_state)
        isaac_poses = node.get_isaac_poses()
    finally:
        if hasattr(node.tf_listener, "executor"):
            node.tf_listener.executor.shutdown()
        if hasattr(node.tf_listener, "dedicated_listener_thread"):
            node.tf_listener.dedicated_listener_thread.join(timeout=1.0)
        node.tf_listener.unregister()
        node.destroy_node()
        rclpy.shutdown()

    joint_names, curobo_poses = compute_curobo_fk(args, joint_state)
    missing = [link for link in CHECK_LINKS if link not in curobo_poses]
    if missing:
        raise RuntimeError(f"cuRobo did not return link poses for: {missing}; returned={list(curobo_poses)}")

    rows: List[dict] = []
    for link in CHECK_LINKS:
        isaac_pos, isaac_quat = isaac_poses[link]
        curobo_pos, curobo_quat = curobo_poses[link]
        pos_err = float(np.linalg.norm(isaac_pos - curobo_pos))
        rot_err = math.degrees(quat_angle_error_rad(isaac_quat, curobo_quat))
        rows.append(
            {
                "link": link,
                "isaac_x": float(isaac_pos[0]),
                "isaac_y": float(isaac_pos[1]),
                "isaac_z": float(isaac_pos[2]),
                "curobo_x": float(curobo_pos[0]),
                "curobo_y": float(curobo_pos[1]),
                "curobo_z": float(curobo_pos[2]),
                "position_error_m": pos_err,
                "rotation_error_deg": rot_err,
            }
        )

    csv_path, txt_path = write_outputs(args, joint_state, rows)
    print(f"cuRobo joint_names: {joint_names}")
    print(f"wrote: {txt_path}")
    print(f"wrote: {csv_path}")
    print(open(txt_path, encoding="utf-8").read())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
