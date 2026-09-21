#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Move the right arm to zero and measure Isaac Sim parent->child origins.

This is intentionally a diagnostic/URDF calibration script, not a planner:
- It publishes only slow joint commands.
- It keeps the left arm and grippers at their current positions.
- It measures the right-arm chain after the right arm settles near zero.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformException, TransformListener


R_JOINTS = ["joint1_R", "joint2_R", "joint3_R", "joint4_R", "joint5_R", "joint6_R"]
L_JOINTS = ["joint1_L", "joint2_L", "joint3_L", "joint4_L", "joint5_L", "joint6_L"]
GRIP_R = "gripper_controller"
GRIP_L = "gripper_left_controller"

CHAIN = [
    ("body", "base_link", "link_body"),
    ("joint1_R", "link_body", "link1_R"),
    ("joint2_R", "link1_R", "link2_R"),
    ("FixedJointR", "link2_R", "link3_R"),
    ("joint3_R", "link3_R", "link4_R"),
    ("joint4_R", "link4_R", "link5_R"),
    ("joint5_R", "link5_R", "link6_R"),
    ("joint6_R", "link6_R", "link7_R"),
    ("link7_R_to_right_tcp", "link7_R", "right_tcp"),
]

RIGHT_TCP_LEFT_FRAME = "gripper01_left1"
RIGHT_TCP_RIGHT_FRAME = "gripper01_right1"
RIGHT_TCP_ORIENT_FRAME = "gripper01_base"

JOINT_LIMITS = {
    "joint1_R": (-2.87979326579, 2.87979326579),
    "joint2_R": (-0.9599310886, 1.6580627894),
    "joint3_R": (-3.0194196060, 0.0872664626),
    "joint4_R": (-2.8797932658, 2.8797932658),
    "joint5_R": (-0.3490658504, 4.6251225178),
    "joint6_R": (-3.1415926536, 3.1415926536),
}


def tf_to_pos_rpy(tf) -> Tuple[np.ndarray, np.ndarray]:
    pos = np.array(
        [tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z],
        dtype=np.float64,
    )
    quat_xyzw = np.array(
        [
            tf.transform.rotation.x,
            tf.transform.rotation.y,
            tf.transform.rotation.z,
            tf.transform.rotation.w,
        ],
        dtype=np.float64,
    )
    rpy = Rotation.from_quat(quat_xyzw).as_euler("xyz", degrees=False)
    return pos, rpy


def parse_xyz_rpy(raw: Optional[str]) -> List[float]:
    if not raw:
        return [0.0, 0.0, 0.0]
    return [float(x) for x in raw.split()]


def load_urdf_origins(urdf_path: str) -> Dict[str, Tuple[List[float], List[float], List[float]]]:
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    out: Dict[str, Tuple[List[float], List[float], List[float]]] = {}
    for joint in root.findall("joint"):
        name = joint.attrib.get("name", "")
        origin = joint.find("origin")
        axis = joint.find("axis")
        xyz = parse_xyz_rpy(origin.attrib.get("xyz") if origin is not None else None)
        rpy = parse_xyz_rpy(origin.attrib.get("rpy") if origin is not None else None)
        axis_xyz = parse_xyz_rpy(axis.attrib.get("xyz") if axis is not None else None)
        out[name] = (xyz, rpy, axis_xyz)
    return out


class ZeroOriginMeasure(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("measure_right_arm_zero_origins")
        self.args = args
        self.pose: Dict[str, float] = {}
        self.left_pose: Dict[str, float] = {}
        self.prev_pose: Dict[str, float] = {}
        self.prev_time: Optional[float] = None
        self.max_observed_step = 0.0
        self.max_observed_speed = 0.0
        self.abort_reason: Optional[str] = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)
        self.pub = self.create_publisher(JointState, "/joint_command", 10)
        self.left_pub = self.create_publisher(JointState, "/joint_left_command", 10)
        self.create_subscription(JointState, "/joint_states", self._joint_cb, 10)
        self.create_subscription(JointState, "/joint_left_states", self._left_joint_cb, 10)

    def _joint_cb(self, msg: JointState):
        now = time.time()
        new_pose = {name: float(pos) for name, pos in zip(msg.name, msg.position)}
        if self.pose and self.prev_time is not None:
            dt = max(now - self.prev_time, 1e-3)
            for joint in R_JOINTS:
                if joint in self.pose and joint in new_pose:
                    step = abs(new_pose[joint] - self.pose[joint])
                    speed = step / dt
                    self.max_observed_step = max(self.max_observed_step, step)
                    self.max_observed_speed = max(self.max_observed_speed, speed)
                    if step > self.args.abort_observed_step:
                        self.abort_reason = f"{joint} observed jump {step:.3f} rad"
                    if speed > self.args.abort_observed_speed:
                        self.abort_reason = f"{joint} observed speed {speed:.3f} rad/s"
        self.prev_pose = self.pose
        self.pose = new_pose
        self.prev_time = now

    def _left_joint_cb(self, msg: JointState):
        self.left_pose = {name: float(pos) for name, pos in zip(msg.name, msg.position)}

    def wait_ready(self, timeout_s: float) -> None:
        tf_children = [child for _, _, child in CHAIN if child != "right_tcp"]
        tf_children.extend([RIGHT_TCP_LEFT_FRAME, RIGHT_TCP_RIGHT_FRAME, RIGHT_TCP_ORIENT_FRAME])
        deadline = time.time() + timeout_s
        missing_tf = list(tf_children)
        while time.time() < deadline and rclpy.ok():
            have_joints = all(j in self.pose for j in R_JOINTS + L_JOINTS + [GRIP_R])
            missing_tf = []
            for child in sorted(set(tf_children)):
                parent = "link7_R" if child in [RIGHT_TCP_LEFT_FRAME, RIGHT_TCP_RIGHT_FRAME, RIGHT_TCP_ORIENT_FRAME] else "base_link"
                try:
                    self.tf_buffer.lookup_transform(
                        parent,
                        child,
                        rclpy.time.Time(),
                        timeout=Duration(seconds=0.01),
                    )
                except TransformException:
                    missing_tf.append(f"{parent}->{child}")
            if have_joints and not missing_tf:
                return
            time.sleep(0.05)
        missing_joints = [j for j in R_JOINTS + L_JOINTS + [GRIP_R] if j not in self.pose]
        raise RuntimeError(f"not ready; missing_joints={missing_joints}, missing_tf={missing_tf}")

    def current_q(self) -> List[float]:
        return [float(self.pose[j]) for j in R_JOINTS]

    def _publish(self, q_right: List[float]) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = R_JOINTS + L_JOINTS + [GRIP_R]
        msg.position = (
            [float(v) for v in q_right]
            + [float(self.pose.get(j, 0.0)) for j in L_JOINTS]
            + [float(self.pose.get(GRIP_R, 0.0))]
        )
        self.pub.publish(msg)

        if GRIP_L in self.left_pose:
            left = JointState()
            left.header.stamp = msg.header.stamp
            left.name = [GRIP_L]
            left.position = [float(self.left_pose.get(GRIP_L, 0.0))]
            self.left_pub.publish(left)

    def move_right_to(self, target: List[float], label: str) -> bool:
        for name, value in zip(R_JOINTS, target):
            lo, hi = JOINT_LIMITS[name]
            if not (lo + self.args.limit_margin <= value <= hi - self.args.limit_margin):
                self.get_logger().error(f"{label}: target {name}={value:.3f} outside safe limits")
                return False
        start = self.current_q()
        max_delta = max(abs(a - b) for a, b in zip(start, target))
        steps = max(1, int(math.ceil(max_delta / self.args.max_joint_step)))
        self.get_logger().info(f"{label}: steps={steps}, max_delta={max_delta:.4f} rad")
        for i in range(steps + 1):
            if self.abort_reason:
                self.get_logger().error(f"ABORT: {self.abort_reason}")
                return False
            alpha = i / steps
            cmd = [(1.0 - alpha) * s + alpha * t for s, t in zip(start, target)]
            self._publish(cmd)
            time.sleep(self.args.dt)
        time.sleep(self.args.settle)
        return True

    def lookup_parent_child(self, parent: str, child: str) -> Tuple[np.ndarray, np.ndarray]:
        tf = self.tf_buffer.lookup_transform(
            parent,
            child,
            rclpy.time.Time(),
            timeout=Duration(seconds=0.2),
        )
        return tf_to_pos_rpy(tf)

    def lookup_right_tcp(self) -> Tuple[np.ndarray, np.ndarray]:
        left_pos, _ = self.lookup_parent_child("link7_R", RIGHT_TCP_LEFT_FRAME)
        right_pos, _ = self.lookup_parent_child("link7_R", RIGHT_TCP_RIGHT_FRAME)
        _, rpy = self.lookup_parent_child("link7_R", RIGHT_TCP_ORIENT_FRAME)
        return (left_pos + right_pos) * 0.5, rpy

    def close(self) -> None:
        if hasattr(self.tf_listener, "executor"):
            self.tf_listener.executor.shutdown()
        if hasattr(self.tf_listener, "dedicated_listener_thread"):
            self.tf_listener.dedicated_listener_thread.join(timeout=1.0)
        self.tf_listener.unregister()


def write_outputs(args: argparse.Namespace, rows: List[dict], final_q: List[float], max_speed: float, max_step: float) -> Tuple[str, str]:
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "right_arm_zero_origin_measurement.csv")
    txt_path = os.path.join(args.out_dir, "right_arm_zero_origin_measurement.txt")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Right arm zero-origin measurement from Isaac Sim TF\n")
        f.write(f"urdf_path: {args.urdf_path}\n")
        f.write(f"final_right_arm_q: {[round(x, 6) for x in final_q]}\n")
        f.write(f"max_observed_speed_rad_s: {max_speed:.6f}\n")
        f.write(f"max_observed_step_rad: {max_step:.6f}\n\n")
        f.write(
            f"{'joint':<22} {'parent->child':<23} {'isaac xyz':>31} {'isaac rpy':>31} "
            f"{'urdf xyz':>31} {'urdf rpy':>31} {'dxyz(m)':>9} {'drpy(deg)':>11}\n"
        )
        for r in rows:
            f.write(
                f"{r['joint']:<22} {r['parent']+'->'+r['child']:<23} "
                f"({r['isaac_x']:+.6f},{r['isaac_y']:+.6f},{r['isaac_z']:+.6f}) "
                f"({r['isaac_roll']:+.6f},{r['isaac_pitch']:+.6f},{r['isaac_yaw']:+.6f}) "
                f"({r['urdf_x']:+.6f},{r['urdf_y']:+.6f},{r['urdf_z']:+.6f}) "
                f"({r['urdf_roll']:+.6f},{r['urdf_pitch']:+.6f},{r['urdf_yaw']:+.6f}) "
                f"{r['origin_xyz_error_m']:>9.5f} {r['origin_rpy_error_deg']:>11.3f}\n"
            )
        f.write("\nSuggested URDF origins:\n")
        for r in rows:
            f.write(
                f"{r['joint']}: xyz=\"{r['isaac_x']:.9g} {r['isaac_y']:.9g} {r['isaac_z']:.9g}\" "
                f"rpy=\"{r['isaac_roll']:.9g} {r['isaac_pitch']:.9g} {r['isaac_yaw']:.9g}\"\n"
            )
    return csv_path, txt_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf-path", default="/home/u-zhuang/ros2_ws/src/isaac/isaac_sim_2026/config/TuringStack_X1_2.urdf")
    parser.add_argument("--out-dir", default="/home/u-zhuang/ros2_ws/src/isaac/isaac_sim_2026/alignment_results")
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--max-joint-step", type=float, default=0.002)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--settle", type=float, default=1.0)
    parser.add_argument("--limit-margin", type=float, default=0.05)
    parser.add_argument("--abort-observed-step", type=float, default=0.12)
    parser.add_argument("--abort-observed-speed", type=float, default=2.0)
    parser.add_argument("--restore", action="store_true")
    args = parser.parse_args()

    rclpy.init()
    node = ZeroOriginMeasure(args)
    rows: List[dict] = []
    try:
        node.wait_ready(args.timeout)
        start_q = node.current_q()
        node.get_logger().info(f"start_q={[round(x, 4) for x in start_q]}")
        target_q = [0.0] * len(R_JOINTS)
        if not node.move_right_to(target_q, "move right arm to zero"):
            return 2

        urdf_origins = load_urdf_origins(args.urdf_path)
        for joint, parent, child in CHAIN:
            if child == "right_tcp":
                isaac_xyz, isaac_rpy = node.lookup_right_tcp()
            else:
                isaac_xyz, isaac_rpy = node.lookup_parent_child(parent, child)
            urdf_xyz, urdf_rpy, urdf_axis = urdf_origins.get(joint, ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]))
            xyz_error = float(np.linalg.norm(isaac_xyz - np.array(urdf_xyz, dtype=np.float64)))
            rpy_error = float(np.linalg.norm(np.degrees(isaac_rpy - np.array(urdf_rpy, dtype=np.float64))))
            rows.append(
                {
                    "joint": joint,
                    "parent": parent,
                    "child": child,
                    "isaac_x": float(isaac_xyz[0]),
                    "isaac_y": float(isaac_xyz[1]),
                    "isaac_z": float(isaac_xyz[2]),
                    "isaac_roll": float(isaac_rpy[0]),
                    "isaac_pitch": float(isaac_rpy[1]),
                    "isaac_yaw": float(isaac_rpy[2]),
                    "urdf_x": float(urdf_xyz[0]),
                    "urdf_y": float(urdf_xyz[1]),
                    "urdf_z": float(urdf_xyz[2]),
                    "urdf_roll": float(urdf_rpy[0]),
                    "urdf_pitch": float(urdf_rpy[1]),
                    "urdf_yaw": float(urdf_rpy[2]),
                    "urdf_axis_x": float(urdf_axis[0]),
                    "urdf_axis_y": float(urdf_axis[1]),
                    "urdf_axis_z": float(urdf_axis[2]),
                    "origin_xyz_error_m": xyz_error,
                    "origin_rpy_error_deg": rpy_error,
                }
            )

        csv_path, txt_path = write_outputs(
            args,
            rows,
            node.current_q(),
            node.max_observed_speed,
            node.max_observed_step,
        )
        print(f"wrote: {txt_path}")
        print(f"wrote: {csv_path}")
        print(open(txt_path, encoding="utf-8").read())

        if args.restore:
            node.move_right_to(start_q, "restore right arm start pose")
        if node.abort_reason:
            print(f"abort_reason={node.abort_reason}")
            return 2
        return 0
    finally:
        try:
            if node.pose:
                node._publish(node.current_q())
        finally:
            node.close()
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
