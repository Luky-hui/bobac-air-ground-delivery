#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Slow right-arm single-joint axis probe.

For each right-arm joint, this script:
- records Isaac Sim TF before motion
- moves only that joint slowly
- records Isaac Sim TF after motion
- computes the observed rotation axis from TF
- computes the URDF/cuRobo predicted rotation axis for the same observed joint delta
- returns the joint to the pre-test pose

It publishes /joint_command, so run only with the simulator visible and ready.
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
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformException, TransformListener


R_JOINTS = ["joint1_R", "joint2_R", "joint3_R", "joint4_R", "joint5_R", "joint6_R"]
L_JOINTS = ["joint1_L", "joint2_L", "joint3_L", "joint4_L", "joint5_L", "joint6_L"]
GRIP_R = "gripper_controller"
GRIP_L = "gripper_left_controller"

JOINT_CHILD = {
    "joint1_R": "link1_R",
    "joint2_R": "link2_R",
    "joint3_R": "link4_R",
    "joint4_R": "link5_R",
    "joint5_R": "link6_R",
    "joint6_R": "link7_R",
}

JOINT_PARENT = {
    "joint1_R": "link_body",
    "joint2_R": "link1_R",
    "joint3_R": "link3_R",
    "joint4_R": "link4_R",
    "joint5_R": "link5_R",
    "joint6_R": "link6_R",
}

JOINT_LIMITS = {
    "joint1_R": (-2.87979326579, 2.87979326579),
    "joint2_R": (-0.9599310886, 1.6580627894),
    "joint3_R": (-3.0194196060, 0.0872664626),
    "joint4_R": (-2.8797932658, 2.8797932658),
    "joint5_R": (-0.3490658504, 4.6251225178),
    "joint6_R": (-3.1415926536, 3.1415926536),
}


def quat_xyzw_to_matrix(q_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = q_xyzw.astype(np.float64)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def quat_wxyz_to_matrix(q_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = q_wxyz.astype(np.float64)
    return quat_xyzw_to_matrix(np.array([x, y, z, w], dtype=np.float64))


def rotation_axis_angle(rot: np.ndarray) -> Tuple[np.ndarray, float]:
    cos_angle = (float(np.trace(rot)) - 1.0) * 0.5
    cos_angle = min(1.0, max(-1.0, cos_angle))
    angle = math.acos(cos_angle)
    if angle < 1e-7:
        return np.array([float("nan"), float("nan"), float("nan")]), 0.0
    axis = np.array(
        [
            rot[2, 1] - rot[1, 2],
            rot[0, 2] - rot[2, 0],
            rot[1, 0] - rot[0, 1],
        ],
        dtype=np.float64,
    )
    axis = axis / (2.0 * math.sin(angle))
    norm = np.linalg.norm(axis)
    if norm > 1e-9:
        axis = axis / norm
    return axis, angle


def axis_error_deg(actual_axis: np.ndarray, urdf_axis: np.ndarray) -> float:
    dot = float(np.dot(actual_axis, urdf_axis))
    dot = min(1.0, max(-1.0, dot))
    return math.degrees(math.acos(dot))


def rpy_to_matrix(rpy: List[float]) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def parse_urdf_joint_data(urdf_path: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for joint in root.findall("joint"):
        name = joint.attrib.get("name", "")
        origin = joint.find("origin")
        axis = joint.find("axis")
        rpy = [0.0, 0.0, 0.0]
        axis_xyz = [0.0, 0.0, 0.0]
        if origin is not None and origin.attrib.get("rpy"):
            rpy = [float(v) for v in origin.attrib["rpy"].split()]
        if axis is not None and axis.attrib.get("xyz"):
            axis_xyz = [float(v) for v in axis.attrib["xyz"].split()]
        out[name] = (rpy_to_matrix(rpy), np.array(axis_xyz, dtype=np.float64))
    return out


def suggested_coordinate_axis(axis_local: np.ndarray) -> Tuple[np.ndarray, float]:
    idx = int(np.argmax(np.abs(axis_local)))
    sign = 1.0 if axis_local[idx] >= 0.0 else -1.0
    suggestion = np.zeros(3, dtype=np.float64)
    suggestion[idx] = sign
    return suggestion, axis_error_deg(axis_local, suggestion)


def axis_to_string(axis: np.ndarray) -> str:
    vals = []
    for v in axis:
        if abs(v) < 0.5:
            vals.append("0")
        elif v > 0:
            vals.append("1")
        else:
            vals.append("-1")
    return " ".join(vals)


def tf_to_rot_xyzw(t) -> np.ndarray:
    return np.array(
        [
            t.transform.rotation.x,
            t.transform.rotation.y,
            t.transform.rotation.z,
            t.transform.rotation.w,
        ],
        dtype=np.float64,
    )


class AxisProbe(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("right_arm_joint_axis_probe")
        self.args = args
        self.pose: Dict[str, float] = {}
        self.left_grip: Dict[str, float] = {}
        self.prev_pose: Dict[str, float] = {}
        self.prev_time: Optional[float] = None
        self.abort_reason: Optional[str] = None
        self.max_observed_step = 0.0
        self.max_observed_speed = 0.0

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
        self.left_grip = {name: float(pos) for name, pos in zip(msg.name, msg.position)}

    def wait_ready(self, timeout_s: float) -> None:
        deadline = time.time() + timeout_s
        links = list(JOINT_CHILD.values()) + list(JOINT_PARENT.values())
        missing_tf = links
        while time.time() < deadline and rclpy.ok():
            have_joints = all(j in self.pose for j in R_JOINTS + L_JOINTS + [GRIP_R])
            missing_tf = []
            for link in links:
                try:
                    self.tf_buffer.lookup_transform(
                        self.args.base_frame,
                        link,
                        rclpy.time.Time(),
                        timeout=Duration(seconds=0.01),
                    )
                except TransformException:
                    missing_tf.append(link)
            if have_joints and not missing_tf:
                return
            time.sleep(0.05)
        missing_joints = [j for j in R_JOINTS + L_JOINTS + [GRIP_R] if j not in self.pose]
        raise RuntimeError(f"not ready; missing_joints={missing_joints}, missing_tf={missing_tf}")

    def current_q(self) -> List[float]:
        return [float(self.pose[j]) for j in R_JOINTS]

    def _publish(self, q_right: List[float]) -> None:
        names = []
        pos = []
        for name, value in zip(R_JOINTS, q_right):
            names.append(name)
            pos.append(float(value))
        for name in L_JOINTS:
            names.append(name)
            pos.append(float(self.pose.get(name, 0.0)))
        names.append(GRIP_R)
        pos.append(float(self.pose.get(GRIP_R, 0.0)))

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = names
        msg.position = pos
        self.pub.publish(msg)

        if self.left_grip or self.args.publish_left_gripper_hold:
            left_msg = JointState()
            left_msg.header.stamp = self.get_clock().now().to_msg()
            left_msg.name = [GRIP_L]
            left_msg.position = [float(self.left_grip.get(GRIP_L, 0.0))]
            self.left_pub.publish(left_msg)

    def move_to(self, target: List[float], label: str) -> bool:
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

    def lookup_rot(self, link: str) -> np.ndarray:
        tf = self.tf_buffer.lookup_transform(
            self.args.base_frame,
            link,
            rclpy.time.Time(),
            timeout=Duration(seconds=0.2),
        )
        return quat_xyzw_to_matrix(tf_to_rot_xyzw(tf))

    def close(self) -> None:
        if hasattr(self.tf_listener, "executor"):
            self.tf_listener.executor.shutdown()
        if hasattr(self.tf_listener, "dedicated_listener_thread"):
            self.tf_listener.dedicated_listener_thread.join(timeout=1.0)
        self.tf_listener.unregister()


class CuroboFK:
    def __init__(self, args: argparse.Namespace, joint_state: Dict[str, float]):
        import torch

        from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelConfig
        from curobo.types.base import TensorDeviceType
        from curobo.util_file import load_yaml

        self.torch = torch
        self.tensor_args = TensorDeviceType()
        robot_cfg = load_yaml(args.robot_cfg_path)["robot_cfg"]
        kin = robot_cfg["kinematics"]
        kin["urdf_path"] = args.urdf_path
        kin["asset_root_path"] = os.path.dirname(args.urdf_path)
        kin["base_link"] = args.base_frame
        kin["ee_link"] = "gripper01_base"
        kin["link_names"] = list(JOINT_CHILD.values())
        kin.setdefault("lock_joints", {})
        kin["lock_joints"]["body"] = float(joint_state.get("body", 0.0))
        model_cfg = CudaRobotModelConfig.from_data_dict(robot_cfg, tensor_args=self.tensor_args)
        self.model = CudaRobotModel(model_cfg)
        self.link_names = list(self.model.link_names)

    def rotations(self, q_right: List[float]) -> Dict[str, np.ndarray]:
        q = self.tensor_args.to_device(np.array(q_right, dtype=np.float32)).view(1, -1)
        state = self.model.get_state(q)
        self.torch.cuda.synchronize()
        quat = state.links_quaternion.detach().cpu().numpy()[0]
        return {name: quat_wxyz_to_matrix(quat[i]) for i, name in enumerate(state.link_names)}


def choose_delta(joint: str, q0: float, desired: float, margin: float, min_delta: float) -> float:
    lo, hi = JOINT_LIMITS[joint]
    if lo + margin <= q0 + desired <= hi - margin:
        return desired
    available_positive = hi - margin - q0
    if desired >= 0.0 and available_positive >= min_delta:
        return available_positive
    if q0 - abs(desired) >= lo + margin:
        return -abs(desired)
    available_negative = q0 - (lo + margin)
    if available_negative >= min_delta:
        return -available_negative
    raise RuntimeError(f"{joint} has no safe delta from q={q0:.4f} within limits [{lo:.4f}, {hi:.4f}]")


def write_outputs(args: argparse.Namespace, rows: List[dict]) -> Tuple[str, str]:
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "right_arm_axis_probe.csv")
    txt_path = os.path.join(args.out_dir, "right_arm_axis_probe.txt")
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Right arm single-joint axis probe\n")
        f.write(f"urdf_path: {args.urdf_path}\n")
        f.write(f"robot_cfg_path: {args.robot_cfg_path}\n")
        f.write(f"delta_requested_rad: {args.delta}\n")
        f.write(f"max_joint_step_rad: {args.max_joint_step}\n")
        f.write(f"dt_s: {args.dt}\n\n")
        f.write(
            f"{'joint':<10} {'child':<8} {'dq_obs':>8} {'tf_angle':>9} "
            f"{'urdf_angle':>10} {'axis_err':>9} {'actual_axis':>30} {'urdf_axis':>30} "
            f"{'local_axis':>30} {'suggest':>9} {'sugg_err':>9}\n"
        )
        for r in rows:
            f.write(
                f"{r['joint']:<10} {r['child']:<8} {r['observed_delta_rad']:>8.4f} "
                f"{r['isaac_rotation_angle_rad']:>9.4f} {r['curobo_rotation_angle_rad']:>10.4f} "
                f"{r['axis_error_deg']:>9.2f} "
                f"({r['isaac_axis_x']:+.3f},{r['isaac_axis_y']:+.3f},{r['isaac_axis_z']:+.3f}) "
                f"({r['curobo_axis_x']:+.3f},{r['curobo_axis_y']:+.3f},{r['curobo_axis_z']:+.3f}) "
                f"({r['axis_local_x']:+.3f},{r['axis_local_y']:+.3f},{r['axis_local_z']:+.3f}) "
                f"{r['suggested_urdf_axis']:>9} {r['suggested_axis_error_deg']:>9.2f}\n"
            )
    return csv_path, txt_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--urdf-path", default="/home/u-zhuang/ros2_ws/src/isaac/isaac_sim_2026/config/TuringStack_X1_2.urdf")
    parser.add_argument(
        "--robot-cfg-path",
        default="/home/u-zhuang/ros2_ws/src/isaac/isaac_sim_2026/config/turingstack_x1_2_right_tcp.yml",
    )
    parser.add_argument("--out-dir", default="/home/u-zhuang/ros2_ws/src/isaac/isaac_sim_2026/alignment_results")
    parser.add_argument("--delta", type=float, default=0.2)
    parser.add_argument("--min-delta", type=float, default=0.04)
    parser.add_argument("--limit-margin", type=float, default=0.05)
    parser.add_argument("--max-joint-step", type=float, default=0.002)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--settle", type=float, default=0.25)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--abort-observed-step", type=float, default=0.12)
    parser.add_argument("--abort-observed-speed", type=float, default=2.0)
    parser.add_argument("--publish-left-gripper-hold", action="store_true")
    args = parser.parse_args()

    rclpy.init()
    node = AxisProbe(args)
    rows: List[dict] = []
    try:
        node.wait_ready(args.timeout)
        fk = CuroboFK(args, node.pose)
        urdf_joint_data = parse_urdf_joint_data(args.urdf_path)
        start_q = node.current_q()
        node.get_logger().info(f"start_q={[round(x, 4) for x in start_q]}")

        for joint in R_JOINTS:
            idx = R_JOINTS.index(joint)
            child = JOINT_CHILD[joint]
            parent = JOINT_PARENT[joint]
            q_before = node.current_q()
            delta = choose_delta(joint, q_before[idx], args.delta, args.limit_margin, args.min_delta)
            q_target = list(q_before)
            q_target[idx] += delta
            node.get_logger().info(f"probe {joint}: command_delta={delta:.4f} rad child={child}")

            isaac_before = node.lookup_rot(child)
            isaac_parent_before = node.lookup_rot(parent)
            curobo_before = fk.rotations(q_before)[child]
            if not node.move_to(q_target, f"{joint} +delta"):
                break
            q_after = node.current_q()
            observed_delta = q_after[idx] - q_before[idx]
            isaac_after = node.lookup_rot(child)
            curobo_after = fk.rotations(q_after)[child]

            isaac_axis, isaac_angle = rotation_axis_angle(isaac_after @ isaac_before.T)
            curobo_axis, curobo_angle = rotation_axis_angle(curobo_after @ curobo_before.T)
            sign = 1.0 if observed_delta >= 0.0 else -1.0
            isaac_axis_pos = isaac_axis * sign
            curobo_axis_pos = curobo_axis * sign
            err = axis_error_deg(isaac_axis_pos, curobo_axis_pos)
            origin_rot, current_axis = urdf_joint_data[joint]
            joint_frame_rot = isaac_parent_before @ origin_rot
            axis_local = joint_frame_rot.T @ isaac_axis_pos
            axis_local = axis_local / max(np.linalg.norm(axis_local), 1e-9)
            suggested_axis, suggested_error = suggested_coordinate_axis(axis_local)

            rows.append(
                {
                    "joint": joint,
                    "parent": parent,
                    "child": child,
                    "command_delta_rad": float(delta),
                    "observed_delta_rad": float(observed_delta),
                    "isaac_rotation_angle_rad": float(isaac_angle),
                    "curobo_rotation_angle_rad": float(curobo_angle),
                    "axis_error_deg": float(err),
                    "isaac_axis_x": float(isaac_axis_pos[0]),
                    "isaac_axis_y": float(isaac_axis_pos[1]),
                    "isaac_axis_z": float(isaac_axis_pos[2]),
                    "curobo_axis_x": float(curobo_axis_pos[0]),
                    "curobo_axis_y": float(curobo_axis_pos[1]),
                    "curobo_axis_z": float(curobo_axis_pos[2]),
                    "axis_local_x": float(axis_local[0]),
                    "axis_local_y": float(axis_local[1]),
                    "axis_local_z": float(axis_local[2]),
                    "current_urdf_axis": " ".join(f"{v:g}" for v in current_axis.tolist()),
                    "suggested_urdf_axis": axis_to_string(suggested_axis),
                    "suggested_axis_error_deg": float(suggested_error),
                }
            )

            if not node.move_to(q_before, f"{joint} return"):
                break

        if rows:
            csv_path, txt_path = write_outputs(args, rows)
            print(f"wrote: {txt_path}")
            print(f"wrote: {csv_path}")
            print(open(txt_path, encoding="utf-8").read())
        print(
            f"max_observed_speed={node.max_observed_speed:.3f} rad/s, "
            f"max_observed_step={node.max_observed_step:.3f} rad"
        )
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
