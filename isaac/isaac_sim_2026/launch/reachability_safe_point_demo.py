#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Safely demonstrate selected right_tcp reachability points in Isaac Sim.

The script moves:
    start_safe_pose -> target_i -> start_safe_pose

for each target row in a CSV. It monitors /joint_states and the actual Isaac TF
right_tcp position, computed as the midpoint of gripper01_left1/right1.
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


R_JOINTS = ["joint1_R", "joint2_R", "joint3_R", "joint4_R", "joint5_R", "joint6_R"]
L_JOINTS = ["joint1_L", "joint2_L", "joint3_L", "joint4_L", "joint5_L", "joint6_L"]
GRIP_R = "gripper_controller"
GRIP_L = "gripper_left_controller"

JOINT_LIMITS = {
    "joint1_R": (-2.87979326579, 2.87979326579),
    "joint2_R": (-0.9599310886, 1.6580627894),
    "joint3_R": (-3.0194196060, 0.0872664626),
    "joint4_R": (-2.8797932658, 2.8797932658),
    "joint5_R": (-0.3490658504, 4.6251225178),
    "joint6_R": (-3.1415926536, 3.1415926536),
}


def tf_translation(tf) -> np.ndarray:
    return np.array(
        [tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z],
        dtype=np.float64,
    )


class SafePointDemo(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("reachability_safe_point_demo")
        self.args = args
        self.pose: Dict[str, float] = {}
        self.left_pose: Dict[str, float] = {}
        self.prev_time: Optional[float] = None
        self.prev_stamp: Optional[float] = None
        self.abort_reason: Optional[str] = None
        self.max_observed_step = 0.0
        self.max_observed_speed = 0.0
        self.speed_spike_count = 0
        self.target_stats: List[dict] = []

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)
        self.pub = self.create_publisher(JointState, "/joint_command", 10)
        self.left_pub = self.create_publisher(JointState, "/joint_left_command", 10)
        self.create_subscription(JointState, "/joint_states", self._joint_cb, 10)
        self.create_subscription(JointState, "/joint_left_states", self._left_joint_cb, 10)

    def _joint_cb(self, msg: JointState):
        now = time.time()
        stamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        if stamp <= 0.0:
            stamp = now
        new_pose = {name: float(pos) for name, pos in zip(msg.name, msg.position)}
        if self.pose and self.prev_stamp is not None:
            dt = stamp - self.prev_stamp
            for joint in R_JOINTS:
                if joint in self.pose and joint in new_pose:
                    step = abs(new_pose[joint] - self.pose[joint])
                    self.max_observed_step = max(self.max_observed_step, step)
                    if step > self.args.abort_observed_step:
                        self.abort_reason = f"{joint} observed step {step:.4f} rad > {self.args.abort_observed_step:.4f}"
                    if dt >= self.args.min_velocity_dt:
                        speed = step / dt
                        self.max_observed_speed = max(self.max_observed_speed, speed)
                        if speed > self.args.abort_observed_speed:
                            self.speed_spike_count += 1
                            if self.speed_spike_count >= self.args.abort_speed_spike_count:
                                self.abort_reason = (
                                    f"{joint} observed speed {speed:.3f} rad/s > "
                                    f"{self.args.abort_observed_speed:.3f} for {self.speed_spike_count} samples"
                                )
                        else:
                            self.speed_spike_count = 0
        self.pose = new_pose
        self.prev_time = now
        self.prev_stamp = stamp

    def _left_joint_cb(self, msg: JointState):
        self.left_pose = {name: float(pos) for name, pos in zip(msg.name, msg.position)}

    def wait_ready(self, timeout_s: float) -> None:
        deadline = time.time() + timeout_s
        missing_tf: List[str] = []
        while time.time() < deadline and rclpy.ok():
            have_joints = all(j in self.pose for j in R_JOINTS + L_JOINTS + [GRIP_R])
            missing_tf = []
            for frame in ["gripper01_left1", "gripper01_right1"]:
                try:
                    self.tf_buffer.lookup_transform(
                        "base_link",
                        frame,
                        rclpy.time.Time(),
                        timeout=Duration(seconds=0.01),
                    )
                except TransformException:
                    missing_tf.append(frame)
            if have_joints and not missing_tf:
                return
            time.sleep(0.05)
        missing_joints = [j for j in R_JOINTS + L_JOINTS + [GRIP_R] if j not in self.pose]
        raise RuntimeError(f"not ready; missing_joints={missing_joints}, missing_tf={missing_tf}")

    def current_q(self) -> List[float]:
        return [float(self.pose[j]) for j in R_JOINTS]

    def right_tcp_position(self) -> np.ndarray:
        left = self.tf_buffer.lookup_transform(
            "base_link",
            "gripper01_left1",
            rclpy.time.Time(),
            timeout=Duration(seconds=0.2),
        )
        right = self.tf_buffer.lookup_transform(
            "base_link",
            "gripper01_right1",
            rclpy.time.Time(),
            timeout=Duration(seconds=0.2),
        )
        return (tf_translation(left) + tf_translation(right)) * 0.5

    def publish_hold(self, q_right: List[float]) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = R_JOINTS + L_JOINTS + [GRIP_R]
        msg.position = (
            [float(v) for v in q_right]
            + [float(self.pose.get(j, 0.0)) for j in L_JOINTS]
            + [float(self.pose.get(GRIP_R, 0.0))]
        )
        self.pub.publish(msg)

        left_msg = JointState()
        left_msg.header.stamp = msg.header.stamp
        left_msg.name = [GRIP_L]
        left_msg.position = [float(self.left_pose.get(GRIP_L, 0.0))]
        self.left_pub.publish(left_msg)

    def validate_q(self, q: List[float], label: str) -> bool:
        for name, value in zip(R_JOINTS, q):
            lo, hi = JOINT_LIMITS[name]
            if value < lo + self.args.joint_limit_margin or value > hi - self.args.joint_limit_margin:
                self.get_logger().error(
                    f"{label}: {name}={value:.4f} outside safe limits "
                    f"[{lo + self.args.joint_limit_margin:.4f}, {hi - self.args.joint_limit_margin:.4f}]"
                )
                return False
        return True

    def move_to(self, q_target: List[float], label: str) -> bool:
        if not self.validate_q(q_target, label):
            return False
        q_start = self.current_q()
        max_delta = max(abs(a - b) for a, b in zip(q_start, q_target))
        steps = max(self.args.min_steps, int(math.ceil(max_delta / self.args.max_joint_step)))
        duration = steps * self.args.dt
        self.get_logger().info(f"move {label}: steps={steps}, duration={duration:.1f}s, max_delta={max_delta:.3f}rad")

        self.speed_spike_count = 0
        for i in range(steps + 1):
            if self.abort_reason:
                self.get_logger().error(f"SOFT STOP before publish: {self.abort_reason}")
                self.publish_hold(self.current_q())
                return False
            alpha = i / steps
            q_cmd = [(1.0 - alpha) * s + alpha * t for s, t in zip(q_start, q_target)]
            self.publish_hold(q_cmd)
            end = time.time() + self.args.dt
            while rclpy.ok() and time.time() < end:
                rclpy.spin_once(self, timeout_sec=0.005)
                if self.abort_reason:
                    self.get_logger().error(f"SOFT STOP while moving {label}: {self.abort_reason}")
                    self.publish_hold(self.current_q())
                    return False

            if i % max(1, steps // 20) == 0:
                observed = self.current_q()
                tracking_err = max(abs(a - b) for a, b in zip(observed, q_cmd))
                if tracking_err > self.args.abort_tracking_error:
                    self.abort_reason = f"tracking error {tracking_err:.3f} rad > {self.args.abort_tracking_error:.3f}"
                    self.publish_hold(observed)
                    return False

        self.get_logger().info(
            f"arrived {label}; max_step={self.max_observed_step:.4f}rad, "
            f"max_speed={self.max_observed_speed:.3f}rad/s"
        )
        return True

    def dwell_and_check_tcp(self, target_xyz: np.ndarray, label: str) -> Tuple[bool, float, np.ndarray]:
        deadline = time.time() + self.args.dwell
        errors: List[float] = []
        last_pos = np.zeros(3, dtype=np.float64)
        while time.time() < deadline and rclpy.ok():
            self.publish_hold(self.current_q())
            rclpy.spin_once(self, timeout_sec=0.02)
            try:
                last_pos = self.right_tcp_position()
                errors.append(float(np.linalg.norm(last_pos - target_xyz)))
            except TransformException as exc:
                self.abort_reason = f"TF lookup failed during dwell {label}: {exc}"
                return False, float("inf"), last_pos
            if self.abort_reason:
                return False, float("inf"), last_pos
        final_error = errors[-1] if errors else float("inf")
        sustained = len([e for e in errors[-min(len(errors), 10):] if e > self.args.abort_tcp_error]) >= min(len(errors), 10)
        if sustained:
            self.abort_reason = f"{label} right_tcp error sustained at {final_error:.3f} m"
            return False, final_error, last_pos
        return True, final_error, last_pos

    def close(self) -> None:
        if hasattr(self.tf_listener, "executor"):
            self.tf_listener.executor.shutdown()
        if hasattr(self.tf_listener, "dedicated_listener_thread"):
            self.tf_listener.dedicated_listener_thread.join(timeout=1.0)
        self.tf_listener.unregister()


def load_targets(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            q = [float(r[f"q_{j}"]) for j in R_JOINTS]
            xyz = np.array([float(r["x"]), float(r["y"]), float(r["z"])], dtype=np.float64)
            rows.append(
                {
                    "category": r.get("category", "target"),
                    "xyz": xyz,
                    "q": q,
                    "orientation": r.get("orientation", ""),
                }
            )
    return rows


def write_report(args: argparse.Namespace, stats: List[dict]) -> str:
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "reachability_safe_point_demo_report.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "category",
            "x",
            "y",
            "z",
            "orientation",
            "final_tcp_error_m",
            "actual_tcp_x",
            "actual_tcp_y",
            "actual_tcp_z",
            "status",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(stats)
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets-csv", default="/home/u-zhuang/ros2_ws/src/isaac/isaac_sim_2026/alignment_results/reachability_demo_targets_5.csv")
    parser.add_argument("--out-dir", default="/home/u-zhuang/ros2_ws/src/isaac/isaac_sim_2026/alignment_results")
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--max-joint-step", type=float, default=0.002)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--min-steps", type=int, default=300)
    parser.add_argument("--dwell", type=float, default=1.0)
    parser.add_argument("--joint-limit-margin", type=float, default=0.05)
    parser.add_argument("--abort-observed-step", type=float, default=0.02)
    parser.add_argument("--abort-observed-speed", type=float, default=0.8)
    parser.add_argument("--abort-speed-spike-count", type=int, default=3)
    parser.add_argument("--min-velocity-dt", type=float, default=0.01)
    parser.add_argument("--abort-tracking-error", type=float, default=0.40)
    parser.add_argument("--abort-tcp-error", type=float, default=0.05)
    args = parser.parse_args()

    targets = load_targets(args.targets_csv)
    if not targets:
        print(f"no targets loaded from {args.targets_csv}")
        return 2

    rclpy.init()
    node = SafePointDemo(args)
    stats: List[dict] = []
    try:
        node.wait_ready(args.timeout)
        start_q = node.current_q()
        start_tcp = node.right_tcp_position()
        node.get_logger().info(f"start_q={[round(x, 4) for x in start_q]}")
        node.get_logger().info(f"start_right_tcp={[round(float(x), 4) for x in start_tcp.tolist()]}")

        for idx, target in enumerate(targets, 1):
            label = f"{idx}/{len(targets)} {target['category']}"
            node.get_logger().info(
                f"target {label}: xyz={[round(float(x), 3) for x in target['xyz'].tolist()]} "
                f"orientation={target['orientation']} q={[round(v, 3) for v in target['q']]}"
            )
            status = "ok"
            if not node.move_to(target["q"], label):
                status = f"abort_move: {node.abort_reason}"
            else:
                ok, tcp_error, actual_tcp = node.dwell_and_check_tcp(target["xyz"], label)
                if not ok:
                    status = f"abort_tcp: {node.abort_reason}"
                stats.append(
                    {
                        "category": target["category"],
                        "x": float(target["xyz"][0]),
                        "y": float(target["xyz"][1]),
                        "z": float(target["xyz"][2]),
                        "orientation": target["orientation"],
                        "final_tcp_error_m": float(tcp_error),
                        "actual_tcp_x": float(actual_tcp[0]),
                        "actual_tcp_y": float(actual_tcp[1]),
                        "actual_tcp_z": float(actual_tcp[2]),
                        "status": status,
                    }
                )
            if status != "ok":
                node.publish_hold(node.current_q())
                break

            if not node.move_to(start_q, f"{label} return_to_start"):
                stats[-1]["status"] = f"abort_return: {node.abort_reason}"
                break
            time.sleep(0.25)

        report = write_report(args, stats)
        node.get_logger().info(f"report written: {report}")
        node.get_logger().info(
            f"demo finished; max_observed_step={node.max_observed_step:.4f}rad, "
            f"max_observed_speed={node.max_observed_speed:.3f}rad/s"
        )
        if node.abort_reason:
            node.get_logger().error(f"abort_reason={node.abort_reason}")
            return 3
        return 0
    finally:
        try:
            if node.pose:
                node.publish_hold(node.current_q())
        finally:
            node.close()
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
