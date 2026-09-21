#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Slowly demonstrate a few cuRobo-reachable right-arm points in Isaac Sim.

Safety posture:
- Publishes only /joint_command as sensor_msgs/JointState.
- Keeps left arm and gripper at their currently observed positions.
- Uses small joint increments and low publish rate.
- Stops immediately if observed joint states jump, exceed limits, or diverge from the command.
"""

import argparse
import csv
import math
import time
from typing import Dict, List, Optional, Tuple

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


R_JOINTS = ["joint1_R", "joint2_R", "joint3_R", "joint4_R", "joint5_R", "joint6_R"]
L_JOINTS = ["joint1_L", "joint2_L", "joint3_L", "joint4_L", "joint5_L", "joint6_L"]
GRIP_R = "gripper_controller"

JOINT_LIMITS = {
    "joint1_R": (-2.879, 2.879),
    "joint2_R": (-0.9599, 1.6581),
    "joint3_R": (-3.0194, 0.0873),
    "joint4_R": (-2.879, 2.879),
    "joint5_R": (-0.349, 4.625),
    "joint6_R": (-3.1416, 3.1416),
}


class ReachabilityDemo(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("curobo_reachability_demo")
        self.args = args
        self.pose: Dict[str, float] = {}
        self.left_pose: Dict[str, float] = {}
        self.prev_pose: Dict[str, float] = {}
        self.prev_time: Optional[float] = None
        self.max_observed_speed = 0.0
        self.max_observed_step = 0.0
        self.abort_reason: Optional[str] = None

        self.pub = self.create_publisher(JointState, "/joint_command", 10)
        self.left_pub = self.create_publisher(JointState, "/joint_left_command", 10)
        self.create_subscription(JointState, "/joint_states", self._joint_cb, 10)
        self.create_subscription(JointState, "/joint_left_states", self._left_joint_cb, 10)

    def _joint_cb(self, msg: JointState):
        now = time.time()
        new_pose = {n: float(p) for n, p in zip(msg.name, msg.position)}
        if self.pose and self.prev_time is not None:
            dt = max(now - self.prev_time, 1e-3)
            for j in R_JOINTS:
                if j in self.pose and j in new_pose:
                    step = abs(new_pose[j] - self.pose[j])
                    speed = step / dt
                    self.max_observed_step = max(self.max_observed_step, step)
                    self.max_observed_speed = max(self.max_observed_speed, speed)
                    if step > self.args.abort_observed_step:
                        self.abort_reason = f"{j} observed jump {step:.3f} rad"
                    if speed > self.args.abort_observed_speed:
                        self.abort_reason = f"{j} observed speed {speed:.3f} rad/s"
        self.prev_pose = self.pose
        self.pose = new_pose
        self.prev_time = now

    def _left_joint_cb(self, msg: JointState):
        self.left_pose = {n: float(p) for n, p in zip(msg.name, msg.position)}

    def wait_ready(self, timeout_s: float = 5.0) -> bool:
        deadline = time.time() + timeout_s
        while rclpy.ok() and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if (
                all(j in self.pose for j in R_JOINTS + L_JOINTS)
                and GRIP_R in self.pose
                and "gripper_left_controller" in self.left_pose
            ):
                return True
        return False

    def current_q(self) -> List[float]:
        return [float(self.pose[j]) for j in R_JOINTS]

    def _publish(self, q_right: List[float]):
        names = []
        pos = []
        for j, q in zip(R_JOINTS, q_right):
            names.append(j)
            pos.append(float(q))
        for j in L_JOINTS:
            names.append(j)
            pos.append(float(self.pose.get(j, 0.0)))
        names.append(GRIP_R)
        pos.append(float(self.pose.get(GRIP_R, 0.0)))

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = names
        msg.position = pos
        self.pub.publish(msg)

        left_msg = JointState()
        left_msg.header.stamp = self.get_clock().now().to_msg()
        left_msg.name = ["gripper_left_controller"]
        left_msg.position = [float(self.left_pose.get("gripper_left_controller", 0.0))]
        self.left_pub.publish(left_msg)

    def validate_target(self, q: List[float]) -> bool:
        for j, v in zip(R_JOINTS, q):
            lo, hi = JOINT_LIMITS[j]
            if not math.isfinite(v) or v < lo + 0.05 or v > hi - 0.05:
                self.get_logger().error(f"target {j}={v:.3f} outside safe limits [{lo:.3f}, {hi:.3f}]")
                return False
        return True

    def move_to(self, q_target: List[float], label: str) -> bool:
        if not self.validate_target(q_target):
            return False
        q_start = self.current_q()
        max_delta = max(abs(a - b) for a, b in zip(q_start, q_target))
        steps = max(1, int(math.ceil(max_delta / self.args.max_joint_step)))
        duration = steps * self.args.dt
        self.get_logger().info(
            f"move {label}: steps={steps}, duration={duration:.1f}s, max_delta={max_delta:.3f}rad"
        )

        for i in range(steps + 1):
            if self.abort_reason:
                self.get_logger().error(f"ABORT before publish: {self.abort_reason}")
                return False
            alpha = i / steps
            q_cmd = [(1.0 - alpha) * s + alpha * t for s, t in zip(q_start, q_target)]
            self._publish(q_cmd)
            end = time.time() + self.args.dt
            while rclpy.ok() and time.time() < end:
                rclpy.spin_once(self, timeout_sec=0.01)
                if self.abort_reason:
                    self.get_logger().error(f"ABORT while moving {label}: {self.abort_reason}")
                    self._publish(self.current_q())
                    return False

            if i % max(1, steps // 10) == 0:
                observed = self.current_q()
                err = max(abs(a - b) for a, b in zip(observed, q_cmd))
                if err > self.args.abort_tracking_error:
                    self.abort_reason = f"tracking error {err:.3f} rad"
                    self.get_logger().error(f"ABORT while moving {label}: {self.abort_reason}")
                    self._publish(observed)
                    return False

        self.get_logger().info(
            f"arrived {label}; max_observed_speed={self.max_observed_speed:.3f}rad/s, "
            f"max_observed_step={self.max_observed_step:.3f}rad"
        )
        return True


def load_targets(csv_path: str, current_q: List[float], max_targets: int, max_dq: float) -> List[Tuple[str, List[float]]]:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["reachable"] != "1":
                continue
            q = [float(r[f"q_{j}"]) for j in R_JOINTS]
            if any(
                q_i < JOINT_LIMITS[j][0] + 0.05 or q_i > JOINT_LIMITS[j][1] - 0.05
                for j, q_i in zip(R_JOINTS, q)
            ):
                continue
            dq = math.sqrt(sum((a - b) ** 2 for a, b in zip(q, current_q)))
            xyz = (float(r["x"]), float(r["y"]), float(r["z"]))
            rows.append((dq, xyz, q))
    rows.sort(key=lambda x: x[0])
    selected = []
    for dq, xyz, q in rows:
        if dq > max_dq:
            continue
        if all(math.dist(xyz, prev_xyz) > 0.09 for _, prev_xyz, _ in selected):
            selected.append((dq, xyz, q))
        if len(selected) >= max_targets:
            break
    return [(f"xyz=({x:.2f},{y:.2f},{z:.2f}) dq={dq:.2f}", q) for dq, (x, y, z), q in selected]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="/home/u-zhuang/ros2_ws/src/isaac/isaac_sim_2026/reachability_results_local_demo/curobo_reachability_scan.csv")
    parser.add_argument("--max-targets", type=int, default=2)
    parser.add_argument("--max-dq", type=float, default=1.35)
    parser.add_argument("--max-joint-step", type=float, default=0.006)
    parser.add_argument("--dt", type=float, default=0.08)
    parser.add_argument("--dwell", type=float, default=1.0)
    parser.add_argument("--abort-observed-step", type=float, default=0.18)
    parser.add_argument("--abort-observed-speed", type=float, default=4.0)
    parser.add_argument("--abort-tracking-error", type=float, default=0.45)
    parser.add_argument("--no-return", action="store_true")
    args = parser.parse_args()

    rclpy.init()
    node = ReachabilityDemo(args)
    try:
        if not node.wait_ready():
            node.get_logger().error("timed out waiting for /joint_states")
            return 2
        start_q = node.current_q()
        node.get_logger().info(f"start_q={[round(x, 3) for x in start_q]}")
        targets = load_targets(args.csv, start_q, args.max_targets, args.max_dq)
        if not targets:
            node.get_logger().error("no safe nearby targets selected")
            return 3
        node.get_logger().info("selected targets:")
        for label, q in targets:
            node.get_logger().info(f"  {label} q={[round(x, 3) for x in q]}")

        for label, q in targets:
            if not node.move_to(q, label):
                return 4
            end = time.time() + args.dwell
            while rclpy.ok() and time.time() < end:
                node._publish(node.current_q())
                rclpy.spin_once(node, timeout_sec=0.05)
                if node.abort_reason:
                    node.get_logger().error(f"ABORT during dwell: {node.abort_reason}")
                    return 5

        if not args.no_return:
            node.move_to(start_q, "return_to_start")
        node.get_logger().info("reachability demo complete")
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
