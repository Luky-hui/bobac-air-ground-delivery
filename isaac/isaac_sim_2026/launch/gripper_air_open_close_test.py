#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Right gripper air open/close test without touching objects.

The script can first move the right arm to a known safe pre_grasp joint pose,
then keeps both arms fixed and publishes only the right gripper controller value
inside /joint_command. It never commands the mimic joints directly.
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

DIRECT_GRIP_R = [
    ("right_gripper_left_joint2", 1.0),
    ("right_gripper_right_join2", -1.0),
    ("right_gripper_left_finger_joint", -1.0),
    ("right_gripper_right_finger_joint", 1.0),
    ("right_gripper_left_outer_finger_joint", -1.0),
    ("gripper_right_outer_finger_joint", -1.0),
    ("right_gripper_right_outer_finger_joint", -1.0),
]

MIMIC_JOINTS = {
    # Observed from /joint_states in Isaac Sim. These signs are the runtime
    # articulation state, which is what matters for monitoring this test.
    "gripper01_base_to_gripper_left2": 1.0,
    "gripper01_base_to_gripper_right2": -1.0,
    "gripper01_base_to_gripper_right3": -1.0,
    "gripper01_right3_to_gripper_right1": 1.0,
    "gripper01_left3_to_gripper_left1": -1.0,
}

GRIPPER_LIMIT = (-0.6999999710, 0.3000000025)
DIRECT_GRIPPER_LIMIT = (-1.50, 0.3000000025)


def parse_floats(raw: str, count: int, label: str) -> List[float]:
    vals = [float(x) for x in raw.replace(",", " ").split()]
    if len(vals) != count:
        raise argparse.ArgumentTypeError(f"{label} must contain {count} numbers")
    return vals


def tf_translation(tf) -> np.ndarray:
    return np.array(
        [tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z],
        dtype=np.float64,
    )


class GripperAirTest(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("gripper_air_open_close_test")
        self.args = args
        self.pose: Dict[str, float] = {}
        self.left_pose: Dict[str, float] = {}
        self.prev_pose: Dict[str, float] = {}
        self.prev_stamp: Optional[float] = None
        self.abort_reason: Optional[str] = None
        self.max_grip_step = 0.0
        self.max_grip_speed = 0.0
        self.max_arm_step = 0.0
        self.max_arm_drift = 0.0
        self.speed_spike_count = 0
        self.arm_hold_q: Optional[List[float]] = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)
        self.pub = self.create_publisher(JointState, "/joint_command", 10)
        self.left_pub = self.create_publisher(JointState, "/joint_left_command", 10)
        self.create_subscription(JointState, "/joint_states", self._joint_cb, 10)
        self.create_subscription(JointState, "/joint_left_states", self._left_cb, 10)

    def _joint_cb(self, msg: JointState) -> None:
        stamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        if stamp <= 0.0:
            stamp = time.time()
        new_pose = {name: float(pos) for name, pos in zip(msg.name, msg.position)}

        if self.pose and self.prev_stamp is not None:
            dt = stamp - self.prev_stamp
            if GRIP_R in self.pose and GRIP_R in new_pose:
                step = abs(new_pose[GRIP_R] - self.pose[GRIP_R])
                self.max_grip_step = max(self.max_grip_step, step)
                if step > self.args.abort_gripper_step:
                    self.abort_reason = (
                        f"{GRIP_R} observed step {step:.4f} rad > "
                        f"{self.args.abort_gripper_step:.4f}"
                    )
                if dt >= self.args.min_velocity_dt:
                    speed = step / dt
                    self.max_grip_speed = max(self.max_grip_speed, speed)
                    if speed > self.args.abort_gripper_speed:
                        self.speed_spike_count += 1
                        if self.speed_spike_count >= self.args.abort_speed_spike_count:
                            self.abort_reason = (
                                f"{GRIP_R} observed speed {speed:.3f} rad/s > "
                                f"{self.args.abort_gripper_speed:.3f}"
                            )
                    else:
                        self.speed_spike_count = 0

            for joint in R_JOINTS:
                if joint in self.pose and joint in new_pose:
                    self.max_arm_step = max(self.max_arm_step, abs(new_pose[joint] - self.pose[joint]))

            if self.arm_hold_q is not None:
                drift = max(abs(new_pose.get(j, hold) - hold) for j, hold in zip(R_JOINTS, self.arm_hold_q))
                self.max_arm_drift = max(self.max_arm_drift, drift)
                if drift > self.args.abort_arm_drift:
                    self.abort_reason = f"right arm drift {drift:.4f} rad > {self.args.abort_arm_drift:.4f}"

        self.prev_pose = self.pose
        self.pose = new_pose
        self.prev_stamp = stamp

    def _left_cb(self, msg: JointState) -> None:
        self.left_pose = {name: float(pos) for name, pos in zip(msg.name, msg.position)}

    def close(self) -> None:
        if hasattr(self.tf_listener, "executor"):
            self.tf_listener.executor.shutdown()
        if hasattr(self.tf_listener, "dedicated_listener_thread"):
            self.tf_listener.dedicated_listener_thread.join(timeout=1.0)
        self.tf_listener.unregister()

    def wait_ready(self, timeout_s: float) -> None:
        deadline = time.time() + timeout_s
        missing_tf: List[str] = []
        while time.time() < deadline and rclpy.ok():
            have_gripper = GRIP_R in self.pose or any(name in self.pose for name, _ in DIRECT_GRIP_R)
            have_joints = all(j in self.pose for j in R_JOINTS + L_JOINTS) and have_gripper
            missing_tf = []
            for frame in ["right_tcp"]:
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
        missing_joints = [j for j in R_JOINTS + L_JOINTS if j not in self.pose]
        if GRIP_R not in self.pose and not any(name in self.pose for name, _ in DIRECT_GRIP_R):
            missing_joints.append(f"{GRIP_R} or direct right_gripper_*")
        raise RuntimeError(f"not ready; missing_joints={missing_joints}, missing_tf={missing_tf}")

    def current_right_q(self) -> List[float]:
        return [float(self.pose[j]) for j in R_JOINTS]

    def current_left_q(self) -> List[float]:
        return [float(self.pose.get(j, 0.0)) for j in L_JOINTS]

    def current_grip(self) -> float:
        if GRIP_R in self.pose:
            return float(self.pose[GRIP_R])
        for name, multiplier in DIRECT_GRIP_R:
            if name in self.pose and abs(multiplier) > 1e-9:
                return float(self.pose[name]) / float(multiplier)
        return 0.0

    def finger_distance(self) -> float:
        try:
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
            return float(np.linalg.norm(tf_translation(left) - tf_translation(right)))
        except TransformException:
            return float("nan")

    def max_mimic_error(self, grip_value: float) -> float:
        errors = []
        for joint, multiplier in MIMIC_JOINTS.items():
            if joint in self.pose:
                errors.append(abs(float(self.pose[joint]) - multiplier * grip_value))
        return max(errors) if errors else float("nan")

    def clamp_gripper(self, value: float) -> float:
        using_direct_runtime = any(name in self.pose for name, _ in DIRECT_GRIP_R)
        lo, hi = DIRECT_GRIPPER_LIMIT if using_direct_runtime else GRIPPER_LIMIT
        lo += self.args.gripper_limit_margin
        hi -= self.args.gripper_limit_margin
        return min(hi, max(lo, float(value)))

    def publish_command(self, q_right: List[float], grip_right: float) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        direct_names = [name for name, _ in DIRECT_GRIP_R if name in self.pose]
        if direct_names:
            msg.name = R_JOINTS + L_JOINTS + direct_names
            direct_positions = []
            for name in direct_names:
                multiplier = next(m for n, m in DIRECT_GRIP_R if n == name)
                direct_positions.append(float(grip_right) * float(multiplier))
            msg.position = [float(v) for v in q_right] + self.current_left_q() + direct_positions
        else:
            msg.name = R_JOINTS + L_JOINTS + [GRIP_R]
            msg.position = [float(v) for v in q_right] + self.current_left_q() + [float(grip_right)]
        self.pub.publish(msg)

        left_msg = JointState()
        left_msg.header.stamp = msg.header.stamp
        left_msg.name = [GRIP_L]
        left_msg.position = [float(self.left_pose.get(GRIP_L, 0.0))]
        self.left_pub.publish(left_msg)

    def move_arm_to(self, target_q: List[float]) -> bool:
        start_q = self.current_right_q()
        start_grip = self.current_grip()
        max_delta = max(abs(a - b) for a, b in zip(start_q, target_q))
        steps = max(self.args.arm_min_steps, int(math.ceil(max_delta / self.args.arm_max_step)))
        self.get_logger().info(
            f"move arm to safe pose: steps={steps}, duration={steps * self.args.dt:.1f}s, "
            f"max_delta={max_delta:.4f}rad"
        )
        self.arm_hold_q = None
        for i in range(steps + 1):
            if self.abort_reason:
                self.get_logger().error(f"SOFT STOP during arm move: {self.abort_reason}")
                self.publish_command(self.current_right_q(), self.current_grip())
                return False
            alpha = i / steps
            q_cmd = [(1.0 - alpha) * s + alpha * t for s, t in zip(start_q, target_q)]
            self.publish_command(q_cmd, start_grip)
            end_time = time.time() + self.args.dt
            while rclpy.ok() and time.time() < end_time:
                rclpy.spin_once(self, timeout_sec=0.005)
        self.get_logger().info("arrived safe arm pose")
        return True

    def move_gripper_to(self, target: float, label: str) -> bool:
        target = self.clamp_gripper(target)
        start = self.current_grip()
        hold_q = self.current_right_q()
        self.arm_hold_q = hold_q
        max_delta = abs(target - start)
        steps = max(self.args.gripper_min_steps, int(math.ceil(max_delta / self.args.gripper_max_step)))
        self.get_logger().info(
            f"move gripper {label}: {start:+.4f} -> {target:+.4f}, "
            f"steps={steps}, duration={steps * self.args.dt:.1f}s"
        )
        self.speed_spike_count = 0
        for i in range(steps + 1):
            if self.abort_reason:
                self.get_logger().error(f"SOFT STOP during gripper move: {self.abort_reason}")
                self.publish_command(self.current_right_q(), self.current_grip())
                return False
            alpha = i / steps
            grip_cmd = (1.0 - alpha) * start + alpha * target
            self.publish_command(hold_q, grip_cmd)
            end_time = time.time() + self.args.dt
            while rclpy.ok() and time.time() < end_time:
                rclpy.spin_once(self, timeout_sec=0.005)
                if self.abort_reason:
                    self.get_logger().error(f"SOFT STOP during gripper move: {self.abort_reason}")
                    self.publish_command(self.current_right_q(), self.current_grip())
                    return False
        return True

    def dwell_measure(self, target: float, label: str) -> dict:
        deadline = time.time() + self.args.dwell
        last_distance = float("nan")
        while time.time() < deadline and rclpy.ok():
            self.publish_command(self.current_right_q(), target)
            rclpy.spin_once(self, timeout_sec=0.02)
            last_distance = self.finger_distance()
            if self.abort_reason:
                break
        actual_grip = self.current_grip()
        return {
            "stage": label,
            "target_gripper": float(target),
            "actual_gripper": float(actual_grip),
            "finger_distance_m": float(last_distance),
            "max_mimic_error_rad": float(self.max_mimic_error(actual_grip)),
            "max_arm_drift_rad": float(self.max_arm_drift),
            "max_gripper_step_rad": float(self.max_grip_step),
            "max_gripper_speed_rad_s": float(self.max_grip_speed),
            "status": "abort" if self.abort_reason else "ok",
        }


def write_report(out_dir: str, rows: List[dict]) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "gripper_air_open_close_report.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "stage",
            "target_gripper",
            "actual_gripper",
            "finger_distance_m",
            "max_mimic_error_rad",
            "max_arm_drift_rad",
            "max_gripper_step_rad",
            "max_gripper_speed_rad_s",
            "status",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="/home/u-zhuang/ros2_ws/src/isaac/isaac_sim_2026/alignment_results")
    parser.add_argument(
        "--pregrasp-q",
        default="0.5650,0.4124,-0.2913,-1.7925,-0.2491,0.8608",
        help="safe right-arm joint pose; use 'current' to skip arm motion",
    )
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--probe-delta", type=float, default=0.10)
    parser.add_argument("--full-delta", type=float, default=0.25)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--dwell", type=float, default=1.0)
    parser.add_argument("--arm-max-step", type=float, default=0.0015)
    parser.add_argument("--arm-min-steps", type=int, default=500)
    parser.add_argument("--gripper-max-step", type=float, default=0.0015)
    parser.add_argument("--gripper-min-steps", type=int, default=80)
    parser.add_argument("--gripper-limit-margin", type=float, default=0.05)
    parser.add_argument("--abort-gripper-step", type=float, default=0.02)
    parser.add_argument("--abort-gripper-speed", type=float, default=0.5)
    parser.add_argument("--abort-speed-spike-count", type=int, default=3)
    parser.add_argument("--min-velocity-dt", type=float, default=0.01)
    parser.add_argument("--abort-arm-drift", type=float, default=0.04)
    args = parser.parse_args()

    rclpy.init()
    node = GripperAirTest(args)
    rows: List[dict] = []
    try:
        node.wait_ready(args.timeout)
        node.get_logger().info(f"start_gripper={node.current_grip():+.4f}")
        node.get_logger().info(f"start_finger_distance={node.finger_distance():.5f}m")

        if args.pregrasp_q.strip().lower() != "current":
            pregrasp_q = parse_floats(args.pregrasp_q, 6, "pregrasp_q")
            if not node.move_arm_to(pregrasp_q):
                raise RuntimeError(node.abort_reason or "arm move failed")

        base_grip = node.current_grip()
        base_distance = node.finger_distance()
        node.get_logger().info(
            f"locked arm pose; base_gripper={base_grip:+.4f}, "
            f"base_finger_distance={base_distance:.5f}m"
        )

        probe_plus = node.clamp_gripper(base_grip + args.probe_delta)
        probe_minus = node.clamp_gripper(base_grip - args.probe_delta)
        for label, target in [("probe_plus", probe_plus), ("probe_minus", probe_minus)]:
            if not node.move_gripper_to(target, label):
                break
            row = node.dwell_measure(target, label)
            rows.append(row)
            node.get_logger().info(
                f"{label}: grip={row['actual_gripper']:+.4f}, "
                f"finger_distance={row['finger_distance_m']:.5f}m, "
                f"mimic_err={row['max_mimic_error_rad']:.4f}rad"
            )
            if node.abort_reason:
                break
            if not node.move_gripper_to(base_grip, f"{label}_return_base"):
                break
            rows.append(node.dwell_measure(base_grip, f"{label}_return_base"))

        if not node.abort_reason and len(rows) >= 3:
            plus_distance = next(r["finger_distance_m"] for r in rows if r["stage"] == "probe_plus")
            minus_distance = next(r["finger_distance_m"] for r in rows if r["stage"] == "probe_minus")
            open_sign = 1.0 if plus_distance >= minus_distance else -1.0
            close_sign = -open_sign
            open_target = node.clamp_gripper(base_grip + open_sign * args.full_delta)
            close_target = node.clamp_gripper(base_grip + close_sign * args.full_delta)
            node.get_logger().info(
                f"detected open_sign={open_sign:+.0f}; open_target={open_target:+.4f}, "
                f"close_target={close_target:+.4f}"
            )
            for label, target in [
                ("full_open", open_target),
                ("full_close", close_target),
                ("full_open_final", open_target),
            ]:
                if not node.move_gripper_to(target, label):
                    break
                rows.append(node.dwell_measure(target, label))
                if node.abort_reason:
                    break

        report = write_report(args.out_dir, rows)
        node.get_logger().info(f"report written: {report}")
        node.get_logger().info(
            f"finished; max_grip_step={node.max_grip_step:.4f}rad, "
            f"max_grip_speed={node.max_grip_speed:.3f}rad/s, "
            f"max_arm_drift={node.max_arm_drift:.4f}rad"
        )
        if node.abort_reason:
            node.get_logger().error(f"abort_reason={node.abort_reason}")
            return 3
        return 0
    finally:
        try:
            if node.pose:
                node.publish_command(node.current_right_q(), node.current_grip())
        finally:
            node.close()
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
