#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sweep right gripper closure values in air and measure finger distance."""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from typing import List

import rclpy

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from gripper_air_open_close_test import GripperAirTest, write_report  # noqa: E402


def parse_values(raw: str) -> List[float]:
    vals = [float(x) for x in raw.replace(",", " ").split()]
    if not vals:
        raise argparse.ArgumentTypeError("values must not be empty")
    return vals


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="/home/u-zhuang/ros2_ws/src/isaac/isaac_sim_2026/alignment_results")
    parser.add_argument("--values", type=parse_values, default=parse_values("+0.25,0.00,-0.10,-0.20,-0.30,-0.40,-0.50,-0.60"))
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--dwell", type=float, default=0.8)
    parser.add_argument("--pregrasp-q", default="current")
    parser.add_argument("--arm-max-step", type=float, default=0.0015)
    parser.add_argument("--arm-min-steps", type=int, default=500)
    parser.add_argument("--gripper-max-step", type=float, default=0.0008)
    parser.add_argument("--gripper-min-steps", type=int, default=180)
    parser.add_argument("--gripper-limit-margin", type=float, default=0.02)
    parser.add_argument("--abort-gripper-step", type=float, default=0.02)
    parser.add_argument("--abort-gripper-speed", type=float, default=0.5)
    parser.add_argument("--abort-speed-spike-count", type=int, default=3)
    parser.add_argument("--min-velocity-dt", type=float, default=0.01)
    parser.add_argument("--abort-arm-drift", type=float, default=0.08)
    args = parser.parse_args()

    rclpy.init()
    node = GripperAirTest(args)
    rows: List[dict] = []
    try:
        node.wait_ready(args.timeout)
        node.get_logger().info(f"start_gripper={node.current_grip():+.4f}")
        node.get_logger().info(f"start_finger_distance={node.finger_distance():.5f}m")

        if args.pregrasp_q.strip().lower() != "current":
            from gripper_air_open_close_test import parse_floats

            pregrasp_q = parse_floats(args.pregrasp_q, 6, "pregrasp_q")
            if not node.move_arm_to(pregrasp_q):
                raise RuntimeError(node.abort_reason or "arm move failed")

        for target in args.values:
            target = node.clamp_gripper(target)
            label = f"sweep_{target:+.2f}"
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
            time.sleep(0.1)

        # Leave the gripper open after the sweep.
        open_target = node.clamp_gripper(0.25)
        if not node.abort_reason:
            node.move_gripper_to(open_target, "return_open")
            rows.append(node.dwell_measure(open_target, "return_open"))

        path = os.path.join(args.out_dir, "gripper_closure_sweep_report.csv")
        os.makedirs(args.out_dir, exist_ok=True)
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
        node.get_logger().info(f"report written: {path}")
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
