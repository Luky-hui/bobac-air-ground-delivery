#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pen-specific dry run: top-down pre_grasp -> approach -> retreat.

This is a no-contact alignment test for a thin upright pen. It keeps the same
right_tcp/curobo safety logic as curobo_pregrasp_approach_dry_run.py, but uses an
explicit approach direction in base_link, defaulting to straight down.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import List

import numpy as np
import rclpy

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from curobo_pregrasp_approach_dry_run import (  # noqa: E402
    DryRunNode,
    parse_xyz,
    solve_waypoint_ik,
    xyzw_to_wxyz,
)


def parse_axis(raw: str) -> np.ndarray:
    axis = parse_xyz(raw)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9:
        raise argparse.ArgumentTypeError("approach axis must be non-zero")
    return axis / norm


def quat_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float32,
    )


def parse_rpy_deg(raw: str) -> np.ndarray:
    vals = parse_xyz(raw)
    return np.radians(vals.astype(np.float64))


def write_pen_report(out_dir: str, rows: List[dict]) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "pen_grasp_alignment_dry_run_report.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "stage",
            "target_x",
            "target_y",
            "target_z",
            "actual_x",
            "actual_y",
            "actual_z",
            "tcp_error_m",
            "ik_position_error_m",
            "ik_rotation_error_rad",
            "status",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-cfg-path", default="/home/u-zhuang/ros2_ws/src/isaac/isaac_sim_2026/config/turingstack_x1_2_right_tcp.yml")
    parser.add_argument("--out-dir", default="/home/u-zhuang/ros2_ws/src/isaac/isaac_sim_2026/alignment_results")
    parser.add_argument("--pen-grasp-xyz", type=parse_xyz, default=parse_xyz("0.45,-0.35,0.77"))
    parser.add_argument(
        "--approach-axis",
        type=parse_axis,
        default=parse_axis("0,0,-1"),
        help="unit direction from pre_grasp toward grasp in base_link; default is straight down",
    )
    parser.add_argument("--approach-distance", type=float, default=0.10)
    parser.add_argument(
        "--tcp-rpy-deg",
        type=parse_rpy_deg,
        default=None,
        help="optional fixed TCP attitude in base_link, degrees. Example: 0,-30,-45",
    )
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--num-seeds", type=int, default=128)
    parser.add_argument("--position-threshold", type=float, default=0.02)
    parser.add_argument("--rotation-threshold", type=float, default=0.15)
    parser.add_argument("--max-joint-step", type=float, default=0.0015)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--min-steps", type=int, default=500)
    parser.add_argument("--approach-min-steps", type=int, default=700)
    parser.add_argument("--dwell", type=float, default=1.0)
    parser.add_argument("--joint-limit-margin", type=float, default=0.05)
    parser.add_argument("--abort-observed-step", type=float, default=0.015)
    parser.add_argument("--abort-observed-speed", type=float, default=0.35)
    parser.add_argument("--abort-speed-spike-count", type=int, default=3)
    parser.add_argument("--min-velocity-dt", type=float, default=0.01)
    parser.add_argument("--abort-tracking-error", type=float, default=0.35)
    parser.add_argument("--abort-tcp-error", type=float, default=0.06)
    args = parser.parse_args()

    rclpy.init()
    node = DryRunNode(args)
    rows: List[dict] = []
    try:
        node.wait_ready(args.timeout)
        q_start = np.array(node.current_q(), dtype=np.float64)
        start_tcp, quat_xyzw, _ = node.lookup_right_tcp_pose()
        grasp_xyz = args.pen_grasp_xyz.astype(np.float64)
        approach_axis = args.approach_axis.astype(np.float64)
        pregrasp_xyz = grasp_xyz - float(args.approach_distance) * approach_axis
        if args.tcp_rpy_deg is None:
            quat_wxyz = xyzw_to_wxyz(quat_xyzw)
            attitude_note = "current"
        else:
            quat_wxyz = quat_from_rpy(float(args.tcp_rpy_deg[0]), float(args.tcp_rpy_deg[1]), float(args.tcp_rpy_deg[2]))
            attitude_note = f"rpy_deg={[round(float(x), 3) for x in np.degrees(args.tcp_rpy_deg).tolist()]}"

        node.get_logger().info(f"start_q={[round(float(x), 4) for x in q_start.tolist()]}")
        node.get_logger().info(f"start_right_tcp={[round(float(x), 4) for x in start_tcp.tolist()]}")
        node.get_logger().info(f"pen_grasp_xyz={[round(float(x), 4) for x in grasp_xyz.tolist()]}")
        node.get_logger().info(f"approach_axis_base={[round(float(x), 4) for x in approach_axis.tolist()]}")
        node.get_logger().info(f"pregrasp_xyz={[round(float(x), 4) for x in pregrasp_xyz.tolist()]}")
        node.get_logger().info(f"tcp_attitude={attitude_note}")

        joint_names, q_solutions, ik_pos_err, ik_rot_err = solve_waypoint_ik(
            args.robot_cfg_path,
            q_start,
            [pregrasp_xyz, grasp_xyz],
            quat_wxyz,
            args.num_seeds,
            args.position_threshold,
            args.rotation_threshold,
        )
        node.get_logger().info(f"curobo_joint_names={joint_names}")
        node.get_logger().info(f"q_pregrasp={[round(float(x), 4) for x in q_solutions[0].tolist()]}")
        node.get_logger().info(f"q_pen_grasp={[round(float(x), 4) for x in q_solutions[1].tolist()]}")

        stages = [
            ("pen_pre_grasp", pregrasp_xyz, q_solutions[0], args.min_steps, ik_pos_err[0], ik_rot_err[0]),
            ("pen_approach", grasp_xyz, q_solutions[1], args.approach_min_steps, ik_pos_err[1], ik_rot_err[1]),
            ("pen_retreat", pregrasp_xyz, q_solutions[0], args.approach_min_steps, ik_pos_err[0], ik_rot_err[0]),
        ]

        for stage, target_xyz, q_target, min_steps, pe, re in stages:
            status = "ok"
            if not node.move_to(q_target.tolist(), stage, min_steps):
                status = f"abort_move: {node.abort_reason}"
            else:
                ok, tcp_error, actual = node.dwell_and_check_tcp(target_xyz, stage)
                if not ok:
                    status = f"abort_tcp: {node.abort_reason}"
                rows.append(
                    {
                        "stage": stage,
                        "target_x": float(target_xyz[0]),
                        "target_y": float(target_xyz[1]),
                        "target_z": float(target_xyz[2]),
                        "actual_x": float(actual[0]),
                        "actual_y": float(actual[1]),
                        "actual_z": float(actual[2]),
                        "tcp_error_m": float(tcp_error),
                        "ik_position_error_m": float(pe),
                        "ik_rotation_error_rad": float(re),
                        "status": status,
                    }
                )
            if status != "ok":
                node.publish_hold(node.current_q())
                break

        if not node.abort_reason:
            node.move_to(q_start.tolist(), "return_start", args.min_steps)

        report = write_pen_report(args.out_dir, rows)
        node.get_logger().info(f"report written: {report}")
        node.get_logger().info(
            f"pen dry run finished; max_step={node.max_observed_step:.4f}rad, "
            f"max_speed={node.max_observed_speed:.3f}rad/s"
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
