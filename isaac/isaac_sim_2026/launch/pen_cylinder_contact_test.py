#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Upright pen/cylinder contact test.

Assumption: a vertical cylinder/pen is already placed in Isaac Sim at the target
grasp center. This script does not spawn the object. It only performs:

open_gripper -> pre_grasp -> approach -> staged close -> optional small lift -> retreat

The first safe run should use --lift-height 0.0 and visually check whether the
fingers sleeve around the pen before enabling a small lift.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import math
import time
from typing import List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.duration import Duration
from tf2_ros import TransformException

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from curobo_pregrasp_approach_dry_run import parse_xyz, solve_waypoint_ik  # noqa: E402
from gripper_air_open_close_test import GripperAirTest  # noqa: E402
from pen_grasp_alignment_dry_run import parse_axis, parse_rpy_deg, quat_from_rpy  # noqa: E402


RIGHT_LIMITS = [
    (-2.87979326579, 2.87979326579),
    (-0.9599310886, 1.6580627894),
    (-3.0194196060, 0.0872664626),
    (-2.8797932658, 2.8797932658),
    (-0.3490658504, 4.6251225178),
    (-3.1415926536, 3.1415926536),
]


def parse_values(raw: str) -> List[float]:
    vals = [float(x) for x in raw.replace(",", " ").split()]
    if not vals:
        raise argparse.ArgumentTypeError("close values must not be empty")
    return vals


def parse_offsets(raw: str) -> List[float]:
    vals = [float(x) for x in raw.replace(",", " ").split()]
    if not vals:
        raise argparse.ArgumentTypeError("y offsets must not be empty")
    return vals


def quat_tilt_deg_xyzw(x: float, y: float, z: float, w: float) -> float:
    # World Z expressed after the object rotation. Tilt is angle from vertical.
    zz = 1.0 - 2.0 * (x * x + y * y)
    zz = max(-1.0, min(1.0, zz))
    return float(math.degrees(math.acos(zz)))


def lookup_object_pose(node: GripperAirTest, frame: str) -> Tuple[np.ndarray, float]:
    tf = node.tf_buffer.lookup_transform(
        "base_link",
        frame,
        rclpy.time.Time(),
        timeout=Duration(seconds=0.2),
    )
    pos = np.array([tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z], dtype=np.float64)
    rot = tf.transform.rotation
    tilt = quat_tilt_deg_xyzw(rot.x, rot.y, rot.z, rot.w)
    return pos, tilt


def maybe_lookup_object_pose(node: GripperAirTest, frame: str) -> Tuple[Optional[np.ndarray], float]:
    if not frame:
        return None, float("nan")
    try:
        return lookup_object_pose(node, frame)
    except TransformException:
        return None, float("nan")


def dwell_hold(node: GripperAirTest, seconds: float) -> None:
    end_time = time.time() + max(0.0, float(seconds))
    while rclpy.ok() and time.time() < end_time:
        node.publish_command(node.current_right_q(), node.current_grip())
        rclpy.spin_once(node, timeout_sec=0.02)
        if node.abort_reason:
            break


def right_tcp_position(node: GripperAirTest) -> np.ndarray:
    tf = node.tf_buffer.lookup_transform(
        "base_link",
        "right_tcp",
        rclpy.time.Time(),
        timeout=Duration(seconds=0.2),
    )
    return np.array([tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z], dtype=np.float64)


def validate_q(q: np.ndarray, margin: float) -> None:
    for i, (value, (lo, hi)) in enumerate(zip(q.tolist(), RIGHT_LIMITS), 1):
        if value < lo + margin or value > hi - margin:
            raise RuntimeError(f"q joint{i}_R={value:.4f} outside safe range [{lo + margin:.4f}, {hi - margin:.4f}]")


def record(
    node: GripperAirTest,
    test_index: int,
    y_offset: float,
    stage: str,
    target_xyz: np.ndarray,
    gripper_target: float,
    note: str,
    object_diameter: float,
    object_before: Optional[np.ndarray],
    object_before_tilt_deg: float,
    object_frame: str,
    manual_observation: str,
) -> dict:
    actual_tcp = right_tcp_position(node)
    actual_grip = node.current_grip()
    finger_distance = node.finger_distance()
    finger_clearance = float("nan")
    if object_diameter > 0.0:
        finger_clearance = float(finger_distance - object_diameter)
    object_after, object_after_tilt_deg = maybe_lookup_object_pose(node, object_frame)
    moved_m = float("nan")
    lifted_or_moved = "unknown"
    if object_before is not None and object_after is not None:
        moved_m = float(np.linalg.norm(object_after - object_before))
        lifted_or_moved = "yes" if moved_m > 0.005 or abs(float(object_after[2] - object_before[2])) > 0.005 else "no"
    return {
        "test_index": int(test_index),
        "y_offset_m": float(y_offset),
        "stage": stage,
        "target_x": float(target_xyz[0]),
        "target_y": float(target_xyz[1]),
        "target_z": float(target_xyz[2]),
        "actual_tcp_x": float(actual_tcp[0]),
        "actual_tcp_y": float(actual_tcp[1]),
        "actual_tcp_z": float(actual_tcp[2]),
        "tcp_error_m": float(np.linalg.norm(actual_tcp - target_xyz)),
        "target_gripper": float(gripper_target),
        "actual_gripper": float(actual_grip),
        "finger_distance_m": float(finger_distance),
        "object_diameter_m": float(object_diameter) if object_diameter > 0.0 else float("nan"),
        "finger_clearance_m": finger_clearance,
        "max_mimic_error_rad": float(node.max_mimic_error(actual_grip)),
        "max_gripper_speed_rad_s": float(node.max_grip_speed),
        "max_arm_drift_rad": float(node.max_arm_drift),
        "object_frame": object_frame,
        "object_before_x": float(object_before[0]) if object_before is not None else float("nan"),
        "object_before_y": float(object_before[1]) if object_before is not None else float("nan"),
        "object_before_z": float(object_before[2]) if object_before is not None else float("nan"),
        "object_after_x": float(object_after[0]) if object_after is not None else float("nan"),
        "object_after_y": float(object_after[1]) if object_after is not None else float("nan"),
        "object_after_z": float(object_after[2]) if object_after is not None else float("nan"),
        "object_moved_m": moved_m,
        "object_tilt_before_deg": float(object_before_tilt_deg),
        "object_tilt_after_deg": float(object_after_tilt_deg),
        "object_lifted_or_moved": lifted_or_moved,
        "manual_observation": manual_observation,
        "status": "abort" if node.abort_reason else "ok",
        "note": note,
    }


def write_report(out_dir: str, rows: List[dict]) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "pen_cylinder_contact_test_report.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "test_index",
            "y_offset_m",
            "stage",
            "target_x",
            "target_y",
            "target_z",
            "actual_tcp_x",
            "actual_tcp_y",
            "actual_tcp_z",
            "tcp_error_m",
            "target_gripper",
            "actual_gripper",
            "finger_distance_m",
            "object_diameter_m",
            "finger_clearance_m",
            "max_mimic_error_rad",
            "max_gripper_speed_rad_s",
            "max_arm_drift_rad",
            "object_frame",
            "object_before_x",
            "object_before_y",
            "object_before_z",
            "object_after_x",
            "object_after_y",
            "object_after_z",
            "object_moved_m",
            "object_tilt_before_deg",
            "object_tilt_after_deg",
            "object_lifted_or_moved",
            "manual_observation",
            "status",
            "note",
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
    parser.add_argument("--y-offset", type=float, default=0.0, help="offset target in base_link Y for biased contact tests")
    parser.add_argument("--y-offsets", type=parse_offsets, default=None, help="comma/space list for batch biased contact tests")
    parser.add_argument("--approach-axis", type=parse_axis, default=parse_axis("0,0,-1"))
    parser.add_argument("--approach-distance", type=float, default=0.10)
    parser.add_argument("--tcp-rpy-deg", type=parse_rpy_deg, default=parse_rpy_deg("0,0,0"))
    parser.add_argument("--open-value", type=float, default=0.25)
    parser.add_argument("--close-values", type=parse_values, default=parse_values("-0.40,-0.50,-0.60"))
    parser.add_argument("--lift-height", type=float, default=0.0)
    parser.add_argument("--object-diameter", type=float, default=0.0, help="optional measured pen/cylinder diameter in meters")
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--num-seeds", type=int, default=192)
    parser.add_argument("--position-threshold", type=float, default=0.02)
    parser.add_argument("--rotation-threshold", type=float, default=0.15)
    parser.add_argument("--joint-limit-margin", type=float, default=0.05)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--arm-max-step", type=float, default=0.001)
    parser.add_argument("--arm-min-steps", type=int, default=800)
    parser.add_argument("--approach-min-steps", type=int, default=900)
    parser.add_argument("--gripper-max-step", type=float, default=0.0008)
    parser.add_argument("--gripper-min-steps", type=int, default=180)
    parser.add_argument("--dwell", type=float, default=1.0)
    parser.add_argument("--object-frame", default="", help="optional TF frame for pen/cylinder pose logging")
    parser.add_argument("--manual-observation", default="unrecorded", help="free text copied into every report row")
    parser.add_argument("--gripper-limit-margin", type=float, default=0.02)
    parser.add_argument("--abort-gripper-step", type=float, default=0.02)
    parser.add_argument("--abort-gripper-speed", type=float, default=0.5)
    parser.add_argument("--abort-speed-spike-count", type=int, default=3)
    parser.add_argument("--min-velocity-dt", type=float, default=0.01)
    parser.add_argument("--abort-arm-drift", type=float, default=0.12)
    args = parser.parse_args()

    # GripperAirTest expects these attributes for its arm and gripper movers.
    args.pregrasp_q = "current"

    y_offsets = args.y_offsets if args.y_offsets is not None else [float(args.y_offset)]
    quat_wxyz = quat_from_rpy(float(args.tcp_rpy_deg[0]), float(args.tcp_rpy_deg[1]), float(args.tcp_rpy_deg[2]))

    rclpy.init()
    node = GripperAirTest(args)
    rows: List[dict] = []
    try:
        node.wait_ready(args.timeout)
        for test_index, y_offset in enumerate(y_offsets, 1):
            grasp_xyz = args.pen_grasp_xyz.astype(np.float64).copy()
            grasp_xyz[1] += float(y_offset)
            pregrasp_xyz = grasp_xyz - float(args.approach_distance) * args.approach_axis.astype(np.float64)
            lift_xyz = grasp_xyz + np.array([0.0, 0.0, float(args.lift_height)], dtype=np.float64)
            q_start = np.array(node.current_right_q(), dtype=np.float32)
            waypoints = [pregrasp_xyz, grasp_xyz]
            if args.lift_height > 1e-6:
                waypoints.append(lift_xyz)

            object_before, object_before_tilt_deg = maybe_lookup_object_pose(node, args.object_frame)
            joint_names, q_solutions, ik_pos_err, ik_rot_err = solve_waypoint_ik(
                args.robot_cfg_path,
                q_start,
                waypoints,
                quat_wxyz,
                args.num_seeds,
                args.position_threshold,
                args.rotation_threshold,
            )
            for q in q_solutions:
                validate_q(q, args.joint_limit_margin)

            node.get_logger().info(f"test {test_index}/{len(y_offsets)}; y_offset={y_offset:+.4f}m")
            node.get_logger().info(f"joint_names={joint_names}")
            node.get_logger().info(f"pen_grasp_xyz={[round(float(x), 4) for x in grasp_xyz.tolist()]}")
            node.get_logger().info(f"pregrasp_xyz={[round(float(x), 4) for x in pregrasp_xyz.tolist()]}")
            node.get_logger().info(f"close_values={args.close_values}; lift_height={args.lift_height:.3f}")

            if not node.move_gripper_to(args.open_value, f"offset_{y_offset:+.3f}_open_before_pregrasp"):
                raise RuntimeError(node.abort_reason or "open failed")
            dwell_hold(node, args.dwell)
            rows.append(record(node, test_index, y_offset, "open_before_pregrasp", right_tcp_position(node), args.open_value, "open in air", args.object_diameter, object_before, object_before_tilt_deg, args.object_frame, args.manual_observation))

            if not node.move_arm_to(q_solutions[0].tolist()):
                raise RuntimeError(node.abort_reason or "pre_grasp move failed")
            dwell_hold(node, args.dwell)
            rows.append(record(node, test_index, y_offset, "pre_grasp", pregrasp_xyz, args.open_value, "above pen, open", args.object_diameter, object_before, object_before_tilt_deg, args.object_frame, args.manual_observation))

            if not node.move_arm_to(q_solutions[1].tolist()):
                raise RuntimeError(node.abort_reason or "approach move failed")
            dwell_hold(node, args.dwell)
            rows.append(record(node, test_index, y_offset, "approach_open", grasp_xyz, args.open_value, "sleeve around pen, still open", args.object_diameter, object_before, object_before_tilt_deg, args.object_frame, args.manual_observation))

            for value in args.close_values:
                if not node.move_gripper_to(value, f"offset_{y_offset:+.3f}_close_{value:+.2f}"):
                    break
                dwell_hold(node, args.dwell)
                rows.append(record(node, test_index, y_offset, f"close_{value:+.2f}", grasp_xyz, value, "graded close at grasp pose", args.object_diameter, object_before, object_before_tilt_deg, args.object_frame, args.manual_observation))
                if node.abort_reason:
                    break

            if not node.abort_reason and args.lift_height > 1e-6:
                if not node.move_arm_to(q_solutions[2].tolist()):
                    raise RuntimeError(node.abort_reason or "lift move failed")
                dwell_hold(node, args.dwell)
                rows.append(record(node, test_index, y_offset, "lift", lift_xyz, node.current_grip(), "small lift test", args.object_diameter, object_before, object_before_tilt_deg, args.object_frame, args.manual_observation))

            if not node.abort_reason:
                if node.move_arm_to(q_solutions[0].tolist()):
                    dwell_hold(node, args.dwell)
                    rows.append(record(node, test_index, y_offset, "retreat", pregrasp_xyz, node.current_grip(), "retreat from pen", args.object_diameter, object_before, object_before_tilt_deg, args.object_frame, args.manual_observation))
                node.move_gripper_to(args.open_value, f"offset_{y_offset:+.3f}_open_after_retreat")
                dwell_hold(node, args.dwell)
                rows.append(record(node, test_index, y_offset, "open_after_retreat", pregrasp_xyz, args.open_value, "leave open", args.object_diameter, object_before, object_before_tilt_deg, args.object_frame, args.manual_observation))

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
    except Exception as exc:
        node.get_logger().error(str(exc))
        if rows:
            report = write_report(args.out_dir, rows)
            node.get_logger().error(f"partial report written: {report}")
        return 2
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
