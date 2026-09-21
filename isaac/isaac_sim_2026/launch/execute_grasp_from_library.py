#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Select and optionally execute a right-arm grasp from right_arm_grasp_library.csv.

Default mode is selection-only. Add --execute to publish commands:

open -> pre_grasp -> target -> close -> lift -> retreat

The library is tied to the current right_tcp/base_link setup. Regenerate it after
changing the robot model, TCP, gripper geometry, or base/object staging policy.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import rclpy
from rclpy.duration import Duration
from std_msgs.msg import String
from tf2_ros import TransformException

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from curobo_pregrasp_approach_dry_run import parse_xyz, solve_waypoint_ik  # noqa: E402
from gripper_air_open_close_test import GripperAirTest  # noqa: E402
from pen_cylinder_contact_test import dwell_hold, right_tcp_position, validate_q  # noqa: E402
from pen_grasp_alignment_dry_run import quat_from_rpy  # noqa: E402


SAFETY_PRIORITY = ["core", "usable", "edge"]


def parse_safety(raw: str) -> List[str]:
    vals = [x.strip() for x in raw.replace(",", " ").split() if x.strip()]
    bad = [x for x in vals if x not in SAFETY_PRIORITY]
    if bad:
        raise argparse.ArgumentTypeError(f"unknown safety class: {bad}")
    return vals


def parse_optional_xyz(raw: str) -> Optional[np.ndarray]:
    if raw.strip().lower() in ("", "none"):
        return None
    return parse_xyz(raw).astype(np.float64)


def row_float(row: Dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except Exception:
        return float("nan")


def load_library(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"empty grasp library: {path}")
    return rows


def lookup_target_from_frame(node: GripperAirTest, frame: str) -> np.ndarray:
    tf = node.tf_buffer.lookup_transform(
        "base_link",
        frame,
        rclpy.time.Time(),
        timeout=Duration(seconds=0.5),
    )
    return np.array([tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z], dtype=np.float64)


def orientation_to_quat_wxyz(label: str, current_quat_wxyz: Optional[np.ndarray]) -> np.ndarray:
    label = (label or "").strip()
    if label == "current":
        if current_quat_wxyz is None:
            raise RuntimeError("library selected orientation=current but current quaternion is unavailable")
        return current_quat_wxyz.astype(np.float32)
    if label.startswith("r0_p") and "_y" in label:
        pitch_s, yaw_s = label[len("r0_p") :].split("_y", 1)
        pitch = math.radians(float(pitch_s))
        yaw = math.radians(float(yaw_s))
        return quat_from_rpy(0.0, pitch, yaw)
    raise RuntimeError(f"cannot parse library orientation label: {label!r}")


def choose_candidate(
    rows: Sequence[Dict[str, str]],
    target_xyz: np.ndarray,
    safety_order: Sequence[str],
    close_value: float,
) -> Tuple[Dict[str, str], float]:
    best_row: Optional[Dict[str, str]] = None
    best_dist = float("inf")
    for safety in safety_order:
        class_rows = [r for r in rows if r.get("safety_class") == safety]
        if not class_rows:
            continue
        close_rows = [r for r in class_rows if abs(row_float(r, "close_value") - close_value) < 1e-6]
        if not close_rows:
            close_rows = sorted(class_rows, key=lambda r: abs(row_float(r, "close_value") - close_value))[: len(class_rows)]
        for row in close_rows:
            p = np.array([row_float(row, "target_x"), row_float(row, "target_y"), row_float(row, "target_z")], dtype=np.float64)
            if np.any(np.isnan(p)):
                continue
            dist = float(np.linalg.norm(p - target_xyz))
            if dist < best_dist:
                best_row = row
                best_dist = dist
        if best_row is not None:
            break
    if best_row is None:
        raise RuntimeError("no candidate found in requested safety classes")
    return best_row, best_dist


def xyz_from_row(row: Dict[str, str], prefix: str) -> np.ndarray:
    return np.array([row_float(row, f"{prefix}_x"), row_float(row, f"{prefix}_y"), row_float(row, f"{prefix}_z")], dtype=np.float64)


def current_tcp_quat_wxyz(node: GripperAirTest) -> np.ndarray:
    tf = node.tf_buffer.lookup_transform(
        "base_link",
        "right_tcp",
        rclpy.time.Time(),
        timeout=Duration(seconds=0.5),
    )
    q = tf.transform.rotation
    return np.array([q.w, q.x, q.y, q.z], dtype=np.float32)


def write_selection_report(out_dir: str, row: Dict[str, str], target_xyz: np.ndarray, distance: float, status: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "selected_grasp.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "status",
            "requested_x",
            "requested_y",
            "requested_z",
            "selected_distance_m",
            *row.keys(),
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "status": status,
                "requested_x": float(target_xyz[0]),
                "requested_y": float(target_xyz[1]),
                "requested_z": float(target_xyz[2]),
                "selected_distance_m": float(distance),
                **row,
            }
        )
    return path


def write_execution_report(out_dir: str, rows: List[Dict[str, object]]) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "grasp_execution_report.csv")
    if not rows:
        return path
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def publish_material_command(node: GripperAirTest, publisher, command: str, wait_s: float) -> None:
    msg = String()
    msg.data = command
    deadline = time.time() + max(0.1, float(wait_s))
    while rclpy.ok() and time.time() < deadline:
        publisher.publish(msg)
        rclpy.spin_once(node, timeout_sec=0.05)
        time.sleep(0.05)


def record_stage(node: GripperAirTest, stage: str, target_xyz: np.ndarray, selected: Dict[str, str]) -> Dict[str, object]:
    actual = right_tcp_position(node)
    return {
        "stage": stage,
        "grasp_id": selected.get("grasp_id", ""),
        "target_x": float(target_xyz[0]),
        "target_y": float(target_xyz[1]),
        "target_z": float(target_xyz[2]),
        "actual_tcp_x": float(actual[0]),
        "actual_tcp_y": float(actual[1]),
        "actual_tcp_z": float(actual[2]),
        "tcp_error_m": float(np.linalg.norm(actual - target_xyz)),
        "actual_gripper": float(node.current_grip()),
        "finger_distance_m": float(node.finger_distance()),
        "max_gripper_speed_rad_s": float(node.max_grip_speed),
        "max_arm_drift_rad": float(node.max_arm_drift),
        "status": "abort" if node.abort_reason else "ok",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library-csv", default="/home/u-zhuang/ros2_ws/src/isaac/isaac_sim_2026/grasp_library_current_pose/right_arm_grasp_library.csv")
    parser.add_argument("--robot-cfg-path", default="/home/u-zhuang/ros2_ws/src/isaac/isaac_sim_2026/config/turingstack_x1_2_right_tcp.yml")
    parser.add_argument("--out-dir", default="/home/u-zhuang/ros2_ws/src/isaac/isaac_sim_2026/grasp_execution_results")
    parser.add_argument("--target-xyz", type=parse_optional_xyz, default=None, help="desired right_tcp grasp point in base_link")
    parser.add_argument("--target-frame", default="", help="optional TF frame whose base_link position is the desired right_tcp grasp point")
    parser.add_argument("--object-bottom-xyz", type=parse_optional_xyz, default=None, help="optional object bottom; target z = bottom + height*ratio")
    parser.add_argument("--object-height", type=float, default=0.12)
    parser.add_argument("--grasp-height-ratio", type=float, default=0.60)
    parser.add_argument("--object-lateral-offset", type=float, default=0.0)
    parser.add_argument("--safety-order", type=parse_safety, default=parse_safety("core,usable,edge"))
    parser.add_argument("--close-value", type=float, default=-1.25)
    parser.add_argument("--max-distance", type=float, default=0.06)
    parser.add_argument("--allow-far-candidate", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--open-after-retreat", action="store_true")
    parser.add_argument("--prepare-material-target", default="", help="optional material_task target to prepare before executing")
    parser.add_argument("--dynamic-after-close", action="store_true", help="set prepared material dynamic after gripper close")
    parser.add_argument("--material-command-wait", type=float, default=0.6)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--num-seeds", type=int, default=192)
    parser.add_argument("--position-threshold", type=float, default=0.02)
    parser.add_argument("--rotation-threshold", type=float, default=0.15)
    parser.add_argument("--joint-limit-margin", type=float, default=0.05)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--arm-max-step", type=float, default=0.001)
    parser.add_argument("--arm-min-steps", type=int, default=800)
    parser.add_argument("--gripper-max-step", type=float, default=0.0004)
    parser.add_argument("--gripper-min-steps", type=int, default=260)
    parser.add_argument("--dwell", type=float, default=1.0)
    parser.add_argument("--gripper-limit-margin", type=float, default=0.0)
    parser.add_argument("--abort-gripper-step", type=float, default=0.02)
    parser.add_argument("--abort-gripper-speed", type=float, default=0.35)
    parser.add_argument("--abort-speed-spike-count", type=int, default=3)
    parser.add_argument("--min-velocity-dt", type=float, default=0.01)
    parser.add_argument("--abort-arm-drift", type=float, default=0.15)
    args = parser.parse_args()
    args.pregrasp_q = "current"

    rows = load_library(args.library_csv)

    rclpy.init()
    node = GripperAirTest(args)
    material_pub = node.create_publisher(String, "/material_task/command", 10)
    report_rows: List[Dict[str, object]] = []
    try:
        node.wait_ready(args.timeout)
        if args.prepare_material_target:
            publish_material_command(
                node,
                material_pub,
                f"prepare_object {args.prepare_material_target}",
                args.material_command_wait,
            )
        if args.target_xyz is not None:
            target_xyz = args.target_xyz.astype(np.float64)
        elif args.target_frame:
            target_xyz = lookup_target_from_frame(node, args.target_frame)
        elif args.object_bottom_xyz is not None:
            target_xyz = args.object_bottom_xyz.astype(np.float64).copy()
            target_xyz[1] += float(args.object_lateral_offset)
            target_xyz[2] += float(args.object_height) * float(args.grasp_height_ratio)
        else:
            raise RuntimeError("provide --target-xyz, --target-frame, or --object-bottom-xyz")

        selected, distance = choose_candidate(rows, target_xyz, args.safety_order, args.close_value)
        status = "selected"
        if distance > args.max_distance and not args.allow_far_candidate:
            status = "too_far"
        selection_path = write_selection_report(args.out_dir, selected, target_xyz, distance, status)

        print(f"requested target: {[round(float(x), 4) for x in target_xyz.tolist()]}")
        print(
            f"selected {selected.get('grasp_id')} safety={selected.get('safety_class')} "
            f"close={selected.get('close_value')} distance={distance:.4f}m"
        )
        print(f"selection report: {selection_path}")

        if status == "too_far":
            print(f"candidate distance exceeds --max-distance {args.max_distance:.3f}m; add --allow-far-candidate to execute")
            return 2
        if not args.execute:
            print("selection-only mode; add --execute to move the robot")
            return 0

        pre_xyz = xyz_from_row(selected, "pre_grasp")
        grasp_xyz = xyz_from_row(selected, "target")
        lift_xyz = xyz_from_row(selected, "lift")
        current_quat = current_tcp_quat_wxyz(node)
        quat_wxyz = orientation_to_quat_wxyz(selected.get("orientation", ""), current_quat)
        q_start = np.array(node.current_right_q(), dtype=np.float32)
        joint_names, q_solutions, _, _ = solve_waypoint_ik(
            args.robot_cfg_path,
            q_start,
            [pre_xyz, grasp_xyz, lift_xyz],
            quat_wxyz,
            args.num_seeds,
            args.position_threshold,
            args.rotation_threshold,
        )
        for q in q_solutions:
            validate_q(q, args.joint_limit_margin)
        node.get_logger().info(f"selected_grasp_id={selected.get('grasp_id')} joint_names={joint_names}")

        open_value = row_float(selected, "open_value")
        close_value = row_float(selected, "close_value")
        if not node.move_gripper_to(open_value, "library_open"):
            raise RuntimeError(node.abort_reason or "open failed")
        dwell_hold(node, args.dwell)
        report_rows.append(record_stage(node, "open", right_tcp_position(node), selected))

        if not node.move_arm_to(q_solutions[0].tolist()):
            raise RuntimeError(node.abort_reason or "pre_grasp failed")
        dwell_hold(node, args.dwell)
        report_rows.append(record_stage(node, "pre_grasp", pre_xyz, selected))

        if not node.move_arm_to(q_solutions[1].tolist()):
            raise RuntimeError(node.abort_reason or "target failed")
        dwell_hold(node, args.dwell)
        report_rows.append(record_stage(node, "target_open", grasp_xyz, selected))

        if not node.move_gripper_to(close_value, "library_close"):
            raise RuntimeError(node.abort_reason or "close failed")
        dwell_hold(node, args.dwell)
        report_rows.append(record_stage(node, "closed", grasp_xyz, selected))

        if args.dynamic_after_close and args.prepare_material_target:
            publish_material_command(
                node,
                material_pub,
                f"object_dynamic {args.prepare_material_target}",
                args.material_command_wait,
            )

        if not node.move_arm_to(q_solutions[2].tolist()):
            raise RuntimeError(node.abort_reason or "lift failed")
        dwell_hold(node, args.dwell)
        report_rows.append(record_stage(node, "lift", lift_xyz, selected))

        if not node.move_arm_to(q_solutions[0].tolist()):
            raise RuntimeError(node.abort_reason or "retreat failed")
        dwell_hold(node, args.dwell)
        report_rows.append(record_stage(node, "retreat", pre_xyz, selected))

        if args.open_after_retreat:
            node.move_gripper_to(open_value, "library_open_after_retreat")
            dwell_hold(node, args.dwell)
            report_rows.append(record_stage(node, "open_after_retreat", pre_xyz, selected))

        exec_path = write_execution_report(args.out_dir, report_rows)
        print(f"execution report: {exec_path}")
        return 0 if not node.abort_reason else 3
    except (RuntimeError, TransformException) as exc:
        print(f"ERROR: {exc}")
        if report_rows:
            exec_path = write_execution_report(args.out_dir, report_rows)
            print(f"partial execution report: {exec_path}")
        return 1
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
