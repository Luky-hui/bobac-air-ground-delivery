#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a first-pass grasp affordance map for an upright pen/cylinder.

Default mode is plan-only: write the 30-trial matrix without moving the robot.
Use --execute to run the trials in Isaac Sim.

The sweep separates:
1. Robot workspace coordinates: object bottom pose in base_link.
2. Object-local grasp parameters: diameter, height ratio, lateral offset, yaw,
   and final gripper close value.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import rclpy

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from curobo_pregrasp_approach_dry_run import parse_xyz, solve_waypoint_ik  # noqa: E402
from gripper_air_open_close_test import GripperAirTest  # noqa: E402
from pen_cylinder_contact_test import (  # noqa: E402
    dwell_hold,
    maybe_lookup_object_pose,
    right_tcp_position,
    validate_q,
)
from pen_grasp_alignment_dry_run import parse_axis, parse_rpy_deg, quat_from_rpy  # noqa: E402


@dataclass
class Trial:
    trial_id: int
    object_type: str
    object_diameter: float
    object_height: float
    object_bottom_xyz: np.ndarray
    grasp_height_ratio: float
    lateral_offset_m: float
    yaw_offset_deg: float
    close_value: float
    tcp_offset_xyz: np.ndarray
    grasp_xyz: np.ndarray
    pregrasp_xyz: np.ndarray
    lift_xyz: np.ndarray


def parse_float_list(raw: str) -> List[float]:
    vals = [float(x) for x in raw.replace(",", " ").split()]
    if not vals:
        raise argparse.ArgumentTypeError("list must not be empty")
    return vals


def parse_int_list(raw: str) -> List[int]:
    vals = [int(x) for x in raw.replace(",", " ").split()]
    if not vals:
        raise argparse.ArgumentTypeError("list must not be empty")
    return vals


def vec_to_str(v: Sequence[float]) -> str:
    return " ".join(f"{float(x):.6f}" for x in v)


def pose_delta(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return float("nan")
    return float(np.linalg.norm(b - a))


def z_delta(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return float("nan")
    return float(b[2] - a[2])


def make_trials(args: argparse.Namespace) -> List[Trial]:
    bottom_xyz = args.object_bottom_xyz.astype(np.float64)
    approach_axis = args.approach_axis.astype(np.float64)
    trials: List[Trial] = []
    trial_id = 1
    for diameter, height, ratio, offset, yaw_deg, close_value in itertools.product(
        args.object_diameters,
        args.object_heights,
        args.grasp_height_ratios,
        args.lateral_offsets,
        args.yaw_offsets_deg,
        args.close_values,
    ):
        tcp_offset = np.array([0.0, float(offset), float(ratio) * float(height)], dtype=np.float64)
        grasp_xyz = bottom_xyz + tcp_offset
        pregrasp_xyz = grasp_xyz - float(args.approach_distance) * approach_axis
        lift_xyz = grasp_xyz + np.array([0.0, 0.0, float(args.lift_height)], dtype=np.float64)
        trials.append(
            Trial(
                trial_id=trial_id,
                object_type=args.object_type,
                object_diameter=float(diameter),
                object_height=float(height),
                object_bottom_xyz=bottom_xyz.copy(),
                grasp_height_ratio=float(ratio),
                lateral_offset_m=float(offset),
                yaw_offset_deg=float(yaw_deg),
                close_value=float(close_value),
                tcp_offset_xyz=tcp_offset,
                grasp_xyz=grasp_xyz,
                pregrasp_xyz=pregrasp_xyz,
                lift_xyz=lift_xyz,
            )
        )
        trial_id += 1
    return trials


def empty_result(trial: Trial, status: str = "planned", note: str = "") -> Dict[str, object]:
    return {
        "trial_id": trial.trial_id,
        "object_type": trial.object_type,
        "object_diameter": trial.object_diameter,
        "object_height": trial.object_height,
        "object_pose_base_x": float(trial.object_bottom_xyz[0]),
        "object_pose_base_y": float(trial.object_bottom_xyz[1]),
        "object_pose_base_z": float(trial.object_bottom_xyz[2]),
        "grasp_height_ratio": trial.grasp_height_ratio,
        "tcp_offset_x": float(trial.tcp_offset_xyz[0]),
        "tcp_offset_y": float(trial.tcp_offset_xyz[1]),
        "tcp_offset_z": float(trial.tcp_offset_xyz[2]),
        "lateral_offset_m": trial.lateral_offset_m,
        "yaw_offset_deg": trial.yaw_offset_deg,
        "pre_grasp_pose": vec_to_str(trial.pregrasp_xyz),
        "grasp_pose": vec_to_str(trial.grasp_xyz),
        "lift_pose": vec_to_str(trial.lift_xyz),
        "close_value": trial.close_value,
        "tcp_error": float("nan"),
        "finger_distance_m": float("nan"),
        "finger_clearance_m": float("nan"),
        "max_joint_speed": float("nan"),
        "max_step_jump": float("nan"),
        "max_gripper_speed": float("nan"),
        "object_moved_before_close": "unknown",
        "object_lifted": "unknown",
        "object_dropped": "unknown",
        "object_tilt_deg": float("nan"),
        "object_move_before_close_m": float("nan"),
        "object_lift_delta_m": float("nan"),
        "object_retreat_delta_m": float("nan"),
        "manual_contact": "unrecorded",
        "manual_knockdown": "unrecorded",
        "manual_lift_success": "unrecorded",
        "manual_stable": "unrecorded",
        "manual_note": "",
        "success_score": -1,
        "success": "unknown",
        "status": status,
        "note": note,
    }


def stage_row(trial: Trial, stage: str, target_xyz: np.ndarray, node: GripperAirTest, object_frame: str) -> Dict[str, object]:
    tcp = right_tcp_position(node)
    obj, tilt = maybe_lookup_object_pose(node, object_frame)
    finger_distance = node.finger_distance()
    return {
        "trial_id": trial.trial_id,
        "stage": stage,
        "target_x": float(target_xyz[0]),
        "target_y": float(target_xyz[1]),
        "target_z": float(target_xyz[2]),
        "actual_tcp_x": float(tcp[0]),
        "actual_tcp_y": float(tcp[1]),
        "actual_tcp_z": float(tcp[2]),
        "tcp_error_m": float(np.linalg.norm(tcp - target_xyz)),
        "actual_gripper": float(node.current_grip()),
        "finger_distance_m": float(finger_distance),
        "object_x": float(obj[0]) if obj is not None else float("nan"),
        "object_y": float(obj[1]) if obj is not None else float("nan"),
        "object_z": float(obj[2]) if obj is not None else float("nan"),
        "object_tilt_deg": float(tilt),
        "max_arm_step_rad": float(node.max_arm_step),
        "max_gripper_step_rad": float(node.max_grip_step),
        "max_gripper_speed_rad_s": float(node.max_grip_speed),
        "status": "abort" if node.abort_reason else "ok",
    }


def score_trial(
    aborted: bool,
    object_before: Optional[np.ndarray],
    object_after_approach: Optional[np.ndarray],
    object_after_lift: Optional[np.ndarray],
    object_after_retreat: Optional[np.ndarray],
    lift_height: float,
) -> Tuple[int, str, str, str, str]:
    if aborted:
        return 0, "no", "unknown", "unknown", "aborted"
    if object_before is None:
        return -1, "unknown", "unknown", "unknown", "no object frame"

    move_before_close = pose_delta(object_before, object_after_approach)
    lifted_delta = z_delta(object_before, object_after_lift)
    retreat_delta = z_delta(object_before, object_after_retreat)
    moved_before = "yes" if not math.isnan(move_before_close) and move_before_close > 0.005 else "no"
    lifted = "yes" if not math.isnan(lifted_delta) and lifted_delta > max(0.012, 0.4 * lift_height) else "no"
    stable_retreat = "yes" if not math.isnan(retreat_delta) and retreat_delta > max(0.012, 0.35 * lift_height) else "no"

    if moved_before == "yes":
        return 1, moved_before, lifted, "unknown", "object moved before close"
    if lifted == "no":
        return 2, moved_before, lifted, "unknown", "closed but did not lift"
    if stable_retreat == "no":
        return 3, moved_before, lifted, "yes", "lifted but not stable after retreat"
    return 5, moved_before, lifted, "no", "stable lift and retreat"


def write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_best_yaml(path: str, rows: List[Dict[str, object]], top_k: int) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ranked = sorted(
        [r for r in rows if int(r.get("success_score", -1)) >= 4],
        key=lambda r: (int(r.get("success_score", -1)), -abs(float(r.get("tcp_offset_y", 0.0)))),
        reverse=True,
    )[:top_k]
    with open(path, "w", encoding="utf-8") as f:
        f.write("best_grasps_pen:\n")
        if not ranked:
            f.write("  []\n")
            return
        for row in ranked:
            f.write(f"  - trial_id: {row['trial_id']}\n")
            f.write(f"    success_score: {row['success_score']}\n")
            f.write(f"    object_diameter: {row['object_diameter']}\n")
            f.write(f"    object_height: {row['object_height']}\n")
            f.write(f"    grasp_height_ratio: {row['grasp_height_ratio']}\n")
            f.write(f"    tcp_offset_y: {row['tcp_offset_y']}\n")
            f.write(f"    yaw_offset_deg: {row['yaw_offset_deg']}\n")
            f.write(f"    close_value: {row['close_value']}\n")
            f.write(f"    grasp_pose: \"{row['grasp_pose']}\"\n")
            f.write(f"    note: \"{row['note']}\"\n")


def interactive_label(row: Dict[str, object]) -> None:
    print("\nManual label for trial", row["trial_id"])
    raw = input("score/contact/knockdown/lift/stable/note (blank to skip): ").strip()
    if not raw:
        return
    parts = raw.split("/", 5)
    if len(parts) >= 1 and parts[0]:
        row["success_score"] = int(parts[0])
        row["success"] = "yes" if int(parts[0]) >= 4 else "no"
    labels = ["manual_contact", "manual_knockdown", "manual_lift_success", "manual_stable", "manual_note"]
    for label, value in zip(labels, parts[1:]):
        row[label] = value


def execute_trial(args: argparse.Namespace, node: GripperAirTest, trial: Trial) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    result = empty_result(trial, status="running")
    stages: List[Dict[str, object]] = []
    object_before, object_tilt_before = maybe_lookup_object_pose(node, args.object_frame)
    try:
        q_start = np.array(node.current_right_q(), dtype=np.float32)
        tcp_rpy = args.tcp_rpy_deg.copy()
        tcp_rpy[2] += math.radians(trial.yaw_offset_deg)
        quat_wxyz = quat_from_rpy(float(tcp_rpy[0]), float(tcp_rpy[1]), float(tcp_rpy[2]))
        joint_names, q_solutions, ik_pos_err, ik_rot_err = solve_waypoint_ik(
            args.robot_cfg_path,
            q_start,
            [trial.pregrasp_xyz, trial.grasp_xyz, trial.lift_xyz],
            quat_wxyz,
            args.num_seeds,
            args.position_threshold,
            args.rotation_threshold,
        )
        for q in q_solutions:
            validate_q(q, args.joint_limit_margin)

        node.get_logger().info(
            f"trial {trial.trial_id}: diameter={trial.object_diameter:.3f}, "
            f"height={trial.object_height:.3f}, ratio={trial.grasp_height_ratio:.2f}, "
            f"offset={trial.lateral_offset_m:+.3f}, yaw={trial.yaw_offset_deg:+.1f}, "
            f"close={trial.close_value:+.2f}"
        )
        node.get_logger().info(f"curobo_joint_names={joint_names}")

        if not node.move_gripper_to(args.open_value, f"trial_{trial.trial_id}_open"):
            raise RuntimeError(node.abort_reason or "open failed")
        dwell_hold(node, args.dwell)
        stages.append(stage_row(trial, "open", right_tcp_position(node), node, args.object_frame))

        if not node.move_arm_to(q_solutions[0].tolist()):
            raise RuntimeError(node.abort_reason or "pregrasp failed")
        dwell_hold(node, args.dwell)
        stages.append(stage_row(trial, "pre_grasp", trial.pregrasp_xyz, node, args.object_frame))

        if not node.move_arm_to(q_solutions[1].tolist()):
            raise RuntimeError(node.abort_reason or "approach failed")
        dwell_hold(node, args.dwell)
        stages.append(stage_row(trial, "approach_open", trial.grasp_xyz, node, args.object_frame))
        object_after_approach, _ = maybe_lookup_object_pose(node, args.object_frame)

        if not node.move_gripper_to(trial.close_value, f"trial_{trial.trial_id}_close"):
            raise RuntimeError(node.abort_reason or "close failed")
        dwell_hold(node, args.dwell)
        closed_row = stage_row(trial, "closed", trial.grasp_xyz, node, args.object_frame)
        stages.append(closed_row)

        if args.lift_height > 1e-6:
            if not node.move_arm_to(q_solutions[2].tolist()):
                raise RuntimeError(node.abort_reason or "lift failed")
            dwell_hold(node, args.dwell)
            stages.append(stage_row(trial, "lift", trial.lift_xyz, node, args.object_frame))
        object_after_lift, object_tilt_after = maybe_lookup_object_pose(node, args.object_frame)

        if node.move_arm_to(q_solutions[0].tolist()):
            dwell_hold(node, args.dwell)
            stages.append(stage_row(trial, "retreat", trial.pregrasp_xyz, node, args.object_frame))
        object_after_retreat, _ = maybe_lookup_object_pose(node, args.object_frame)

        node.move_gripper_to(args.open_value, f"trial_{trial.trial_id}_open_after")
        dwell_hold(node, args.dwell)
        stages.append(stage_row(trial, "open_after_retreat", trial.pregrasp_xyz, node, args.object_frame))

        score, moved_before, lifted, dropped, note = score_trial(
            bool(node.abort_reason),
            object_before,
            object_after_approach,
            object_after_lift,
            object_after_retreat,
            args.lift_height,
        )
        result.update(
            {
                "tcp_error": float(closed_row["tcp_error_m"]),
                "finger_distance_m": float(closed_row["finger_distance_m"]),
                "finger_clearance_m": float(closed_row["finger_distance_m"] - trial.object_diameter),
                "max_joint_speed": float("nan"),
                "max_step_jump": float(node.max_arm_step),
                "max_gripper_speed": float(node.max_grip_speed),
                "object_moved_before_close": moved_before,
                "object_lifted": lifted,
                "object_dropped": dropped,
                "object_tilt_deg": float(object_tilt_after if object_after_lift is not None else object_tilt_before),
                "object_move_before_close_m": pose_delta(object_before, object_after_approach),
                "object_lift_delta_m": z_delta(object_before, object_after_lift),
                "object_retreat_delta_m": z_delta(object_before, object_after_retreat),
                "success_score": score,
                "success": "yes" if score >= 4 else ("unknown" if score < 0 else "no"),
                "status": "abort" if node.abort_reason else "ok",
                "note": note,
            }
        )
        if args.interactive_label:
            interactive_label(result)
        return result, stages
    except Exception as exc:
        result.update({"status": "error", "success_score": 0, "success": "no", "note": str(exc)})
        try:
            node.publish_command(node.current_right_q(), node.current_grip())
        except Exception:
            pass
        return result, stages


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-cfg-path", default="/home/u-zhuang/ros2_ws/src/isaac/isaac_sim_2026/config/turingstack_x1_2_right_tcp.yml")
    parser.add_argument("--out-dir", default="/home/u-zhuang/ros2_ws/src/isaac/isaac_sim_2026/grasp_affordance_results")
    parser.add_argument("--execute", action="store_true", help="actually move the robot; default only writes the plan")
    parser.add_argument("--trial-ids", type=parse_int_list, default=None, help="optional subset to execute, e.g. 7,8,9")
    parser.add_argument("--object-type", default="upright_pen")
    parser.add_argument("--object-bottom-xyz", type=parse_xyz, default=parse_xyz("0.45,-0.35,0.70"))
    parser.add_argument("--object-diameters", type=parse_float_list, default=parse_float_list("0.030,0.035"))
    parser.add_argument("--object-heights", type=parse_float_list, default=parse_float_list("0.12"))
    parser.add_argument("--grasp-height-ratios", type=parse_float_list, default=parse_float_list("0.60"))
    parser.add_argument("--lateral-offsets", type=parse_float_list, default=parse_float_list("-0.010,-0.005,0,0.005,0.010"))
    parser.add_argument("--yaw-offsets-deg", type=parse_float_list, default=parse_float_list("0"))
    parser.add_argument("--close-values", type=parse_float_list, default=parse_float_list("-0.50,-0.60,-0.65"))
    parser.add_argument("--tcp-rpy-deg", type=parse_rpy_deg, default=parse_rpy_deg("0,0,0"))
    parser.add_argument("--approach-axis", type=parse_axis, default=parse_axis("0,0,-1"))
    parser.add_argument("--approach-distance", type=float, default=0.10)
    parser.add_argument("--lift-height", type=float, default=0.03)
    parser.add_argument("--open-value", type=float, default=0.25)
    parser.add_argument("--object-frame", default="", help="optional TF frame for automatic object pose scoring")
    parser.add_argument("--interactive-label", action="store_true", help="ask for manual score after each executed trial")
    parser.add_argument("--pause-between-trials", type=float, default=0.0)
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
    parser.add_argument("--dwell", type=float, default=1.2)
    parser.add_argument("--gripper-limit-margin", type=float, default=0.02)
    parser.add_argument("--abort-gripper-step", type=float, default=0.02)
    parser.add_argument("--abort-gripper-speed", type=float, default=0.35)
    parser.add_argument("--abort-speed-spike-count", type=int, default=3)
    parser.add_argument("--min-velocity-dt", type=float, default=0.01)
    parser.add_argument("--abort-arm-drift", type=float, default=0.12)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    args.pregrasp_q = "current"

    trials = make_trials(args)
    if args.trial_ids is not None:
        selected = set(args.trial_ids)
        trials = [trial for trial in trials if trial.trial_id in selected]
    trial_rows = [empty_result(t) for t in trials]
    stage_rows: List[Dict[str, object]] = []
    os.makedirs(args.out_dir, exist_ok=True)

    trial_csv = os.path.join(args.out_dir, "grasp_affordance_trials.csv")
    stage_csv = os.path.join(args.out_dir, "grasp_affordance_stage_log.csv")
    map_csv = os.path.join(args.out_dir, "grasp_affordance_success_map.csv")
    best_yaml = os.path.join(args.out_dir, "best_grasps_pen.yaml")

    if not args.execute:
        write_csv(trial_csv, trial_rows)
        write_csv(map_csv, trial_rows)
        write_best_yaml(best_yaml, trial_rows, args.top_k)
        print(f"plan written: {trial_csv}")
        print(f"trials: {len(trials)}")
        print("add --execute to run the robot")
        return 0

    rclpy.init()
    node = GripperAirTest(args)
    try:
        node.wait_ready(args.timeout)
        trial_rows = []
        for trial in trials:
            result, stages = execute_trial(args, node, trial)
            trial_rows.append(result)
            stage_rows.extend(stages)
            write_csv(trial_csv, trial_rows)
            if stage_rows:
                write_csv(stage_csv, stage_rows)
            write_csv(map_csv, trial_rows)
            write_best_yaml(best_yaml, trial_rows, args.top_k)
            if node.abort_reason:
                break
            if args.pause_between_trials > 0.0:
                time.sleep(args.pause_between_trials)
        return 0 if not node.abort_reason else 3
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
