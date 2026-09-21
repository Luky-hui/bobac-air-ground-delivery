#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert a cuRobo reachability scan into a callable right-arm grasp library."""

from __future__ import annotations

import argparse
import csv
import math
import os
from typing import Dict, Iterable, List


def parse_xyz(raw: str) -> List[float]:
    vals = [float(x) for x in raw.replace(",", " ").split()]
    if len(vals) != 3:
        raise argparse.ArgumentTypeError("expected 3 numbers")
    return vals


def dist_to_box_margin(x: float, y: float, z: float, core: Dict[str, float]) -> float:
    return min(
        x - core["x_min"],
        core["x_max"] - x,
        y - core["y_min"],
        core["y_max"] - y,
        z - core["z_min"],
        core["z_max"] - z,
    )


def row_float(row: Dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except ValueError:
        return float("nan")


def safety_class(x: float, y: float, z: float, core: Dict[str, float], usable: Dict[str, float]) -> str:
    core_margin = dist_to_box_margin(x, y, z, core)
    if core_margin >= 0.025:
        return "core"
    usable_margin = dist_to_box_margin(x, y, z, usable)
    if usable_margin >= 0.0:
        return "usable"
    return "edge"


def recommended_close_values(finger_diameter: float) -> List[float]:
    # Bobac's Lebai gripper is stable around -0.85 rad when only the two main
    # gripper joints are commanded in the Isaac articulation.
    # articulation.  Deeper values such as -1.0 over-close the mimic chain and
    # can destabilize PhysX before the lift validation stage.
    return [-0.85]


def make_library(args: argparse.Namespace) -> List[Dict[str, object]]:
    core = {
        "x_min": args.core_x[0],
        "x_max": args.core_x[1],
        "y_min": args.core_y[0],
        "y_max": args.core_y[1],
        "z_min": args.core_z[0],
        "z_max": args.core_z[1],
    }
    usable = {
        "x_min": args.usable_x[0],
        "x_max": args.usable_x[1],
        "y_min": args.usable_y[0],
        "y_max": args.usable_y[1],
        "z_min": args.usable_z[0],
        "z_max": args.usable_z[1],
    }
    approach = args.approach_axis
    close_values = recommended_close_values(args.object_diameter)
    out: List[Dict[str, object]] = []
    with open(args.scan_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        q_fields = [name for name in (reader.fieldnames or []) if name.startswith("q_")]
        for row in reader:
            if row.get("reachable") != "1":
                continue
            x = row_float(row, "x")
            y = row_float(row, "y")
            z = row_float(row, "z")
            if math.isnan(x) or math.isnan(y) or math.isnan(z):
                continue
            cls = safety_class(x, y, z, core, usable)
            if cls == "edge" and not args.include_edge:
                continue
            pre = [
                x - args.pregrasp_distance * approach[0],
                y - args.pregrasp_distance * approach[1],
                z - args.pregrasp_distance * approach[2],
            ]
            lift = [x, y, z + args.lift_height]
            for close in close_values:
                grasp_id = f"g_{len(out) + 1:05d}"
                library_row: Dict[str, object] = {
                    "grasp_id": grasp_id,
                    "safety_class": cls,
                    "target_x": x,
                    "target_y": y,
                    "target_z": z,
                    "pre_grasp_x": pre[0],
                    "pre_grasp_y": pre[1],
                    "pre_grasp_z": pre[2],
                    "lift_x": lift[0],
                    "lift_y": lift[1],
                    "lift_z": lift[2],
                    "approach_axis_x": approach[0],
                    "approach_axis_y": approach[1],
                    "approach_axis_z": approach[2],
                    "pregrasp_distance_m": args.pregrasp_distance,
                    "lift_height_m": args.lift_height,
                    "orientation": row.get("orientation", ""),
                    "close_value": close,
                    "open_value": args.open_value,
                    "object_diameter_m": args.object_diameter if args.object_diameter > 0 else "",
                    "position_error_m": row.get("position_error_m", ""),
                    "rotation_error": row.get("rotation_error", ""),
                    "source_scan": args.scan_csv,
                }
                for field in q_fields:
                    library_row[field] = row.get(field, "")
                out.append(library_row)
    return out


def write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        raise RuntimeError("no grasp rows to write")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_yaml(path: str, rows: List[Dict[str, object]], max_rows: int) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    selected = rows[:max_rows]
    with open(path, "w", encoding="utf-8") as f:
        f.write("right_arm_grasp_library:\n")
        f.write(f"  count_csv: {len(rows)}\n")
        f.write("  convention:\n")
        f.write("    frame: base_link\n")
        f.write("    ee_link: right_tcp\n")
        f.write("    approach: pre_grasp -> target along approach_axis\n")
        f.write("  grasps_preview:\n")
        for row in selected:
            f.write(f"    - grasp_id: {row['grasp_id']}\n")
            f.write(f"      safety_class: {row['safety_class']}\n")
            f.write(f"      target_xyz: [{row['target_x']:.4f}, {row['target_y']:.4f}, {row['target_z']:.4f}]\n")
            f.write(f"      pre_grasp_xyz: [{row['pre_grasp_x']:.4f}, {row['pre_grasp_y']:.4f}, {row['pre_grasp_z']:.4f}]\n")
            f.write(f"      lift_xyz: [{row['lift_x']:.4f}, {row['lift_y']:.4f}, {row['lift_z']:.4f}]\n")
            f.write(f"      orientation: \"{row['orientation']}\"\n")
            f.write(f"      close_value: {row['close_value']}\n")
            q_keys = [key for key in row.keys() if key.startswith("q_")]
            q_text = ", ".join(f"{float(row[key]):.5f}" for key in q_keys)
            f.write(f"      q: [{q_text}]\n")


def write_summary(path: str, rows: List[Dict[str, object]]) -> None:
    counts: Dict[str, int] = {}
    close_counts: Dict[str, int] = {}
    for row in rows:
        counts[str(row["safety_class"])] = counts.get(str(row["safety_class"]), 0) + 1
        close_counts[str(row["close_value"])] = close_counts.get(str(row["close_value"]), 0) + 1
    with open(path, "w", encoding="utf-8") as f:
        f.write("right_arm grasp library summary\n")
        f.write(f"total_grasps: {len(rows)}\n")
        f.write(f"by_safety_class: {counts}\n")
        f.write(f"by_close_value: {close_counts}\n")
        f.write("usage: choose safety_class=core first, then usable, avoid edge unless needed.\n")
        f.write("frame: base_link; ee_link: right_tcp\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-csv", required=True)
    parser.add_argument("--out-dir", default="/home/u-zhuang/ros2_ws/src/isaac/isaac_sim_2026/grasp_library_current_pose")
    parser.add_argument("--object-diameter", type=float, default=0.03)
    parser.add_argument("--approach-axis", type=parse_xyz, default=parse_xyz("0,0,-1"))
    parser.add_argument("--pregrasp-distance", type=float, default=0.10)
    parser.add_argument("--lift-height", type=float, default=0.03)
    parser.add_argument("--open-value", type=float, default=0.25)
    parser.add_argument("--core-x", type=parse_xyz_pair, default=None)
    parser.add_argument("--core-y", type=parse_xyz_pair, default=None)
    parser.add_argument("--core-z", type=parse_xyz_pair, default=None)
    parser.add_argument("--usable-x", type=parse_xyz_pair, default=None)
    parser.add_argument("--usable-y", type=parse_xyz_pair, default=None)
    parser.add_argument("--usable-z", type=parse_xyz_pair, default=None)
    parser.add_argument("--include-edge", action="store_true")
    parser.add_argument("--yaml-preview", type=int, default=50)
    args = parser.parse_args()

    args.core_x = args.core_x or [0.40, 0.55]
    args.core_y = args.core_y or [-0.45, -0.30]
    args.core_z = args.core_z or [0.70, 0.90]
    args.usable_x = args.usable_x or [0.35, 0.60]
    args.usable_y = args.usable_y or [-0.50, -0.20]
    args.usable_z = args.usable_z or [0.60, 1.00]

    rows = make_library(args)
    csv_path = os.path.join(args.out_dir, "right_arm_grasp_library.csv")
    yaml_path = os.path.join(args.out_dir, "right_arm_grasp_library_preview.yaml")
    summary_path = os.path.join(args.out_dir, "right_arm_grasp_library_summary.txt")
    write_csv(csv_path, rows)
    write_yaml(yaml_path, rows, args.yaml_preview)
    write_summary(summary_path, rows)
    print(f"library rows: {len(rows)}")
    print(f"csv: {csv_path}")
    print(f"yaml preview: {yaml_path}")
    print(f"summary: {summary_path}")
    return 0


def parse_xyz_pair(raw: str) -> List[float]:
    vals = [float(x) for x in raw.replace(",", " ").split()]
    if len(vals) != 2:
        raise argparse.ArgumentTypeError("expected 2 numbers")
    return vals


if __name__ == "__main__":
    raise SystemExit(main())
