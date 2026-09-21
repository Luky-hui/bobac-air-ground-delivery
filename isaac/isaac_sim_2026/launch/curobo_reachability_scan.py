#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Offline cuRobo reachability scan for the current Isaac Sim/ROS2 robot state.

This script only subscribes to /joint_states and runs batched IK locally.
It does not publish joint commands or move the robot.
"""

import argparse
import csv
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


DEFAULT_JOINT_NAMES = ["joint1_R", "joint2_R", "joint3_R", "joint4_R", "joint5_R", "joint6_R"]
CUROBO_SRC = "/home/u-zhuang/ros2_ws/src/isaac/isaac_sim_2026/curobo/src"

JOINT_LIMITS = {
    "joint1_R": (-2.87979326579, 2.87979326579),
    "joint2_R": (-0.9599310886, 1.6580627894),
    "joint3_R": (-3.0194196060, 0.0872664626),
    "joint4_R": (-2.8797932658, 2.8797932658),
    "joint5_R": (-0.3490658504, 4.6251225178),
    "joint6_R": (-3.1415926536, 3.1415926536),
    "joint_1": (-3.11, 3.11),
    "joint_2": (-3.11, 2.36),
    "joint_3": (-2.79, 2.53),
    "joint_4": (-3.11, 3.11),
    "joint_5": (-3.11, 3.11),
    "joint_6": (-6.28, 6.28),
}


@dataclass
class ScanRange:
    start: float
    stop: float
    step: float

    def values(self) -> np.ndarray:
        count = int(math.floor((self.stop - self.start) / self.step + 0.5)) + 1
        return np.array([self.start + i * self.step for i in range(count)], dtype=np.float32)


def parse_range(raw: str) -> ScanRange:
    parts = [float(x) for x in raw.split(":")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("range must be start:stop:step")
    if parts[2] <= 0:
        raise argparse.ArgumentTypeError("range step must be positive")
    if parts[1] < parts[0]:
        raise argparse.ArgumentTypeError("range stop must be >= start")
    return ScanRange(parts[0], parts[1], parts[2])


def parse_joint_names(raw: str) -> List[str]:
    names = [item.strip() for item in raw.replace(",", " ").split() if item.strip()]
    if not names:
        raise argparse.ArgumentTypeError("joint names must not be empty")
    return names


def wait_for_joint_state(topic: str, joint_names: Sequence[str], timeout_s: float) -> Dict[str, float]:
    import rclpy
    from sensor_msgs.msg import JointState

    rclpy.init(args=None)
    node = rclpy.create_node("curobo_reachability_scan_reader")
    result: Dict[str, float] = {}

    def cb(msg: JointState):
        nonlocal result
        result = {name: float(pos) for name, pos in zip(msg.name, msg.position)}

    sub = node.create_subscription(JointState, topic, cb, 10)
    deadline = time.time() + timeout_s
    while rclpy.ok() and time.time() < deadline and not result:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_subscription(sub)
    node.destroy_node()
    rclpy.shutdown()
    if not result:
        raise RuntimeError(f"Timed out waiting for {topic}")
    missing = [j for j in joint_names if j not in result]
    if missing:
        raise RuntimeError(f"{topic} is missing arm joints: {missing}")
    return result


def chunked(seq: Sequence[np.ndarray], size: int) -> Iterable[Sequence[np.ndarray]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def make_grid(x_range: ScanRange, y_range: ScanRange, z_range: ScanRange) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs = x_range.values()
    ys = y_range.values()
    zs = z_range.values()
    points = np.array([(x, y, z) for z in zs for y in ys for x in xs], dtype=np.float32)
    return xs, ys, zs, points


def quat_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float32,
    )


def make_orientation_set(mode: str, current_quat: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    mode = mode.strip().lower()
    if mode == "current":
        return [("current", current_quat.astype(np.float32))]
    if mode == "grasp":
        # Candidate gripper-base attitudes. These are intentionally broad; the CSV tells us
        # which one cuRobo actually selected at each reachable point.
        out = [("current", current_quat.astype(np.float32))]
        pitch_values = [math.radians(v) for v in (-90, -60, -30, 0, 30, 60, 90)]
        yaw_values = [math.radians(v) for v in (-90, -45, 0, 45, 90, 135, 180, -135)]
        for p in pitch_values:
            for y in yaw_values:
                out.append((f"r0_p{round(math.degrees(p))}_y{round(math.degrees(y))}", quat_from_rpy(0.0, p, y)))
        return out
    raise ValueError(f"unknown orientation mode: {mode}")


def within_safe_limits(q: np.ndarray, joint_names: Sequence[str], margin: float) -> bool:
    for name, value in zip(joint_names, q.tolist()):
        if name not in JOINT_LIMITS:
            continue
        lo, hi = JOINT_LIMITS[name]
        if value < lo + margin or value > hi - margin:
            return False
    return True


def solve_reachability(args: argparse.Namespace) -> Tuple[dict, List[dict]]:
    import torch

    if CUROBO_SRC not in sys.path:
        sys.path.insert(0, CUROBO_SRC)

    from curobo.geom.sdf.world import CollisionCheckerType
    from curobo.geom.types import WorldConfig
    from curobo.types.base import TensorDeviceType
    from curobo.types.math import Pose
    from curobo.types.robot import JointState as CuJointState
    from curobo.util_file import load_yaml
    from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

    joint_names = list(args.joint_names)
    joint_state = wait_for_joint_state(args.joint_topic, joint_names, args.joint_timeout)
    q_current = np.array([joint_state[j] for j in joint_names], dtype=np.float32)

    tensor_args = TensorDeviceType()
    robot_cfg = load_yaml(args.robot_cfg_path)["robot_cfg"]
    # cuRobo's primitive collision checker expects at least one obstacle. Keep this far away so
    # the scan remains a pure robot-kinematics/self-collision reachability test.
    world_cfg = WorldConfig.from_dict(
        {
            "cuboid": {
                "far_dummy": {
                    "dims": [0.01, 0.01, 0.01],
                    "pose": [100.0, 100.0, 100.0, 1.0, 0.0, 0.0, 0.0],
                }
            }
        }
    )

    ik_config = IKSolverConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        position_threshold=args.position_threshold,
        rotation_threshold=args.rotation_threshold,
        num_seeds=args.num_seeds,
        self_collision_check=not args.no_self_collision,
        self_collision_opt=not args.no_self_collision,
        tensor_args=tensor_args,
        use_cuda_graph=False,
        collision_checker_type=CollisionCheckerType.PRIMITIVE,
    )
    ik_solver = IKSolver(ik_config)

    current_js = CuJointState.from_position(
        tensor_args.to_device(q_current).view(1, -1),
        joint_names=joint_names,
    ).get_ordered_joint_state(ik_solver.kinematics.joint_names)

    fk = ik_solver.fk(current_js.position)
    current_pos = fk.ee_position.detach().cpu().numpy()[0]
    current_quat = fk.ee_quaternion.detach().cpu().numpy()[0]

    xs, ys, zs, points = make_grid(args.x_range, args.y_range, args.z_range)
    rows: List[dict] = []
    success_flags: List[bool] = []
    orientation_set = make_orientation_set(args.orientation_mode, current_quat)

    for batch_points in chunked(list(points), args.batch_size):
        batch_np = np.asarray(batch_points, dtype=np.float32)
        best = [
            {
                "ok": False,
                "position_error_m": float("inf"),
                "rotation_error": float("inf"),
                "orientation": "",
                "q": None,
            }
            for _ in range(len(batch_np))
        ]
        for orient_name, orient_quat in orientation_set:
            pos = tensor_args.to_device(batch_np)
            quat = tensor_args.to_device(np.tile(orient_quat, (len(batch_np), 1)).astype(np.float32))
            goal_pose = Pose(position=pos, quaternion=quat)
            retract = current_js.position.repeat(len(batch_np), 1)
            result = ik_solver.solve_batch(goal_pose, retract_config=retract)

            success = result.success.detach().cpu().numpy().reshape(-1).astype(bool)
            pos_err = result.position_error.detach().cpu().numpy().reshape(-1)
            rot_err = result.rotation_error.detach().cpu().numpy().reshape(-1)
            q_sol = result.solution.detach().cpu().numpy().reshape(len(batch_np), -1)

            for i, (ok, pe, re, sol) in enumerate(zip(success, pos_err, rot_err, q_sol)):
                safe = (not args.filter_joint_limits) or within_safe_limits(
                    sol, ik_solver.kinematics.joint_names, args.joint_limit_margin
                )
                if bool(ok) and safe and float(pe) < best[i]["position_error_m"]:
                    best[i] = {
                        "ok": True,
                        "position_error_m": float(pe),
                        "rotation_error": float(re),
                        "orientation": orient_name,
                        "q": sol.copy(),
                    }
            torch.cuda.synchronize()

        for point, item in zip(batch_np, best):
            success_flags.append(bool(item["ok"]))
            sol = item["q"]
            if sol is None:
                sol = np.full((len(ik_solver.kinematics.joint_names),), np.nan, dtype=np.float32)
            rows.append(
                {
                    "x": float(point[0]),
                    "y": float(point[1]),
                    "z": float(point[2]),
                    "reachable": int(bool(item["ok"])),
                    "orientation": item["orientation"],
                    "position_error_m": float(item["position_error_m"]),
                    "rotation_error": float(item["rotation_error"]),
                    **{f"q_{name}": float(val) for name, val in zip(ik_solver.kinematics.joint_names, sol)},
                }
            )

    flags = np.array(success_flags, dtype=bool).reshape(len(zs), len(ys), len(xs))
    summary = {
        "robot_cfg_path": args.robot_cfg_path,
        "joint_names": ik_solver.kinematics.joint_names,
        "current_joint_state": {j: float(q) for j, q in zip(joint_names, q_current)},
        "current_ee_position": current_pos.tolist(),
        "current_ee_quaternion_wxyz": current_quat.tolist(),
        "orientation_mode": args.orientation_mode,
        "orientation_count": len(orientation_set),
        "filter_joint_limits": bool(args.filter_joint_limits),
        "x_values": xs.tolist(),
        "y_values": ys.tolist(),
        "z_values": zs.tolist(),
        "total_points": int(flags.size),
        "reachable_points": int(flags.sum()),
        "reachable_ratio": float(flags.mean()),
        "z_slices": [],
    }

    for zi, z in enumerate(zs):
        layer = flags[zi]
        summary["z_slices"].append(
            {
                "z": float(z),
                "reachable": int(layer.sum()),
                "total": int(layer.size),
                "ratio": float(layer.mean()),
                "map": render_layer(layer, xs, ys),
            }
        )
    return summary, rows


def render_layer(layer: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> str:
    lines = []
    lines.append("y\\x " + " ".join(f"{x:4.2f}" for x in xs))
    for yi in range(len(ys) - 1, -1, -1):
        cells = ["O" if ok else "." for ok in layer[yi]]
        lines.append(f"{ys[yi]:4.2f} " + " ".join(cells))
    return "\n".join(lines)


def write_outputs(summary: dict, rows: List[dict], out_dir: str) -> Tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "curobo_reachability_scan.csv")
    txt_path = os.path.join(out_dir, "curobo_reachability_summary.txt")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("cuRobo reachability scan\n")
        f.write(f"robot_cfg_path: {summary['robot_cfg_path']}\n")
        f.write(f"joint_names: {summary['joint_names']}\n")
        f.write(f"current_joint_state: {summary['current_joint_state']}\n")
        f.write(f"current_ee_position: {summary['current_ee_position']}\n")
        f.write(f"current_ee_quaternion_wxyz: {summary['current_ee_quaternion_wxyz']}\n")
        f.write(f"orientation_mode: {summary['orientation_mode']} ({summary['orientation_count']} candidates)\n")
        f.write(f"filter_joint_limits: {summary['filter_joint_limits']}\n")
        f.write(
            f"reachable: {summary['reachable_points']}/{summary['total_points']} "
            f"({summary['reachable_ratio'] * 100.0:.1f}%)\n\n"
        )
        f.write("Legend: O reachable, . unreachable. Coordinates are in base_link meters.\n\n")
        for z_slice in summary["z_slices"]:
            f.write(
                f"z={z_slice['z']:.3f}: {z_slice['reachable']}/{z_slice['total']} "
                f"({z_slice['ratio'] * 100.0:.1f}%)\n"
            )
            f.write(z_slice["map"])
            f.write("\n\n")
    return csv_path, txt_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--robot-cfg-path",
        default="/home/u-zhuang/ros2_ws/src/isaac/isaac_sim_2026/config/turingstack_x1_2_right_tcp.yml",
    )
    parser.add_argument("--joint-topic", default="/joint_states")
    parser.add_argument("--joint-names", type=parse_joint_names, default=DEFAULT_JOINT_NAMES)
    parser.add_argument("--x-range", type=parse_range, default=parse_range("-0.10:0.75:0.05"))
    parser.add_argument("--y-range", type=parse_range, default=parse_range("-0.65:0.25:0.05"))
    parser.add_argument("--z-range", type=parse_range, default=parse_range("0.55:1.25:0.10"))
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--num-seeds", type=int, default=64)
    parser.add_argument("--position-threshold", type=float, default=0.015)
    parser.add_argument("--rotation-threshold", type=float, default=0.20)
    parser.add_argument("--orientation-mode", choices=["current", "grasp"], default="current")
    parser.add_argument("--filter-joint-limits", action="store_true")
    parser.add_argument("--joint-limit-margin", type=float, default=0.05)
    parser.add_argument("--joint-timeout", type=float, default=5.0)
    parser.add_argument("--out-dir", default="/home/u-zhuang/ros2_ws/src/isaac/isaac_sim_2026/reachability_results")
    parser.add_argument("--no-self-collision", action="store_true")
    args = parser.parse_args()

    summary, rows = solve_reachability(args)
    csv_path, txt_path = write_outputs(summary, rows, args.out_dir)

    print(f"reachable {summary['reachable_points']}/{summary['total_points']} ({summary['reachable_ratio'] * 100.0:.1f}%)")
    print(f"csv: {csv_path}")
    print(f"summary: {txt_path}")
    print("")
    for z_slice in summary["z_slices"]:
        print(
            f"z={z_slice['z']:.3f}: {z_slice['reachable']}/{z_slice['total']} "
            f"({z_slice['ratio'] * 100.0:.1f}%)"
        )


if __name__ == "__main__":
    main()
