#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dry-run a safe right_tcp pre_grasp -> approach -> retreat motion.

This script does not close the gripper. It:
1. Reads the current right-arm state and Isaac TF.
2. Uses the current right_tcp attitude as the grasp attitude.
3. Computes pre_grasp = grasp - approach_distance * tool_forward_axis.
4. Solves cuRobo IK for pre_grasp and grasp.
5. Publishes slow joint commands with soft-stop monitoring.
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
from scipy.spatial.transform import Rotation
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


def transform_pos_quat_xyzw(tf) -> Tuple[np.ndarray, np.ndarray]:
    pos = np.array(
        [tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z],
        dtype=np.float64,
    )
    quat = np.array(
        [
            tf.transform.rotation.x,
            tf.transform.rotation.y,
            tf.transform.rotation.z,
            tf.transform.rotation.w,
        ],
        dtype=np.float64,
    )
    return pos, quat


def xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float32)


def parse_xyz(raw: str) -> np.ndarray:
    parts = [float(x) for x in raw.replace(",", " ").split()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("xyz must contain exactly three numbers")
    return np.array(parts, dtype=np.float64)


class DryRunNode(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("curobo_pregrasp_approach_dry_run")
        self.args = args
        self.pose: Dict[str, float] = {}
        self.left_pose: Dict[str, float] = {}
        self.prev_stamp: Optional[float] = None
        self.abort_reason: Optional[str] = None
        self.max_observed_step = 0.0
        self.max_observed_speed = 0.0
        self.speed_spike_count = 0

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)
        self.pub = self.create_publisher(JointState, "/joint_command", 10)
        self.left_pub = self.create_publisher(JointState, "/joint_left_command", 10)
        self.create_subscription(JointState, "/joint_states", self._joint_cb, 10)
        self.create_subscription(JointState, "/joint_left_states", self._left_joint_cb, 10)

    def _joint_cb(self, msg: JointState) -> None:
        stamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        if stamp <= 0.0:
            stamp = time.time()
        new_pose = {name: float(pos) for name, pos in zip(msg.name, msg.position)}
        if self.pose and self.prev_stamp is not None:
            dt = stamp - self.prev_stamp
            for joint in R_JOINTS:
                if joint not in self.pose or joint not in new_pose:
                    continue
                step = abs(new_pose[joint] - self.pose[joint])
                self.max_observed_step = max(self.max_observed_step, step)
                if step > self.args.abort_observed_step:
                    self.abort_reason = (
                        f"{joint} observed step {step:.4f} rad > "
                        f"{self.args.abort_observed_step:.4f}"
                    )
                if dt >= self.args.min_velocity_dt:
                    speed = step / dt
                    self.max_observed_speed = max(self.max_observed_speed, speed)
                    if speed > self.args.abort_observed_speed:
                        self.speed_spike_count += 1
                        if self.speed_spike_count >= self.args.abort_speed_spike_count:
                            self.abort_reason = (
                                f"{joint} observed speed {speed:.3f} rad/s > "
                                f"{self.args.abort_observed_speed:.3f}"
                            )
                    else:
                        self.speed_spike_count = 0
        self.pose = new_pose
        self.prev_stamp = stamp

    def _left_joint_cb(self, msg: JointState) -> None:
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
            have_joints = all(j in self.pose for j in R_JOINTS + L_JOINTS + [GRIP_R])
            missing_tf = []
            for frame in ["gripper01_left1", "gripper01_right1", "gripper01_base"]:
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

    def lookup_right_tcp_pose(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        left_tf = self.tf_buffer.lookup_transform(
            "base_link",
            "gripper01_left1",
            rclpy.time.Time(),
            timeout=Duration(seconds=0.2),
        )
        right_tf = self.tf_buffer.lookup_transform(
            "base_link",
            "gripper01_right1",
            rclpy.time.Time(),
            timeout=Duration(seconds=0.2),
        )
        base_tf = self.tf_buffer.lookup_transform(
            "base_link",
            "gripper01_base",
            rclpy.time.Time(),
            timeout=Duration(seconds=0.2),
        )
        left_pos, _ = transform_pos_quat_xyzw(left_tf)
        right_pos, _ = transform_pos_quat_xyzw(right_tf)
        base_pos, base_quat_xyzw = transform_pos_quat_xyzw(base_tf)
        tcp_pos = (left_pos + right_pos) * 0.5
        return tcp_pos, base_quat_xyzw, base_pos

    def tool_forward_axis(self) -> np.ndarray:
        tcp_pos, _, gripper_base_pos = self.lookup_right_tcp_pose()
        axis = tcp_pos - gripper_base_pos
        norm = float(np.linalg.norm(axis))
        if norm < 1e-6:
            raise RuntimeError("cannot infer tool forward axis from gripper base to right_tcp")
        return axis / norm

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
            lo += self.args.joint_limit_margin
            hi -= self.args.joint_limit_margin
            if value < lo or value > hi:
                self.get_logger().error(f"{label}: {name}={value:.4f} outside [{lo:.4f}, {hi:.4f}]")
                return False
        return True

    def move_to(self, q_target: List[float], label: str, min_steps: int) -> bool:
        if not self.validate_q(q_target, label):
            return False
        q_start = self.current_q()
        max_delta = max(abs(a - b) for a, b in zip(q_start, q_target))
        steps = max(min_steps, int(math.ceil(max_delta / self.args.max_joint_step)))
        duration = steps * self.args.dt
        self.get_logger().info(
            f"move {label}: steps={steps}, duration={duration:.1f}s, max_delta={max_delta:.4f}rad"
        )

        self.speed_spike_count = 0
        for i in range(steps + 1):
            if self.abort_reason:
                self.get_logger().error(f"SOFT STOP before publish: {self.abort_reason}")
                self.publish_hold(self.current_q())
                return False
            alpha = i / steps
            q_cmd = [(1.0 - alpha) * s + alpha * t for s, t in zip(q_start, q_target)]
            self.publish_hold(q_cmd)

            end_time = time.time() + self.args.dt
            while rclpy.ok() and time.time() < end_time:
                rclpy.spin_once(self, timeout_sec=0.005)
                if self.abort_reason:
                    self.get_logger().error(f"SOFT STOP while moving {label}: {self.abort_reason}")
                    self.publish_hold(self.current_q())
                    return False

            if i % max(1, steps // 20) == 0:
                observed = self.current_q()
                tracking_err = max(abs(a - b) for a, b in zip(observed, q_cmd))
                if tracking_err > self.args.abort_tracking_error:
                    self.abort_reason = (
                        f"{label} tracking error {tracking_err:.3f} rad > "
                        f"{self.args.abort_tracking_error:.3f}"
                    )
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
        actual = np.zeros(3, dtype=np.float64)
        while time.time() < deadline and rclpy.ok():
            self.publish_hold(self.current_q())
            rclpy.spin_once(self, timeout_sec=0.02)
            actual, _, _ = self.lookup_right_tcp_pose()
            errors.append(float(np.linalg.norm(actual - target_xyz)))
            if self.abort_reason:
                return False, float("inf"), actual
        final_error = errors[-1] if errors else float("inf")
        tail = errors[-min(len(errors), 10):]
        sustained = bool(tail) and all(e > self.args.abort_tcp_error for e in tail)
        if sustained:
            self.abort_reason = f"{label} right_tcp error sustained at {final_error:.3f} m"
            return False, final_error, actual
        return True, final_error, actual


def solve_waypoint_ik(
    robot_cfg_path: str,
    q_current: np.ndarray,
    waypoint_positions: List[np.ndarray],
    quat_wxyz: np.ndarray,
    num_seeds: int,
    position_threshold: float,
    rotation_threshold: float,
) -> Tuple[List[str], List[np.ndarray], List[float], List[float]]:
    import torch

    from curobo.geom.sdf.world import CollisionCheckerType
    from curobo.geom.types import WorldConfig
    from curobo.types.base import TensorDeviceType
    from curobo.types.math import Pose
    from curobo.types.robot import JointState as CuJointState
    from curobo.util_file import load_yaml
    from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

    tensor_args = TensorDeviceType()
    robot_cfg = load_yaml(robot_cfg_path)["robot_cfg"]
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
        position_threshold=position_threshold,
        rotation_threshold=rotation_threshold,
        num_seeds=num_seeds,
        self_collision_check=True,
        self_collision_opt=True,
        tensor_args=tensor_args,
        use_cuda_graph=False,
        collision_checker_type=CollisionCheckerType.PRIMITIVE,
    )
    ik_solver = IKSolver(ik_config)

    current_js = CuJointState.from_position(
        tensor_args.to_device(q_current.astype(np.float32)).view(1, -1),
        joint_names=R_JOINTS,
    ).get_ordered_joint_state(ik_solver.kinematics.joint_names)
    retract = current_js.position

    solutions: List[np.ndarray] = []
    pos_errors: List[float] = []
    rot_errors: List[float] = []
    for pos_np in waypoint_positions:
        goal_pose = Pose(
            position=tensor_args.to_device(pos_np.astype(np.float32)).view(1, 3),
            quaternion=tensor_args.to_device(quat_wxyz.astype(np.float32)).view(1, 4),
        )
        result = ik_solver.solve_batch(goal_pose, retract_config=retract)
        success = bool(result.success.detach().cpu().numpy().reshape(-1)[0])
        pos_err = float(result.position_error.detach().cpu().numpy().reshape(-1)[0])
        rot_err = float(result.rotation_error.detach().cpu().numpy().reshape(-1)[0])
        q_sol = result.solution.detach().cpu().numpy().reshape(1, -1)[0].astype(np.float64)
        torch.cuda.synchronize()
        if not success:
            raise RuntimeError(
                f"IK failed for waypoint {pos_np.tolist()} "
                f"(pos_err={pos_err:.4f}, rot_err={rot_err:.4f})"
            )
        solutions.append(q_sol)
        pos_errors.append(pos_err)
        rot_errors.append(rot_err)
        retract = tensor_args.to_device(q_sol.astype(np.float32)).view(1, -1)
    return list(ik_solver.kinematics.joint_names), solutions, pos_errors, rot_errors


def write_report(args: argparse.Namespace, rows: List[dict]) -> str:
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "pregrasp_approach_dry_run_report.csv")
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
    parser.add_argument("--grasp-xyz", type=parse_xyz, default=parse_xyz("0.45,-0.38,0.78"))
    parser.add_argument("--approach-distance", type=float, default=0.10)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--num-seeds", type=int, default=128)
    parser.add_argument("--position-threshold", type=float, default=0.01)
    parser.add_argument("--rotation-threshold", type=float, default=0.10)
    parser.add_argument("--max-joint-step", type=float, default=0.0015)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--min-steps", type=int, default=500)
    parser.add_argument("--approach-min-steps", type=int, default=600)
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
        forward_axis = node.tool_forward_axis()
        grasp_xyz = args.grasp_xyz.astype(np.float64)
        pregrasp_xyz = grasp_xyz - float(args.approach_distance) * forward_axis
        quat_wxyz = xyzw_to_wxyz(quat_xyzw)

        node.get_logger().info(f"start_q={[round(float(x), 4) for x in q_start.tolist()]}")
        node.get_logger().info(f"start_right_tcp={[round(float(x), 4) for x in start_tcp.tolist()]}")
        node.get_logger().info(f"tool_forward_axis_base={[round(float(x), 4) for x in forward_axis.tolist()]}")
        node.get_logger().info(f"pregrasp_xyz={[round(float(x), 4) for x in pregrasp_xyz.tolist()]}")
        node.get_logger().info(f"grasp_xyz={[round(float(x), 4) for x in grasp_xyz.tolist()]}")

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
        node.get_logger().info(f"q_grasp={[round(float(x), 4) for x in q_solutions[1].tolist()]}")

        stages = [
            ("pre_grasp", pregrasp_xyz, q_solutions[0], args.min_steps, ik_pos_err[0], ik_rot_err[0]),
            ("approach", grasp_xyz, q_solutions[1], args.approach_min_steps, ik_pos_err[1], ik_rot_err[1]),
            ("retreat", pregrasp_xyz, q_solutions[0], args.approach_min_steps, ik_pos_err[0], ik_rot_err[0]),
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

        # Return to the original safe pose after a successful retreat.
        if not node.abort_reason:
            node.move_to(q_start.tolist(), "return_start", args.min_steps)

        report = write_report(args, rows)
        node.get_logger().info(f"report written: {report}")
        node.get_logger().info(
            f"dry run finished; max_step={node.max_observed_step:.4f}rad, "
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
