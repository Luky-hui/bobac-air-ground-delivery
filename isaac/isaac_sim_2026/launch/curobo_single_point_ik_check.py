#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only cuRobo single point IK smoke test for right_tcp."""

from __future__ import annotations

import argparse
import time
from typing import Dict, List

import numpy as np


RIGHT_ARM_JOINTS = ["joint1_R", "joint2_R", "joint3_R", "joint4_R", "joint5_R", "joint6_R"]

JOINT_LIMITS = {
    "joint1_R": (-2.87979326579, 2.87979326579),
    "joint2_R": (-0.9599310886, 1.6580627894),
    "joint3_R": (-3.0194196060, 0.0872664626),
    "joint4_R": (-2.8797932658, 2.8797932658),
    "joint5_R": (-0.3490658504, 4.6251225178),
    "joint6_R": (-3.1415926536, 3.1415926536),
}


def wait_for_joint_state(timeout_s: float) -> Dict[str, float]:
    import rclpy
    from sensor_msgs.msg import JointState

    rclpy.init(args=None)
    node = rclpy.create_node("curobo_single_point_ik_check_reader")
    result: Dict[str, float] = {}

    def cb(msg: JointState):
        nonlocal result
        result = {name: float(pos) for name, pos in zip(msg.name, msg.position)}

    sub = node.create_subscription(JointState, "/joint_states", cb, 10)
    deadline = time.time() + timeout_s
    while rclpy.ok() and time.time() < deadline and not result:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_subscription(sub)
    node.destroy_node()
    rclpy.shutdown()
    if not result:
        raise RuntimeError("Timed out waiting for /joint_states")
    missing = [j for j in RIGHT_ARM_JOINTS if j not in result]
    if missing:
        raise RuntimeError(f"/joint_states is missing right-arm joints: {missing}")
    return result


def within_limits(q: np.ndarray, joint_names: List[str], margin: float) -> bool:
    for name, value in zip(joint_names, q.tolist()):
        lo, hi = JOINT_LIMITS[name]
        if value < lo + margin or value > hi - margin:
            return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-cfg-path", default="/home/u-zhuang/ros2_ws/src/isaac/isaac_sim_2026/config/turingstack_x1_2_right_tcp.yml")
    parser.add_argument("--joint-timeout", type=float, default=5.0)
    parser.add_argument("--num-seeds", type=int, default=64)
    parser.add_argument("--position-threshold", type=float, default=0.01)
    parser.add_argument("--rotation-threshold", type=float, default=0.10)
    parser.add_argument("--joint-limit-margin", type=float, default=0.05)
    args = parser.parse_args()

    from curobo.geom.sdf.world import CollisionCheckerType
    from curobo.geom.types import WorldConfig
    from curobo.types.base import TensorDeviceType
    from curobo.types.math import Pose
    from curobo.types.robot import JointState as CuJointState
    from curobo.util_file import load_yaml
    from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

    joint_state = wait_for_joint_state(args.joint_timeout)
    q_current = np.array([joint_state[j] for j in RIGHT_ARM_JOINTS], dtype=np.float32)

    tensor_args = TensorDeviceType()
    robot_cfg = load_yaml(args.robot_cfg_path)["robot_cfg"]
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
        self_collision_check=True,
        self_collision_opt=True,
        tensor_args=tensor_args,
        use_cuda_graph=False,
        collision_checker_type=CollisionCheckerType.PRIMITIVE,
    )
    ik_solver = IKSolver(ik_config)
    current_js = CuJointState.from_position(
        tensor_args.to_device(q_current).view(1, -1),
        joint_names=RIGHT_ARM_JOINTS,
    ).get_ordered_joint_state(ik_solver.kinematics.joint_names)

    fk = ik_solver.fk(current_js.position)
    current_pos = fk.ee_position.detach().cpu().numpy()[0]
    current_quat = fk.ee_quaternion.detach().cpu().numpy()[0]
    goal_pose = Pose(position=fk.ee_position.clone(), quaternion=fk.ee_quaternion.clone())
    result = ik_solver.solve_batch(goal_pose, retract_config=current_js.position.clone())

    success = bool(result.success.detach().cpu().numpy().reshape(-1)[0])
    pos_err = float(result.position_error.detach().cpu().numpy().reshape(-1)[0])
    rot_err = float(result.rotation_error.detach().cpu().numpy().reshape(-1)[0])
    q_sol = result.solution.detach().cpu().numpy().reshape(1, -1)[0]
    safe = within_limits(q_sol, ik_solver.kinematics.joint_names, args.joint_limit_margin)

    print("cuRobo single-point IK check")
    print(f"robot_cfg_path: {args.robot_cfg_path}")
    print(f"joint_names: {ik_solver.kinematics.joint_names}")
    print(f"current_q: {[round(float(x), 6) for x in q_current.tolist()]}")
    print(f"current_right_tcp_position: {[round(float(x), 6) for x in current_pos.tolist()]}")
    print(f"current_right_tcp_quat_wxyz: {[round(float(x), 6) for x in current_quat.tolist()]}")
    print(f"success: {success}")
    print(f"within_limits_margin_{args.joint_limit_margin}: {safe}")
    print(f"position_error_m: {pos_err:.8f}")
    print(f"rotation_error_rad: {rot_err:.8f}")
    print(f"q_solution_active: {[round(float(x), 6) for x in q_sol.tolist()]}")
    return 0 if success and safe else 2


if __name__ == "__main__":
    raise SystemExit(main())
