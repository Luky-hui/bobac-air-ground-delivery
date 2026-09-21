#!/usr/bin/env python3
import math
import os
import sys
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.node import Node
from sensor_msgs.msg import JointState

from grasp_demo_interfaces.action import PlanToPose
from grasp_demo_pkg.common import pose_from_point_and_axes


class PlanToPoseNode(Node):
    def __init__(self):
        super().__init__('plan_to_pose_node')
        self.declare_parameter('base_frame', 'base_link_arm')
        self.declare_parameter('workspace_min', [-0.80, -0.70, 0.02])
        self.declare_parameter('workspace_max', [0.80, 0.70, 1.20])
        self.declare_parameter('commanded_pose_topic', '/demo_grasp/commanded_pose')

        self.declare_parameter('joint_state_topic', '/bobac_joint_states')
        self.declare_parameter('command_topic', '/hand_command')
        self.declare_parameter('joint_names', ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6'])
        self.declare_parameter('reference_pose', [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.declare_parameter('robot_cfg_path', '/home/u-zhuang/ros2_ws/src/isaac/isaac_sim_2026/config/bobac_eco65_right_tcp.yml')
        self.declare_parameter('num_seeds', 192)
        self.declare_parameter('position_threshold', 0.015)
        self.declare_parameter('rotation_threshold', 0.20)
        self.declare_parameter('execute_steps', 60)
        self.declare_parameter('execute_rate_hz', 20.0)
        self.declare_parameter('max_joint_step', 0.012)
        self.declare_parameter('hold_seconds', 0.5)
        self.declare_parameter('local_seed_count', 24)
        self.declare_parameter('local_seed_noise', 0.015)
        self.declare_parameter('prefer_local_seed', True)
        self.declare_parameter('max_solution_jump', 1.80)

        self.pose_pub = self.create_publisher(PoseStamped, self.get_parameter('commanded_pose_topic').value, 10)
        self.joint_pub = self.create_publisher(JointState, self.get_parameter('command_topic').value, 10)
        self.current_demo_pose = [float(v) for v in self.get_parameter('reference_pose').value]
        self.current_joints = {}
        self.last_commanded_q = None
        self.pose_solution_cache = []
        self.create_subscription(
            JointState,
            self.get_parameter('joint_state_topic').value,
            self._on_joint_state,
            10,
        )

        self.server = ActionServer(
            self,
            PlanToPose,
            '/demo_plan_to_pose',
            execute_callback=self.execute_callback,
            goal_callback=lambda goal: GoalResponse.ACCEPT,
            cancel_callback=lambda goal: CancelResponse.ACCEPT,
        )
        self.get_logger().info(
            'plan_to_pose_node started; official perception target is executed with cuRobo IK and smooth JointState commands'
        )

    def _on_joint_state(self, msg):
        for name, value in zip(msg.name, msg.position):
            self.current_joints[str(name)] = float(value)

    def _in_workspace(self, xyz):
        lo = np.array(self.get_parameter('workspace_min').value, dtype=np.float64)
        hi = np.array(self.get_parameter('workspace_max').value, dtype=np.float64)
        return bool(np.all(xyz >= lo) and np.all(xyz <= hi))

    @staticmethod
    def _clip(value, lo, hi):
        return max(lo, min(hi, float(value)))

    def _publish_joint_pose(self, names, positions):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(names)
        msg.position = [float(v) for v in positions]
        self.joint_pub.publish(msg)

    def _current_q(self):
        names = list(self.get_parameter('joint_names').value)
        if self.last_commanded_q is not None and len(self.last_commanded_q) == len(names):
            return [float(v) for v in self.last_commanded_q]
        if all(name in self.current_joints for name in names):
            return [float(self.current_joints[name]) for name in names]
        ref = [float(v) for v in self.get_parameter('reference_pose').value]
        if len(ref) != len(names):
            raise RuntimeError('reference_pose length must match joint_names length')
        return ref

    @staticmethod
    def _joint_delta(a, b):
        return math.atan2(math.sin(float(b) - float(a)), math.cos(float(b) - float(a)))

    def _solution_jump(self, q_ref, q_sol):
        return max(abs(self._joint_delta(a, b)) for a, b in zip(q_ref, q_sol))

    def _make_local_seed_config(self, tensor_args, q_current, dof):
        import torch

        count = max(1, int(self.get_parameter('local_seed_count').value))
        noise = max(0.0, float(self.get_parameter('local_seed_noise').value))
        base = np.array(q_current, dtype=np.float32).reshape(1, 1, dof)
        seeds = np.repeat(base, count, axis=1)
        if count > 1 and noise > 0.0:
            offsets = np.linspace(-noise, noise, count, dtype=np.float32).reshape(1, count, 1)
            pattern = np.array([1.0, -0.7, 0.5, -0.4, 0.25, -0.2], dtype=np.float32).reshape(1, 1, dof)
            seeds = seeds + offsets * pattern
            seeds[:, 0, :] = base[:, 0, :]
        return tensor_args.to_device(seeds)

    @staticmethod
    def _install_isaac_python_paths():
        isaac_root = os.environ.get('ISAAC_SIM_ROOT', '/home/u-zhuang/isaac-sim-4.5.0')
        for path in (
            '/home/u-zhuang/ros2_ws/src/isaac/isaac_sim_2026/curobo/src',
            os.path.join(isaac_root, 'kit/python/lib/python3.10/site-packages'),
            os.path.join(isaac_root, 'python_packages'),
            os.path.join(isaac_root, 'exts/omni.pip.compute/pip_prebundle'),
            os.path.join(isaac_root, 'exts/omni.pip.cloud/pip_prebundle'),
        ):
            if os.path.isdir(path) and path not in sys.path:
                sys.path.insert(0, path)

    def _solve_curobo_ik(self, xyz, quat_wxyz):
        self._install_isaac_python_paths()
        import torch
        from curobo.geom.sdf.world import CollisionCheckerType
        from curobo.geom.types import WorldConfig
        from curobo.types.base import TensorDeviceType
        from curobo.types.math import Pose
        from curobo.types.robot import JointState as CuJointState
        from curobo.util_file import load_yaml
        from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

        tensor_args = TensorDeviceType()
        robot_cfg = load_yaml(str(self.get_parameter('robot_cfg_path').value))['robot_cfg']
        world_cfg = WorldConfig.from_dict(
            {'cuboid': {'far_dummy': {'dims': [0.01, 0.01, 0.01], 'pose': [100.0, 100.0, 100.0, 1.0, 0.0, 0.0, 0.0]}}}
        )
        ik_config = IKSolverConfig.load_from_robot_config(
            robot_cfg,
            world_cfg,
            position_threshold=float(self.get_parameter('position_threshold').value),
            rotation_threshold=float(self.get_parameter('rotation_threshold').value),
            num_seeds=int(self.get_parameter('num_seeds').value),
            self_collision_check=True,
            self_collision_opt=True,
            tensor_args=tensor_args,
            use_cuda_graph=False,
            collision_checker_type=CollisionCheckerType.PRIMITIVE,
        )
        ik_solver = IKSolver(ik_config)
        q_current = np.array(self._current_q(), dtype=np.float32)
        current_js = CuJointState.from_position(
            tensor_args.to_device(q_current).view(1, -1),
            joint_names=list(self.get_parameter('joint_names').value),
        ).get_ordered_joint_state(ik_solver.kinematics.joint_names)
        goal_pose = Pose(
            position=tensor_args.to_device(np.array(xyz, dtype=np.float32)).view(1, 3),
            quaternion=tensor_args.to_device(np.array(quat_wxyz, dtype=np.float32)).view(1, 4),
        )
        result = None
        if bool(self.get_parameter('prefer_local_seed').value):
            seed_config = self._make_local_seed_config(
                tensor_args,
                current_js.position.detach().cpu().numpy().reshape(-1),
                len(ik_solver.kinematics.joint_names),
            )
            result = ik_solver.solve_batch(
                goal_pose,
                retract_config=current_js.position,
                seed_config=seed_config,
                num_seeds=int(seed_config.shape[1]),
                use_nn_seed=False,
            )
        if result is None or not bool(result.success.detach().cpu().numpy().reshape(-1)[0]):
            result = ik_solver.solve_batch(
                goal_pose,
                retract_config=current_js.position,
                num_seeds=int(self.get_parameter('num_seeds').value),
                use_nn_seed=False,
            )
        success = bool(result.success.detach().cpu().numpy().reshape(-1)[0])
        pos_err = float(result.position_error.detach().cpu().numpy().reshape(-1)[0])
        rot_err = float(result.rotation_error.detach().cpu().numpy().reshape(-1)[0])
        q_sol = result.solution.detach().cpu().numpy().reshape(1, -1)[0].astype(np.float64)
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            torch.cuda.synchronize()
        if not success:
            raise RuntimeError(f'cuRobo IK failed: pos_err={pos_err:.4f}, rot_err={rot_err:.4f}')
        max_jump = float(self.get_parameter('max_solution_jump').value)
        jump = self._solution_jump(current_js.position.detach().cpu().numpy().reshape(-1), q_sol)
        if jump > max_jump:
            raise RuntimeError(f'cuRobo IK changed branch: max_joint_jump={jump:.3f} > {max_jump:.3f}')
        return list(ik_solver.kinematics.joint_names), q_sol.tolist(), pos_err, rot_err

    def _execute_motion(self, goal_handle, target_pose):
        names = list(self.get_parameter('joint_names').value)
        if len(names) != len(target_pose):
            raise RuntimeError('joint_names length must match generated target pose length')

        start = self._current_q()
        max_delta = max(abs(a - b) for a, b in zip(start, target_pose))
        by_step = int(math.ceil(max_delta / max(1e-4, float(self.get_parameter('max_joint_step').value))))
        steps = max(2, int(self.get_parameter('execute_steps').value), by_step)
        rate_hz = max(1.0, float(self.get_parameter('execute_rate_hz').value))
        period = 1.0 / rate_hz

        feedback = PlanToPose.Feedback()
        feedback.state = 'EXECUTING_CUROBO_TRAJECTORY'
        goal_handle.publish_feedback(feedback)

        for i in range(steps + 1):
            if goal_handle.is_cancel_requested:
                return False
            t = i / steps
            s = t * t * (3.0 - 2.0 * t)
            pose = [a + (b - a) * s for a, b in zip(start, target_pose)]
            self._publish_joint_pose(names, pose)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(period)

        hold = max(0.0, float(self.get_parameter('hold_seconds').value))
        end = time.time() + hold
        while rclpy.ok() and time.time() < end:
            if goal_handle.is_cancel_requested:
                return False
            self._publish_joint_pose(names, target_pose)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(period)

        self.last_commanded_q = [float(v) for v in target_pose]
        return True

    def _find_cached_solution(self, xyz):
        target = np.array(xyz, dtype=np.float64)
        best = None
        best_dist = 1e9
        for item in self.pose_solution_cache:
            dist = float(np.linalg.norm(target - item['xyz']))
            if dist < best_dist:
                best = item
                best_dist = dist
        if best is not None and best_dist < 0.002:
            return best
        return None

    def _cache_solution(self, label, xyz, joint_names, target_joints, pos_err, rot_err):
        self.pose_solution_cache.append(
            {
                'label': str(label),
                'xyz': np.array(xyz, dtype=np.float64),
                'joint_names': list(joint_names),
                'target_joints': [float(v) for v in target_joints],
                'pos_err': float(pos_err),
                'rot_err': float(rot_err),
            }
        )
        self.pose_solution_cache = self.pose_solution_cache[-16:]

    async def execute_callback(self, goal_handle):
        goal = goal_handle.request
        result = PlanToPose.Result()
        feedback = PlanToPose.Feedback()

        feedback.state = 'CHECKING_WORKSPACE'
        goal_handle.publish_feedback(feedback)
        xyz = np.array([goal.target.point.x, goal.target.point.y, goal.target.point.z], dtype=np.float64)
        lo = np.array(self.get_parameter('workspace_min').value, dtype=np.float64)
        hi = np.array(self.get_parameter('workspace_max').value, dtype=np.float64)
        self.get_logger().info(
            f'{goal.label}: target={xyz.tolist()}, workspace_min={lo.tolist()}, workspace_max={hi.tolist()}, execute={goal.execute}'
        )
        if not self._in_workspace(xyz):
            result.success = False
            result.message = f'target outside workspace: {xyz.tolist()}'
            goal_handle.abort()
            return result

        feedback.state = 'BUILDING_CANDIDATE_POSE'
        goal_handle.publish_feedback(feedback)
        normal = np.array([goal.normal.vector.x, goal.normal.vector.y, goal.normal.vector.z], dtype=np.float64)
        axis = np.array([goal.long_axis.vector.x, goal.long_axis.vector.y, goal.long_axis.vector.z], dtype=np.float64)
        frame_id = goal.target.header.frame_id or self.get_parameter('base_frame').value
        result.commanded_pose = pose_from_point_and_axes(frame_id, self.get_clock().now().to_msg(), xyz, normal, axis)
        self.pose_pub.publish(result.commanded_pose)

        qx = result.commanded_pose.pose.orientation.x
        qy = result.commanded_pose.pose.orientation.y
        qz = result.commanded_pose.pose.orientation.z
        qw = result.commanded_pose.pose.orientation.w
        quat_wxyz = np.array([qw, qx, qy, qz], dtype=np.float32)
        cached = self._find_cached_solution(xyz)
        if cached is not None:
            joint_names = cached['joint_names']
            target_joints = cached['target_joints']
            pos_err = cached['pos_err']
            rot_err = cached['rot_err']
            self.get_logger().info(
                f'{goal.label}: reusing cached cuRobo solution from {cached["label"]}, '
                f'q={[round(v, 3) for v in target_joints]}'
            )
        else:
            joint_names, target_joints, pos_err, rot_err = self._solve_curobo_ik(xyz, quat_wxyz)
            self._cache_solution(goal.label, xyz, joint_names, target_joints, pos_err, rot_err)
            self.get_logger().info(
                f'{goal.label}: curobo_joints={joint_names}, q={[round(v, 3) for v in target_joints]}, '
                f'ik_pos_err={pos_err:.4f}, ik_rot_err={rot_err:.4f}'
            )

        if goal.execute:
            ok = self._execute_motion(goal_handle, target_joints)
            if not ok:
                result.success = False
                result.message = 'cuRobo trajectory canceled'
                goal_handle.canceled()
                return result
            result.message = f'cuRobo IK executed; pos_err={pos_err:.4f}, rot_err={rot_err:.4f}'
        else:
            feedback.state = 'POSE_READY'
            goal_handle.publish_feedback(feedback)
            result.message = f'candidate pose generated; cuRobo IK ok pos_err={pos_err:.4f}, rot_err={rot_err:.4f}'

        result.success = True
        goal_handle.succeed()
        return result


def main(args=None):
    rclpy.init(args=args)
    node = PlanToPoseNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
