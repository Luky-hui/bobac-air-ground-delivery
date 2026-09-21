#!/usr/bin/env python3
import json
import math
import time

import rclpy
from geometry_msgs.msg import PointStamped, Twist, Vector3Stamped
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from grasp_demo_interfaces.action import DetectObject, GripperCommand, PlanToPose


class PickPlaceStateMachine(Node):
    def __init__(self):
        super().__init__('pick_place_state_machine')
        self.declare_parameter('command_topic', '/hand_command')
        self.declare_parameter('joint_state_topic', '/demo_grasp/arm_joint_states')
        self.declare_parameter('joint_names', ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6'])
        self.declare_parameter('observation_pose', [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.declare_parameter('observation_rate_hz', 50.0)
        self.declare_parameter('observation_max_joint_step', 0.012)
        self.declare_parameter('target_classes', ['pencil', 'pen'])
        self.declare_parameter('confidence_threshold', 0.01)
        self.declare_parameter('detect_server_timeout_sec', 60.0)
        self.declare_parameter('detect_goal_timeout_sec', 15.0)
        self.declare_parameter('object_point_topic', '/demo_grasp/object_point_base')
        self.declare_parameter('normal_topic', '/demo_grasp/normal_base')
        self.declare_parameter('long_axis_topic', '/demo_grasp/long_axis_base')
        self.declare_parameter('pre_grasp_offset', 0.08)
        self.declare_parameter('finger_tip_offset', 0.03)
        self.declare_parameter('lift_offset', 0.12)
        self.declare_parameter('min_grasp_z', 0.0)
        self.declare_parameter('min_pre_grasp_z', 0.0)
        self.declare_parameter('min_lift_z', 0.0)
        self.declare_parameter('lift_steps', 6)
        self.declare_parameter('place_position', [0.65, -0.22, 0.18])
        self.declare_parameter('material_command_topic', '/material_task/command')
        self.declare_parameter('material_report_topic', '/material_task/report')
        self.declare_parameter('cargo_command_topic', '/cargo_bay/command')
        self.declare_parameter('cargo_target_id', 'white_pencil')
        self.declare_parameter('place_pre_offset', 0.12)
        self.declare_parameter('place_retreat_offset', 0.16)
        self.declare_parameter('object_place_z_offset', 0.015)
        self.declare_parameter('place_scene_point_is_tcp', True)
        self.declare_parameter('control_cargo_doors', False)
        self.declare_parameter('retreat_after_place', False)
        self.declare_parameter('min_place_z_from_cargo_body', False)
        self.declare_parameter('cargo_body_tcp_clearance', 0.12)
        self.declare_parameter('place_stand_before_place', False)
        self.declare_parameter('place_stand_pose', [3.88, -0.38, 0.0])
        self.declare_parameter('place_stand_xy_tolerance', 0.035)
        self.declare_parameter('place_stand_yaw_tolerance', 0.06)
        self.declare_parameter('place_stand_speed', 0.10)
        self.declare_parameter('place_stand_angular_speed', 0.35)
        self.declare_parameter('place_stand_timeout', 18.0)
        self.declare_parameter('reposition_base_before_place', True)
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('cmd_vel_topic', '/bobac_kinematic_cmd_vel')
        self.declare_parameter('place_base_target_xy', [0.30, 0.42])
        self.declare_parameter('place_base_xy_tolerance', 0.035)
        self.declare_parameter('place_base_max_step', 0.22)
        self.declare_parameter('place_base_reposition_iterations', 2)
        self.declare_parameter('place_base_reposition_speed', 0.08)
        self.declare_parameter('place_base_reposition_timeout', 8.0)
        self.declare_parameter('retreat_base_before_lift', False)
        self.declare_parameter('retreat_base_before_lift_delta', [-0.32, 0.0])
        self.declare_parameter('retreat_base_before_lift_speed', 0.08)
        self.declare_parameter('retreat_base_before_lift_timeout', 10.0)
        self.declare_parameter('escape_lift_before_base_retreat', False)
        self.declare_parameter('escape_lift_z', 0.24)
        self.declare_parameter('post_retreat_lift_from_measured_tcp', False)
        self.declare_parameter('carry_pose_before_place', False)
        self.declare_parameter('carry_pose', [0.0, 0.58, -1.67, -0.50, 1.51, 0.0])
        self.declare_parameter('carry_pose_hold_seconds', 0.6)
        self.declare_parameter('close_side_door_after_place', True)
        self.declare_parameter('place_validation_tolerance', 0.25)
        self.declare_parameter('video_place_enabled', False)
        self.declare_parameter('video_place_face_yaw_offset', 0.0)
        self.declare_parameter('video_place_yaw_tolerance', 0.05)
        self.declare_parameter('video_place_yaw_speed', 0.25)
        self.declare_parameter('video_place_target_base_xy', [0.42, 0.0])
        self.declare_parameter('video_place_base_xy_tolerance', 0.04)
        self.declare_parameter('video_place_base_max_step', 0.08)
        self.declare_parameter('video_place_base_reposition_iterations', 6)
        self.declare_parameter('video_place_base_reposition_speed', 0.04)
        self.declare_parameter('video_place_base_reposition_timeout', 7.0)
        self.declare_parameter('video_place_normal_base', [0.0, 0.0, -1.0])
        self.declare_parameter('video_place_long_axis_base', [0.0, 1.0, 0.0])
        self.declare_parameter('video_place_pre_offset_x', 0.16)
        self.declare_parameter('video_place_retreat_offset_x', 0.18)
        self.declare_parameter('video_place_post_release_lift', 0.12)
        self.declare_parameter('video_place_tcp_clearance_from_cargo_body', 0.10)
        self.declare_parameter('video_place_close_side_door_after_place', False)
        self.declare_parameter('planning_frame', 'base_link_arm')
        self.declare_parameter('place_only_long_axis', [0.772, -0.635, -0.001])
        self.declare_parameter('execute_grasp', True)
        self.declare_parameter('stop_after_lift', True)
        self.declare_parameter('place_only', False)
        self.declare_parameter('require_object_in_workspace', True)
        self.declare_parameter('object_workspace_min', [-0.80, -0.70, 0.02])
        self.declare_parameter('object_workspace_max', [0.80, 0.70, 1.20])
        self.declare_parameter('validate_lift_with_scene_points', True)
        self.declare_parameter('lift_validation_target_id', 'white_pencil')
        self.declare_parameter('lift_validation_min_world_dz', 0.08)
        self.declare_parameter('lift_validation_timeout', 6.0)

        self.object_point = None
        self.normal = None
        self.long_axis = None
        self.current_joints = {}
        self.last_odom = None
        self.material_reports = []
        self.hand_pub = self.create_publisher(JointState, self.get_parameter('command_topic').value, 10)
        self.cmd_vel_pub = self.create_publisher(
            Twist, self.get_parameter('cmd_vel_topic').value, 10
        )
        self.material_command_pub = self.create_publisher(
            String, self.get_parameter('material_command_topic').value, 10
        )
        self.cargo_command_pub = self.create_publisher(
            String, self.get_parameter('cargo_command_topic').value, 10
        )
        self.create_subscription(JointState, self.get_parameter('joint_state_topic').value, self.on_joint_state, 10)
        self.create_subscription(Odometry, self.get_parameter('odom_topic').value, self.on_odom, 10)
        self.create_subscription(PointStamped, self.get_parameter('object_point_topic').value, self.on_point, 10)
        self.create_subscription(Vector3Stamped, self.get_parameter('normal_topic').value, self.on_normal, 10)
        self.create_subscription(Vector3Stamped, self.get_parameter('long_axis_topic').value, self.on_axis, 10)
        self.create_subscription(
            String, self.get_parameter('material_report_topic').value, self.on_material_report, 10
        )
        self.detect_client = ActionClient(self, DetectObject, '/demo_detect_object')
        self.plan_client = ActionClient(self, PlanToPose, '/demo_plan_to_pose')
        self.gripper_client = ActionClient(self, GripperCommand, '/demo_gripper_command')
        self.get_logger().info('official grasp state machine started; perception outputs drive pregrasp, approach, close, and lift')

    def on_point(self, msg):
        self.object_point = msg

    def on_normal(self, msg):
        self.normal = msg

    def on_axis(self, msg):
        self.long_axis = msg

    def on_joint_state(self, msg):
        for name, value in zip(msg.name, msg.position):
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                self.current_joints[str(name)] = value

    def on_odom(self, msg):
        self.last_odom = msg

    def on_material_report(self, msg):
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        self.material_reports.append((time.time(), payload))
        self.material_reports = self.material_reports[-32:]

    @staticmethod
    def _fmt_point(msg):
        return f'({msg.point.x:.3f}, {msg.point.y:.3f}, {msg.point.z:.3f}) frame={msg.header.frame_id}'

    @staticmethod
    def _fmt_vector(msg):
        return f'({msg.vector.x:.3f}, {msg.vector.y:.3f}, {msg.vector.z:.3f}) frame={msg.header.frame_id}'

    def _move_to_joint_pose(self, target, label, hold_seconds=0.6):
        names = list(self.get_parameter('joint_names').value)
        target = [float(v) for v in target]
        if len(target) != len(names):
            raise RuntimeError(f'{label} length must match joint_names length')
        start_deadline = time.time() + 2.0
        while rclpy.ok() and time.time() < start_deadline:
            if all(name in self.current_joints for name in names):
                break
            rclpy.spin_once(self, timeout_sec=0.05)
        start = [self.current_joints.get(name, target[i]) for i, name in enumerate(names)]
        max_delta = max(abs(a - b) for a, b in zip(start, target))
        steps = max(2, int(math.ceil(max_delta / max(1e-4, float(self.get_parameter('observation_max_joint_step').value)))))
        rate_hz = max(1.0, float(self.get_parameter('observation_rate_hz').value))
        period = 1.0 / rate_hz
        msg = JointState()
        msg.name = names
        for i in range(steps + 1):
            t = i / steps
            s = t * t * (3.0 - 2.0 * t)
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.position = [a + (b - a) * s for a, b in zip(start, target)]
            self.hand_pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(period)
        end_time = time.time() + hold_seconds
        while rclpy.ok() and time.time() < end_time:
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.position = target
            self.hand_pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(0.1)
        self.get_logger().info(f'{label} command sent')

    def move_to_observation_pose(self, hold_seconds=3.0):
        self._move_to_joint_pose(
            self.get_parameter('observation_pose').value,
            'observation pose',
            hold_seconds=hold_seconds,
        )

    def move_to_carry_pose(self):
        if not bool(self.get_parameter('carry_pose_before_place').value):
            return
        self._move_to_joint_pose(
            self.get_parameter('carry_pose').value,
            'carry pose',
            hold_seconds=float(self.get_parameter('carry_pose_hold_seconds').value),
        )

    def detect_object(self):
        server_timeout = float(self.get_parameter('detect_server_timeout_sec').value)
        self.get_logger().info(
            f'waiting for /demo_detect_object action server up to {server_timeout:.1f}s'
        )
        if not self.detect_client.wait_for_server(timeout_sec=server_timeout):
            raise RuntimeError('/demo_detect_object unavailable')
        goal = DetectObject.Goal()
        goal.target_classes = [str(v) for v in self.get_parameter('target_classes').value]
        goal.confidence_threshold = float(self.get_parameter('confidence_threshold').value)
        goal.timeout = float(self.get_parameter('detect_goal_timeout_sec').value)
        future = self.detect_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            raise RuntimeError('detect goal rejected')
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result().result
        if not result.success:
            raise RuntimeError(f'detect failed: {result.message}')
        self.get_logger().info(f'detected {result.detected_class}, confidence={result.confidence:.3f}')

    def _wait_for_inputs(self, timeout=10.0):
        start = time.time()
        while time.time() - start < timeout and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.object_point is not None and self.normal is not None and self.long_axis is not None:
                return True
        return False

    def _normalized_normal(self):
        if self.normal is None:
            return (0.0, 0.0, -1.0)
        nx, ny, nz = self.normal.vector.x, self.normal.vector.y, self.normal.vector.z
        norm = math.sqrt(nx * nx + ny * ny + nz * nz)
        if norm < 1e-9:
            return (0.0, 0.0, -1.0)
        return (nx / norm, ny / norm, nz / norm)

    def _offset_along_approach(self, source, distance):
        nx, ny, nz = self._normalized_normal()
        msg = PointStamped()
        msg.header = source.header
        msg.point.x = source.point.x - nx * distance
        msg.point.y = source.point.y - ny * distance
        msg.point.z = source.point.z - nz * distance
        return msg

    def _point_from_xyz(self, xyz):
        msg = PointStamped()
        if self.object_point is not None:
            msg.header = self.object_point.header
        else:
            msg.header.frame_id = str(self.get_parameter('planning_frame').value)
        msg.point.x = float(xyz[0])
        msg.point.y = float(xyz[1])
        msg.point.z = float(xyz[2])
        return msg

    def _vector_from_xyz(self, xyz):
        values = [float(v) for v in xyz]
        norm = math.sqrt(sum(v * v for v in values))
        if norm < 1.0e-9:
            raise RuntimeError(f'invalid zero vector: {xyz}')
        msg = Vector3Stamped()
        msg.header.frame_id = str(self.get_parameter('planning_frame').value)
        msg.vector.x = values[0] / norm
        msg.vector.y = values[1] / norm
        msg.vector.z = values[2] / norm
        return msg

    def _ensure_place_vectors(self):
        frame_id = str(self.get_parameter('planning_frame').value)
        if self.normal is None:
            self.normal = Vector3Stamped()
            self.normal.header.frame_id = frame_id
            self.normal.vector.x = 0.0
            self.normal.vector.y = 0.0
            self.normal.vector.z = -1.0
        if self.long_axis is None:
            axis = [float(v) for v in self.get_parameter('place_only_long_axis').value]
            norm = math.sqrt(sum(v * v for v in axis))
            if norm < 1e-9:
                axis = [1.0, 0.0, 0.0]
                norm = 1.0
            self.long_axis = Vector3Stamped()
            self.long_axis.header.frame_id = frame_id
            self.long_axis.vector.x = axis[0] / norm
            self.long_axis.vector.y = axis[1] / norm
            self.long_axis.vector.z = axis[2] / norm

    def _run_cargo_place_stage(self, execute):
        raise RuntimeError(
            'legacy cargo place stage is disabled; redesign placement before enabling this path'
        )

    def _run_cargo_place_stage_legacy(self, execute):
        # Deprecated: this old place path assumes the pencil is already rigidly
        # carried by the gripper and drives directly to a cargo-bay TCP point.
        # It is kept only as reference while the placement strategy is rebuilt.
        self._ensure_place_vectors()
        control_cargo_doors = bool(self.get_parameter('control_cargo_doors').value)
        if control_cargo_doors:
            self._send_cargo_command('bottom_close')
            self._send_cargo_command('side_open')
        self._move_to_place_stand()
        self._reposition_base_for_place()
        targets = self._cargo_place_targets()
        pre_place = self._point_from_xyz(targets['pre_place'])
        place = self._point_from_xyz(targets['tcp_place'])
        retreat = self._point_from_xyz(targets['retreat'])
        self.get_logger().info(f'pre_place={self._fmt_point(pre_place)}')
        self.get_logger().info(f'place={self._fmt_point(place)}')
        self._send_plan(pre_place, 'pre_place', execute=execute)
        self._send_plan(place, 'place', execute=execute)
        self._send_gripper('open', 100.0)
        if not self._validate_cargo_place(targets['target_id'], targets['object_place_base']):
            raise RuntimeError('cargo place validation failed')
        if bool(self.get_parameter('retreat_after_place').value):
            self._send_plan(retreat, 'retreat_after_place', execute=execute)
        if control_cargo_doors and bool(self.get_parameter('close_side_door_after_place').value):
            self._send_cargo_command('side_close')

    def _request_material_report(self, command, task, timeout=15.0):
        start = time.time()
        last_publish = 0.0
        msg = String()
        msg.data = str(command)
        while rclpy.ok() and time.time() - start < timeout:
            now = time.time()
            if now - last_publish >= 0.5:
                self.material_command_pub.publish(msg)
                last_publish = now
            rclpy.spin_once(self, timeout_sec=0.1)
            for stamp, payload in reversed(self.material_reports):
                if stamp >= start and payload.get('task') == task:
                    return payload
        raise RuntimeError(f'material report timed out: command={command}, task={task}')

    def _send_cargo_command(self, command, settle_sec=0.4):
        msg = String()
        msg.data = str(command)
        self.cargo_command_pub.publish(msg)
        self.get_logger().info(f'cargo command sent: {command}')
        end = time.time() + float(settle_sec)
        while rclpy.ok() and time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

    @staticmethod
    def _vec_sub(a, b):
        return [float(a[i]) - float(b[i]) for i in range(3)]

    @staticmethod
    def _vec_add(a, b):
        return [float(a[i]) + float(b[i]) for i in range(3)]

    @staticmethod
    def _dist(a, b):
        return math.sqrt(sum((float(a[i]) - float(b[i])) ** 2 for i in range(3)))

    @staticmethod
    def _yaw_from_odom(msg):
        q = msg.pose.pose.orientation
        return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    @staticmethod
    def _wrap_angle(angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def _current_odom_pose(self, timeout=3.0):
        deadline = time.time() + float(timeout)
        while rclpy.ok() and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            msg = self.last_odom
            if msg is None:
                continue
            p = msg.pose.pose.position
            yaw = self._yaw_from_odom(msg)
            values = [p.x, p.y, p.z, yaw]
            if all(math.isfinite(float(v)) for v in values):
                return float(p.x), float(p.y), float(p.z), float(yaw)
        raise RuntimeError('odom unavailable for base reposition')

    def _publish_base_stop(self, count=10):
        stop = Twist()
        for _ in range(max(1, int(count))):
            self.cmd_vel_pub.publish(stop)
            rclpy.spin_once(self, timeout_sec=0.02)

    def _drive_base_world_delta(self, delta_world, speed=None, timeout=None):
        start = self._current_odom_pose()
        target_x = start[0] + float(delta_world[0])
        target_y = start[1] + float(delta_world[1])
        if speed is None:
            speed = float(self.get_parameter('place_base_reposition_speed').value)
        if timeout is None:
            timeout = float(self.get_parameter('place_base_reposition_timeout').value)
        speed = max(0.02, abs(float(speed)))
        timeout = max(1.0, float(timeout))
        deadline = time.time() + timeout
        while rclpy.ok() and time.time() < deadline:
            x, y, z, yaw = self._current_odom_pose(timeout=0.2)
            dx = target_x - x
            dy = target_y - y
            dist = math.hypot(dx, dy)
            if dist <= 0.015:
                self._publish_base_stop()
                return True
            cmd = Twist()
            v = min(speed, max(0.025, dist * 0.8))
            cmd.linear.x = v * dx / max(dist, 1.0e-6)
            cmd.linear.y = v * dy / max(dist, 1.0e-6)
            self.cmd_vel_pub.publish(cmd)
            rclpy.spin_once(self, timeout_sec=0.03)
        self._publish_base_stop()
        return False

    def _drive_base_yaw_to(self, target_yaw, speed=None, tolerance=None, timeout=None):
        if speed is None:
            speed = float(self.get_parameter('video_place_yaw_speed').value)
        if tolerance is None:
            tolerance = float(self.get_parameter('video_place_yaw_tolerance').value)
        if timeout is None:
            timeout = float(self.get_parameter('place_stand_timeout').value)
        speed = max(0.03, abs(float(speed)))
        tolerance = max(0.01, abs(float(tolerance)))
        deadline = time.time() + max(1.0, float(timeout))
        while rclpy.ok() and time.time() < deadline:
            _x, _y, _z, yaw = self._current_odom_pose(timeout=0.2)
            yaw_err = self._wrap_angle(float(target_yaw) - yaw)
            if abs(yaw_err) <= tolerance:
                self._publish_base_stop()
                self.get_logger().info(
                    f'video place yaw reached: yaw={yaw:.3f}, target={float(target_yaw):.3f}, '
                    f'error={abs(yaw_err):.3f}'
                )
                return True
            cmd = Twist()
            cmd.angular.z = math.copysign(min(speed, max(0.04, abs(yaw_err) * 0.9)), yaw_err)
            self.cmd_vel_pub.publish(cmd)
            rclpy.spin_once(self, timeout_sec=0.03)
        self._publish_base_stop()
        return False

    def _retreat_base_before_lift(self):
        if not bool(self.get_parameter('retreat_base_before_lift').value):
            return
        delta = [float(v) for v in self.get_parameter('retreat_base_before_lift_delta').value]
        if len(delta) != 2:
            raise RuntimeError('retreat_base_before_lift_delta must contain [dx, dy]')
        self.get_logger().info(
            f'pre-lift base retreat command: delta_world=({delta[0]:.3f}, {delta[1]:.3f})'
        )
        ok = self._drive_base_world_delta(
            delta,
            speed=float(self.get_parameter('retreat_base_before_lift_speed').value),
            timeout=float(self.get_parameter('retreat_base_before_lift_timeout').value),
        )
        if not ok:
            raise RuntimeError('pre-lift base retreat failed')

    def _copy_point_with_z(self, point, z_value):
        msg = PointStamped()
        msg.header = point.header
        msg.point.x = point.point.x
        msg.point.y = point.point.y
        msg.point.z = float(z_value)
        return msg

    def _measured_right_tcp_point_base(self):
        measure = self._request_material_report('measure_tcp', 'measure_gripper_tcp')
        right_tcp = measure.get('right', {})
        tcp_base = right_tcp.get('tcp_base')
        if tcp_base is None:
            tcp_base = right_tcp.get('tcp_local_arm')
        if tcp_base is None:
            raise RuntimeError('measure_tcp report did not include right.tcp_base or right.tcp_local_arm')
        msg = PointStamped()
        msg.header.frame_id = str(self.get_parameter('planning_frame').value)
        msg.point.x = float(tcp_base[0])
        msg.point.y = float(tcp_base[1])
        msg.point.z = float(tcp_base[2])
        return msg

    def _post_retreat_lift_points(self, fallback_start, fallback_end):
        if not bool(self.get_parameter('post_retreat_lift_from_measured_tcp').value):
            return self._staged_lift_points(fallback_start, fallback_end)
        measured = self._measured_right_tcp_point_base()
        target_z = max(
            measured.point.z + max(0.0, fallback_end.point.z - fallback_start.point.z),
            float(self.get_parameter('min_lift_z').value),
        )
        lifted = self._copy_point_with_z(measured, target_z)
        self.get_logger().info(
            'lift uses measured tcp start: '
            f'start={self._fmt_point(measured)}, lifted={self._fmt_point(lifted)}'
        )
        return self._staged_lift_points(measured, lifted)

    def _drive_base_world_pose(self, target_pose):
        if len(target_pose) != 3:
            raise RuntimeError('place_stand_pose must contain [x, y, yaw]')
        target_x, target_y, target_yaw = [float(v) for v in target_pose]
        xy_tol = max(0.005, float(self.get_parameter('place_stand_xy_tolerance').value))
        yaw_tol = max(0.01, float(self.get_parameter('place_stand_yaw_tolerance').value))
        speed = max(0.02, abs(float(self.get_parameter('place_stand_speed').value)))
        angular_speed = max(0.03, abs(float(self.get_parameter('place_stand_angular_speed').value)))
        timeout = max(1.0, float(self.get_parameter('place_stand_timeout').value))
        deadline = time.time() + timeout
        while rclpy.ok() and time.time() < deadline:
            x, y, _z, yaw = self._current_odom_pose(timeout=0.2)
            dx = target_x - x
            dy = target_y - y
            dist = math.hypot(dx, dy)
            yaw_err = self._wrap_angle(target_yaw - yaw)
            if dist <= xy_tol and abs(yaw_err) <= yaw_tol:
                self._publish_base_stop()
                self.get_logger().info(
                    f'place stand reached: pose=({x:.3f}, {y:.3f}, {yaw:.3f}), '
                    f'error_xy={dist:.3f}, error_yaw={abs(yaw_err):.3f}'
                )
                return True
            cmd = Twist()
            if dist > xy_tol:
                v = min(speed, max(0.025, dist * 0.8))
                cmd.linear.x = v * dx / max(dist, 1.0e-6)
                cmd.linear.y = v * dy / max(dist, 1.0e-6)
            if abs(yaw_err) > yaw_tol:
                w = min(angular_speed, max(0.05, abs(yaw_err) * 0.9))
                cmd.angular.z = math.copysign(w, yaw_err)
            self.cmd_vel_pub.publish(cmd)
            rclpy.spin_once(self, timeout_sec=0.03)
        self._publish_base_stop()
        self.get_logger().warning(
            f'place stand timed out before pose=({target_x:.3f}, {target_y:.3f}, {target_yaw:.3f})'
        )
        return False

    def _move_to_place_stand(self):
        if not bool(self.get_parameter('place_stand_before_place').value):
            return
        target_pose = [float(v) for v in self.get_parameter('place_stand_pose').value]
        self.get_logger().info(
            f'place stand command: target=({target_pose[0]:.3f}, {target_pose[1]:.3f}, {target_pose[2]:.3f})'
        )
        if not self._drive_base_world_pose(target_pose):
            raise RuntimeError('place stand positioning failed')

    def _place_base_from_scene(self):
        scene = self._request_material_report('scene_points', 'scene_points')
        place_base = scene.get('place_base')
        if place_base is None or len(place_base) < 2:
            raise RuntimeError('scene_points report did not include place_base for base reposition')
        return scene, [float(v) for v in place_base]

    def _reposition_base_for_place(self):
        if not bool(self.get_parameter('reposition_base_before_place').value):
            return
        target_xy = [float(v) for v in self.get_parameter('place_base_target_xy').value]
        if len(target_xy) != 2:
            raise RuntimeError('place_base_target_xy must have two values')
        tol = max(0.005, float(self.get_parameter('place_base_xy_tolerance').value))
        max_step = max(0.0, float(self.get_parameter('place_base_max_step').value))
        iterations = max(0, int(self.get_parameter('place_base_reposition_iterations').value))
        for index in range(iterations):
            _, place_base = self._place_base_from_scene()
            # Moving the mobile base in +X/+Y makes a fixed scene target shift in
            # the opposite direction in base_link_arm. Move by current - target so
            # the measured place point converges toward the desired reachable window.
            err_base = [place_base[0] - target_xy[0], place_base[1] - target_xy[1]]
            err = math.hypot(err_base[0], err_base[1])
            self.get_logger().info(
                f'place base reposition check {index + 1}/{iterations}: '
                f'place_base_xy=({place_base[0]:.3f}, {place_base[1]:.3f}), '
                f'target_xy=({target_xy[0]:.3f}, {target_xy[1]:.3f}), error={err:.3f}'
            )
            if err <= tol:
                return
            if max_step <= 0.0:
                return
            scale = min(1.0, max_step / max(err, 1.0e-6))
            step_base = [err_base[0] * scale, err_base[1] * scale]
            _, _, _, yaw = self._current_odom_pose()
            cos_yaw = math.cos(yaw)
            sin_yaw = math.sin(yaw)
            delta_world = [
                cos_yaw * step_base[0] - sin_yaw * step_base[1],
                sin_yaw * step_base[0] + cos_yaw * step_base[1],
            ]
            self.get_logger().info(
                f'base reposition command: step_base=({step_base[0]:.3f}, {step_base[1]:.3f}), '
                f'delta_world=({delta_world[0]:.3f}, {delta_world[1]:.3f})'
            )
            if not self._drive_base_world_delta(delta_world):
                self.get_logger().warning('base reposition timed out before reaching computed delta')

    def _reposition_base_for_video_place(self):
        target_xy = [float(v) for v in self.get_parameter('video_place_target_base_xy').value]
        if len(target_xy) != 2:
            raise RuntimeError('video_place_target_base_xy must have two values')
        tol = max(0.005, float(self.get_parameter('video_place_base_xy_tolerance').value))
        max_step = max(0.0, float(self.get_parameter('video_place_base_max_step').value))
        iterations = max(0, int(self.get_parameter('video_place_base_reposition_iterations').value))
        speed = float(self.get_parameter('video_place_base_reposition_speed').value)
        timeout = float(self.get_parameter('video_place_base_reposition_timeout').value)
        for index in range(iterations):
            _scene, place_base = self._place_base_from_scene()
            err_base = [place_base[0] - target_xy[0], place_base[1] - target_xy[1]]
            err = math.hypot(err_base[0], err_base[1])
            self.get_logger().info(
                f'video place base check {index + 1}/{iterations}: '
                f'place_base_xy=({place_base[0]:.3f}, {place_base[1]:.3f}), '
                f'target_xy=({target_xy[0]:.3f}, {target_xy[1]:.3f}), error={err:.3f}'
            )
            if err <= tol or max_step <= 0.0:
                return
            scale = min(1.0, max_step / max(err, 1.0e-6))
            step_base = [err_base[0] * scale, err_base[1] * scale]
            _x, _y, _z, yaw = self._current_odom_pose()
            cos_yaw = math.cos(yaw)
            sin_yaw = math.sin(yaw)
            delta_world = [
                cos_yaw * step_base[0] - sin_yaw * step_base[1],
                sin_yaw * step_base[0] + cos_yaw * step_base[1],
            ]
            self.get_logger().info(
                f'video place base command: step_base=({step_base[0]:.3f}, {step_base[1]:.3f}), '
                f'delta_world=({delta_world[0]:.3f}, {delta_world[1]:.3f})'
            )
            if not self._drive_base_world_delta(delta_world, speed=speed, timeout=timeout):
                self.get_logger().warning('video place base reposition timed out before reaching computed delta')

    def _cargo_place_targets(self):
        target_id = str(self.get_parameter('cargo_target_id').value)
        scene = self._request_material_report('scene_points', 'scene_points')
        measure = self._request_material_report('measure_tcp', 'measure_gripper_tcp')
        points = {str(item.get('id')): item for item in scene.get('points', [])}
        target = points.get(target_id)
        if not target or target.get('base') is None:
            raise RuntimeError(f'no base position for cargo target: {target_id}')
        place_base = scene.get('place_base')
        if place_base is None:
            raise RuntimeError('scene_points report did not include place_base')
        place_base = [float(v) for v in place_base]
        if bool(self.get_parameter('min_place_z_from_cargo_body').value):
            cargo_body_base = scene.get('cargo_body_base')
            if cargo_body_base is not None and len(cargo_body_base) >= 3:
                clearance_z = (
                    float(cargo_body_base[2])
                    + float(self.get_parameter('cargo_body_tcp_clearance').value)
                )
                if place_base[2] < clearance_z:
                    self.get_logger().info(
                        'place z raised from scene target by cargo_body clearance: '
                        f'{place_base[2]:.3f} -> {clearance_z:.3f}'
                    )
                    place_base[2] = clearance_z
            else:
                self.get_logger().warning(
                    'min_place_z_from_cargo_body enabled but scene_points has no cargo_body_base'
                )
        right_tcp = measure.get('right', {})
        tcp_base = right_tcp.get('tcp_base')
        if tcp_base is None:
            tcp_base = right_tcp.get('tcp_local_arm')
        if tcp_base is None:
            raise RuntimeError('measure_tcp report did not include right.tcp_base or right.tcp_local_arm')

        object_base = target['base']
        held_offset = self._vec_sub(object_base, tcp_base)
        if bool(self.get_parameter('place_scene_point_is_tcp').value):
            tcp_place = [float(v) for v in place_base]
            object_place_base = self._vec_add(tcp_place, held_offset)
        else:
            object_place_offset = [0.0, 0.0, float(self.get_parameter('object_place_z_offset').value)]
            object_place_base = self._vec_add(place_base, object_place_offset)
            tcp_place = self._vec_sub(object_place_base, held_offset)
        normal = self._normalized_normal()
        pre_offset = float(self.get_parameter('place_pre_offset').value)
        retreat_offset = float(self.get_parameter('place_retreat_offset').value)

        def offset_against_approach(point, distance):
            return [
                float(point[0]) - normal[0] * distance,
                float(point[1]) - normal[1] * distance,
                float(point[2]) - normal[2] * distance,
            ]

        pre_place = offset_against_approach(tcp_place, pre_offset)
        retreat = offset_against_approach(tcp_place, retreat_offset)
        self.get_logger().info(
            'cargo place geometry: '
            f'object_base={[round(v, 3) for v in object_base]}, '
            f'tcp_base={[round(v, 3) for v in tcp_base]}, '
            f'held_offset={[round(v, 3) for v in held_offset]}, '
            f'place_base={[round(v, 3) for v in place_base]}, '
            f'place_scene_point_is_tcp={bool(self.get_parameter("place_scene_point_is_tcp").value)}, '
            f'tcp_place={[round(v, 3) for v in tcp_place]}'
        )
        return {
            'target_id': target_id,
            'object_base': object_base,
            'tcp_base': tcp_base,
            'held_offset': held_offset,
            'place_base': place_base,
            'object_place_base': object_place_base,
            'tcp_place': tcp_place,
            'pre_place': pre_place,
            'retreat': retreat,
        }

    def _validate_cargo_place(self, target_id, expected_base):
        scene = self._request_material_report('scene_points', 'scene_points')
        points = {str(item.get('id')): item for item in scene.get('points', [])}
        target = points.get(target_id)
        if not target or target.get('base') is None:
            self.get_logger().warning(f'cargo place validation missing target={target_id}')
            return False
        error = self._dist(target['base'], expected_base)
        tol = float(self.get_parameter('place_validation_tolerance').value)
        self.get_logger().info(
            f'cargo place validation: target={target_id}, error={error:.3f}m, tolerance={tol:.3f}m'
        )
        return error <= tol

    def _face_cargo_for_video_place(self):
        scene = self._request_material_report('scene_points', 'scene_points')
        target_world = scene.get('cargo_body_world') or scene.get('place_world')
        if target_world is None or len(target_world) < 2:
            raise RuntimeError('scene_points report did not include cargo/place world target')
        x, y, _z, _yaw = self._current_odom_pose()
        target_yaw = math.atan2(float(target_world[1]) - y, float(target_world[0]) - x)
        target_yaw += float(self.get_parameter('video_place_face_yaw_offset').value)
        self.get_logger().info(
            f'video place face cargo: robot=({x:.3f}, {y:.3f}), '
            f'target_world=({float(target_world[0]):.3f}, {float(target_world[1]):.3f}), '
            f'yaw_target={target_yaw:.3f}'
        )
        if not self._drive_base_yaw_to(target_yaw):
            raise RuntimeError('video place base yaw alignment failed')

    def _video_place_tcp_targets(self):
        scene = self._request_material_report('scene_points', 'scene_points')
        place_base = scene.get('place_base')
        if place_base is None or len(place_base) < 3:
            raise RuntimeError('scene_points report did not include place_base')
        tcp_place = [float(v) for v in place_base]
        cargo_body_base = scene.get('cargo_body_base')
        if cargo_body_base is not None and len(cargo_body_base) >= 3:
            min_z = float(cargo_body_base[2]) + float(
                self.get_parameter('video_place_tcp_clearance_from_cargo_body').value
            )
            if tcp_place[2] < min_z:
                self.get_logger().info(
                    f'video place z raised by cargo_body clearance: {tcp_place[2]:.3f} -> {min_z:.3f}'
                )
                tcp_place[2] = min_z
        pre_offset = float(self.get_parameter('video_place_pre_offset_x').value)
        retreat_offset = float(self.get_parameter('video_place_retreat_offset_x').value)
        post_lift = float(self.get_parameter('video_place_post_release_lift').value)
        pre_place = [tcp_place[0] - pre_offset, tcp_place[1], tcp_place[2]]
        release_lift = [tcp_place[0], tcp_place[1], tcp_place[2] + max(0.02, post_lift)]
        retreat = [tcp_place[0] - retreat_offset, tcp_place[1], release_lift[2]]
        return {
            'tcp_place': tcp_place,
            'pre_place': pre_place,
            'release_lift': release_lift,
            'retreat': retreat,
        }

    def _run_video_place_stage(self, execute):
        self.get_logger().info('video-style cargo place stage started: lift -> face cargo -> horizontalize -> base adjust -> forward place -> lift retreat')
        self._send_cargo_command('bottom_close')
        self._send_cargo_command('side_open')
        self._face_cargo_for_video_place()

        normal = self._vector_from_xyz(self.get_parameter('video_place_normal_base').value)
        long_axis = self._vector_from_xyz(self.get_parameter('video_place_long_axis_base').value)

        high_tcp = self._measured_right_tcp_point_base()
        self.get_logger().info(
            f'video place horizontalize at high tcp={self._fmt_point(high_tcp)}'
        )
        self._send_plan(
            high_tcp,
            'video_place_horizontalize',
            execute=execute,
            normal=normal,
            long_axis=long_axis,
        )

        self._reposition_base_for_video_place()
        targets = self._video_place_tcp_targets()
        pre_place = self._point_from_xyz(targets['pre_place'])
        place = self._point_from_xyz(targets['tcp_place'])
        release_lift = self._point_from_xyz(targets['release_lift'])
        retreat = self._point_from_xyz(targets['retreat'])
        self.get_logger().info(f'video pre_place={self._fmt_point(pre_place)}')
        self.get_logger().info(f'video place={self._fmt_point(place)}')
        self.get_logger().info(f'video release_lift={self._fmt_point(release_lift)}')
        self.get_logger().info(f'video retreat={self._fmt_point(retreat)}')
        self._send_plan(pre_place, 'video_pre_place', execute=execute, normal=normal, long_axis=long_axis)
        self._send_plan(place, 'video_place', execute=execute, normal=normal, long_axis=long_axis)
        self._send_gripper('open', 100.0)
        self._send_plan(release_lift, 'video_release_lift', execute=execute, normal=normal, long_axis=long_axis)
        self._send_plan(retreat, 'video_retreat_after_place', execute=execute, normal=normal, long_axis=long_axis)
        if bool(self.get_parameter('video_place_close_side_door_after_place').value):
            self._send_cargo_command('side_close')
        self.get_logger().info('video-style cargo place stage completed')

    def _clamp_min_z(self, point, min_z, label):
        min_z = float(min_z)
        if min_z <= 0.0 or point.point.z >= min_z:
            return point
        msg = PointStamped()
        msg.header = point.header
        msg.point.x = point.point.x
        msg.point.y = point.point.y
        msg.point.z = min_z
        self.get_logger().info(
            f'{label}: raised z from {point.point.z:.3f} to {min_z:.3f} for table clearance'
        )
        return msg

    def _validate_object_in_workspace(self, point):
        if not bool(self.get_parameter('require_object_in_workspace').value):
            return
        mn = [float(v) for v in self.get_parameter('object_workspace_min').value]
        mx = [float(v) for v in self.get_parameter('object_workspace_max').value]
        xyz = [float(point.point.x), float(point.point.y), float(point.point.z)]
        labels = ['x', 'y', 'z']
        outside = [
            f'{labels[i]}={xyz[i]:.3f} not in [{mn[i]:.3f}, {mx[i]:.3f}]'
            for i in range(3)
            if xyz[i] < mn[i] or xyz[i] > mx[i]
        ]
        if outside:
            raise RuntimeError(
                'object_point_base outside arm workspace; run office navigation to the workbench first: '
                + ', '.join(outside)
            )

    def _scene_point_for_target(self, target_id, timeout=None):
        timeout = float(timeout if timeout is not None else self.get_parameter('lift_validation_timeout').value)
        scene = self._request_material_report('scene_points', 'scene_points', timeout=timeout)
        for item in scene.get('points', []):
            if item.get('id') == target_id:
                return item
        raise RuntimeError(f'scene_points report did not include target id={target_id}')

    @staticmethod
    def _world_z(scene_point):
        world = scene_point.get('world')
        if not isinstance(world, list) or len(world) < 3:
            raise RuntimeError('scene_points target has no world z')
        return float(world[2])

    def _validate_lifted_object(self, initial_scene_point):
        if not bool(self.get_parameter('validate_lift_with_scene_points').value):
            return
        target_id = str(self.get_parameter('lift_validation_target_id').value)
        min_dz = float(self.get_parameter('lift_validation_min_world_dz').value)
        end = time.time() + 1.0
        while rclpy.ok() and time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
        after = self._scene_point_for_target(target_id)
        dz = self._world_z(after) - self._world_z(initial_scene_point)
        self.get_logger().info(
            f'lift validation: target={target_id}, world_dz={dz:.3f}, '
            f'before_world={initial_scene_point.get("world")}, after_world={after.get("world")}'
        )
        if dz < min_dz:
            raise RuntimeError(
                f'lift validation failed: {target_id} world dz={dz:.3f} < {min_dz:.3f}; '
                'gripper did not carry the object'
            )

    def _staged_lift_points(self, start, end):
        steps = max(1, int(self.get_parameter('lift_steps').value))
        if steps == 1:
            return [end]
        points = []
        for index in range(1, steps + 1):
            ratio = index / steps
            msg = PointStamped()
            msg.header = end.header
            msg.point.x = start.point.x + (end.point.x - start.point.x) * ratio
            msg.point.y = start.point.y + (end.point.y - start.point.y) * ratio
            msg.point.z = start.point.z + (end.point.z - start.point.z) * ratio
            points.append(msg)
        return points

    def _send_plan(self, point, label, execute=True, normal=None, long_axis=None):
        if not self.plan_client.wait_for_server(timeout_sec=5.0):
            raise RuntimeError('/demo_plan_to_pose unavailable')
        goal = PlanToPose.Goal()
        goal.target = point
        goal.normal = normal if normal is not None else self.normal
        goal.long_axis = long_axis if long_axis is not None else self.long_axis
        goal.execute = execute
        goal.label = label
        future = self.plan_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            raise RuntimeError(f'plan goal rejected: {label}')
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result().result
        if not result.success:
            raise RuntimeError(f'plan failed: {label}: {result.message}')
        self.get_logger().info(f'{label}: {result.message}')

    def _send_gripper(self, command, position):
        if not self.gripper_client.wait_for_server(timeout_sec=5.0):
            raise RuntimeError('/demo_gripper_command unavailable')
        goal = GripperCommand.Goal()
        goal.command = command
        goal.position = float(position)
        goal.speed = 50.0
        goal.wait_for_completion = True
        future = self.gripper_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            raise RuntimeError(f'gripper goal rejected: {command}')
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result().result
        if not result.success:
            raise RuntimeError(f'gripper failed: {result.message}')
        self.get_logger().info(f'gripper {command}: final_position={result.final_position:.1f}')

    def run_once(self):
        try:
            execute = bool(self.get_parameter('execute_grasp').value)
            if bool(self.get_parameter('place_only').value):
                self.get_logger().info('place_only enabled; assuming object is already grasped and lifted')
                self._run_cargo_place_stage(execute)
                self.get_logger().info('cargo place sequence completed')
                return True
            self.move_to_observation_pose()
            self.detect_object()
            if not self._wait_for_inputs():
                raise RuntimeError('perception outputs timed out')
            self.get_logger().info(f'object_point_base={self._fmt_point(self.object_point)}')
            self.get_logger().info(f'normal_base={self._fmt_vector(self.normal)}')
            self.get_logger().info(f'long_axis_base={self._fmt_vector(self.long_axis)}')
            self._validate_object_in_workspace(self.object_point)
            initial_scene_point = None
            if bool(self.get_parameter('validate_lift_with_scene_points').value):
                initial_scene_point = self._scene_point_for_target(
                    str(self.get_parameter('lift_validation_target_id').value)
                )
            tip = float(self.get_parameter('finger_tip_offset').value)
            pre = float(self.get_parameter('pre_grasp_offset').value)
            lift = float(self.get_parameter('lift_offset').value)
            pre_grasp = self._offset_along_approach(self.object_point, tip + pre)
            grasp = self._offset_along_approach(self.object_point, tip)
            lifted = self._offset_along_approach(self.object_point, tip + lift)
            grasp = self._clamp_min_z(grasp, self.get_parameter('min_grasp_z').value, 'grasp')
            pre_grasp = self._clamp_min_z(
                pre_grasp, self.get_parameter('min_pre_grasp_z').value, 'pre_grasp'
            )
            lifted = self._clamp_min_z(lifted, self.get_parameter('min_lift_z').value, 'lifted')
            self.get_logger().info(f'pre_grasp={self._fmt_point(pre_grasp)}')
            self.get_logger().info(f'grasp={self._fmt_point(grasp)}')
            self.get_logger().info(f'lifted={self._fmt_point(lifted)}')
            self._send_gripper('open', 100.0)
            self._send_plan(pre_grasp, 'pre_grasp', execute=execute)
            self._send_plan(grasp, 'grasp', execute=execute)
            self._send_gripper('close', 0.0)
            if bool(self.get_parameter('escape_lift_before_base_retreat').value):
                escape_z = max(grasp.point.z, float(self.get_parameter('escape_lift_z').value))
                escape = self._copy_point_with_z(grasp, escape_z)
                self.get_logger().info(f'escape_lift target={self._fmt_point(escape)}')
                self._send_plan(escape, 'escape_lift', execute=execute)
                lift_start = escape
            else:
                lift_start = grasp
            self._retreat_base_before_lift()
            lift_points = self._post_retreat_lift_points(lift_start, lifted)
            for index, lift_point in enumerate(lift_points, start=1):
                label = f'lift_{index:02d}' if len(lift_points) > 1 else 'lift'
                self.get_logger().info(f'{label} target={self._fmt_point(lift_point)}')
                self._send_plan(lift_point, label, execute=execute)
            if initial_scene_point is not None:
                self._validate_lifted_object(initial_scene_point)
            self.move_to_carry_pose()
            if bool(self.get_parameter('video_place_enabled').value):
                self._run_video_place_stage(execute)
                return True
            if bool(self.get_parameter('stop_after_lift').value):
                self.get_logger().info('stable grasp sequence completed at lifted pose; place stage disabled by task requirement')
                return True
            self._run_cargo_place_stage(execute)
        except RuntimeError as exc:
            self.get_logger().error(str(exc))
            return False
        self.get_logger().info('perception pipeline demo completed')
        return True


def main(args=None):
    rclpy.init(args=args)
    node = PickPlaceStateMachine()
    try:
        node.run_once()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
