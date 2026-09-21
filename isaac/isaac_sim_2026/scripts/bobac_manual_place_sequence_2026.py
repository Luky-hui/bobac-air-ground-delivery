#!/usr/bin/env python3
"""Replay the manually measured Bobac place sequence.

This script only publishes ROS commands:
- /joint_command for right arm and gripper JointState targets
- /bobac_kinematic_cmd_vel for base odom servoing

It does not modify USD prims or material objects.
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Sequence, Tuple

import rclpy
from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String


ARM_JOINTS = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
GRIPPER_JOINTS = ["gripper_l_joint1", "gripper_r_joint1"]

PLACE_ARM_WAYPOINTS = [
    [-0.023600, 0.685000, -0.770400, -0.400700, 1.582600, 0.621400],
    [-0.023300, 0.685000, -0.380200, -0.400800, 1.582500, 0.621400],
    [-0.023200, 0.685000, -0.379900, -0.403000, 1.584200, 0.096400],
    [-1.973200, 0.685000, -0.379900, -0.403000, 1.584500, 0.096400],
    [-1.973900, 0.115000, -0.379900, -0.403100, 1.585200, 0.096100],
    [-1.973700, 0.114900, -0.380500, -0.399400, 1.578700, 1.326300],
    [-1.973600, -0.130100, -0.380500, -0.399400, 1.578600, 1.326400],
    # [-1.973600, -0.155100, -0.380500, -0.399400, 1.578600, 1.326400],
]

DEFAULT_BASE_TARGET = [3.827835, -0.360926, -0.051663]
DEFAULT_RESET_ARM_POSE = PLACE_ARM_WAYPOINTS[0]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def shortest_angle_delta(target: float, actual: float) -> float:
    return math.atan2(math.sin(target - actual), math.cos(target - actual))


def quat_to_yaw(msg: Odometry) -> float:
    q = msg.pose.pose.orientation
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def smoothstep(t: float) -> float:
    t = clamp(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


class BobacManualPlaceSequence(Node):
    def __init__(self) -> None:
        super().__init__("bobac_manual_place_sequence_2026")

        self.declare_parameter("joint_state_topic", "/bobac_joint_states")
        self.declare_parameter("joint_command_topic", "/joint_command")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("cmd_vel_topic", "/bobac_kinematic_cmd_vel")
        self.declare_parameter("cargo_command_topic", "/cargo_bay/command")
        self.declare_parameter("cargo_status_topic", "/cargo_bay/status")
        self.declare_parameter("drone_command_topic", "/drone/command")
        self.declare_parameter("drone_status_topic", "/drone/status")
        self.declare_parameter("drone_world_position_topic", "/drone/world_position")
        self.declare_parameter("base_target", DEFAULT_BASE_TARGET)
        self.declare_parameter("reset_arm_pose", DEFAULT_RESET_ARM_POSE)
        self.declare_parameter("arm_segment_duration", 8.0)
        self.declare_parameter("reset_duration", 4.0)
        self.declare_parameter("rate_hz", 50.0)
        self.declare_parameter("base_max_linear", 0.10)
        self.declare_parameter("base_max_angular", 0.35)
        self.declare_parameter("base_xy_tolerance", 0.025)
        self.declare_parameter("base_yaw_tolerance", 0.035)
        self.declare_parameter("base_timeout", 35.0)
        self.declare_parameter("base_kp_linear", 0.8)
        self.declare_parameter("base_kp_angular", 1.1)
        self.declare_parameter("execute_base_move", False)
        self.declare_parameter("open_gripper", True)
        self.declare_parameter("open_gripper_angle", 0.30)
        self.declare_parameter("hold_gripper_closed_until_release", True)
        self.declare_parameter("closed_gripper_angle", -0.85)
        self.declare_parameter("release_hold_sec", 1.0)
        self.declare_parameter("reset_after_release", True)
        self.declare_parameter("open_side_door_before_place", True)
        self.declare_parameter("close_side_door_after_place", True)
        self.declare_parameter("close_bottom_door_before_place", True)
        self.declare_parameter("lock_payload_after_place", False)
        self.declare_parameter("cargo_prepare_before_waypoint", 4)
        self.declare_parameter("cargo_status_wait_sec", 2.0)
        self.declare_parameter("cargo_close_settle_sec", 2.2)
        self.declare_parameter("enable_post_place_drone_rise", False)
        self.declare_parameter("post_place_drone_rise", 0.30)
        self.declare_parameter("post_place_drone_rise_wait_sec", 4.0)
        self.declare_parameter("dry_run", False)

        self.joint_pub = self.create_publisher(
            JointState, str(self.get_parameter("joint_command_topic").value), 10
        )
        self.cmd_pub = self.create_publisher(
            Twist, str(self.get_parameter("cmd_vel_topic").value), 10
        )
        self.cargo_pub = self.create_publisher(
            String, str(self.get_parameter("cargo_command_topic").value), 10
        )
        self.drone_pub = self.create_publisher(
            String, str(self.get_parameter("drone_command_topic").value), 10
        )
        self.current_joints: Optional[Dict[str, float]] = None
        self.current_odom: Optional[Odometry] = None
        self.current_drone_position: Optional[PointStamped] = None
        self.drone_status = ""
        self.cargo_status = ""
        self.released = False
        self.create_subscription(
            JointState,
            str(self.get_parameter("joint_state_topic").value),
            self._joint_cb,
            10,
        )
        self.create_subscription(
            Odometry,
            str(self.get_parameter("odom_topic").value),
            self._odom_cb,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("cargo_status_topic").value),
            self._cargo_status_cb,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("drone_status_topic").value),
            self._drone_status_cb,
            10,
        )
        self.create_subscription(
            PointStamped,
            str(self.get_parameter("drone_world_position_topic").value),
            self._drone_position_cb,
            10,
        )

    def _joint_cb(self, msg: JointState) -> None:
        values = dict(zip(msg.name, msg.position))
        if all(name in values for name in ARM_JOINTS):
            self.current_joints = {name: float(values[name]) for name in ARM_JOINTS}

    def _odom_cb(self, msg: Odometry) -> None:
        self.current_odom = msg

    def _cargo_status_cb(self, msg: String) -> None:
        self.cargo_status = msg.data
        self.get_logger().info(f"cargo status: {msg.data}")

    def _drone_status_cb(self, msg: String) -> None:
        self.drone_status = msg.data
        self.get_logger().info(f"drone status: {msg.data}")

    def _drone_position_cb(self, msg: PointStamped) -> None:
        self.current_drone_position = msg

    def send_cargo_command(self, command: str, expected: str) -> bool:
        timeout = max(0.0, float(self.get_parameter("cargo_status_wait_sec").value))
        self.cargo_status = ""
        deadline = time.time() + 1.0
        while (
            rclpy.ok()
            and time.time() < deadline
            and self.cargo_pub.get_subscription_count() == 0
        ):
            rclpy.spin_once(self, timeout_sec=0.05)

        self.get_logger().info(f"cargo command: {command}")
        if not bool(self.get_parameter("dry_run").value):
            msg = String()
            msg.data = command
            for _ in range(3):
                self.cargo_pub.publish(msg)
                rclpy.spin_once(self, timeout_sec=0.05)

        if timeout <= 0.0:
            return True
        end = time.time() + timeout
        while rclpy.ok() and time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.1)
            if expected and expected in self.cargo_status:
                if "close" in command:
                    end_settle = time.time() + float(
                        self.get_parameter("cargo_close_settle_sec").value
                    )
                    while rclpy.ok() and time.time() < end_settle:
                        rclpy.spin_once(self, timeout_sec=0.05)
                return True
        self.get_logger().warning(
            f"cargo command '{command}' not confirmed as '{expected}' within {timeout:.1f}s"
        )
        return False

    def _wait_ready(self, timeout_sec: float = 10.0) -> bool:
        end = time.time() + timeout_sec
        while rclpy.ok() and time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.current_joints is not None and self.current_odom is not None:
                return True
        self.get_logger().error(
            f"not ready: joints={self.current_joints is not None}, odom={self.current_odom is not None}"
        )
        return False

    def _publish_arm(self, q: Sequence[float]) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(ARM_JOINTS)
        msg.position = [float(v) for v in q]
        if (
            bool(self.get_parameter("hold_gripper_closed_until_release").value)
            and not self.released
        ):
            left_angle = float(self.get_parameter("closed_gripper_angle").value)
            msg.name.extend(GRIPPER_JOINTS)
            msg.position.extend([left_angle, -left_angle])
        if not bool(self.get_parameter("dry_run").value):
            self.joint_pub.publish(msg)

    def _publish_gripper(self, left_angle: float) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(GRIPPER_JOINTS)
        msg.position = [float(left_angle), -float(left_angle)]
        if not bool(self.get_parameter("dry_run").value):
            self.joint_pub.publish(msg)

    def _publish_stop(self, count: int = 10) -> None:
        stop = Twist()
        for _ in range(count):
            if not bool(self.get_parameter("dry_run").value):
                self.cmd_pub.publish(stop)
            rclpy.spin_once(self, timeout_sec=0.02)

    def rise_drone_after_close(self) -> None:
        if not bool(self.get_parameter("enable_post_place_drone_rise").value):
            self.get_logger().info(
                "post_place_drone_rise disabled; PX4 mission owns drone takeoff"
            )
            return
        height = float(self.get_parameter("post_place_drone_rise").value)
        if height <= 0.0:
            return
        wait_sec = max(0.0, float(self.get_parameter("post_place_drone_rise_wait_sec").value))
        deadline = time.time() + 3.0
        while (
            rclpy.ok()
            and time.time() < deadline
            and (self.current_drone_position is None or self.drone_pub.get_subscription_count() == 0)
        ):
            rclpy.spin_once(self, timeout_sec=0.05)
        if self.current_drone_position is None:
            self.get_logger().warning("post_place_drone_rise skipped: no /drone/world_position")
            return
        p = self.current_drone_position.point
        start_z = float(p.z)
        target_z = start_z + height
        command = f"goto {p.x:.6f} {p.y:.6f} {target_z:.6f} 0.000000"
        self.get_logger().info(
            f"post_place_drone_rise: before=[{p.x:.3f},{p.y:.3f},{p.z:.3f}], command={command}"
        )
        if not bool(self.get_parameter("dry_run").value):
            msg = String()
            msg.data = command
            for _ in range(3):
                self.drone_pub.publish(msg)
                rclpy.spin_once(self, timeout_sec=0.05)
        end = time.time() + wait_sec
        while rclpy.ok() and time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.current_drone_position is not None:
                if self.current_drone_position.point.z >= target_z - 0.04:
                    break
        if self.current_drone_position is not None:
            p = self.current_drone_position.point
            self.get_logger().info(
                f"post_place_drone_rise: after=[{p.x:.3f},{p.y:.3f},{p.z:.3f}], dz={p.z - start_z:.3f}"
            )

    def _current_arm_q(self) -> Optional[List[float]]:
        if self.current_joints is None:
            return None
        return [self.current_joints[name] for name in ARM_JOINTS]

    def move_arm_to(self, label: str, target: Sequence[float], duration: float) -> bool:
        start = self._current_arm_q()
        if start is None:
            self.get_logger().error(f"{label}: no arm joint state")
            return False
        target = [float(v) for v in target]
        if len(target) != len(ARM_JOINTS):
            self.get_logger().error(f"{label}: target must contain 6 values")
            return False

        rate_hz = max(1.0, float(self.get_parameter("rate_hz").value))
        duration = max(0.1, float(duration))
        steps = max(1, int(duration * rate_hz))
        self.get_logger().info(
            f"{label}: moving arm over {duration:.1f}s to "
            f"[{', '.join(f'{v:.4f}' for v in target)}]"
        )
        for i in range(steps + 1):
            if not rclpy.ok():
                return False
            s = smoothstep(i / steps)
            q = [a + (b - a) * s for a, b in zip(start, target)]
            self._publish_arm(q)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(1.0 / rate_hz)
        return True

    def _pose(self) -> Optional[Tuple[float, float, float]]:
        if self.current_odom is None:
            return None
        p = self.current_odom.pose.pose.position
        yaw = quat_to_yaw(self.current_odom)
        if not all(math.isfinite(v) for v in (p.x, p.y, yaw)):
            return None
        return float(p.x), float(p.y), float(yaw)

    def drive_base_to(self, target: Sequence[float]) -> bool:
        if len(target) != 3:
            self.get_logger().error("base_target must contain [x, y, yaw]")
            return False
        target_x, target_y, target_yaw = [float(v) for v in target]
        xy_tol = abs(float(self.get_parameter("base_xy_tolerance").value))
        yaw_tol = abs(float(self.get_parameter("base_yaw_tolerance").value))
        max_v = abs(float(self.get_parameter("base_max_linear").value))
        max_w = abs(float(self.get_parameter("base_max_angular").value))
        kp_v = float(self.get_parameter("base_kp_linear").value)
        kp_w = float(self.get_parameter("base_kp_angular").value)
        timeout = max(1.0, float(self.get_parameter("base_timeout").value))
        rate_hz = max(1.0, float(self.get_parameter("rate_hz").value))
        period = 1.0 / rate_hz
        deadline = time.time() + timeout
        next_log = 0.0

        self.get_logger().info(
            f"base: driving to [{target_x:.6f}, {target_y:.6f}, {target_yaw:.6f}]"
        )
        while rclpy.ok() and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.0)
            pose = self._pose()
            if pose is None:
                self.get_logger().error("base: odom unavailable")
                self._publish_stop()
                return False
            x, y, yaw = pose
            dx = target_x - x
            dy = target_y - y
            dist = math.hypot(dx, dy)
            yaw_error = shortest_angle_delta(target_yaw, yaw)
            if dist <= xy_tol and abs(yaw_error) <= yaw_tol:
                self._publish_stop()
                self.get_logger().info(
                    f"base: arrived [{x:.6f}, {y:.6f}, {yaw:.6f}], "
                    f"dist={dist:.3f}, yaw_err={yaw_error:.3f}"
                )
                return True

            cmd = Twist()
            if dist > xy_tol:
                cmd.linear.x = clamp(kp_v * dx, -max_v, max_v)
                cmd.linear.y = clamp(kp_v * dy, -max_v, max_v)
            if abs(yaw_error) > yaw_tol:
                cmd.angular.z = clamp(kp_w * yaw_error, -max_w, max_w)
            if not bool(self.get_parameter("dry_run").value):
                self.cmd_pub.publish(cmd)

            now = time.time()
            if now >= next_log:
                self.get_logger().info(
                    f"base: pose=[{x:.3f},{y:.3f},{yaw:.3f}], "
                    f"dist={dist:.3f}, yaw_err={yaw_error:.3f}, "
                    f"cmd=({cmd.linear.x:.3f},{cmd.linear.y:.3f},{cmd.angular.z:.3f})"
                )
                next_log = now + 2.0
            time.sleep(period)

        self._publish_stop()
        self.get_logger().error("base: target timeout")
        return False

    def open_gripper(self) -> None:
        angle = float(self.get_parameter("open_gripper_angle").value)
        hold = max(0.0, float(self.get_parameter("release_hold_sec").value))
        end = time.time() + hold
        self.get_logger().info(f"release: opening gripper to {angle:.3f} rad")
        self.released = True
        while rclpy.ok() and time.time() < end:
            self._publish_gripper(angle)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(1.0 / max(1.0, float(self.get_parameter("rate_hz").value)))

    def run(self) -> bool:
        if not self._wait_ready():
            return False
        cargo_prepared = False
        prepare_index = int(self.get_parameter("cargo_prepare_before_waypoint").value)
        prepare_index = max(1, min(len(PLACE_ARM_WAYPOINTS), prepare_index))
        duration = float(self.get_parameter("arm_segment_duration").value)
        for index, target in enumerate(PLACE_ARM_WAYPOINTS, start=1):
            if index >= prepare_index and not cargo_prepared:
                if bool(self.get_parameter("close_bottom_door_before_place").value):
                    self.send_cargo_command("bottom_close", "bottom_closed")
                if bool(self.get_parameter("open_side_door_before_place").value):
                    self.send_cargo_command("left_open", "left_opened")
                cargo_prepared = True
            if not self.move_arm_to(f"arm_waypoint_{index}", target, duration):
                return False
            if index == 3 and bool(self.get_parameter("execute_base_move").value):
                if not self.drive_base_to(self.get_parameter("base_target").value):
                    return False
        if bool(self.get_parameter("open_gripper").value):
            self.open_gripper()
        if bool(self.get_parameter("reset_after_release").value):
            if not self.move_arm_to(
                "arm_reset_after_release",
                self.get_parameter("reset_arm_pose").value,
                float(self.get_parameter("reset_duration").value),
            ):
                return False
        if bool(self.get_parameter("close_side_door_after_place").value):
            self.send_cargo_command("left_close", "left_closed")
        if bool(self.get_parameter("lock_payload_after_place").value):
            self.send_cargo_command("payload_lock", "payload_locked=True")
        self.rise_drone_after_close()
        self._publish_stop()
        self.get_logger().info("manual place sequence completed")
        return True


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BobacManualPlaceSequence()
    try:
        node.run()
    finally:
        node._publish_stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
