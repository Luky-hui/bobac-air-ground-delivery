#!/usr/bin/env python3
"""Two-stage odom navigation for Bobac sorting-task waypoint marking."""

from __future__ import annotations

import math
import time
from typing import List, Optional, Sequence, Tuple

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


DEFAULT_WAYPOINTS = [
    1.760000,
    -1.100000,
    1.661177,
    3.436775,
    -0.727569,
    1.661177,
    3.970959,
    -0.368233,
    1.577041,
]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def shortest_angle_delta(target: float, actual: float) -> float:
    return math.atan2(math.sin(target - actual), math.cos(target - actual))


def wrap_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def clockwise_distance(current: float, target: float) -> float:
    return (current - target) % (2.0 * math.pi)


def quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def parse_waypoints(flat_values: Sequence[float]) -> List[Tuple[float, float, float]]:
    values = [float(v) for v in flat_values]
    if len(values) == 0 or len(values) % 3 != 0:
        raise ValueError("waypoints must be a flat [x, y, yaw, ...] list")
    return [(values[i], values[i + 1], values[i + 2]) for i in range(0, len(values), 3)]


class BobacSortOdomNav(Node):
    def __init__(self) -> None:
        super().__init__("bobac_sort_odom_nav_2026")

        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("waypoints", DEFAULT_WAYPOINTS)
        self.declare_parameter("prestop_xy_tolerance", 0.12)
        self.declare_parameter("final_xy_tolerance", 0.055)
        self.declare_parameter("yaw_tolerance", 0.08)
        self.declare_parameter("max_linear", 1.20)
        self.declare_parameter("min_linear", 0.30)
        self.declare_parameter("fine_min_linear", 0.10)
        self.declare_parameter("max_angular", 1.60)
        self.declare_parameter("min_angular", 0.32)
        self.declare_parameter("linear_kp", 1.00)
        self.declare_parameter("angular_kp", 1.10)
        self.declare_parameter("fine_radius", 0.28)
        self.declare_parameter("align_yaw_tolerance", 0.12)
        self.declare_parameter("forward_only_mode", True)
        self.declare_parameter("world_frame_cmd", False)
        self.declare_parameter("max_abs_z", 0.12)
        self.declare_parameter("settle_sec", 0.4)
        self.declare_parameter("goal_timeout_sec", 90.0)
        self.declare_parameter("publish_hz", 30.0)
        self.declare_parameter("publish_stop_count", 15)
        self.declare_parameter("use_scan_safety", True)
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("scan_stale_sec", 0.8)
        self.declare_parameter("front_stop_range", 0.34)
        self.declare_parameter("front_slow_range", 0.55)
        self.declare_parameter("front_angle_width", 0.70)
        self.declare_parameter("scan_slow_scale", 0.35)

        self.last_odom: Optional[Odometry] = None
        self.last_scan: Optional[LaserScan] = None
        self.last_scan_time = 0.0
        self.create_subscription(
            Odometry,
            str(self.get_parameter("odom_topic").value),
            self._odom_cb,
            10,
        )
        self.create_subscription(
            LaserScan,
            str(self.get_parameter("scan_topic").value),
            self._scan_cb,
            10,
        )
        self.cmd_pub = self.create_publisher(
            Twist, str(self.get_parameter("cmd_vel_topic").value), 10
        )

    def _odom_cb(self, msg: Odometry) -> None:
        self.last_odom = msg

    def _scan_cb(self, msg: LaserScan) -> None:
        self.last_scan = msg
        self.last_scan_time = time.time()

    def _front_clearance(self) -> Optional[float]:
        if not bool(self.get_parameter("use_scan_safety").value):
            return None
        scan = self.last_scan
        if scan is None or not scan.ranges:
            return None
        if time.time() - self.last_scan_time > float(self.get_parameter("scan_stale_sec").value):
            return None

        half_width = abs(float(self.get_parameter("front_angle_width").value)) * 0.5
        best = math.inf
        for index, value in enumerate(scan.ranges):
            distance = float(value)
            if not math.isfinite(distance):
                continue
            if distance < scan.range_min or distance > scan.range_max:
                continue
            angle = scan.angle_min + float(index) * scan.angle_increment
            if abs(angle) <= half_width:
                best = min(best, distance)
        if math.isfinite(best):
            return best
        return None

    def _current_pose(self) -> Optional[Tuple[float, float, float, float]]:
        msg = self.last_odom
        if msg is None:
            return None
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = quat_to_yaw(q.x, q.y, q.z, q.w)
        values = [
            p.x,
            p.y,
            p.z,
            yaw,
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.angular.z,
        ]
        if not all(math.isfinite(v) for v in values):
            return None
        return p.x, p.y, p.z, yaw

    def _publish_stop(self) -> None:
        stop = Twist()
        for _ in range(max(1, int(self.get_parameter("publish_stop_count").value))):
            try:
                self.cmd_pub.publish(stop)
                rclpy.spin_once(self, timeout_sec=0.02)
            except Exception:
                return

    def _wait_for_odom(self) -> bool:
        deadline = time.time() + 10.0
        while rclpy.ok() and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self._current_pose() is not None:
                return True
        return False

    def _settle_and_report(self, label: str, target: Tuple[float, float, float]) -> bool:
        self._publish_stop()
        end = time.time() + max(0.0, float(self.get_parameter("settle_sec").value))
        while rclpy.ok() and time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

        pose = self._current_pose()
        if pose is None:
            self.get_logger().error(f"{label}: odom invalid during settle")
            return False
        xy_error = math.hypot(target[0] - pose[0], target[1] - pose[1])
        yaw_error = abs(shortest_angle_delta(target[2], pose[3]))
        self.get_logger().info(
            f"{label}: settled pose [{pose[0]:.6f}, {pose[1]:.6f}, {pose[3]:.6f}], "
            f"error xy={xy_error:.3f}m yaw={yaw_error:.3f}rad"
        )
        return True

    def _drive_to(self, label: str, target: Tuple[float, float, float], final: bool) -> bool:
        timeout = float(self.get_parameter("goal_timeout_sec").value)
        publish_hz = max(1.0, float(self.get_parameter("publish_hz").value))
        period = 1.0 / publish_hz
        xy_tol = float(
            self.get_parameter("final_xy_tolerance" if final else "prestop_xy_tolerance").value
        )
        yaw_tol = float(self.get_parameter("yaw_tolerance").value)
        max_v = abs(float(self.get_parameter("max_linear").value))
        min_v = abs(float(self.get_parameter("min_linear").value))
        fine_min_v = abs(float(self.get_parameter("fine_min_linear").value))
        max_w = abs(float(self.get_parameter("max_angular").value))
        min_w = abs(float(self.get_parameter("min_angular").value))
        linear_kp = float(self.get_parameter("linear_kp").value)
        angular_kp = float(self.get_parameter("angular_kp").value)
        fine_radius = abs(float(self.get_parameter("fine_radius").value))
        align_yaw_tol = abs(float(self.get_parameter("align_yaw_tolerance").value))
        forward_only = bool(self.get_parameter("forward_only_mode").value)
        world_frame_cmd = bool(self.get_parameter("world_frame_cmd").value)
        max_z = abs(float(self.get_parameter("max_abs_z").value))

        target_x, target_y, target_yaw = target
        self.get_logger().info(
            f"{label}: target x={target_x:.6f}, y={target_y:.6f}, yaw={target_yaw:.6f}, "
            f"xy_tol={xy_tol:.3f}, min_v={min_v:.3f}"
        )

        next_log_time = 0.0
        deadline = time.time() + timeout
        while rclpy.ok() and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            pose = self._current_pose()
            if pose is None:
                self.get_logger().error(f"{label}: /odom invalid")
                self._publish_stop()
                return False

            x, y, z, yaw = pose
            if abs(z) > max_z:
                self.get_logger().error(f"{label}: base z={z:.3f} exceeds {max_z:.3f}")
                self._publish_stop()
                return False

            dx = target_x - x
            dy = target_y - y
            dist = math.hypot(dx, dy)
            yaw_error = shortest_angle_delta(target_yaw, yaw)
            if dist <= xy_tol and abs(yaw_error) <= yaw_tol:
                return self._settle_and_report(label, target)

            cos_yaw = math.cos(yaw)
            sin_yaw = math.sin(yaw)
            # Bobac imported wheel frame is rotated relative to the published
            # odom frame: cmd.x drives odom -Y and cmd.y drives odom +X at yaw=0.
            body_x = sin_yaw * dx - cos_yaw * dy
            body_y = cos_yaw * dx + sin_yaw * dy

            cmd = Twist()
            if world_frame_cmd:
                if dist > xy_tol:
                    requested_speed = min(max_v, linear_kp * dist)
                    floor = fine_min_v if dist < fine_radius else min_v
                    requested_speed = max(floor, requested_speed)
                    cmd.linear.x = requested_speed * dx / max(dist, 1.0e-6)
                    cmd.linear.y = requested_speed * dy / max(dist, 1.0e-6)
                else:
                    requested_w = clamp(angular_kp * yaw_error, -max_w, max_w)
                    if abs(yaw_error) > yaw_tol:
                        if 0.0 < abs(requested_w) < min_w:
                            requested_w = math.copysign(min_w, requested_w)
                        cmd.angular.z = requested_w
            elif forward_only:
                if dist > xy_tol:
                    target_heading = math.atan2(dy, dx)
                    desired_yaw = wrap_pi(target_heading + math.pi / 2.0)
                    cw = clockwise_distance(yaw, desired_yaw)
                    if cw > align_yaw_tol:
                        cmd.angular.z = clamp(angular_kp * cw, min_w, max_w)
                    else:
                        requested_speed = min(max_v, linear_kp * dist)
                        floor = fine_min_v if dist < fine_radius else min_v
                        cmd.linear.x = max(floor, requested_speed)
                else:
                    cw = clockwise_distance(yaw, target_yaw)
                    if cw > yaw_tol:
                        cmd.angular.z = clamp(angular_kp * cw, min_w, max_w)
            else:
                if dist > xy_tol:
                    requested_speed = min(max_v, linear_kp * dist)
                    floor = fine_min_v if dist < fine_radius else min_v
                    requested_speed = max(floor, requested_speed)
                    cmd.linear.x = requested_speed * body_x / max(dist, 1.0e-6)
                    cmd.linear.y = requested_speed * body_y / max(dist, 1.0e-6)

                requested_w = clamp(angular_kp * yaw_error, -max_w, max_w)
                if abs(yaw_error) > yaw_tol:
                    if 0.0 < abs(requested_w) < min_w:
                        requested_w = math.copysign(min_w, requested_w)
                    cmd.angular.z = -requested_w

            clearance = self._front_clearance()
            if clearance is not None:
                stop_range = abs(float(self.get_parameter("front_stop_range").value))
                slow_range = max(stop_range, abs(float(self.get_parameter("front_slow_range").value)))
                slow_scale = clamp(
                    float(self.get_parameter("scan_slow_scale").value), 0.05, 1.0
                )
                if world_frame_cmd:
                    # Project a world-frame command onto Bobac's physical front
                    # direction, so the laser safety blocks only motion that
                    # keeps pushing into the obstacle seen by the front scan.
                    forward_component = cmd.linear.x * math.sin(yaw) - cmd.linear.y * math.cos(yaw)
                else:
                    forward_component = cmd.linear.x
                translating_forward = forward_component > 0.01
                translating_any = abs(cmd.linear.x) > 0.01 or abs(cmd.linear.y) > 0.01
                if clearance < stop_range and translating_forward:
                    cmd.linear.x = 0.0
                    cmd.linear.y = 0.0
                    if abs(cmd.angular.z) < min_w:
                        cmd.angular.z = min_w
                elif clearance < slow_range and translating_forward:
                    cmd.linear.x *= slow_scale
                    cmd.linear.y *= slow_scale

            self.cmd_pub.publish(cmd)
            now = time.time()
            if now >= next_log_time:
                scan_text = "none" if clearance is None else f"{clearance:.2f}m"
                self.get_logger().info(
                    f"{label}: dist={dist:.3f} yaw_err={yaw_error:.3f} "
                    f"front={scan_text} "
                    f"cmd=({cmd.linear.x:.3f},{cmd.linear.y:.3f},{cmd.angular.z:.3f})"
                )
                next_log_time = now + 2.0
            time.sleep(period)

        self._publish_stop()
        pose = self._current_pose()
        if pose is None:
            self.get_logger().error(f"{label}: timed out and odom invalid")
            return False
        dist = math.hypot(target_x - pose[0], target_y - pose[1])
        yaw_error = abs(shortest_angle_delta(target_yaw, pose[3]))
        self.get_logger().error(
            f"{label}: timed out, pose [{pose[0]:.6f}, {pose[1]:.6f}, {pose[3]:.6f}], "
            f"error xy={dist:.3f}m yaw={yaw_error:.3f}rad"
        )
        return False

    def run(self) -> bool:
        if not self._wait_for_odom():
            self.get_logger().error("no valid /odom; is Bobac scene playing?")
            return False

        waypoints = parse_waypoints(self.get_parameter("waypoints").value)
        labels = [f"coarse_{i + 1}" for i in range(max(0, len(waypoints) - 1))] + [
            "final_stop"
        ]
        for index, (label, waypoint) in enumerate(zip(labels, waypoints)):
            if not self._drive_to(label, waypoint, final=index == len(waypoints) - 1):
                return False

        pose = self._current_pose()
        if pose is None:
            return False
        final = waypoints[-1]
        xy_error = math.hypot(final[0] - pose[0], final[1] - pose[1])
        yaw_error = abs(shortest_angle_delta(final[2], pose[3]))
        self.get_logger().info(
            f"RESULT pose=[{pose[0]:.6f}, {pose[1]:.6f}, {pose[3]:.6f}] "
            f"target=[{final[0]:.6f}, {final[1]:.6f}, {final[2]:.6f}] "
            f"error_xy={xy_error:.3f} error_yaw={yaw_error:.3f}"
        )
        return True


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BobacSortOdomNav()
    ok = False
    try:
        ok = node.run()
    finally:
        node._publish_stop()
        node.destroy_node()
        rclpy.shutdown()
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
