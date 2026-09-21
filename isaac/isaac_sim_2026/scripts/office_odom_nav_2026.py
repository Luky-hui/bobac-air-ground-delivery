#!/usr/bin/env python3
"""Low-speed odom waypoint navigation for the office/material area."""

import math
import time
from typing import List, Optional, Sequence, Tuple

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


DEFAULT_WAYPOINTS = [
    4.279537,
    -0.458060,
    -1.350366,
    4.555933,
    -1.358829,
    -0.823705,
    5.141703,
    -2.005444,
    -0.208234,
]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def shortest_angle_delta(target: float, actual: float) -> float:
    return math.atan2(math.sin(target - actual), math.cos(target - actual))


def quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def parse_waypoints(flat_values: Sequence[float]) -> List[Tuple[float, float, float]]:
    values = [float(v) for v in flat_values]
    if len(values) == 0 or len(values) % 3 != 0:
        raise ValueError("waypoints must be a flat [x, y, yaw, ...] list")
    return [(values[i], values[i + 1], values[i + 2]) for i in range(0, len(values), 3)]


class OfficeOdomNav(Node):
    def __init__(self) -> None:
        super().__init__("office_odom_nav_2026")

        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("waypoints", DEFAULT_WAYPOINTS)
        self.declare_parameter("prestop_xy_tolerance", 0.12)
        self.declare_parameter("final_xy_tolerance", 0.10)
        self.declare_parameter("yaw_tolerance", 0.15)
        self.declare_parameter("max_linear", 0.20)
        self.declare_parameter("max_angular", 0.60)
        self.declare_parameter("front_stop_range", 0.34)
        self.declare_parameter("front_slow_range", 0.55)
        self.declare_parameter("max_abs_z", 0.10)
        self.declare_parameter("goal_timeout_sec", 240.0)
        self.declare_parameter("publish_stop_count", 20)

        self.last_odom: Optional[Odometry] = None
        self.last_scan: Optional[LaserScan] = None

        self.create_subscription(Odometry, str(self.get_parameter("odom_topic").value), self._odom_cb, 10)
        self.create_subscription(LaserScan, str(self.get_parameter("scan_topic").value), self._scan_cb, 10)
        self.cmd_pub = self.create_publisher(Twist, str(self.get_parameter("cmd_vel_topic").value), 10)

    def _odom_cb(self, msg: Odometry) -> None:
        self.last_odom = msg

    def _scan_cb(self, msg: LaserScan) -> None:
        self.last_scan = msg

    def _current_pose(self) -> Optional[Tuple[float, float, float, float]]:
        msg = self.last_odom
        if msg is None:
            return None
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = quat_to_yaw(q.x, q.y, q.z, q.w)
        values = [p.x, p.y, p.z, yaw, msg.twist.twist.linear.x, msg.twist.twist.angular.z]
        if not all(math.isfinite(v) for v in values):
            return None
        return p.x, p.y, p.z, yaw

    def _front_clearance(self) -> Tuple[float, float, float]:
        scan = self.last_scan
        if scan is None or not scan.ranges:
            return math.inf, math.inf, math.inf

        front = []
        left = []
        right = []
        angle = scan.angle_min
        for r in scan.ranges:
            if math.isfinite(r) and scan.range_min <= r <= scan.range_max:
                if abs(angle) <= math.radians(28):
                    front.append(r)
                elif math.radians(28) < angle <= math.radians(95):
                    left.append(r)
                elif -math.radians(95) <= angle < -math.radians(28):
                    right.append(r)
            angle += scan.angle_increment

        return (
            min(front) if front else math.inf,
            min(left) if left else math.inf,
            min(right) if right else math.inf,
        )

    def _publish_stop(self) -> None:
        stop = Twist()
        for _ in range(max(1, int(self.get_parameter("publish_stop_count").value))):
            self.cmd_pub.publish(stop)
            rclpy.spin_once(self, timeout_sec=0.02)

    def _wait_for_inputs(self) -> bool:
        deadline = time.time() + 10.0
        while rclpy.ok() and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.last_odom is not None and self.last_scan is not None:
                return True
        return False

    def _drive_to(self, label: str, target: Tuple[float, float, float], final: bool) -> bool:
        timeout = float(self.get_parameter("goal_timeout_sec").value)
        xy_tol = float(
            self.get_parameter("final_xy_tolerance" if final else "prestop_xy_tolerance").value
        )
        yaw_tol = float(self.get_parameter("yaw_tolerance").value)
        max_v = abs(float(self.get_parameter("max_linear").value))
        max_w = abs(float(self.get_parameter("max_angular").value))
        max_z = abs(float(self.get_parameter("max_abs_z").value))
        stop_range = float(self.get_parameter("front_stop_range").value)
        slow_range = float(self.get_parameter("front_slow_range").value)
        deadline = time.time() + timeout
        target_x, target_y, target_yaw = target

        self.get_logger().info(
            f"{label}: drive target x={target_x:.6f}, y={target_y:.6f}, yaw={target_yaw:.6f}"
        )
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
                self._publish_stop()
                self.get_logger().info(
                    f"{label}: reached, error xy={dist:.3f}m, yaw={abs(yaw_error):.3f}rad"
                )
                return True

            cmd = Twist()
            front, left, right = self._front_clearance()
            if front < stop_range:
                turn_dir = -1.0 if right > left else 1.0
                cmd.angular.z = turn_dir * min(max_w, 0.16)
                self.cmd_pub.publish(cmd)
                continue

            if dist > xy_tol:
                heading = math.atan2(dy, dx)
                heading_error = shortest_angle_delta(heading, yaw)
                if abs(heading_error) > 0.24:
                    cmd.angular.z = clamp(0.85 * heading_error, -max_w, max_w)
                else:
                    speed_scale = 0.45 if front < slow_range else 1.0
                    cmd.linear.x = clamp(0.42 * dist, 0.025, max_v) * speed_scale
                    cmd.angular.z = clamp(0.80 * heading_error, -max_w, max_w)
            else:
                cmd.angular.z = clamp(0.65 * yaw_error, -max_w, max_w)

            self.cmd_pub.publish(cmd)

        self._publish_stop()
        pose = self._current_pose()
        if pose is None:
            self.get_logger().error(f"{label}: timed out and /odom invalid")
            return False
        dist = math.hypot(target_x - pose[0], target_y - pose[1])
        yaw_error = abs(shortest_angle_delta(target_yaw, pose[3]))
        self.get_logger().error(f"{label}: timed out, error xy={dist:.3f}m, yaw={yaw_error:.3f}rad")
        return False

    def run(self) -> bool:
        if not self._wait_for_inputs():
            self.get_logger().error("missing /odom or /scan; is Isaac Sim playing?")
            return False

        waypoints = parse_waypoints(self.get_parameter("waypoints").value)
        labels = [f"prestop_{i + 1}" for i in range(max(0, len(waypoints) - 1))] + ["final_stop"]
        for index, (label, waypoint) in enumerate(zip(labels, waypoints)):
            if not self._drive_to(label, waypoint, final=index == len(waypoints) - 1):
                return False
            time.sleep(0.4)

        pose = self._current_pose()
        if pose is None:
            return False
        final = waypoints[-1]
        xy_error = math.hypot(final[0] - pose[0], final[1] - pose[1])
        yaw_error = abs(shortest_angle_delta(final[2], pose[3]))
        self.get_logger().info(
            f"final odom x={pose[0]:.6f}, y={pose[1]:.6f}, yaw={pose[3]:.6f}"
        )
        self.get_logger().info(f"final error xy={xy_error:.3f}m, yaw={yaw_error:.3f}rad")
        self.get_logger().info("office odom navigation completed successfully")
        return True


def main(args=None) -> None:
    rclpy.init(args=args)
    node = OfficeOdomNav()
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
