#!/usr/bin/env python3
"""Navigate to the office/material area through measured odom waypoints."""

import math
import time
from typing import List, Optional, Sequence, Tuple

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.node import Node


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


def yaw_to_quat(yaw: float) -> Tuple[float, float, float, float]:
    return 0.0, 0.0, math.sin(yaw * 0.5), math.cos(yaw * 0.5)


def quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def shortest_angle_delta(target: float, actual: float) -> float:
    return math.atan2(math.sin(target - actual), math.cos(target - actual))


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def parse_waypoints(flat_values: Sequence[float]) -> List[Tuple[float, float, float]]:
    values = [float(v) for v in flat_values]
    if len(values) == 0 or len(values) % 3 != 0:
        raise ValueError("waypoints must be a flat [x1, y1, yaw1, x2, y2, yaw2, ...] list")
    return [(values[i], values[i + 1], values[i + 2]) for i in range(0, len(values), 3)]


class OfficeWaypointNav(Node):
    def __init__(self) -> None:
        super().__init__("office_waypoint_nav_2026")

        self.declare_parameter("frame_id", "odom")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("waypoints", DEFAULT_WAYPOINTS)
        self.declare_parameter("action_wait_sec", 20.0)
        self.declare_parameter("goal_timeout_sec", 180.0)
        self.declare_parameter("settle_sec", 1.0)
        self.declare_parameter("nav_handoff_xy_tolerance", 0.45)
        self.declare_parameter("final_nav_handoff_xy_tolerance", 1.20)
        self.declare_parameter("final_failure_handoff_xy_tolerance", 1.30)
        self.declare_parameter("final_xy_tolerance", 0.30)
        self.declare_parameter("final_yaw_tolerance", 0.45)
        self.declare_parameter("fine_dock_enabled", True)
        self.declare_parameter("fine_dock_timeout_sec", 180.0)
        self.declare_parameter("fine_dock_xy_tolerance", 0.10)
        self.declare_parameter("fine_dock_yaw_tolerance", 0.15)
        self.declare_parameter("fine_dock_max_linear", 0.055)
        self.declare_parameter("fine_dock_max_angular", 0.18)
        self.declare_parameter("max_abs_z", 0.10)
        self.declare_parameter("publish_stop_count", 15)

        self.last_odom: Optional[Odometry] = None
        self.create_subscription(
            Odometry,
            str(self.get_parameter("odom_topic").value),
            self._odom_cb,
            10,
        )
        self.stop_pub = self.create_publisher(
            Twist,
            str(self.get_parameter("cmd_vel_topic").value),
            10,
        )
        self.nav_client = ActionClient(self, NavigateToPose, "navigate_to_pose")

    def _odom_cb(self, msg: Odometry) -> None:
        self.last_odom = msg

    def _spin_until(self, future, timeout_sec: float):
        deadline = time.time() + timeout_sec
        while rclpy.ok() and not future.done():
            if time.time() > deadline:
                return None
            rclpy.spin_once(self, timeout_sec=0.1)
        return future.result() if future.done() else None

    def _wait_for_odom(self, timeout_sec: float = 10.0) -> bool:
        deadline = time.time() + timeout_sec
        while rclpy.ok() and self.last_odom is None and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        return self.last_odom is not None

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

    def _publish_stop(self) -> None:
        stop = Twist()
        count = max(1, int(self.get_parameter("publish_stop_count").value))
        for _ in range(count):
            self.stop_pub.publish(stop)
            rclpy.spin_once(self, timeout_sec=0.02)

    def _pose_is_safe(self, pose: Optional[Tuple[float, float, float, float]]) -> bool:
        if pose is None:
            return False
        return abs(pose[2]) <= abs(float(self.get_parameter("max_abs_z").value))

    def _cancel_goal(self, goal_handle) -> None:
        self._spin_until(goal_handle.cancel_goal_async(), timeout_sec=2.0)
        self._publish_stop()

    def _wait_for_nav_result(
        self,
        label: str,
        goal_handle,
        timeout_sec: float,
        waypoint: Tuple[float, float, float],
    ):
        deadline = time.time() + timeout_sec
        max_abs_z = abs(float(self.get_parameter("max_abs_z").value))
        handoff_param = (
            "final_nav_handoff_xy_tolerance"
            if label == "final_stop"
            else "nav_handoff_xy_tolerance"
        )
        handoff_xy = abs(float(self.get_parameter(handoff_param).value))
        result_future = goal_handle.get_result_async()
        while rclpy.ok() and not result_future.done():
            if time.time() > deadline:
                self.get_logger().error(f"{label}: navigation timed out after {timeout_sec:.1f}s")
                self._cancel_goal(goal_handle)
                return None
            rclpy.spin_once(self, timeout_sec=0.1)
            pose = self._current_pose()
            if pose is None:
                self.get_logger().error(f"{label}: /odom became non-finite; canceling goal")
                self._cancel_goal(goal_handle)
                return None
            if abs(pose[2]) > max_abs_z:
                self.get_logger().error(
                    f"{label}: base z={pose[2]:.3f} exceeds limit {max_abs_z:.3f}; canceling goal"
                )
                self._cancel_goal(goal_handle)
                return None
            xy_error = math.hypot(waypoint[0] - pose[0], waypoint[1] - pose[1])
            if xy_error <= handoff_xy:
                self.get_logger().info(
                    f"{label}: odom handoff reached, error xy={xy_error:.3f}m"
                )
                self._cancel_goal(goal_handle)
                return "handoff"
        return result_future.result() if result_future.done() else None

    def _fine_dock(self, target: Tuple[float, float, float]) -> bool:
        if not bool(self.get_parameter("fine_dock_enabled").value):
            return True

        xy_tol = float(self.get_parameter("fine_dock_xy_tolerance").value)
        yaw_tol = float(self.get_parameter("fine_dock_yaw_tolerance").value)
        timeout = float(self.get_parameter("fine_dock_timeout_sec").value)
        max_v = abs(float(self.get_parameter("fine_dock_max_linear").value))
        max_w = abs(float(self.get_parameter("fine_dock_max_angular").value))
        deadline = time.time() + timeout
        target_x, target_y, target_yaw = target

        self.get_logger().info(
            f"fine_dock: target xy_tol={xy_tol:.3f}m, yaw_tol={yaw_tol:.3f}rad"
        )
        while rclpy.ok() and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            pose = self._current_pose()
            if not self._pose_is_safe(pose):
                self.get_logger().error("fine_dock: /odom is invalid or base z exceeded limit")
                self._publish_stop()
                return False

            x, y, _z, yaw = pose
            dx = target_x - x
            dy = target_y - y
            xy_error = math.hypot(dx, dy)
            yaw_error = shortest_angle_delta(target_yaw, yaw)
            if xy_error <= xy_tol and abs(yaw_error) <= yaw_tol:
                self._publish_stop()
                self.get_logger().info(
                    f"fine_dock: aligned, error xy={xy_error:.3f}m, yaw={abs(yaw_error):.3f}rad"
                )
                return True

            cmd = Twist()
            if xy_error > xy_tol:
                target_heading = math.atan2(dy, dx)
                heading_error = shortest_angle_delta(target_heading, yaw)
                if abs(heading_error) > 0.35:
                    cmd.angular.z = clamp(0.70 * heading_error, -max_w, max_w)
                else:
                    forward_error = math.cos(yaw) * dx + math.sin(yaw) * dy
                    cmd.linear.x = clamp(0.28 * forward_error, -0.035, max_v)
                    cmd.angular.z = clamp(0.55 * heading_error, -max_w, max_w)
            else:
                cmd.angular.z = clamp(0.55 * yaw_error, -max_w, max_w)

            self.stop_pub.publish(cmd)

        self._publish_stop()
        pose = self._current_pose()
        if pose is None:
            self.get_logger().error("fine_dock: timed out and /odom is invalid")
            return False
        xy_error = math.hypot(target_x - pose[0], target_y - pose[1])
        yaw_error = abs(shortest_angle_delta(target_yaw, pose[3]))
        self.get_logger().error(
            f"fine_dock: timed out, error xy={xy_error:.3f}m, yaw={yaw_error:.3f}rad"
        )
        return False

    def _make_goal(self, xyz: Tuple[float, float, float]) -> NavigateToPose.Goal:
        x, y, yaw = xyz
        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = str(self.get_parameter("frame_id").value)
        # Keep the stamp at zero so Nav2 transforms the odom-frame goal with the
        # latest TF during replanning instead of an expired map->odom transform.
        goal.pose.header.stamp.sec = 0
        goal.pose.header.stamp.nanosec = 0
        goal.pose.pose.position.x = x
        goal.pose.pose.position.y = y
        qx, qy, qz, qw = yaw_to_quat(yaw)
        goal.pose.pose.orientation.x = qx
        goal.pose.pose.orientation.y = qy
        goal.pose.pose.orientation.z = qz
        goal.pose.pose.orientation.w = qw
        return goal

    def _navigate_one(self, label: str, waypoint: Tuple[float, float, float]) -> bool:
        timeout = float(self.get_parameter("goal_timeout_sec").value)
        self.get_logger().info(
            f"{label}: send goal x={waypoint[0]:.6f}, y={waypoint[1]:.6f}, yaw={waypoint[2]:.6f}"
        )
        goal_handle = self._spin_until(
            self.nav_client.send_goal_async(self._make_goal(waypoint)),
            timeout_sec=10.0,
        )
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error(f"{label}: goal rejected or send timed out")
            self._publish_stop()
            return False

        result = self._wait_for_nav_result(label, goal_handle, timeout_sec=timeout, waypoint=waypoint)
        self._publish_stop()
        if result is None:
            return False
        if result != "handoff" and result.status != GoalStatus.STATUS_SUCCEEDED:
            pose = self._current_pose()
            if label == "final_stop" and pose is not None:
                xy_error = math.hypot(waypoint[0] - pose[0], waypoint[1] - pose[1])
                failure_handoff = abs(
                    float(self.get_parameter("final_failure_handoff_xy_tolerance").value)
                )
                if xy_error <= failure_handoff:
                    self.get_logger().warn(
                        f"{label}: Nav2 status={result.status}, taking fine dock handoff at xy={xy_error:.3f}m"
                    )
                    return True
            self.get_logger().error(f"{label}: navigation failed with status={result.status}")
            return False

        pose = self._current_pose()
        if pose is not None:
            dx = waypoint[0] - pose[0]
            dy = waypoint[1] - pose[1]
            dyaw = shortest_angle_delta(waypoint[2], pose[3])
            self.get_logger().info(
                f"{label}: reached, error xy={math.hypot(dx, dy):.3f}m, yaw={abs(dyaw):.3f}rad"
            )
        return True

    def run(self) -> bool:
        if not self._wait_for_odom():
            self.get_logger().error("no /odom; is Isaac Sim playing and ROS_DOMAIN_ID correct?")
            return False

        pose = self._current_pose()
        max_abs_z = abs(float(self.get_parameter("max_abs_z").value))
        if pose is None or abs(pose[2]) > max_abs_z:
            self.get_logger().error("initial /odom is invalid; reset Isaac Sim before navigating")
            return False

        wait_sec = float(self.get_parameter("action_wait_sec").value)
        self.get_logger().info("waiting for Nav2 navigate_to_pose action")
        if not self.nav_client.wait_for_server(timeout_sec=wait_sec):
            self.get_logger().error("navigate_to_pose action is unavailable; start Nav2 first")
            return False

        waypoints = parse_waypoints(self.get_parameter("waypoints").value)
        labels = [f"prestop_{i + 1}" for i in range(max(0, len(waypoints) - 1))] + ["final_stop"]

        for label, waypoint in zip(labels, waypoints):
            if not self._navigate_one(label, waypoint):
                return False
            if label == "final_stop" and not self._fine_dock(waypoint):
                return False
            settle = max(0.0, float(self.get_parameter("settle_sec").value))
            end = time.time() + settle
            while rclpy.ok() and time.time() < end:
                self._publish_stop()
                rclpy.spin_once(self, timeout_sec=0.05)

        final = waypoints[-1]
        pose = self._current_pose()
        if pose is None:
            self.get_logger().error("final /odom is invalid")
            return False
        xy_error = math.hypot(final[0] - pose[0], final[1] - pose[1])
        yaw_error = abs(shortest_angle_delta(final[2], pose[3]))
        self.get_logger().info(
            f"final odom x={pose[0]:.6f}, y={pose[1]:.6f}, yaw={pose[3]:.6f}"
        )
        self.get_logger().info(
            f"final error xy={xy_error:.3f}m, yaw={yaw_error:.3f}rad"
        )
        if xy_error > float(self.get_parameter("final_xy_tolerance").value):
            self.get_logger().error("final xy tolerance failed")
            return False
        if yaw_error > float(self.get_parameter("final_yaw_tolerance").value):
            self.get_logger().error("final yaw tolerance failed")
            return False

        self.get_logger().info("office waypoint navigation completed successfully")
        return True


def main(args=None) -> None:
    rclpy.init(args=args)
    node = OfficeWaypointNav()
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
