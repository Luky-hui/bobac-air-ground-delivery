#!/usr/bin/env python3
"""Publish a short, slow /cmd_vel motion and print before/after odom."""

import math
import time
from typing import Optional, Tuple

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node


def yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def odom_pose(msg: Odometry) -> Tuple[float, float, float, float]:
    p = msg.pose.pose.position
    q = msg.pose.pose.orientation
    return p.x, p.y, p.z, yaw_from_quat(q.x, q.y, q.z, q.w)


def finite_odom(msg: Odometry, max_abs_z: float) -> bool:
    x, y, z, yaw = odom_pose(msg)
    values = [
        x,
        y,
        z,
        yaw,
        msg.twist.twist.linear.x,
        msg.twist.twist.linear.y,
        msg.twist.twist.angular.z,
    ]
    return all(math.isfinite(v) for v in values) and abs(z) <= max_abs_z


class SlowBaseMove(Node):
    def __init__(self) -> None:
        super().__init__("slow_base_move_2026")

        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("linear", 0.03)
        self.declare_parameter("angular", 0.0)
        self.declare_parameter("duration", 1.0)
        self.declare_parameter("rate_hz", 20.0)
        self.declare_parameter("ramp_sec", 0.5)
        self.declare_parameter("max_linear", 0.05)
        self.declare_parameter("max_angular", 0.15)
        self.declare_parameter("max_abs_z", 0.08)

        self.pub = self.create_publisher(Twist, str(self.get_parameter("cmd_vel_topic").value), 10)
        self.last_odom: Optional[Odometry] = None
        self.create_subscription(
            Odometry,
            str(self.get_parameter("odom_topic").value),
            self._odom_cb,
            10,
        )

    def _odom_cb(self, msg: Odometry) -> None:
        self.last_odom = msg

    def wait_for_odom(self, timeout_sec: float = 5.0) -> Optional[Odometry]:
        deadline = time.time() + timeout_sec
        while rclpy.ok() and self.last_odom is None and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        return self.last_odom

    def publish_stop(self, count: int = 10) -> None:
        stop = Twist()
        for _ in range(count):
            self.pub.publish(stop)
            rclpy.spin_once(self, timeout_sec=0.02)

    def run(self) -> bool:
        max_abs_z = abs(float(self.get_parameter("max_abs_z").value))
        start = self.wait_for_odom()
        if start is None:
            self.get_logger().error("no /odom received; is Isaac Sim playing?")
            self.publish_stop()
            return False
        if not finite_odom(start, max_abs_z):
            self.get_logger().error("initial /odom is invalid; reset Isaac Sim first")
            self.publish_stop()
            return False

        requested_linear = float(self.get_parameter("linear").value)
        requested_angular = float(self.get_parameter("angular").value)
        max_linear = abs(float(self.get_parameter("max_linear").value))
        max_angular = abs(float(self.get_parameter("max_angular").value))
        linear = max(-max_linear, min(max_linear, requested_linear))
        angular = max(-max_angular, min(max_angular, requested_angular))

        duration = max(0.0, float(self.get_parameter("duration").value))
        rate_hz = max(1.0, float(self.get_parameter("rate_hz").value))
        ramp_sec = max(0.0, float(self.get_parameter("ramp_sec").value))

        x0, y0, z0, yaw0 = odom_pose(start)
        self.get_logger().info(
            f"before: x={x0:.6f}, y={y0:.6f}, z={z0:.6f}, yaw={yaw0:.6f}"
        )
        self.get_logger().info(
            f"moving: linear={linear:.3f} m/s, angular={angular:.3f} rad/s, duration={duration:.2f}s"
        )

        begin = time.time()
        period = 1.0 / rate_hz
        ok = True
        while rclpy.ok() and time.time() - begin < duration:
            elapsed = time.time() - begin
            scale = 1.0
            if ramp_sec > 0.0:
                scale = min(1.0, elapsed / ramp_sec)
                remaining = duration - elapsed
                scale = min(scale, max(0.0, remaining / ramp_sec))

            msg = Twist()
            msg.linear.x = linear * scale
            msg.angular.z = angular * scale
            self.pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.0)

            if self.last_odom is not None and not finite_odom(self.last_odom, max_abs_z):
                self.get_logger().error("invalid /odom while moving; stopping")
                ok = False
                break
            time.sleep(period)

        self.publish_stop()
        time.sleep(0.2)
        rclpy.spin_once(self, timeout_sec=0.1)

        end = self.last_odom
        if end is not None and finite_odom(end, max_abs_z):
            x1, y1, z1, yaw1 = odom_pose(end)
            self.get_logger().info(
                f"after:  x={x1:.6f}, y={y1:.6f}, z={z1:.6f}, yaw={yaw1:.6f}"
            )
            self.get_logger().info(
                f"delta:  dx={x1 - x0:.6f}, dy={y1 - y0:.6f}, "
                f"dyaw={yaw1 - yaw0:.6f}, dist={math.hypot(x1 - x0, y1 - y0):.6f}"
            )
            print(f"[{x1:.6f}, {y1:.6f}, {yaw1:.6f}]")
        else:
            ok = False
            self.get_logger().error("final /odom is invalid; reset Isaac Sim before continuing")
        return ok


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SlowBaseMove()
    try:
        node.run()
    finally:
        node.publish_stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
