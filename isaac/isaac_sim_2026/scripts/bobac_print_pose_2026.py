#!/usr/bin/env python3
"""Print Bobac odometry pose as [x, y, yaw]."""

import math

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node


def yaw_from_odom(msg: Odometry) -> float:
    q = msg.pose.pose.orientation
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class BobacPrintPose(Node):
    def __init__(self) -> None:
        super().__init__("bobac_print_pose_2026")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("once", True)
        self.declare_parameter("rate_hz", 2.0)
        self.once = bool(self.get_parameter("once").value)
        self.done = False
        self.create_subscription(
            Odometry, str(self.get_parameter("odom_topic").value), self._odom_cb, 10
        )

    def _odom_cb(self, msg: Odometry) -> None:
        if self.once and self.done:
            return
        p = msg.pose.pose.position
        yaw = yaw_from_odom(msg)
        print(f"[{p.x:.6f}, {p.y:.6f}, {yaw:.6f}]")
        self.done = True


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BobacPrintPose()
    try:
        if node.once:
            while rclpy.ok() and not node.done:
                rclpy.spin_once(node, timeout_sec=0.1)
        else:
            period = 1.0 / max(0.1, float(node.get_parameter("rate_hz").value))
            while rclpy.ok():
                rclpy.spin_once(node, timeout_sec=period)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
