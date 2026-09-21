#!/usr/bin/env python3
"""Convert Bobac ROS cmd_vel commands into three omni-wheel joint velocities."""

from __future__ import annotations

import math
import time
from typing import Tuple

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import JointState


def clamp(value: float, limit: float) -> float:
    if limit <= 0.0:
        return value
    return max(-limit, min(limit, value))


class BobacCmdVelToJoint(Node):
    def __init__(self) -> None:
        super().__init__("bobac_cmd_vel_to_joint_2026")

        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("joint_command_topic", "/joint_command")
        self.declare_parameter("front_wheel_joint", "front_wheel_joint")
        self.declare_parameter("left_wheel_joint", "left_wheel_joint")
        self.declare_parameter("right_wheel_joint", "right_wheel_joint")
        self.declare_parameter("wheel_radius", 0.1)
        self.declare_parameter("base_radius", 0.185)
        self.declare_parameter("max_linear", 0.30)
        self.declare_parameter("max_angular", 0.50)
        self.declare_parameter("max_wheel_speed", 8.0)
        self.declare_parameter("publish_hz", 30.0)
        self.declare_parameter("command_timeout", 0.35)
        self.declare_parameter("front_sign", 1.0)
        self.declare_parameter("left_sign", 1.0)
        self.declare_parameter("right_sign", 1.0)

        self.joint_names = [
            str(self.get_parameter("front_wheel_joint").value),
            str(self.get_parameter("left_wheel_joint").value),
            str(self.get_parameter("right_wheel_joint").value),
        ]
        self.wheel_radius = abs(float(self.get_parameter("wheel_radius").value))
        self.base_radius = abs(float(self.get_parameter("base_radius").value))
        self.max_linear = abs(float(self.get_parameter("max_linear").value))
        self.max_angular = abs(float(self.get_parameter("max_angular").value))
        self.max_wheel_speed = abs(float(self.get_parameter("max_wheel_speed").value))
        self.command_timeout = max(0.05, float(self.get_parameter("command_timeout").value))
        self.signs = (
            float(self.get_parameter("front_sign").value),
            float(self.get_parameter("left_sign").value),
            float(self.get_parameter("right_sign").value),
        )

        if self.wheel_radius <= 1e-6:
            raise ValueError("wheel_radius must be positive")

        self.last_twist = Twist()
        self.last_command_time = 0.0
        self.pub = self.create_publisher(
            JointState, str(self.get_parameter("joint_command_topic").value), 10
        )
        self.create_subscription(
            Twist, str(self.get_parameter("cmd_vel_topic").value), self._cmd_vel_cb, 10
        )

        publish_hz = max(1.0, float(self.get_parameter("publish_hz").value))
        self.create_timer(1.0 / publish_hz, self._publish_joint_command)
        self.get_logger().info(
            "Bobac cmd_vel bridge ready: "
            f"{self.get_parameter('cmd_vel_topic').value} -> "
            f"{self.get_parameter('joint_command_topic').value}, joints={self.joint_names}"
        )

    def _cmd_vel_cb(self, msg: Twist) -> None:
        self.last_twist = msg
        self.last_command_time = time.monotonic()

    def _wheel_speeds(self, twist: Twist) -> Tuple[float, float, float]:
        vx = clamp(float(twist.linear.x), self.max_linear)
        vy = clamp(float(twist.linear.y), self.max_linear)
        wz = clamp(float(twist.angular.z), self.max_angular)

        r = self.wheel_radius
        radius_term = self.base_radius * wz / r
        front = vx / r + radius_term
        left = -0.5 * vx / r + (math.sqrt(3.0) * 0.5) * vy / r + radius_term
        right = -0.5 * vx / r - (math.sqrt(3.0) * 0.5) * vy / r + radius_term
        signed = (
            front * self.signs[0],
            left * self.signs[1],
            right * self.signs[2],
        )

        largest = max(abs(value) for value in signed)
        if self.max_wheel_speed > 0.0 and largest > self.max_wheel_speed:
            scale = self.max_wheel_speed / largest
            signed = tuple(value * scale for value in signed)
        return signed

    def _publish_joint_command(self) -> None:
        if time.monotonic() - self.last_command_time > self.command_timeout:
            speeds = (0.0, 0.0, 0.0)
        else:
            speeds = self._wheel_speeds(self.last_twist)

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.velocity = list(speeds)
        self.pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BobacCmdVelToJoint()
    try:
        rclpy.spin(node)
    finally:
        stop = JointState()
        stop.header.stamp = node.get_clock().now().to_msg()
        stop.name = node.joint_names
        stop.velocity = [0.0, 0.0, 0.0]
        for _ in range(5):
            node.pub.publish(stop)
            rclpy.spin_once(node, timeout_sec=0.02)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
