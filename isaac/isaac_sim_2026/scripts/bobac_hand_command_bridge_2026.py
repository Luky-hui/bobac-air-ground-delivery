#!/usr/bin/env python3
"""Bridge the official Bobac demo arm command topic to this USD scene.

The official grasp_demo_pkg publishes sensor_msgs/JointState on /hand_command.
The 2026 Bobac USD ActionGraph listens on /joint_command.  This node keeps the
official demo package untouched and only adapts the scene interface.
"""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class BobacHandCommandBridge(Node):
    def __init__(self) -> None:
        super().__init__("bobac_hand_command_bridge_2026")
        self.declare_parameter("input_topic", "/hand_command")
        self.declare_parameter("output_topic", "/joint_command")

        input_topic = str(self.get_parameter("input_topic").value)
        output_topic = str(self.get_parameter("output_topic").value)
        self.pub = self.create_publisher(JointState, output_topic, 10)
        self.create_subscription(JointState, input_topic, self._on_command, 10)
        self.forwarded = 0
        self.get_logger().info(f"bridging {input_topic} -> {output_topic}")

    def _on_command(self, msg: JointState) -> None:
        out = JointState()
        out.header = msg.header
        out.name = list(msg.name)
        out.position = list(msg.position)
        out.velocity = list(msg.velocity)
        out.effort = list(msg.effort)
        self.pub.publish(out)
        self.forwarded += 1
        if self.forwarded == 1:
            self.get_logger().info(
                f"first command forwarded: joints={list(msg.name)}"
            )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BobacHandCommandBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
