#!/usr/bin/env python3
"""Print current Bobac right-arm joint angles."""

import math
from typing import Dict

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


JOINT_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]


class BobacPrintJoints(Node):
    def __init__(self) -> None:
        super().__init__("bobac_print_joints_2026")
        self.declare_parameter("joint_state_topic", "/bobac_joint_states")
        self.declare_parameter("once", True)
        self.done = False
        self.once = bool(self.get_parameter("once").value)
        self.create_subscription(
            JointState, str(self.get_parameter("joint_state_topic").value), self._joint_cb, 10
        )

    def _joint_cb(self, msg: JointState) -> None:
        if self.once and self.done:
            return
        values: Dict[str, float] = dict(zip(msg.name, msg.position))
        if not all(name in values for name in JOINT_NAMES):
            return
        rad = [float(values[name]) for name in JOINT_NAMES]
        deg = [math.degrees(v) for v in rad]
        print("joint_names:", JOINT_NAMES)
        print("rad:", "[" + ", ".join(f"{v:.6f}" for v in rad) + "]")
        print("deg:", "[" + ", ".join(f"{v:.2f}" for v in deg) + "]")
        print(
            "ros2_param_target:=\"["
            + ", ".join(f"{v:.6f}" for v in rad)
            + "]\""
        )
        self.done = True


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BobacPrintJoints()
    try:
        if node.once:
            while rclpy.ok() and not node.done:
                rclpy.spin_once(node, timeout_sec=0.1)
        else:
            while rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0.5)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
