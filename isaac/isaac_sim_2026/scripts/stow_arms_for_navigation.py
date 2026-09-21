#!/usr/bin/env python3
"""Move the Mercury X1 arms into the folded navigation posture.

The official scene starts with the arms in a wide posture.  Nav2 plans with a
compact base footprint, so the arms should be tucked before driving through
the office corridor.  The default targets match the folded pose from the Isaac
Sim joint panel; values are in radians.  This node keeps both grippers at their
current positions and only moves the arm joints.
"""

import time
from typing import Dict, List

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


RIGHT_ARM = ["joint1_R", "joint2_R", "joint3_R", "joint4_R", "joint5_R", "joint6_R"]
LEFT_ARM = ["joint1_L", "joint2_L", "joint3_L", "joint4_L", "joint5_L", "joint6_L"]
RIGHT_GRIPPER = "gripper_controller"
LEFT_GRIPPER = "gripper_left_controller"


class StowArmsForNavigation(Node):
    def __init__(self) -> None:
        super().__init__("stow_arms_for_navigation")

        self.declare_parameter("wait_sec", 8.0)
        self.declare_parameter("duration_sec", 5.0)
        self.declare_parameter("publish_hz", 50.0)
        self.declare_parameter("hold_sec", 0.5)
        self.declare_parameter("right_target", [0.0, 0.9861, -0.4677, 0.3962, 1.5708, 0.0])
        self.declare_parameter("left_target", [0.0, 0.7575, -0.7540, 0.5027, 2.1607, 0.1728])

        self.main_state: Dict[str, float] = {}
        self.left_state: Dict[str, float] = {}

        self.pub_main = self.create_publisher(JointState, "/joint_command", 10)
        self.pub_left = self.create_publisher(JointState, "/joint_left_command", 10)
        self.create_subscription(JointState, "/joint_states", self._main_cb, 10)
        self.create_subscription(JointState, "/joint_left_states", self._left_cb, 10)

    def _main_cb(self, msg: JointState) -> None:
        for name, pos in zip(msg.name, msg.position):
            self.main_state[name] = float(pos)

    def _left_cb(self, msg: JointState) -> None:
        for name, pos in zip(msg.name, msg.position):
            self.left_state[name] = float(pos)

    def _wait_for_state(self) -> bool:
        deadline = time.time() + float(self.get_parameter("wait_sec").value)
        required = set(RIGHT_ARM + LEFT_ARM)
        while rclpy.ok() and time.time() < deadline:
            if required.issubset(self.main_state):
                return True
            rclpy.spin_once(self, timeout_sec=0.05)
        missing = sorted(required.difference(self.main_state))
        self.get_logger().warn(
            f"joint state timeout; missing {missing}. "
            "Arm stow skipped. Is Isaac Sim playing and publishing /joint_states?"
        )
        return False

    def _publish(self, right: List[float], left: List[float]) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = RIGHT_ARM + LEFT_ARM
        msg.position = list(right) + list(left)

        if RIGHT_GRIPPER in self.main_state:
            msg.name.append(RIGHT_GRIPPER)
            msg.position.append(float(self.main_state[RIGHT_GRIPPER]))
        self.pub_main.publish(msg)

        if LEFT_GRIPPER in self.left_state:
            left_msg = JointState()
            left_msg.header.stamp = msg.header.stamp
            left_msg.name = [LEFT_GRIPPER]
            left_msg.position = [float(self.left_state[LEFT_GRIPPER])]
            self.pub_left.publish(left_msg)

    def run(self) -> bool:
        self.get_logger().info("waiting for current arm joint states")
        if not self._wait_for_state():
            return False

        right_target = [float(v) for v in self.get_parameter("right_target").value]
        left_target = [float(v) for v in self.get_parameter("left_target").value]
        if len(right_target) != 6 or len(left_target) != 6:
            raise RuntimeError("right_target and left_target must contain 6 values")

        right_start = [float(self.main_state[j]) for j in RIGHT_ARM]
        left_start = [float(self.main_state[j]) for j in LEFT_ARM]
        duration = max(0.1, float(self.get_parameter("duration_sec").value))
        hz = max(1.0, float(self.get_parameter("publish_hz").value))
        steps = max(2, int(duration * hz))

        self.get_logger().info(
            "stowing arms for navigation: "
            f"right={right_target}, left={left_target}, duration={duration:.1f}s"
        )
        for i in range(steps + 1):
            alpha = i / steps
            right = [s + (e - s) * alpha for s, e in zip(right_start, right_target)]
            left = [s + (e - s) * alpha for s, e in zip(left_start, left_target)]
            self._publish(right, left)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(1.0 / hz)

        hold_until = time.time() + max(0.0, float(self.get_parameter("hold_sec").value))
        while rclpy.ok() and time.time() < hold_until:
            self._publish(right_target, left_target)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(1.0 / hz)

        self.get_logger().info("arms are in navigation stow posture")
        return True


def main(args=None) -> None:
    rclpy.init(args=args)
    node = StowArmsForNavigation()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
