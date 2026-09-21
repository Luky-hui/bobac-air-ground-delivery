#!/usr/bin/env python3
"""Execute a fixed Bobac right-arm joint pose.

Target is provided as a ROS parameter:
  ros2 run isaac_sim_2026 bobac_execute_joint_pose_2026.py --ros-args -p target:="[0,0,0,0,0,0]"
"""

import time
from typing import Dict, List, Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


JOINT_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
JOINT_LIMITS = {
    "joint_1": (-3.11, 3.11),
    "joint_2": (-3.11, 2.36),
    "joint_3": (-2.79, 2.53),
    "joint_4": (-3.11, 3.11),
    "joint_5": (-3.11, 3.11),
    "joint_6": (-6.28, 6.28),
}


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class BobacExecuteJointPose(Node):
    def __init__(self) -> None:
        super().__init__("bobac_execute_joint_pose_2026")
        self.declare_parameter("joint_state_topic", "/bobac_joint_states")
        self.declare_parameter("command_topic", "/joint_command")
        self.declare_parameter("target", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.declare_parameter("duration", 3.0)
        self.declare_parameter("rate_hz", 50.0)
        self.declare_parameter("hold_sec", 0.5)

        self.pub = self.create_publisher(
            JointState, str(self.get_parameter("command_topic").value), 10
        )
        self.current: Optional[Dict[str, float]] = None
        self.create_subscription(
            JointState, str(self.get_parameter("joint_state_topic").value), self._joint_cb, 10
        )

    def _joint_cb(self, msg: JointState) -> None:
        values = dict(zip(msg.name, msg.position))
        if all(name in values for name in JOINT_NAMES):
            self.current = {name: float(values[name]) for name in JOINT_NAMES}

    def wait_for_state(self, timeout_sec: float = 5.0) -> bool:
        end = time.time() + timeout_sec
        while rclpy.ok() and self.current is None and time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.1)
        return self.current is not None

    def publish_positions(self, positions: List[float]) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(JOINT_NAMES)
        msg.position = list(positions)
        self.pub.publish(msg)

    def run(self) -> bool:
        if not self.wait_for_state():
            self.get_logger().error(
                f"no {self.get_parameter('joint_state_topic').value} received; not commanding arm"
            )
            return False

        raw_target = [float(v) for v in self.get_parameter("target").value]
        if len(raw_target) != len(JOINT_NAMES):
            self.get_logger().error("target must contain exactly 6 joint angles")
            return False

        target: List[float] = []
        for name, value in zip(JOINT_NAMES, raw_target):
            low, high = JOINT_LIMITS[name]
            target.append(clamp(value, low, high))

        assert self.current is not None
        start = [self.current[name] for name in JOINT_NAMES]
        duration = max(0.1, float(self.get_parameter("duration").value))
        rate_hz = max(1.0, float(self.get_parameter("rate_hz").value))
        steps = max(1, int(duration * rate_hz))

        self.get_logger().info(
            "executing target rad=[" + ", ".join(f"{v:.6f}" for v in target) + "]"
        )
        for i in range(steps + 1):
            if not rclpy.ok():
                return False
            ratio = i / steps
            smooth = ratio * ratio * (3.0 - 2.0 * ratio)
            positions = [s + (t - s) * smooth for s, t in zip(start, target)]
            self.publish_positions(positions)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(1.0 / rate_hz)

        hold_end = time.time() + max(0.0, float(self.get_parameter("hold_sec").value))
        while rclpy.ok() and time.time() < hold_end:
            self.publish_positions(target)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(1.0 / rate_hz)
        return True


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BobacExecuteJointPose()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
