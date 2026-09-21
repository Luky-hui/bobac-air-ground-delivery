#!/usr/bin/env python3
import time
import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class ObservationPoseNode(Node):
    def __init__(self):
        super().__init__('observation_pose_node')
        self.declare_parameter('command_topic', '/hand_command')
        self.declare_parameter('joint_state_topic', '/demo_grasp/arm_joint_states')
        self.declare_parameter('joint_names', ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6'])
        self.declare_parameter('observation_pose', [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.declare_parameter('publish_rate_hz', 50.0)
        self.declare_parameter('hold_seconds', 3.0)
        self.declare_parameter('max_joint_step', 0.012)
        self.pub = self.create_publisher(JointState, self.get_parameter('command_topic').value, 10)
        self.current_joints = {}
        self.create_subscription(JointState, self.get_parameter('joint_state_topic').value, self._on_joint_state, 10)
        self.get_logger().info('observation_pose_node started')

    def _on_joint_state(self, msg):
        for name, value in zip(msg.name, msg.position):
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                self.current_joints[str(name)] = value

    def _current_or_target(self, names, target):
        deadline = time.time() + 2.0
        while rclpy.ok() and time.time() < deadline:
            if all(name in self.current_joints for name in names):
                return [self.current_joints[name] for name in names]
            rclpy.spin_once(self, timeout_sec=0.05)
        return list(target)

    def publish_pose(self, names, positions):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(names)
        msg.position = [float(v) for v in positions]
        self.pub.publish(msg)

    def run(self):
        names = list(self.get_parameter('joint_names').value)
        target = [float(v) for v in self.get_parameter('observation_pose').value]
        start = self._current_or_target(names, target)
        rate_hz = max(1.0, float(self.get_parameter('publish_rate_hz').value))
        hold_seconds = max(0.1, float(self.get_parameter('hold_seconds').value))
        period = 1.0 / rate_hz
        max_delta = max(abs(a - b) for a, b in zip(start, target)) if start else 0.0
        steps = max(2, int(math.ceil(max_delta / max(1e-4, float(self.get_parameter('max_joint_step').value)))))
        self.get_logger().info(f'moving to observation pose smoothly: steps={steps}, max_delta={max_delta:.4f}rad')
        for i in range(steps + 1):
            t = i / steps
            s = t * t * (3.0 - 2.0 * t)
            pose = [a + (b - a) * s for a, b in zip(start, target)]
            self.publish_pose(names, pose)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(period)
        end_time = time.time() + hold_seconds
        while rclpy.ok() and time.time() < end_time:
            self.publish_pose(names, target)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(period)
        self.get_logger().info('observation pose command published')


def main(args=None):
    rclpy.init(args=args)
    node = ObservationPoseNode()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
