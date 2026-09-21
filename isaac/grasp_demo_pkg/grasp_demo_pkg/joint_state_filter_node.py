#!/usr/bin/env python3
import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class JointStateFilterNode(Node):
    def __init__(self):
        super().__init__('joint_state_filter_node')
        self.declare_parameter('input_topic', '/bobac_joint_states')
        self.declare_parameter('output_topic', '/demo_grasp/arm_joint_states')
        self.declare_parameter('joint_names', ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6'])
        self.declare_parameter('fallback_pose', [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.names = [str(v) for v in self.get_parameter('joint_names').value]
        fallback = [float(v) for v in self.get_parameter('fallback_pose').value]
        self.last = {name: fallback[i] for i, name in enumerate(self.names)}
        self.pub = self.create_publisher(JointState, self.get_parameter('output_topic').value, 10)
        self.create_subscription(JointState, self.get_parameter('input_topic').value, self._on_joint_state, 10)
        self.get_logger().info(f'filtering finite arm joints {self.names}')

    def _on_joint_state(self, msg):
        changed = False
        raw = {str(name): value for name, value in zip(msg.name, msg.position)}
        for name in self.names:
            if name not in raw:
                continue
            try:
                value = float(raw[name])
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                self.last[name] = value
                changed = True
        if not changed and not self.last:
            return
        out = JointState()
        out.header = msg.header
        out.name = list(self.names)
        out.position = [float(self.last[name]) for name in self.names]
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = JointStateFilterNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
