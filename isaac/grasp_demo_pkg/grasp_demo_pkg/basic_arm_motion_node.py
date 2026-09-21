#!/usr/bin/env python3
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class BasicArmMotionNode(Node):
    def __init__(self):
        super().__init__('basic_arm_motion_node')
        self.declare_parameter('command_topic', '/hand_command')
        self.declare_parameter('joint_names', ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6'])
        self.declare_parameter('start_pose', [0.0, 0.58, -1.67, -0.5, 1.51, 0.0])
        self.declare_parameter('demo_pose', [0.0, 0.35, -1.30, -0.35, 1.35, 0.0])
        self.declare_parameter('steps', 60)
        self.declare_parameter('rate_hz', 20.0)
        self.declare_parameter('hold_seconds', 1.0)
        self.pub = self.create_publisher(JointState, self.get_parameter('command_topic').value, 10)
        self.get_logger().info('basic_arm_motion_node started')

    def _publish_pose(self, names, positions):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(names)
        msg.position = [float(v) for v in positions]
        self.pub.publish(msg)

    def _move_linear(self, names, start, goal):
        steps = max(2, int(self.get_parameter('steps').value))
        rate_hz = max(1.0, float(self.get_parameter('rate_hz').value))
        period = 1.0 / rate_hz
        for i in range(steps + 1):
            if not rclpy.ok():
                return
            t = i / steps
            # Smoothstep interpolation reduces sudden starts/stops while staying simple.
            s = t * t * (3.0 - 2.0 * t)
            pos = [a + (b - a) * s for a, b in zip(start, goal)]
            self._publish_pose(names, pos)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(period)

    def _hold(self, names, pose):
        hold = max(0.0, float(self.get_parameter('hold_seconds').value))
        rate_hz = max(1.0, float(self.get_parameter('rate_hz').value))
        end = time.time() + hold
        while rclpy.ok() and time.time() < end:
            self._publish_pose(names, pose)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(1.0 / rate_hz)

    def run(self):
        names = list(self.get_parameter('joint_names').value)
        start = [float(v) for v in self.get_parameter('start_pose').value]
        demo = [float(v) for v in self.get_parameter('demo_pose').value]
        if len(names) != len(start) or len(names) != len(demo):
            raise RuntimeError('joint_names, start_pose, and demo_pose must have the same length')
        self.get_logger().info(f'moving fixed demo trajectory: start={start} demo={demo}')
        self._hold(names, start)
        self._move_linear(names, start, demo)
        self._hold(names, demo)
        self._move_linear(names, demo, start)
        self._hold(names, start)
        self.get_logger().info('basic arm motion demo completed')


def main(args=None):
    rclpy.init(args=args)
    node = BasicArmMotionNode()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
