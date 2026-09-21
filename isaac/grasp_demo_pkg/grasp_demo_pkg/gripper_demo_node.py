#!/usr/bin/env python3
import time

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.node import Node
from sensor_msgs.msg import JointState

from grasp_demo_interfaces.action import GripperCommand


class GripperDemoNode(Node):
    def __init__(self):
        super().__init__('gripper_demo_node')
        self.declare_parameter('joint_name', 'gripper_l_joint1')
        self.declare_parameter('joint_names', ['gripper_l_joint1', 'gripper_r_joint1'])
        self.declare_parameter('joint_multipliers', [1.0, -1.0])
        self.declare_parameter('command_topic', '/hand_command')
        self.declare_parameter('open_position', 100.0)
        self.declare_parameter('close_position', 0.0)
        self.declare_parameter('open_angle', 0.78)
        self.declare_parameter('close_angle', 0.0)
        self.current_position = float(self.get_parameter('open_position').value)
        self.pub = self.create_publisher(JointState, self.get_parameter('command_topic').value, 10)
        self.server = ActionServer(
            self,
            GripperCommand,
            '/demo_gripper_command',
            execute_callback=self.execute_callback,
            goal_callback=lambda goal: GoalResponse.ACCEPT,
            cancel_callback=lambda goal: CancelResponse.ACCEPT,
        )
        self.get_logger().info('gripper_demo_node started')

    def _target_from_goal(self, goal):
        if goal.command == 'open':
            return float(self.get_parameter('open_position').value)
        if goal.command == 'close':
            return float(self.get_parameter('close_position').value)
        if goal.command == 'position':
            return max(0.0, min(100.0, float(goal.position)))
        return None

    def _percent_to_angle(self, percent):
        open_pos = float(self.get_parameter('open_position').value)
        close_pos = float(self.get_parameter('close_position').value)
        open_angle = float(self.get_parameter('open_angle').value)
        close_angle = float(self.get_parameter('close_angle').value)
        if abs(open_pos - close_pos) < 1e-6:
            return open_angle
        ratio = (float(percent) - close_pos) / (open_pos - close_pos)
        ratio = max(0.0, min(1.0, ratio))
        return close_angle + (open_angle - close_angle) * ratio

    def _publish_joint(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        names = list(self.get_parameter('joint_names').value)
        if not names:
            names = [self.get_parameter('joint_name').value]
        multipliers = [float(v) for v in self.get_parameter('joint_multipliers').value]
        if len(multipliers) != len(names):
            multipliers = [1.0] * len(names)
        angle = self._percent_to_angle(self.current_position)
        msg.name = names
        msg.position = [float(angle) * m for m in multipliers]
        self.pub.publish(msg)

    async def execute_callback(self, goal_handle):
        goal = goal_handle.request
        result = GripperCommand.Result()
        target = self._target_from_goal(goal)
        if target is None:
            result.success = False
            result.message = f'unknown command: {goal.command}'
            result.final_position = self.current_position
            goal_handle.abort()
            return result
        feedback = GripperCommand.Feedback()
        feedback.state = 'MOVING'
        steps = 10 if goal.wait_for_completion else 1
        start = self.current_position
        speed = goal.speed if goal.speed > 0.0 else 50.0
        delay = max(0.02, 0.15 * (50.0 / speed))
        for i in range(1, steps + 1):
            if goal_handle.is_cancel_requested:
                result.success = False
                result.message = 'canceled'
                result.final_position = self.current_position
                goal_handle.canceled()
                return result
            self.current_position = start + (target - start) * (i / steps)
            self._publish_joint()
            feedback.current_position = float(self.current_position)
            goal_handle.publish_feedback(feedback)
            time.sleep(delay)
        feedback.state = 'REACHED'
        feedback.current_position = float(self.current_position)
        goal_handle.publish_feedback(feedback)
        result.success = True
        result.message = 'gripper command completed'
        result.final_position = float(self.current_position)
        goal_handle.succeed()
        return result


def main(args=None):
    rclpy.init(args=args)
    node = GripperDemoNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
