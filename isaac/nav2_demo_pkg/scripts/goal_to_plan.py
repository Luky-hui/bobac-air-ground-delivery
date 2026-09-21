#!/usr/bin/env python3
import rclpy
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import ComputePathToPose
from nav_msgs.msg import Path
from rclpy.action import ActionClient
from rclpy.node import Node


class GoalToPlan(Node):
    def __init__(self):
        super().__init__('goal_to_plan')
        self.declare_parameter('goal_topic', '/goal_pose')
        self.declare_parameter('plan_topic', '/plan')
        self.declare_parameter('planner_action', '/compute_path_to_pose')
        self.declare_parameter('planner_id', 'GridBased')
        self.plan_pub = self.create_publisher(Path, self.get_parameter('plan_topic').value, 10)
        self.action_client = ActionClient(
            self,
            ComputePathToPose,
            self.get_parameter('planner_action').value,
        )
        self.goal_sub = self.create_subscription(
            PoseStamped,
            self.get_parameter('goal_topic').value,
            self._goal_callback,
            10,
        )
        self.get_logger().info('goal_to_plan started; RViz goal poses will request ComputePathToPose')

    def _goal_callback(self, msg):
        if not self.action_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().warn('ComputePathToPose action server is not available')
            return
        goal = ComputePathToPose.Goal()
        goal.goal = msg
        goal.planner_id = self.get_parameter('planner_id').value
        self.get_logger().info(
            f'planning to x={msg.pose.position.x:.3f}, y={msg.pose.position.y:.3f}'
        )
        future = self.action_client.send_goal_async(goal)
        future.add_done_callback(self._goal_response_callback)

    def _goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('planner rejected goal')
            return
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._result_callback)

    def _result_callback(self, future):
        result = future.result().result
        if result.path.poses:
            self.plan_pub.publish(result.path)
            self.get_logger().info(f'published plan with {len(result.path.poses)} poses')
        else:
            self.get_logger().warn('planner returned an empty path')


def main(args=None):
    rclpy.init(args=args)
    node = GoalToPlan()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
