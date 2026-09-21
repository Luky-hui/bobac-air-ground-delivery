#!/usr/bin/env python3
"""
发送导航目标点脚本
功能：通过命令行向 Nav2 发送导航目标点
用法：python3 send_goal.py --x 2.0 --y 3.0 --yaw 1.57
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
import argparse
import math


class NavGoalSender(Node):
    def __init__(self):
        super().__init__('nav_goal_sender')
        self.action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

    def send_goal(self, x, y, yaw):
        """发送导航目标点"""
        goal_msg = NavigateToPose.Goal()

        # 设置目标位姿
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        goal_msg.pose.pose.position.z = 0.0

        # 将 yaw 角转换为四元数
        goal_msg.pose.pose.orientation.x = 0.0
        goal_msg.pose.pose.orientation.y = 0.0
        goal_msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal_msg.pose.pose.orientation.w = math.cos(yaw / 2.0)

        self.get_logger().info(f'发送导航目标: x={x}, y={y}, yaw={yaw}')

        # 等待 action server
        self.action_client.wait_for_server()

        # 发送目标
        self.send_goal_future = self.action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        self.send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        """目标响应回调"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('目标被拒绝')
            return

        self.get_logger().info('目标已接受，开始导航...')
        self.result_future = goal_handle.get_result_async()
        self.result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        """结果回调"""
        result = future.result().result
        self.get_logger().info('导航完成！')
        rclpy.shutdown()

    def feedback_callback(self, feedback_msg):
        """反馈回调"""
        feedback = feedback_msg.feedback
        # self.get_logger().info(f'当前距离目标: {feedback.distance_remaining:.2f}m')


def main():
    parser = argparse.ArgumentParser(description='发送导航目标点到 Nav2')
    parser.add_argument('--x', type=float, required=True, help='目标 X 坐标（米）')
    parser.add_argument('--y', type=float, required=True, help='目标 Y 坐标（米）')
    parser.add_argument('--yaw', type=float, default=0.0, help='目标朝向角（弧度）')

    args = parser.parse_args()

    rclpy.init()
    nav_goal_sender = NavGoalSender()

    try:
        nav_goal_sender.send_goal(args.x, args.y, args.yaw)
        rclpy.spin(nav_goal_sender)
    except KeyboardInterrupt:
        pass
    finally:
        nav_goal_sender.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
