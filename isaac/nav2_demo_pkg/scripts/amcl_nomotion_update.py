#!/usr/bin/env python3
import math
import time

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.node import Node
from std_srvs.srv import Empty


class AmclInitializer(Node):
    def __init__(self):
        super().__init__('amcl_initializer')
        self.declare_parameter('initialpose_topic', '/initialpose')
        self.declare_parameter('nomotion_service', '/request_nomotion_update')
        self.declare_parameter('startup_delay_sec', 3.0)
        self.declare_parameter('initial_x', 1.60)
        self.declare_parameter('initial_y', 0.26)
        self.declare_parameter('initial_yaw', 0.05)
        self.declare_parameter('cov_xx', 0.04)
        self.declare_parameter('cov_yy', 0.04)
        self.declare_parameter('cov_aa', 0.02)
        self.declare_parameter('update_count', 20)
        self.declare_parameter('update_interval_sec', 0.5)
        self.initialpose_topic = self.get_parameter('initialpose_topic').value
        self.initialpose_pub = self.create_publisher(PoseWithCovarianceStamped, self.initialpose_topic, 10)
        self.initialpose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            self.initialpose_topic,
            self._initialpose_callback,
            10,
        )
        self.nomotion_client = self.create_client(Empty, self.get_parameter('nomotion_service').value)
        self._handling_initialpose = False
        self._last_initialpose_stamp = None

    def _build_default_initial_pose(self):
        x = float(self.get_parameter('initial_x').value)
        y = float(self.get_parameter('initial_y').value)
        yaw = float(self.get_parameter('initial_yaw').value)
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        msg.pose.pose.position.z = 0.0
        msg.pose.pose.orientation.z = math.sin(yaw * 0.5)
        msg.pose.pose.orientation.w = math.cos(yaw * 0.5)
        msg.pose.covariance[0] = float(self.get_parameter('cov_xx').value)
        msg.pose.covariance[7] = float(self.get_parameter('cov_yy').value)
        msg.pose.covariance[35] = float(self.get_parameter('cov_aa').value)
        return msg

    def _publish_default_initial_pose(self):
        msg = self._build_default_initial_pose()
        self._handling_initialpose = True
        for _ in range(5):
            self.initialpose_pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.1)
        self._handling_initialpose = False
        self.get_logger().info(
            'published default initial pose: '
            f'x={msg.pose.pose.position.x:.3f}, y={msg.pose.pose.position.y:.3f}'
        )

    def _initialpose_callback(self, msg):
        if self._handling_initialpose:
            return
        stamp_key = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        if stamp_key == self._last_initialpose_stamp:
            return
        self._last_initialpose_stamp = stamp_key
        self.get_logger().info(
            'received initial pose from RViz: '
            f'x={msg.pose.pose.position.x:.3f}, y={msg.pose.pose.position.y:.3f}'
        )
        self._run_nomotion_updates()

    def _request_nomotion_update(self, index, count):
        future = self.nomotion_client.call_async(Empty.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        if future.result() is None:
            self.get_logger().warn(f'no-motion update {index}/{count} did not complete')
        else:
            self.get_logger().info(f'no-motion update {index}/{count}')

    def _run_nomotion_updates(self):
        if not self.nomotion_client.service_is_ready():
            if not self.nomotion_client.wait_for_service(timeout_sec=5.0):
                self.get_logger().warn('AMCL no-motion update service is not available')
                return
        count = int(self.get_parameter('update_count').value)
        interval = float(self.get_parameter('update_interval_sec').value)
        for index in range(1, count + 1):
            self._request_nomotion_update(index, count)
            time.sleep(interval)

    def run(self):
        time.sleep(float(self.get_parameter('startup_delay_sec').value))
        if not self.nomotion_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().warn('AMCL no-motion update service is not available')
            return
        self._publish_default_initial_pose()
        self._run_nomotion_updates()
        self.get_logger().info('listening for RViz 2D Pose Estimate on /initialpose')
        rclpy.spin(self)


def main(args=None):
    rclpy.init(args=args)
    node = AmclInitializer()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
