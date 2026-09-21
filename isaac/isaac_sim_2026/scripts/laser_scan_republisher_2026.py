#!/usr/bin/env python3
import math

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan, PointCloud2
from sensor_msgs_py import point_cloud2


class LaserScanRepublisher2026(Node):
    def __init__(self):
        super().__init__("laser_scan_republisher_2026")
        self.declare_parameter("input_topic", "/laser_scan")
        self.declare_parameter("output_topic", "/laser_scan_nav")
        self.declare_parameter("target_frame", "lidar")
        self.declare_parameter("cloud_topic", "/laser_obstacles_cloud")
        self.declare_parameter("publish_cloud", True)
        self.declare_parameter("cloud_range_max", 5.0)
        self.declare_parameter("cloud_z", 0.10)
        self.declare_parameter("restamp_with_now", True)
        self.declare_parameter("replace_negative_with_inf", True)

        input_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        output_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.publisher = self.create_publisher(
            LaserScan, self.get_parameter("output_topic").value, output_qos
        )
        self.cloud_publisher = self.create_publisher(
            PointCloud2, self.get_parameter("cloud_topic").value, output_qos
        )
        self.subscription = self.create_subscription(
            LaserScan,
            self.get_parameter("input_topic").value,
            self.on_scan,
            input_qos,
        )
        self.get_logger().info(
            "laser_scan_republisher_2026: "
            f"{self.get_parameter('input_topic').value} -> "
            f"{self.get_parameter('output_topic').value}, "
            f"{self.get_parameter('cloud_topic').value}"
        )

    def on_scan(self, msg: LaserScan) -> None:
        out = LaserScan()
        out.header = msg.header

        frame = str(self.get_parameter("target_frame").value)
        if frame:
            out.header.frame_id = frame
        if bool(self.get_parameter("restamp_with_now").value):
            out.header.stamp = self.get_clock().now().to_msg()

        out.angle_min = msg.angle_min
        out.angle_max = msg.angle_max
        out.angle_increment = msg.angle_increment
        out.time_increment = msg.time_increment
        out.scan_time = msg.scan_time
        out.range_min = msg.range_min
        out.range_max = msg.range_max
        out.intensities = msg.intensities

        if bool(self.get_parameter("replace_negative_with_inf").value):
            out.ranges = [
                math.inf if math.isfinite(value) and value < out.range_min else value
                for value in msg.ranges
            ]
        else:
            out.ranges = msg.ranges

        self.publisher.publish(out)

        if bool(self.get_parameter("publish_cloud").value):
            self.publish_cloud(out)

    def publish_cloud(self, scan: LaserScan) -> None:
        cloud_range_max = float(self.get_parameter("cloud_range_max").value)
        cloud_z = float(self.get_parameter("cloud_z").value)
        points = []

        for index, value in enumerate(scan.ranges):
            distance = float(value)
            if not math.isfinite(distance):
                continue
            if distance < scan.range_min or distance > min(scan.range_max, cloud_range_max):
                continue
            angle = scan.angle_min + index * scan.angle_increment
            points.append(
                [
                    distance * math.cos(angle),
                    distance * math.sin(angle),
                    cloud_z,
                ]
            )

        cloud = point_cloud2.create_cloud_xyz32(scan.header, points)
        self.cloud_publisher.publish(cloud)


def main(args=None):
    rclpy.init(args=args)
    node = LaserScanRepublisher2026()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
