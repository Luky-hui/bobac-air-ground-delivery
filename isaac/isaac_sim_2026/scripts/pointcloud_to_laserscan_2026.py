#!/usr/bin/env python3
import math
from typing import Iterable, Tuple

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, PointCloud2
from sensor_msgs_py import point_cloud2


class PointCloudToLaserScan2026(Node):
    def __init__(self):
        super().__init__("pointcloud_to_laserscan_2026")
        self.declare_parameter("cloud_topic", "/avoidance/lidar/pointcloud")
        self.declare_parameter("scan_topic", "/laser_scan_fuse")
        self.declare_parameter("secondary_scan_topic", "/scan")
        self.declare_parameter("target_frame", "avoidance_lidar")
        self.declare_parameter("angle_min", -math.pi)
        self.declare_parameter("angle_max", math.pi)
        self.declare_parameter("angle_increment", math.radians(0.5))
        self.declare_parameter("range_min", 0.08)
        self.declare_parameter("range_max", 12.0)
        self.declare_parameter("min_height", -0.25)
        self.declare_parameter("max_height", 1.25)
        self.declare_parameter("use_inf", True)
        self.declare_parameter("min_valid_points", 8)

        self.scan_pub = self.create_publisher(
            LaserScan, self.get_parameter("scan_topic").value, 10
        )
        self.secondary_pub = self.create_publisher(
            LaserScan, self.get_parameter("secondary_scan_topic").value, 10
        )
        self.create_subscription(
            PointCloud2,
            self.get_parameter("cloud_topic").value,
            self.on_cloud,
            10,
        )
        self.get_logger().info(
            "pointcloud_to_laserscan_2026: "
            f"{self.get_parameter('cloud_topic').value} -> "
            f"{self.get_parameter('scan_topic').value}, "
            f"{self.get_parameter('secondary_scan_topic').value}"
        )

    @staticmethod
    def _iter_xyz(msg: PointCloud2) -> Iterable[Tuple[float, float, float]]:
        return point_cloud2.read_points(
            msg, field_names=("x", "y", "z"), skip_nans=True
        )

    def on_cloud(self, msg: PointCloud2) -> None:
        angle_min = float(self.get_parameter("angle_min").value)
        angle_max = float(self.get_parameter("angle_max").value)
        angle_increment = float(self.get_parameter("angle_increment").value)
        range_min = float(self.get_parameter("range_min").value)
        range_max = float(self.get_parameter("range_max").value)
        min_height = float(self.get_parameter("min_height").value)
        max_height = float(self.get_parameter("max_height").value)
        use_inf = bool(self.get_parameter("use_inf").value)
        min_valid_points = max(0, int(self.get_parameter("min_valid_points").value))

        count = int(math.ceil((angle_max - angle_min) / angle_increment))
        fill = math.inf if use_inf else range_max + 1.0
        ranges = [fill] * count
        valid_points = 0

        for x, y, z in self._iter_xyz(msg):
            z = float(z)
            if z < min_height or z > max_height:
                continue
            x = float(x)
            y = float(y)
            distance = math.hypot(x, y)
            if distance < range_min or distance > range_max:
                continue
            angle = math.atan2(y, x)
            if angle < angle_min or angle >= angle_max:
                continue
            index = int((angle - angle_min) / angle_increment)
            if 0 <= index < count and distance < ranges[index]:
                ranges[index] = distance
                valid_points += 1

        if valid_points < min_valid_points:
            return

        scan = LaserScan()
        scan.header = msg.header
        frame = str(self.get_parameter("target_frame").value)
        if frame:
            scan.header.frame_id = frame
        scan.angle_min = angle_min
        scan.angle_max = angle_min + (count - 1) * angle_increment
        scan.angle_increment = angle_increment
        scan.time_increment = 0.0
        scan.scan_time = 0.1
        scan.range_min = range_min
        scan.range_max = range_max
        scan.ranges = ranges

        self.scan_pub.publish(scan)
        self.secondary_pub.publish(scan)


def main(args=None):
    rclpy.init(args=args)
    node = PointCloudToLaserScan2026()
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
