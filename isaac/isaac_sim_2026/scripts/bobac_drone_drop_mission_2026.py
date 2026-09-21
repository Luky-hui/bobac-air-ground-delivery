#!/usr/bin/env python3
"""Waypoint mission for the Bobac scene drone and cargo bay."""

from __future__ import annotations

import argparse
import math
import time
from typing import Iterable

import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from std_msgs.msg import String


WAYPOINTS = [
    ("takeoff_clear", (4.35, -0.38, 1.60, 0.0), 0.10, 12.0),
    ("wall_left_entry", (2.60, -1.70, 1.80, 0.0), 0.10, 16.0),
    ("wall_left_exit", (2.60, -4.70, 1.80, 0.0), 0.10, 16.0),
    ("drop_align", (4.790769, -4.700881, 1.23, 0.0), 0.08, 16.0),
]


class BobacDroneDropMission(Node):
    def __init__(self) -> None:
        super().__init__("bobac_drone_drop_mission_2026")
        self.drone_pub = self.create_publisher(String, "/drone/command", 10)
        self.cargo_pub = self.create_publisher(String, "/cargo_bay/command", 10)
        self.position = None
        self.create_subscription(PointStamped, "/drone/world_position", self._on_position, 10)
        self.create_subscription(String, "/drone/status", self._on_drone_status, 10)
        self.create_subscription(String, "/cargo_bay/status", self._on_cargo_status, 10)

    def _on_position(self, msg: PointStamped) -> None:
        self.position = msg.point

    def _on_drone_status(self, msg: String) -> None:
        self.get_logger().info(f"drone: {msg.data}")

    def _on_cargo_status(self, msg: String) -> None:
        self.get_logger().info(f"cargo: {msg.data}")

    def _wait_for_links(self) -> None:
        deadline = time.time() + 5.0
        while rclpy.ok() and time.time() < deadline:
            if self.drone_pub.get_subscription_count() and self.cargo_pub.get_subscription_count():
                return
            rclpy.spin_once(self, timeout_sec=0.05)
        raise RuntimeError("missing /drone/command or /cargo_bay/command subscriber")

    def _publish_string(self, pub, command: str) -> None:
        msg = String()
        msg.data = command
        pub.publish(msg)

    def cargo(self, command: str, wait_sec: float = 2.0) -> None:
        self.get_logger().info(f"cargo command: {command}")
        self._publish_string(self.cargo_pub, command)
        end = time.time() + wait_sec
        while rclpy.ok() and time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.1)

    def takeoff_prime(self, height: float = 0.35, wait_sec: float = 4.0) -> None:
        command = f"takeoff {height:.6f}"
        self.get_logger().info(f"takeoff prime: {command}")
        self._publish_string(self.drone_pub, command)
        end = time.time() + wait_sec
        while rclpy.ok() and time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.1)

    def goto(self, name: str, target: Iterable[float], tolerance: float, timeout_sec: float) -> None:
        x, y, z, yaw = [float(v) for v in target]
        command = f"goto {x:.6f} {y:.6f} {z:.6f} {yaw:.6f}"
        self.get_logger().info(f"{name}: {command}")
        self._publish_string(self.drone_pub, command)
        deadline = time.time() + timeout_sec
        last_log = 0.0
        while rclpy.ok() and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.position is None:
                continue
            dx = x - self.position.x
            dy = y - self.position.y
            dz = z - self.position.z
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            now = time.time()
            if now - last_log > 1.0:
                self.get_logger().info(
                    f"{name}: pos=[{self.position.x:.3f},{self.position.y:.3f},{self.position.z:.3f}], dist={dist:.3f}"
                )
                last_log = now
            if dist <= tolerance:
                self.get_logger().info(f"{name}: reached")
                return
        raise RuntimeError(f"{name}: timeout before reaching target")

    def run(self, drop: bool) -> None:
        self._wait_for_links()
        self.cargo("left_close", wait_sec=2.0)
        self.takeoff_prime()
        for name, target, tolerance, timeout_sec in WAYPOINTS:
            self.goto(name, target, tolerance, timeout_sec)
        if drop:
            self.cargo("bottom_open", wait_sec=3.0)
            self.cargo("bottom_close", wait_sec=2.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drop", action="store_true", help="open and close the bottom bay at the final waypoint")
    args = parser.parse_args()

    rclpy.init()
    node = BobacDroneDropMission()
    try:
        node.run(drop=args.drop)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
