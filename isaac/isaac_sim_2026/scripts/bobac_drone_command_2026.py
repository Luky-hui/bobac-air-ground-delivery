#!/usr/bin/env python3
"""Send simple runtime commands to the Bobac scene drone."""

from __future__ import annotations

import argparse
import time

import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from std_msgs.msg import String


class BobacDroneCommand(Node):
    def __init__(self) -> None:
        super().__init__("bobac_drone_command_2026")
        self.pub = self.create_publisher(String, "/drone/command", 10)
        self.status = []
        self.last_position = None
        self.create_subscription(String, "/drone/status", self._on_status, 10)
        self.create_subscription(PointStamped, "/drone/world_position", self._on_position, 10)

    def _on_status(self, msg: String) -> None:
        self.status.append(msg.data)
        print(msg.data)

    def _on_position(self, msg: PointStamped) -> None:
        self.last_position = msg.point

    def send(self, command: str, wait_sec: float) -> None:
        deadline = time.time() + 2.0
        while rclpy.ok() and time.time() < deadline and self.pub.get_subscription_count() == 0:
            rclpy.spin_once(self, timeout_sec=0.05)

        msg = String()
        msg.data = command
        op = command.split()[0].lower() if command.split() else ""
        count = 1 if op in {"takeoff", "goto", "land", "hold"} else 3
        for _ in range(count):
            self.pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.05)

        end = time.time() + max(0.0, wait_sec)
        while rclpy.ok() and time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.1)

        if self.last_position is not None:
            p = self.last_position
            print(f"drone_world=[{p.x:.6f}, {p.y:.6f}, {p.z:.6f}]")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    takeoff = sub.add_parser("takeoff")
    takeoff.add_argument("height", nargs="?", type=float, default=0.5)

    goto = sub.add_parser("goto")
    goto.add_argument("x", type=float)
    goto.add_argument("y", type=float)
    goto.add_argument("z", type=float)
    goto.add_argument("yaw", nargs="?", type=float)

    land = sub.add_parser("land")
    land.add_argument("z", nargs="?", type=float, default=1.02)

    sub.add_parser("hold")
    sub.add_parser("status")
    parser.add_argument("--wait-sec", type=float, default=3.0)
    args = parser.parse_args()

    if args.command == "takeoff":
        command = f"takeoff {args.height:.6f}"
    elif args.command == "goto":
        command = f"goto {args.x:.6f} {args.y:.6f} {args.z:.6f}"
        if args.yaw is not None:
            command += f" {args.yaw:.6f}"
    elif args.command == "land":
        command = f"land {args.z:.6f}"
    else:
        command = args.command

    rclpy.init()
    node = BobacDroneCommand()
    try:
        node.send(command, args.wait_sec)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
