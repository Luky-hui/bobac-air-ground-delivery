#!/usr/bin/env python3
"""Keyboard teleop for Bobac scene drone.

Keys:
  w/s: +x / -x
  a/d: +y / -y
  r/f: +z / -z
  q/e: yaw left / yaw right
  p: print latest world position
  space: stop
"""

from __future__ import annotations

import select
import sys
import termios
import tty
from typing import Optional

import rclpy
from geometry_msgs.msg import PointStamped, Twist
from rclpy.node import Node


HELP = """
Bobac drone keyboard teleop
  w/s: +x / -x
  a/d: +y / -y
  r/f: +z / -z
  q/e: yaw left / yaw right
  p: print latest /drone/world_position
  space: stop
  Ctrl-C: exit
"""


class BobacDroneKeyboardTeleop(Node):
    def __init__(self) -> None:
        super().__init__("bobac_keyboard_drone_teleop_2026")
        self.declare_parameter("cmd_vel_topic", "/drone/cmd_vel")
        self.declare_parameter("world_position_topic", "/drone/world_position")
        self.declare_parameter("linear_speed", 0.35)
        self.declare_parameter("vertical_speed", 0.22)
        self.declare_parameter("turn_speed", 0.45)
        self.pub = self.create_publisher(
            Twist, str(self.get_parameter("cmd_vel_topic").value), 10
        )
        self.last_position: Optional[PointStamped] = None
        self.create_subscription(
            PointStamped,
            str(self.get_parameter("world_position_topic").value),
            self._on_position,
            10,
        )

    def _on_position(self, msg: PointStamped) -> None:
        self.last_position = msg

    def make_cmd(self, key: str) -> Twist:
        linear = float(self.get_parameter("linear_speed").value)
        vertical = float(self.get_parameter("vertical_speed").value)
        turn = float(self.get_parameter("turn_speed").value)
        msg = Twist()
        if key == "w":
            msg.linear.x = linear
        elif key == "s":
            msg.linear.x = -linear
        elif key == "a":
            msg.linear.y = linear
        elif key == "d":
            msg.linear.y = -linear
        elif key == "r":
            msg.linear.z = vertical
        elif key == "f":
            msg.linear.z = -vertical
        elif key == "q":
            msg.angular.z = turn
        elif key == "e":
            msg.angular.z = -turn
        return msg

    def print_position(self) -> None:
        if self.last_position is None:
            print("No /drone/world_position yet")
            return
        p = self.last_position.point
        print(f"drone_world=[{p.x:.6f}, {p.y:.6f}, {p.z:.6f}]")


def read_key(timeout: float = 0.1) -> str:
    readable, _, _ = select.select([sys.stdin], [], [], timeout)
    if not readable:
        return ""
    return sys.stdin.read(1)


def main() -> None:
    rclpy.init()
    node = BobacDroneKeyboardTeleop()
    settings = termios.tcgetattr(sys.stdin)
    print(HELP)
    try:
        tty.setcbreak(sys.stdin.fileno())
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.0)
            key = read_key(0.08)
            if not key:
                continue
            if key == "\x03":
                break
            if key == "p":
                node.print_position()
                continue
            if key == " ":
                node.pub.publish(Twist())
                continue
            if key in "wsadrfqe":
                node.pub.publish(node.make_cmd(key))
    finally:
        node.pub.publish(Twist())
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
