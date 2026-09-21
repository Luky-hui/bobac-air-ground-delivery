#!/usr/bin/env python3
"""Keyboard teleop for the Bobac omnidirectional base.

Publishes geometry_msgs/Twist only. It does not modify the USD scene.
"""

import math
import select
import sys
import termios
import time
import tty
from typing import Optional

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node


BASE_KEY_BINDINGS = {
    "w": (1.0, 0.0, 0.0),
    "s": (-1.0, 0.0, 0.0),
    "a": (0.0, 1.0, 0.0),
    "d": (0.0, -1.0, 0.0),
    "q": (0.0, 0.0, 1.0),
    "e": (0.0, 0.0, -1.0),
}


HELP = """
Bobac base keyboard teleop

  w : forward
  s : backward
  a : move left
  d : move right
  q : rotate left in place
  e : rotate right in place
  x or space : stop
  p : print current [x, y, yaw] from /odom
  +/- : speed up / slow down
  Ctrl-C : stop and exit
"""


def yaw_from_odom(msg: Odometry) -> float:
    q = msg.pose.pose.orientation
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class BobacKeyboardBaseTeleop(Node):
    def __init__(self) -> None:
        super().__init__("bobac_keyboard_base_teleop_2026")
        self.declare_parameter("cmd_vel_topic", "/bobac_kinematic_cmd_vel")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("publish_hz", 30.0)
        self.declare_parameter("linear_speed", 0.12)
        self.declare_parameter("angular_speed", 0.35)
        self.declare_parameter("deadman_timeout", 0.35)

        self.pub = self.create_publisher(
            Twist, str(self.get_parameter("cmd_vel_topic").value), 10
        )
        self.odom: Optional[Odometry] = None
        self.create_subscription(
            Odometry, str(self.get_parameter("odom_topic").value), self._odom_cb, 10
        )

        self.linear_speed = float(self.get_parameter("linear_speed").value)
        self.angular_speed = float(self.get_parameter("angular_speed").value)
        self.deadman_timeout = float(self.get_parameter("deadman_timeout").value)
        self.vx = 0.0
        self.vy = 0.0
        self.wz = 0.0
        self.last_key_time = 0.0

    def _odom_cb(self, msg: Odometry) -> None:
        self.odom = msg

    def set_cmd(self, vx_scale: float = 0.0, vy_scale: float = 0.0, wz_scale: float = 0.0) -> None:
        self.vx = vx_scale * self.linear_speed
        self.vy = vy_scale * self.linear_speed
        self.wz = wz_scale * self.angular_speed
        self.last_key_time = time.monotonic()

    def stop(self) -> None:
        self.vx = 0.0
        self.vy = 0.0
        self.wz = 0.0
        self.pub.publish(Twist())

    def publish(self) -> None:
        if time.monotonic() - self.last_key_time > self.deadman_timeout:
            self.vx = 0.0
            self.vy = 0.0
            self.wz = 0.0
        msg = Twist()
        msg.linear.x = self.vx
        msg.linear.y = self.vy
        msg.angular.z = self.wz
        self.pub.publish(msg)

    def print_pose(self) -> None:
        if self.odom is None:
            print("No /odom received yet")
            return
        p = self.odom.pose.pose.position
        yaw = yaw_from_odom(self.odom)
        print(f"[{p.x:.6f}, {p.y:.6f}, {yaw:.6f}]")


def read_key(timeout: float) -> str:
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    if not ready:
        return ""
    return sys.stdin.read(1)


def self_test() -> None:
    expected = {
        "w": (1.0, 0.0, 0.0),
        "s": (-1.0, 0.0, 0.0),
        "a": (0.0, 1.0, 0.0),
        "d": (0.0, -1.0, 0.0),
        "q": (0.0, 0.0, 1.0),
        "e": (0.0, 0.0, -1.0),
    }
    assert BASE_KEY_BINDINGS == expected, BASE_KEY_BINDINGS
    print("base key mapping self-test OK")


def main(args=None) -> None:
    if "--self-test" in sys.argv:
        self_test()
        return

    rclpy.init(args=args)
    node = BobacKeyboardBaseTeleop()
    old_settings = termios.tcgetattr(sys.stdin)
    rate_hz = max(1.0, float(node.get_parameter("publish_hz").value))
    period = 1.0 / rate_hz

    print(HELP)
    try:
        tty.setraw(sys.stdin.fileno())
        while rclpy.ok():
            key = read_key(period)
            if key == "\x03":
                break
            if key in BASE_KEY_BINDINGS:
                vx_scale, vy_scale, wz_scale = BASE_KEY_BINDINGS[key]
                node.set_cmd(vx_scale=vx_scale, vy_scale=vy_scale, wz_scale=wz_scale)
            elif key in {"x", " "}:
                node.stop()
            elif key == "p":
                node.print_pose()
            elif key in {"+", "="}:
                node.linear_speed *= 1.15
                node.angular_speed *= 1.15
                print(f"speed: linear={node.linear_speed:.3f}, angular={node.angular_speed:.3f}")
            elif key in {"-", "_"}:
                node.linear_speed *= 0.85
                node.angular_speed *= 0.85
                print(f"speed: linear={node.linear_speed:.3f}, angular={node.angular_speed:.3f}")

            rclpy.spin_once(node, timeout_sec=0.0)
            node.publish()
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        for _ in range(10):
            node.stop()
            time.sleep(0.02)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
