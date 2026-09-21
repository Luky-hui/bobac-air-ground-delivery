#!/usr/bin/env python3
"""Slow, deadman-style base jogger for measuring Isaac Sim office waypoints."""

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


MOVE_BINDINGS = {
    "i": (1.0, 0.0),
    ",": (-1.0, 0.0),
    "j": (0.0, 1.0),
    "l": (0.0, -1.0),
    "u": (1.0, 1.0),
    "o": (1.0, -1.0),
    "m": (-1.0, -1.0),
    ".": (-1.0, 1.0),
}


HELP = """
Safe base jog for waypoint capture

Hold keys to move, release to auto-stop:
  u    i    o       forward arcs / forward / forward arcs
  j    k    l       rotate left / stop / rotate right
  m    ,    .       reverse arcs / reverse / reverse arcs

  p: print current [x, y, yaw] from /odom
  space or k: stop
  Ctrl-C: stop and exit
"""


def yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def clamp_step(current: float, target: float, max_step: float) -> float:
    if target > current:
        return min(target, current + max_step)
    return max(target, current - max_step)


class SafeBaseJog(Node):
    def __init__(self) -> None:
        super().__init__("safe_base_jog_2026")

        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("publish_hz", 20.0)
        self.declare_parameter("max_linear", 0.035)
        self.declare_parameter("max_angular", 0.10)
        self.declare_parameter("linear_accel", 0.035)
        self.declare_parameter("angular_accel", 0.12)
        self.declare_parameter("deadman_timeout", 0.45)
        self.declare_parameter("max_abs_z", 0.08)

        self.max_linear = abs(float(self.get_parameter("max_linear").value))
        self.max_angular = abs(float(self.get_parameter("max_angular").value))
        self.linear_accel = abs(float(self.get_parameter("linear_accel").value))
        self.angular_accel = abs(float(self.get_parameter("angular_accel").value))
        self.deadman_timeout = max(0.1, float(self.get_parameter("deadman_timeout").value))
        self.max_abs_z = abs(float(self.get_parameter("max_abs_z").value))

        self.target_linear = 0.0
        self.target_angular = 0.0
        self.current_linear = 0.0
        self.current_angular = 0.0
        self.last_key_time = 0.0
        self.last_publish_time = time.monotonic()
        self.last_odom: Optional[Odometry] = None
        self.bad_odom = False

        self.pub = self.create_publisher(Twist, str(self.get_parameter("cmd_vel_topic").value), 10)
        self.create_subscription(Odometry, str(self.get_parameter("odom_topic").value), self._odom_cb, 10)

        hz = max(1.0, float(self.get_parameter("publish_hz").value))
        self.timer = self.create_timer(1.0 / hz, self._publish)

    def _odom_cb(self, msg: Odometry) -> None:
        self.last_odom = msg
        p = msg.pose.pose.position
        values = [
            p.x,
            p.y,
            p.z,
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.angular.z,
        ]
        bad = not all(math.isfinite(v) for v in values) or abs(p.z) > self.max_abs_z
        if bad and not self.bad_odom:
            self.get_logger().error(
                "invalid /odom detected; publishing stop only. Reset Isaac Sim before continuing."
            )
        self.bad_odom = bad

    def set_motion(self, linear_scale: float, angular_scale: float) -> None:
        if self.bad_odom:
            self.stop()
            return
        self.target_linear = linear_scale * self.max_linear
        self.target_angular = angular_scale * self.max_angular
        self.last_key_time = time.monotonic()

    def stop(self) -> None:
        self.target_linear = 0.0
        self.target_angular = 0.0
        self.current_linear = 0.0
        self.current_angular = 0.0
        self._publish_zero()

    def print_pose(self) -> None:
        msg = self.last_odom
        if msg is None:
            print("No /odom received yet")
            return
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = yaw_from_quat(q.x, q.y, q.z, q.w)
        if not all(math.isfinite(v) for v in (p.x, p.y, p.z, yaw)):
            print("Current /odom contains NaN/Inf; reset Isaac Sim")
            return
        print(f"[{p.x:.6f}, {p.y:.6f}, {yaw:.6f}]  z={p.z:.6f}")

    def _publish_zero(self) -> None:
        self.pub.publish(Twist())

    def _publish(self) -> None:
        now = time.monotonic()
        if now - self.last_key_time > self.deadman_timeout or self.bad_odom:
            self.target_linear = 0.0
            self.target_angular = 0.0

        dt = max(1e-3, now - self.last_publish_time)
        self.last_publish_time = now
        self.current_linear = clamp_step(
            self.current_linear, self.target_linear, self.linear_accel * dt
        )
        self.current_angular = clamp_step(
            self.current_angular, self.target_angular, self.angular_accel * dt
        )

        msg = Twist()
        msg.linear.x = self.current_linear
        msg.angular.z = self.current_angular
        self.pub.publish(msg)


class RawTerminal:
    def __enter__(self):
        self.settings = termios.tcgetattr(sys.stdin)
        tty.setraw(sys.stdin.fileno())
        return self

    def __exit__(self, exc_type, exc, tb):
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)

    @staticmethod
    def read_key(timeout: float = 0.05) -> Optional[str]:
        readable, _, _ = select.select([sys.stdin], [], [], timeout)
        if not readable:
            return None
        return sys.stdin.read(1)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SafeBaseJog()
    print(HELP)
    try:
        with RawTerminal():
            while rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0.0)
                key = RawTerminal.read_key(0.03)
                if key is None:
                    continue
                if key == "\x03":
                    break
                if key in MOVE_BINDINGS:
                    node.set_motion(*MOVE_BINDINGS[key])
                elif key in (" ", "k"):
                    node.stop()
                elif key == "p":
                    node.print_pose()
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
