#!/usr/bin/env python3
"""Keyboard jogger for Bobac right arm joints.

Publishes sensor_msgs/JointState to /joint_command. It does not run IK and
does not modify the USD scene.
"""

import math
import select
import sys
import termios
import time
import tty
from typing import Dict, List, Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


JOINT_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
JOINT_LIMITS = {
    "joint_1": (-3.11, 3.11),
    "joint_2": (-3.11, 2.36),
    "joint_3": (-2.79, 2.53),
    "joint_4": (-3.11, 3.11),
    "joint_5": (-3.11, 3.11),
    "joint_6": (-6.28, 6.28),
}
HELP = """
Bobac right arm joint keyboard jogger

  1 : select joint_1
  2 : select joint_2
  3 : select joint_3
  4 : select joint_4
  5 : select joint_5
  6 : select joint_6

  + or = : increase selected joint
  -      : decrease selected joint

  [ / ] : smaller / larger jog step
  p : print current target as radians and degrees
  h : print this help
  Ctrl-C : stop and exit

The script waits for /bobac_joint_states first, then sends complete joint targets.
"""


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def step_towards(current: float, target: float, max_step: float) -> float:
    if target > current:
        return min(target, current + max_step)
    return max(target, current - max_step)


class BobacKeyboardArmJoints(Node):
    def __init__(self) -> None:
        super().__init__("bobac_keyboard_arm_joints_2026")
        self.declare_parameter("joint_state_topic", "/bobac_joint_states")
        self.declare_parameter("command_topic", "/joint_command")
        self.declare_parameter("publish_hz", 50.0)
        self.declare_parameter("jog_step_rad", 0.015)
        self.declare_parameter("max_publish_step_rad", 0.010)

        self.pub = self.create_publisher(
            JointState, str(self.get_parameter("command_topic").value), 10
        )
        self.create_subscription(
            JointState, str(self.get_parameter("joint_state_topic").value), self._joint_cb, 10
        )

        self.have_state = False
        self.current: Dict[str, float] = {name: 0.0 for name in JOINT_NAMES}
        self.target: Dict[str, float] = {name: 0.0 for name in JOINT_NAMES}
        self.commanded: Dict[str, float] = {name: 0.0 for name in JOINT_NAMES}
        self.jog_step = float(self.get_parameter("jog_step_rad").value)
        self.max_publish_step = float(self.get_parameter("max_publish_step_rad").value)
        self.selected_joint = "joint_1"
        self.warned_no_state = False

    def _joint_cb(self, msg: JointState) -> None:
        values = dict(zip(msg.name, msg.position))
        if not all(name in values for name in JOINT_NAMES):
            return
        for name in JOINT_NAMES:
            self.current[name] = float(values[name])
        if not self.have_state:
            self.have_state = True
            self.target.update(self.current)
            self.commanded.update(self.current)
            self.get_logger().info("received joint states; keyboard arm jog is ready")

    def select_joint(self, joint_name: str) -> None:
        self.selected_joint = joint_name
        print(f"selected {joint_name}")

    def jog(self, joint_name: str, direction: float, verbose: bool = False) -> None:
        if not self.have_state:
            if not self.warned_no_state:
                print(
                    f"No {self.get_parameter('joint_state_topic').value} yet; "
                    "not commanding arm"
                )
                self.warned_no_state = True
            return
        low, high = JOINT_LIMITS[joint_name]
        self.target[joint_name] = clamp(
            self.target[joint_name] + direction * self.jog_step, low, high
        )
        if verbose:
            print(
                f"{joint_name}: target={self.target[joint_name]:.6f} rad "
                f"({math.degrees(self.target[joint_name]):.2f} deg)"
            )

    def publish(self) -> None:
        if not self.have_state:
            return
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(JOINT_NAMES)
        positions: List[float] = []
        for name in JOINT_NAMES:
            self.commanded[name] = step_towards(
                self.commanded[name], self.target[name], self.max_publish_step
            )
            positions.append(self.commanded[name])
        msg.position = positions
        self.pub.publish(msg)

    def print_target(self) -> None:
        source = self.target if self.have_state else self.current
        rad = [source[name] for name in JOINT_NAMES]
        deg = [math.degrees(v) for v in rad]
        print("rad:", "[" + ", ".join(f"{v:.6f}" for v in rad) + "]")
        print("deg:", "[" + ", ".join(f"{v:.2f}" for v in deg) + "]")


def read_key(timeout: float) -> str:
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    if not ready:
        return ""
    return sys.stdin.read(1)


def self_test() -> None:
    assert joint_select_keys() == {
        "1": "joint_1",
        "2": "joint_2",
        "3": "joint_3",
        "4": "joint_4",
        "5": "joint_5",
        "6": "joint_6",
    }
    assert "+" in PLUS_KEYS and "=" in PLUS_KEYS
    assert "-" in MINUS_KEYS and "_" in MINUS_KEYS
    print("arm key mapping self-test OK")


PLUS_KEYS = {"+", "="}
MINUS_KEYS = {"-", "_"}


def joint_select_keys():
    return {
        "1": "joint_1",
        "2": "joint_2",
        "3": "joint_3",
        "4": "joint_4",
        "5": "joint_5",
        "6": "joint_6",
    }


def main(args=None) -> None:
    if "--self-test" in sys.argv:
        self_test()
        return

    rclpy.init(args=args)
    node = BobacKeyboardArmJoints()
    old_settings = termios.tcgetattr(sys.stdin)
    rate_hz = max(1.0, float(node.get_parameter("publish_hz").value))
    period = 1.0 / rate_hz
    select_keys = joint_select_keys()
    print(HELP)
    try:
        tty.setraw(sys.stdin.fileno())
        while rclpy.ok():
            key = read_key(period)
            if key == "\x03":
                break
            if key in select_keys:
                node.select_joint(select_keys[key])
            elif key in PLUS_KEYS:
                node.jog(node.selected_joint, 1.0)
            elif key in MINUS_KEYS:
                node.jog(node.selected_joint, -1.0)
            elif key == "[":
                node.jog_step = max(0.001, node.jog_step * 0.5)
                print(f"jog_step_rad={node.jog_step:.6f}")
            elif key == "]":
                node.jog_step = min(0.10, node.jog_step * 2.0)
                print(f"jog_step_rad={node.jog_step:.6f}")
            elif key == "p":
                node.print_target()
            elif key == "h":
                print(HELP)
            rclpy.spin_once(node, timeout_sec=0.0)
            node.publish()
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
