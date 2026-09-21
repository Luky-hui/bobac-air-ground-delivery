#!/usr/bin/env python3
"""Keyboard PX4 Offboard teleop for marking drone waypoints.

Controls are setpoint based, not direct prim motion. The node continuously
publishes /fmu/in/offboard_control_mode and /fmu/in/trajectory_setpoint.
"""

from __future__ import annotations

import math
import select
import subprocess
import sys
import termios
import time
import tty
from typing import Iterable

import rclpy
from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import TrajectorySetpoint
from px4_msgs.msg import VehicleCommand
from px4_msgs.msg import VehicleLocalPosition
from px4_msgs.msg import VehicleStatus
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String


PX4_IN_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)

PX4_OUT_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)

HELP = """
PX4 keyboard teleop, world/ENU increments:
  t: takeoff and enter Offboard
  w/s: world +x / -x
  a/d: world +y / -y
  r/f: up / down
  q/e: yaw left / yaw right
  h: hover at current position
  l: land
  p: print current pose and target
  [/]: decrease / increase position step
  {/}: decrease / increase yaw step
  1/2: side door close/open
  3/4: bottom door close/open
  space: hold current target
  x: exit
"""


def finite_position(msg: VehicleLocalPosition) -> bool:
    return math.isfinite(msg.x) and math.isfinite(msg.y) and math.isfinite(msg.z)


def world_enu_delta_to_px4_ned_delta(delta: Iterable[float]) -> list[float]:
    dx, dy, dz = [float(v) for v in delta]
    return [dy, dx, -dz]


def normalize_yaw(yaw: float) -> float:
    return math.atan2(math.sin(yaw), math.cos(yaw))


class RawTerminal:
    def __enter__(self):
        self.settings = termios.tcgetattr(sys.stdin)
        tty.setraw(sys.stdin.fileno())
        return self

    def __exit__(self, exc_type, exc, tb):
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)


class Px4KeyboardTeleop(Node):
    def __init__(self) -> None:
        super().__init__("bobac_px4_keyboard_teleop_2026")
        self.declare_parameter("takeoff_height", 0.8)
        self.declare_parameter("step", 0.15)
        self.declare_parameter("yaw_step", 0.15)
        self.declare_parameter("px4_param_tool", "/home/u-zhuang/PX4-Autopilot/build/px4_sitl_default/bin/px4-param")
        self.declare_parameter("cargo_command_topic", "/cargo_bay/command")

        self.offboard_pub = self.create_publisher(OffboardControlMode, "/fmu/in/offboard_control_mode", PX4_IN_QOS)
        self.setpoint_pub = self.create_publisher(TrajectorySetpoint, "/fmu/in/trajectory_setpoint", PX4_IN_QOS)
        self.command_pub = self.create_publisher(VehicleCommand, "/fmu/in/vehicle_command", PX4_IN_QOS)
        self.cargo_pub = self.create_publisher(String, str(self.get_parameter("cargo_command_topic").value), 10)
        self.create_subscription(VehicleLocalPosition, "/fmu/out/vehicle_local_position", self._on_position, PX4_OUT_QOS)
        self.create_subscription(VehicleStatus, "/fmu/out/vehicle_status", self._on_status, PX4_OUT_QOS)
        self.create_timer(0.05, self._publish_stream)

        self.position: VehicleLocalPosition | None = None
        self.status: VehicleStatus | None = None
        self.target: list[float] | None = None
        self.target_yaw: float | None = None
        self.stream_enabled = False
        self.step = float(self.get_parameter("step").value)
        self.yaw_step = float(self.get_parameter("yaw_step").value)

    def _on_position(self, msg: VehicleLocalPosition) -> None:
        if finite_position(msg):
            self.position = msg

    def _on_status(self, msg: VehicleStatus) -> None:
        self.status = msg

    def timestamp_us(self) -> int:
        return int(self.get_clock().now().nanoseconds / 1000)

    def _publish_stream(self) -> None:
        if not self.stream_enabled or self.target is None:
            return
        now = self.timestamp_us()
        mode = OffboardControlMode()
        mode.timestamp = now
        mode.position = True
        self.offboard_pub.publish(mode)

        msg = TrajectorySetpoint()
        msg.timestamp = now
        msg.position = [float(self.target[0]), float(self.target[1]), float(self.target[2])]
        if self.target_yaw is not None:
            msg.yaw = float(self.target_yaw)
        self.setpoint_pub.publish(msg)

    def spin_for(self, seconds: float) -> None:
        end = time.monotonic() + seconds
        while rclpy.ok() and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.02)

    def configure_px4(self) -> None:
        tool = str(self.get_parameter("px4_param_tool").value)
        for name, value in (("COM_RC_IN_MODE", "4"), ("COM_RCL_EXCEPT", "4")):
            try:
                subprocess.run(
                    [tool, "set", name, value],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=3.0,
                )
            except Exception:
                pass

    def send_vehicle_command(self, command: int, param1: float = 0.0, param2: float = 0.0, count: int = 3) -> None:
        for _ in range(max(1, count)):
            msg = VehicleCommand()
            msg.timestamp = self.timestamp_us()
            msg.command = int(command)
            msg.param1 = float(param1)
            msg.param2 = float(param2)
            msg.target_system = 1
            msg.target_component = 1
            msg.source_system = 1
            msg.source_component = 1
            msg.from_external = True
            self.command_pub.publish(msg)
            self.spin_for(0.08)

    def set_target_from_current(self, dz: float = 0.0) -> bool:
        if self.position is None:
            print("No /fmu/out/vehicle_local_position yet")
            return False
        self.target = [
            float(self.position.x),
            float(self.position.y),
            float(self.position.z) + float(dz),
        ]
        self.target_yaw = float(self.position.heading) if math.isfinite(self.position.heading) else 0.0
        self.stream_enabled = True
        return True

    def takeoff(self) -> None:
        if self.position is None:
            print("No PX4 local position yet")
            return
        self.configure_px4()
        height = max(0.05, float(self.get_parameter("takeoff_height").value))
        self.set_target_from_current(dz=-height)
        print(f"takeoff target NED={self.target}, streaming before Offboard/arm")
        self.spin_for(1.5)
        self.send_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0, count=5)
        self.send_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0, 0.0, count=3)
        self.send_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0, count=5)

    def move_enu(self, dx: float, dy: float, dz: float) -> None:
        if self.target is None:
            if not self.set_target_from_current():
                return
        nd = world_enu_delta_to_px4_ned_delta([dx, dy, dz])
        assert self.target is not None
        self.target = [
            self.target[0] + nd[0],
            self.target[1] + nd[1],
            self.target[2] + nd[2],
        ]
        self.stream_enabled = True
        self.print_state(prefix="target updated")

    def yaw(self, delta: float) -> None:
        if self.target_yaw is None:
            self.target_yaw = float(self.position.heading) if self.position else 0.0
        self.target_yaw = normalize_yaw(self.target_yaw + delta)
        self.stream_enabled = True
        self.print_state(prefix="yaw updated")

    def hover(self) -> None:
        self.set_target_from_current()
        self.print_state(prefix="hover")

    def land(self) -> None:
        self.send_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND, count=1)
        print("land command sent")

    def send_cargo(self, command: str) -> None:
        msg = String()
        msg.data = command
        for _ in range(3):
            self.cargo_pub.publish(msg)
            self.spin_for(0.05)
        print(f"cargo command sent: {command}")

    def print_state(self, prefix: str = "state") -> None:
        if self.position is None:
            print(f"{prefix}: no position")
            return
        enu = [self.position.y, self.position.x, -self.position.z]
        target = self.target if self.target is not None else None
        print(
            f"{prefix}: px4_ned=[{self.position.x:.6f}, {self.position.y:.6f}, {self.position.z:.6f}], "
            f"world_enu=[{enu[0]:.6f}, {enu[1]:.6f}, {enu[2]:.6f}], "
            f"heading={self.position.heading:.6f}, target={target}, "
            f"step={self.step:.3f}, yaw_step={self.yaw_step:.3f}"
        )
        if self.status is not None:
            print(
                f"  status: arming={self.status.arming_state}, nav={self.status.nav_state}, "
                f"preflight={bool(self.status.pre_flight_checks_pass)}, failsafe={bool(self.status.failsafe)}"
            )

    def key(self, ch: str) -> bool:
        if ch == "x":
            return False
        if ch == "t":
            self.takeoff()
        elif ch == "w":
            self.move_enu(self.step, 0.0, 0.0)
        elif ch == "s":
            self.move_enu(-self.step, 0.0, 0.0)
        elif ch == "a":
            self.move_enu(0.0, self.step, 0.0)
        elif ch == "d":
            self.move_enu(0.0, -self.step, 0.0)
        elif ch == "r":
            self.move_enu(0.0, 0.0, self.step)
        elif ch == "f":
            self.move_enu(0.0, 0.0, -self.step)
        elif ch == "q":
            self.yaw(self.yaw_step)
        elif ch == "e":
            self.yaw(-self.yaw_step)
        elif ch == "h" or ch == " ":
            self.hover()
        elif ch == "l":
            self.land()
        elif ch == "p":
            self.print_state(prefix="print")
        elif ch == "[":
            self.step = max(0.02, self.step * 0.8)
            print(f"step={self.step:.3f}")
        elif ch == "]":
            self.step = min(1.0, self.step * 1.25)
            print(f"step={self.step:.3f}")
        elif ch == "{":
            self.yaw_step = max(0.02, self.yaw_step * 0.8)
            print(f"yaw_step={self.yaw_step:.3f}")
        elif ch == "}":
            self.yaw_step = min(1.0, self.yaw_step * 1.25)
            print(f"yaw_step={self.yaw_step:.3f}")
        elif ch == "1":
            self.send_cargo("left_close")
        elif ch == "2":
            self.send_cargo("left_open")
        elif ch == "3":
            self.send_cargo("bottom_close")
        elif ch == "4":
            self.send_cargo("bottom_open")
        return True


def read_key(timeout: float = 0.05) -> str:
    readable, _, _ = select.select([sys.stdin], [], [], timeout)
    if not readable:
        return ""
    return sys.stdin.read(1)


def main() -> None:
    rclpy.init()
    node = Px4KeyboardTeleop()
    print(HELP)
    try:
        with RawTerminal():
            running = True
            while rclpy.ok() and running:
                rclpy.spin_once(node, timeout_sec=0.01)
                ch = read_key(0.02)
                if ch:
                    running = node.key(ch)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
