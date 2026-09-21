#!/usr/bin/env python3
"""Print current PX4 local pose for drone waypoint marking.

PX4 local position is NED:
  x north/forward, y east/right, z down.
For the helper scripts, user-facing move deltas are world/ENU:
  x world +x, y world +y, z up.
"""

from __future__ import annotations

import math

import rclpy
from px4_msgs.msg import VehicleLocalPosition, VehicleStatus
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy


PX4_OUT_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


class Px4PosePrinter(Node):
    def __init__(self) -> None:
        super().__init__("bobac_px4_print_pose_2026")
        self.position: VehicleLocalPosition | None = None
        self.status: VehicleStatus | None = None
        self.create_subscription(
            VehicleLocalPosition,
            "/fmu/out/vehicle_local_position",
            self._on_position,
            PX4_OUT_QOS,
        )
        self.create_subscription(
            VehicleStatus,
            "/fmu/out/vehicle_status",
            self._on_status,
            PX4_OUT_QOS,
        )

    def _on_position(self, msg: VehicleLocalPosition) -> None:
        if math.isfinite(msg.x) and math.isfinite(msg.y) and math.isfinite(msg.z):
            self.position = msg

    def _on_status(self, msg: VehicleStatus) -> None:
        self.status = msg


def main() -> None:
    rclpy.init()
    node = Px4PosePrinter()
    try:
        deadline = node.get_clock().now().nanoseconds + int(5.0 * 1e9)
        while rclpy.ok() and node.position is None:
            rclpy.spin_once(node, timeout_sec=0.1)
            if node.get_clock().now().nanoseconds > deadline:
                break
        pos = node.position
        if pos is None:
            raise SystemExit("No /fmu/out/vehicle_local_position received")

        # ENU values are printed in the same convention used by move_enu.
        enu_x = float(pos.y)
        enu_y = float(pos.x)
        enu_z = -float(pos.z)
        print(
            "px4_ned: "
            f"[{pos.x:.6f}, {pos.y:.6f}, {pos.z:.6f}, heading={pos.heading:.6f}]"
        )
        print(f"world_enu: [{enu_x:.6f}, {enu_y:.6f}, {enu_z:.6f}]")
        print(
            "ros2_param_abs_ned:="
            f"\"[{pos.x:.6f}, {pos.y:.6f}, {pos.z:.6f}]\""
        )
        print(
            "valid: "
            f"xy={bool(pos.xy_valid)}, z={bool(pos.z_valid)}, "
            f"vxy={bool(pos.v_xy_valid)}, vz={bool(pos.v_z_valid)}, "
            f"heading_good={bool(pos.heading_good_for_control)}"
        )
        if node.status is not None:
            print(
                "status: "
                f"arming={node.status.arming_state}, nav={node.status.nav_state}, "
                f"preflight={bool(node.status.pre_flight_checks_pass)}, "
                f"failsafe={bool(node.status.failsafe)}"
            )
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
