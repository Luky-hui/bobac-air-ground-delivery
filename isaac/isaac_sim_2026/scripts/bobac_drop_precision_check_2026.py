#!/usr/bin/env python3
"""Measure final payload drop precision without moving any scene prim."""

from __future__ import annotations

import argparse
import json
import math
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class BobacDropPrecisionCheck(Node):
    def __init__(self, timeout_sec: float) -> None:
        super().__init__("bobac_drop_precision_check_2026")
        self.timeout_sec = float(timeout_sec)
        self.command_pub = self.create_publisher(String, "/material_task/command", 10)
        self.create_subscription(String, "/material_task/report", self._on_report, 10)
        self.report = None
        self.started_at = time.monotonic()
        self.command_sent = False

    def _on_report(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        if payload.get("task") == "scene_points":
            self.report = payload

    def _publish_scene_points(self) -> None:
        msg = String()
        msg.data = "scene_points"
        self.command_pub.publish(msg)
        self.command_sent = True
        self.get_logger().info("send command: scene_points")

    def tick(self) -> bool:
        if not self.command_sent and self.command_pub.get_subscription_count() > 0:
            self._publish_scene_points()
        if self.report is not None:
            return True
        if time.monotonic() - self.started_at > self.timeout_sec:
            raise TimeoutError(f"drop precision check timed out after {self.timeout_sec:.1f}s")
        return False


def _find_point(report: dict, target_id: str):
    for point in report.get("points", []):
        if point.get("id") == target_id:
            return point.get("world")
    return None


def _score_band(error_xy: float) -> str:
    if error_xy <= 0.2:
        return "0.2m"
    if error_xy <= 0.4:
        return "0.4m"
    if error_xy <= 0.6:
        return "0.6m"
    return "miss"


def _print_result(report: dict) -> int:
    white = report.get("white_pencil_world") or _find_point(report, "white_pencil")
    target = report.get("drop_target_world")
    error_xy = report.get("drop_error_xy")
    band = report.get("drop_score_band")

    if white is None or target is None:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print("ERROR: missing white_pencil_world or drop_target_world in scene_points report")
        return 2

    if error_xy is None:
        error_xy = math.hypot(float(white[0]) - float(target[0]), float(white[1]) - float(target[1]))
    else:
        error_xy = float(error_xy)
    if not band:
        band = _score_band(error_xy)

    result = {
        "white_pencil_world": white,
        "drop_target_prim": report.get("drop_target_prim"),
        "drop_target_world": target,
        "drop_error_xy": round(error_xy, 6),
        "drop_score_band": band,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if error_xy <= 0.2:
        print("PASS: payload center is within 0.2m scoring radius")
        return 0
    if error_xy <= 0.6:
        print("WARN: payload is inside a lower scoring radius; tighten visual alignment or lower drop height")
        return 3
    print("FAIL: payload is outside 0.6m scoring radius")
    return 4


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout-sec", type=float, default=8.0)
    args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    node = BobacDropPrecisionCheck(args.timeout_sec)
    exit_code = 1
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            if node.tick():
                exit_code = _print_result(node.report)
                break
    except Exception as exc:
        node.get_logger().error(str(exc))
        exit_code = 1
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
