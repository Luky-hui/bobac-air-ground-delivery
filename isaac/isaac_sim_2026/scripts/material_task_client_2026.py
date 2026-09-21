#!/usr/bin/env python3
"""Run Isaac-side material detection and pick-place task."""

from __future__ import annotations

import argparse
import json
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class MaterialTaskClient(Node):
    def __init__(self, target: str, detect_only: bool, real_grasp: bool, timeout_sec: float):
        super().__init__("material_task_client_2026")
        self.target = target
        self.detect_only = detect_only
        self.real_grasp = real_grasp
        self.timeout_sec = timeout_sec
        self.command_pub = self.create_publisher(String, "/material_task/command", 10)
        self.create_subscription(String, "/material_task/detections", self._on_detections, 10)
        self.create_subscription(String, "/material_task/status", self._on_status, 10)
        self.create_subscription(String, "/material_task/report", self._on_report, 10)
        self.detections = None
        self.report = None
        self.status = None
        self.started_at = time.monotonic()
        self.command_sent = False
        self.detect_command_sent = False
        self.grasp_command_sent = False

    def _parse_json(self, msg: String):
        try:
            return json.loads(msg.data)
        except json.JSONDecodeError:
            return {"raw": msg.data}

    def _on_detections(self, msg: String):
        self.detections = self._parse_json(msg)
        count = self.detections.get("detected_count", 0)
        found = self.detections.get("all_targets_found", False)
        self.get_logger().info(f"detections: count={count}, all_targets_found={found}")

    def _on_status(self, msg: String):
        self.status = self._parse_json(msg)
        self.get_logger().info(f"status: {self.status}")

    def _on_report(self, msg: String):
        self.report = self._parse_json(msg)
        self.get_logger().info(f"report: {self.report}")

    def _publish_command(self, text: str):
        msg = String()
        msg.data = text
        self.command_pub.publish(msg)

    def tick(self):
        if self.command_pub.get_subscription_count() > 0 and not self.command_sent:
            if self.detect_only:
                command = "detect"
            elif self.real_grasp:
                command = "detect"
                self.detect_command_sent = True
            else:
                command = f"run {self.target}"
            self.get_logger().info(f"send command: {command}")
            self._publish_command(command)
            self.command_sent = True
        if (
            self.real_grasp
            and self.detect_command_sent
            and not self.grasp_command_sent
            and self.detections is not None
        ):
            if not self.detections.get("all_targets_found", False):
                raise RuntimeError(f"material detection failed before grasp: {self.detections}")
            command = f"real_grasp {self.target}"
            self.get_logger().info(f"send command after detection: {command}")
            self._publish_command(command)
            self.grasp_command_sent = True
        if time.monotonic() - self.started_at > self.timeout_sec:
            raise TimeoutError(f"material task timed out after {self.timeout_sec:.1f}s")
        if self.detect_only:
            if self.detections is not None:
                return True
        elif self.report is not None:
            if not self.real_grasp:
                return True
            if "ok" in self.report or self.report.get("status") in {"completed", "failed"}:
                return True
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="red_pencil", help="red_pencil or white_pencil")
    parser.add_argument("--detect-only", action="store_true")
    parser.add_argument("--real-grasp", action="store_true", help="use physics contact grasp; no object bind/teleport")
    parser.add_argument("--timeout-sec", type=float, default=30.0)
    args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    node = MaterialTaskClient(args.target, args.detect_only, args.real_grasp, args.timeout_sec)
    exit_code = 1
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            if node.tick():
                if args.detect_only:
                    ok = bool(node.detections and node.detections.get("all_targets_found"))
                    print(json.dumps(node.detections, ensure_ascii=False, indent=2))
                else:
                    ok = bool(node.report and node.report.get("ok"))
                    print(json.dumps(node.report, ensure_ascii=False, indent=2))
                exit_code = 0 if ok else 2
                break
    except Exception as exc:
        node.get_logger().error(str(exc))
        exit_code = 1
    finally:
        node.destroy_node()
        rclpy.shutdown()
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
