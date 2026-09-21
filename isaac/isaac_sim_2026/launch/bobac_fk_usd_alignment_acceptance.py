#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bobac cuRobo FK vs USD measured TCP acceptance test.

For each safe joint sample this script:
- publishes joint_1..joint_6 slowly to /joint_command
- waits until /bobac_joint_states is close to the requested q
- asks Isaac Sim /material_task to run "fk_check <target>"
- records the USD measured TCP, cuRobo FK TCP, and the position error

The 2 cm acceptance criterion is intentionally checked on Isaac's measured
Bobac gripper TCP in base_link_arm coordinates, because this is the value that
must agree with cuRobo before any real grasp execution can be trusted.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from typing import Dict, List, Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String


JOINTS = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]

DEFAULT_SAMPLES = {
    "zero": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "retract": [0.0, 0.58, -1.67, -0.50, 1.51, 0.0],
    "table_mild": [0.35, -0.40, -0.85, -0.20, 1.25, 0.10],
    "table_grasp_a": [1.00, -0.80, -1.00, -0.40, 1.40, -0.0588],
    "table_grasp_b": [0.82, -0.68, -1.18, -0.28, 1.48, 0.18],
}

LIMITS = {
    "joint_1": (-3.11, 3.11),
    "joint_2": (-3.11, 2.36),
    "joint_3": (-2.79, 2.53),
    "joint_4": (-3.11, 3.11),
    "joint_5": (-3.11, 3.11),
    "joint_6": (-6.28, 6.28),
}


class BobacFKAcceptance(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("bobac_fk_usd_alignment_acceptance")
        self.args = args
        self.current: Dict[str, float] = {}
        self.report: Optional[dict] = None
        self.report_stamp = 0.0
        self.status: Optional[dict] = None

        self.joint_pub = self.create_publisher(JointState, "/joint_command", 10)
        self.command_pub = self.create_publisher(String, "/material_task/command", 10)
        self.create_subscription(JointState, "/bobac_joint_states", self._on_joint_state, 10)
        self.create_subscription(String, "/material_task/report", self._on_report, 10)
        self.create_subscription(String, "/material_task/status", self._on_status, 10)

    @staticmethod
    def _parse_json(msg: String) -> dict:
        try:
            return json.loads(msg.data)
        except json.JSONDecodeError:
            return {"raw": msg.data}

    def _on_joint_state(self, msg: JointState) -> None:
        for name, value in zip(msg.name, msg.position):
            if name in JOINTS:
                self.current[name] = float(value)

    def _on_report(self, msg: String) -> None:
        payload = self._parse_json(msg)
        if payload.get("task") == "bobac_curobo_usd_tcp_alignment":
            self.report = payload
            self.report_stamp = time.monotonic()
            self.get_logger().info(
                "fk report: error=%s pass_2cm=%s q=%s"
                % (payload.get("tcp_error_m"), payload.get("pass_2cm"), payload.get("q"))
            )

    def _on_status(self, msg: String) -> None:
        self.status = self._parse_json(msg)

    def spin_until(self, predicate, timeout: float, label: str) -> None:
        deadline = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if predicate():
                return
        raise TimeoutError(f"timeout waiting for {label}")

    def wait_ready(self) -> None:
        self.spin_until(
            lambda: all(j in self.current for j in JOINTS)
            and self.command_pub.get_subscription_count() > 0,
            self.args.timeout,
            "/bobac_joint_states and /material_task/command subscriber",
        )

    def current_q(self) -> List[float]:
        if all(j in self.current for j in JOINTS):
            return [float(self.current[j]) for j in JOINTS]
        return [0.0] * len(JOINTS)

    def validate_q(self, q: List[float], label: str) -> None:
        if len(q) != len(JOINTS):
            raise ValueError(f"{label}: expected {len(JOINTS)} joints, got {len(q)}")
        for name, value in zip(JOINTS, q):
            lo, hi = LIMITS[name]
            if not (lo <= float(value) <= hi):
                raise ValueError(f"{label}: {name}={value:.4f} outside [{lo}, {hi}]")

    def publish_q(self, q: List[float]) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(JOINTS)
        msg.position = [float(v) for v in q]
        self.joint_pub.publish(msg)

    def move_to(self, q_target: List[float], label: str) -> None:
        self.validate_q(q_target, label)
        q_start = self.current_q()
        max_delta = max(abs(float(a) - float(b)) for a, b in zip(q_start, q_target))
        steps = max(1, int(math.ceil(max_delta / max(1.0e-4, self.args.max_joint_step))))
        self.get_logger().info(f"{label}: move steps={steps}, max_delta={max_delta:.4f}")
        for i in range(steps + 1):
            alpha = i / steps
            q = [(1.0 - alpha) * a + alpha * b for a, b in zip(q_start, q_target)]
            self.publish_q(q)
            end = time.monotonic() + self.args.dt
            while rclpy.ok() and time.monotonic() < end:
                rclpy.spin_once(self, timeout_sec=0.01)

        deadline = time.monotonic() + self.args.settle
        while rclpy.ok() and time.monotonic() < deadline:
            self.publish_q(q_target)
            rclpy.spin_once(self, timeout_sec=0.05)

        def close_enough() -> bool:
            now = self.current_q()
            return max(abs(a - b) for a, b in zip(now, q_target)) <= self.args.joint_tolerance

        self.spin_until(close_enough, self.args.timeout, f"{label} joint convergence")

    def run_fk_check(self, label: str) -> dict:
        previous_stamp = self.report_stamp
        msg = String()
        msg.data = f"fk_check {self.args.target}"
        self.command_pub.publish(msg)
        self.spin_until(
            lambda: self.report is not None and self.report_stamp > previous_stamp,
            self.args.timeout,
            f"{label} fk_check report",
        )
        assert self.report is not None
        return dict(self.report)


def write_outputs(args: argparse.Namespace, rows: List[dict], reports: List[dict]) -> None:
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "bobac_fk_usd_alignment_acceptance.csv")
    txt_path = os.path.join(args.out_dir, "bobac_fk_usd_alignment_acceptance.txt")
    json_path = os.path.join(args.out_dir, "bobac_fk_usd_alignment_acceptance.json")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "label",
                "requested_q",
                "reported_q",
                "usd_tcp_base",
                "curobo_tcp_base",
                "tcp_error_m",
                "pass_2cm",
                "finger_distance_m",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    max_error = max((float(r["tcp_error_m"]) for r in rows), default=float("inf"))
    passed = bool(rows) and all(bool(r["pass_2cm"]) for r in rows)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Bobac FK acceptance: cuRobo TCP vs USD measured TCP\n")
        f.write(f"target: {args.target}\n")
        f.write(f"acceptance: max tcp_error_m < {args.acceptance_m:.3f} m\n")
        f.write(f"result: {'PASS' if passed else 'FAIL'}\n")
        f.write(f"max_error_m: {max_error:.6f}\n\n")
        for row in rows:
            f.write(
                f"{row['label']}: error={float(row['tcp_error_m']):.6f} m, "
                f"pass_2cm={row['pass_2cm']}, "
                f"usd={row['usd_tcp_base']}, curobo={row['curobo_tcp_base']}, "
                f"q={row['reported_q']}\n"
            )

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "target": args.target,
                "acceptance_m": args.acceptance_m,
                "passed": passed,
                "max_error_m": max_error,
                "rows": rows,
                "reports": reports,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"wrote: {txt_path}")
    print(f"wrote: {csv_path}")
    print(f"wrote: {json_path}")
    print(open(txt_path, encoding="utf-8").read())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="white_pencil")
    parser.add_argument("--out-dir", default="/home/u-zhuang/ros2_ws/src/isaac/isaac_sim_2026/alignment_results")
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--max-joint-step", type=float, default=0.010)
    parser.add_argument("--dt", type=float, default=0.04)
    parser.add_argument("--settle", type=float, default=1.0)
    parser.add_argument("--joint-tolerance", type=float, default=0.025)
    parser.add_argument("--acceptance-m", type=float, default=0.020)
    parser.add_argument(
        "--samples-json",
        default="",
        help="Optional JSON object mapping labels to six joint values.",
    )
    args, ros_args = parser.parse_known_args()

    samples = DEFAULT_SAMPLES
    if args.samples_json:
        samples = {str(k): [float(x) for x in v] for k, v in json.loads(args.samples_json).items()}

    rclpy.init(args=ros_args)
    node = BobacFKAcceptance(args)
    rows: List[dict] = []
    reports: List[dict] = []
    try:
        node.wait_ready()
        for label, q in samples.items():
            node.move_to([float(v) for v in q], label)
            report = node.run_fk_check(label)
            reports.append(report)
            measured = report.get("measured_tcp", {})
            curobo = report.get("curobo_fk", {})
            err = report.get("tcp_error_m")
            row = {
                "label": label,
                "requested_q": json.dumps([round(float(v), 6) for v in q]),
                "reported_q": json.dumps(report.get("q", [])),
                "usd_tcp_base": json.dumps(measured.get("tcp_local_arm")),
                "curobo_tcp_base": json.dumps(curobo.get("tcp_base")),
                "tcp_error_m": float(err) if err is not None else float("inf"),
                "pass_2cm": bool(err is not None and float(err) < args.acceptance_m),
                "finger_distance_m": measured.get("finger_distance_m"),
            }
            rows.append(row)
        write_outputs(args, rows, reports)
        return 0 if rows and all(r["pass_2cm"] for r in rows) else 2
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
