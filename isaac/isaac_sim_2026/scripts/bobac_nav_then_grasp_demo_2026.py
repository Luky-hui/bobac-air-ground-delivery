#!/usr/bin/env python3
"""Run Bobac sorting navigation, then start the official grasp perception demo."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import rclpy
from rclpy.node import Node


DEFAULT_WAYPOINTS = (
    "[1.760000, -1.100000, 1.661177, "
    "3.436775, -0.727569, 1.661177, "
    "3.970959, -0.368233, 1.577041]"
)


class TopicWaiter(Node):
    def __init__(self) -> None:
        super().__init__("bobac_after_nav_topic_waiter")

    def wait_for_topics(self, topics: list[str], timeout_sec: float) -> bool:
        deadline = time.time() + timeout_sec
        missing = list(topics)
        while rclpy.ok() and time.time() < deadline:
            available = {name for name, _types in self.get_topic_names_and_types()}
            missing = [topic for topic in topics if topic not in available]
            if not missing:
                return True
            self.get_logger().info(f"waiting for topics: {', '.join(missing)}")
            rclpy.spin_once(self, timeout_sec=1.0)
        self.get_logger().error(f"missing topics after {timeout_sec:.1f}s: {', '.join(missing)}")
        return False


def run_checked(cmd: list[str], env: dict[str, str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    completed = subprocess.run(cmd, env=env, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-nav", action="store_true", help="Only run post-nav perception.")
    parser.add_argument("--keep-running", action="store_true", help="Keep the perception launch alive.")
    parser.add_argument("--camera-timeout", type=float, default=45.0)
    parser.add_argument("--perception-launch", default="perception_pipeline_demo.launch.py")
    parser.add_argument("--debug-dir", default="/tmp/grasp_demo_debug")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = dict(os.environ)
    env.setdefault("ROS_DOMAIN_ID", "0")

    if not args.skip_nav:
        run_checked(
            [
                "ros2",
                "run",
                "isaac_sim_2026",
                "bobac_sort_odom_nav_2026.py",
                "--ros-args",
                "-p",
                "odom_topic:=/odom",
                "-p",
                "cmd_vel_topic:=/bobac_kinematic_cmd_vel",
                "-p",
                "world_frame_cmd:=true",
                "-p",
                "forward_only_mode:=false",
                "-p",
                f"waypoints:={DEFAULT_WAYPOINTS}",
                "-p",
                "max_linear:=0.60",
                "-p",
                "min_linear:=0.15",
                "-p",
                "fine_min_linear:=0.05",
                "-p",
                "max_angular:=0.80",
                "-p",
                "min_angular:=0.18",
                "-p",
                "prestop_xy_tolerance:=0.14",
                "-p",
                "final_xy_tolerance:=0.055",
                "-p",
                "yaw_tolerance:=0.08",
                "-p",
                "goal_timeout_sec:=180.0",
            ],
            env,
        )

    rclpy.init()
    waiter = TopicWaiter()
    try:
        topics_ready = waiter.wait_for_topics(
            ["/arm_camera/rgb", "/arm_camera/depth", "/arm_camera/camera_info"],
            args.camera_timeout,
        )
    finally:
        waiter.destroy_node()
        rclpy.shutdown()
    if not topics_ready:
        raise SystemExit(2)

    debug_dir = Path(args.debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    for stale_name in (
        "last_detection.txt",
        "last_detection_annotated.png",
        "last_detection_mask.png",
    ):
        stale_path = debug_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()
    cmd = [
        "ros2",
        "launch",
        "grasp_demo_pkg",
        args.perception_launch,
        "debug_image_dir:=" + args.debug_dir,
    ]
    print("+ " + " ".join(cmd), flush=True)
    process = subprocess.Popen(cmd, env=env)
    if args.keep_running:
        try:
            process.wait()
        except KeyboardInterrupt:
            process.send_signal(signal.SIGINT)
            process.wait(timeout=10)
        raise SystemExit(process.returncode or 0)

    deadline = time.time() + 35.0
    detection_file = debug_dir / "last_detection.txt"
    while time.time() < deadline and process.poll() is None:
        if detection_file.exists():
            print(detection_file.read_text(encoding="utf-8"), flush=True)
            process.send_signal(signal.SIGINT)
            process.wait(timeout=10)
            return
        time.sleep(0.5)

    if process.poll() is None:
        process.send_signal(signal.SIGINT)
        process.wait(timeout=10)
    print(f"detection debug file not produced: {detection_file}", file=sys.stderr)
    raise SystemExit(3)


if __name__ == "__main__":
    main()
