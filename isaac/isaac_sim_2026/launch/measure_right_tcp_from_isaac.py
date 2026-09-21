#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Measure a right TCP candidate from Isaac Sim TF.

The TCP position is the midpoint between gripper01_left1 and gripper01_right1,
expressed in link7_R. The orientation is copied from gripper01_base relative to
link7_R, which matches both finger endpoint frames in the current Isaac scene.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from typing import Tuple

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from tf2_ros import Buffer, TransformException, TransformListener


def tf_to_pos_quat_xyzw(tf) -> Tuple[np.ndarray, np.ndarray]:
    pos = np.array(
        [tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z],
        dtype=np.float64,
    )
    quat = np.array(
        [
            tf.transform.rotation.x,
            tf.transform.rotation.y,
            tf.transform.rotation.z,
            tf.transform.rotation.w,
        ],
        dtype=np.float64,
    )
    return pos, quat


class TcpMeasure(Node):
    def __init__(self, parent: str, left: str, right: str, orient_frame: str):
        super().__init__("measure_right_tcp_from_isaac")
        self.parent = parent
        self.left = left
        self.right = right
        self.orient_frame = orient_frame
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)

    def lookup(self, child: str):
        return self.tf_buffer.lookup_transform(
            self.parent,
            child,
            rclpy.time.Time(),
            timeout=Duration(seconds=0.2),
        )

    def wait_ready(self, timeout_s: float) -> None:
        deadline = time.time() + timeout_s
        missing = [self.left, self.right, self.orient_frame]
        while time.time() < deadline and rclpy.ok():
            missing = []
            for child in [self.left, self.right, self.orient_frame]:
                try:
                    self.lookup(child)
                except TransformException:
                    missing.append(child)
            if not missing:
                return
            time.sleep(0.05)
        raise RuntimeError(f"Timed out waiting for {self.parent} -> {missing}")

    def close(self) -> None:
        if hasattr(self.tf_listener, "executor"):
            self.tf_listener.executor.shutdown()
        if hasattr(self.tf_listener, "dedicated_listener_thread"):
            self.tf_listener.dedicated_listener_thread.join(timeout=1.0)
        self.tf_listener.unregister()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", default="link7_R")
    parser.add_argument("--left", default="gripper01_left1")
    parser.add_argument("--right", default="gripper01_right1")
    parser.add_argument("--orient-frame", default="gripper01_base")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--out-dir", default="/home/u-zhuang/ros2_ws/src/isaac/isaac_sim_2026/alignment_results")
    args = parser.parse_args()

    rclpy.init()
    node = TcpMeasure(args.parent, args.left, args.right, args.orient_frame)
    try:
        node.wait_ready(args.timeout)
        left_pos, left_quat = tf_to_pos_quat_xyzw(node.lookup(args.left))
        right_pos, right_quat = tf_to_pos_quat_xyzw(node.lookup(args.right))
        orient_pos, orient_quat = tf_to_pos_quat_xyzw(node.lookup(args.orient_frame))
    finally:
        node.close()
        node.destroy_node()
        rclpy.shutdown()

    tcp_pos = (left_pos + right_pos) * 0.5
    rpy = Rotation.from_quat(orient_quat).as_euler("xyz", degrees=False)
    rpy_deg = np.degrees(rpy)
    finger_distance = float(np.linalg.norm(left_pos - right_pos))

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "right_tcp_measurement.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("right_tcp measurement from Isaac Sim TF\n")
        f.write(f"parent: {args.parent}\n")
        f.write(f"left_finger_frame: {args.left}\n")
        f.write(f"right_finger_frame: {args.right}\n")
        f.write(f"orientation_source_frame: {args.orient_frame}\n\n")
        f.write(f"left_xyz:  {left_pos.tolist()}\n")
        f.write(f"right_xyz: {right_pos.tolist()}\n")
        f.write(f"finger_distance_m: {finger_distance:.9f}\n")
        f.write(f"right_tcp_xyz: {tcp_pos.tolist()}\n")
        f.write(f"right_tcp_rpy_rad: {rpy.tolist()}\n")
        f.write(f"right_tcp_rpy_deg: {rpy_deg.tolist()}\n")
        f.write(f"gripper_base_xyz_for_reference: {orient_pos.tolist()}\n")
        f.write(f"gripper_base_quat_xyzw: {orient_quat.tolist()}\n\n")
        f.write("URDF snippet:\n")
        f.write('  <link name="right_tcp"/>\n')
        f.write('  <joint name="link7_R_to_right_tcp" type="fixed">\n')
        f.write('    <parent link="link7_R"/>\n')
        f.write('    <child link="right_tcp"/>\n')
        f.write(
            f'    <origin xyz="{tcp_pos[0]:.9g} {tcp_pos[1]:.9g} {tcp_pos[2]:.9g}" '
            f'rpy="{rpy[0]:.9g} {rpy[1]:.9g} {rpy[2]:.9g}"/>\n'
        )
        f.write("  </joint>\n")

    print(f"wrote: {out_path}")
    print(open(out_path, encoding="utf-8").read())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
