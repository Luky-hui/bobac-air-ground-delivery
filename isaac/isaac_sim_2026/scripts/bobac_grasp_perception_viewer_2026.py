#!/usr/bin/env python3
"""Viewer for the official Bobac grasp perception chain.

This node is intentionally read-only: it subscribes to the arm camera,
official YOLOE debug image, bbox, depth point, and base-frame point, then
draws a compact evidence overlay for demos.
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped, Vector3Stamped
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Float32MultiArray, String


def image_to_bgr(msg: Image) -> np.ndarray:
    encoding = msg.encoding.lower()
    dtype = np.uint8 if encoding not in {"16uc1", "mono16"} else np.uint16
    channels_by_encoding = {
        "rgb8": 3,
        "bgr8": 3,
        "rgba8": 4,
        "bgra8": 4,
        "mono8": 1,
        "8uc1": 1,
    }
    channels = channels_by_encoding.get(encoding)
    if channels is None:
        channels = max(1, int(msg.step / max(1, msg.width)))
        channels = 4 if channels >= 4 else 3 if channels >= 3 else 1
    arr = np.frombuffer(msg.data, dtype=dtype)
    if channels == 1:
        arr = arr.reshape((msg.height, msg.width))
        if arr.dtype != np.uint8:
            arr = cv2.convertScaleAbs(arr)
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    arr = arr.reshape((msg.height, msg.width, channels))
    if channels == 4:
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR if encoding == "rgba8" else cv2.COLOR_BGRA2BGR)
    if encoding == "rgb8":
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return arr.copy()


def fmt_point(point: Optional[PointStamped]) -> str:
    if point is None:
        return "none"
    return f"{point.point.x:+.3f},{point.point.y:+.3f},{point.point.z:+.3f} [{point.header.frame_id}]"


def fmt_vec(vec: Optional[Vector3Stamped]) -> str:
    if vec is None:
        return "none"
    norm = math.sqrt(vec.vector.x**2 + vec.vector.y**2 + vec.vector.z**2)
    return f"{vec.vector.x:+.3f},{vec.vector.y:+.3f},{vec.vector.z:+.3f} |n|={norm:.2f}"


class BobacGraspPerceptionViewer(Node):
    def __init__(self) -> None:
        super().__init__("bobac_grasp_perception_viewer_2026")
        self.declare_parameter("raw_image_topic", "/arm_camera/rgb")
        self.declare_parameter("debug_image_topic", "/demo_grasp/debug_image")
        self.declare_parameter("bbox_topic", "/demo_grasp/bbox")
        self.declare_parameter("label_topic", "/demo_grasp/label")
        self.declare_parameter("confidence_topic", "/demo_grasp/confidence")
        self.declare_parameter("point_camera_topic", "/demo_grasp/object_point_camera")
        self.declare_parameter("point_base_topic", "/demo_grasp/object_point_base")
        self.declare_parameter("long_axis_base_topic", "/demo_grasp/long_axis_base")
        self.declare_parameter("save_path", "/tmp/bobac_grasp_perception_view.png")
        self.declare_parameter("window_name", "Bobac grasp perception")
        self.declare_parameter("show_window", True)
        self.declare_parameter("save_period_sec", 0.4)

        self.raw_image: Optional[np.ndarray] = None
        self.debug_image: Optional[np.ndarray] = None
        self.bbox: Optional[list[float]] = None
        self.label = ""
        self.confidence: Optional[float] = None
        self.point_camera: Optional[PointStamped] = None
        self.point_base: Optional[PointStamped] = None
        self.long_axis_base: Optional[Vector3Stamped] = None
        self.display_enabled = bool(self.get_parameter("show_window").value)
        self.last_save = 0.0
        self.last_log = 0.0

        self.create_subscription(Image, str(self.get_parameter("raw_image_topic").value), self.on_raw, 10)
        self.create_subscription(Image, str(self.get_parameter("debug_image_topic").value), self.on_debug, 10)
        self.create_subscription(Float32MultiArray, str(self.get_parameter("bbox_topic").value), self.on_bbox, 10)
        self.create_subscription(String, str(self.get_parameter("label_topic").value), self.on_label, 10)
        self.create_subscription(Float32, str(self.get_parameter("confidence_topic").value), self.on_confidence, 10)
        self.create_subscription(PointStamped, str(self.get_parameter("point_camera_topic").value), self.on_point_camera, 10)
        self.create_subscription(PointStamped, str(self.get_parameter("point_base_topic").value), self.on_point_base, 10)
        self.create_subscription(Vector3Stamped, str(self.get_parameter("long_axis_base_topic").value), self.on_axis, 10)
        self.create_timer(0.1, self.render)
        self.get_logger().info("grasp perception viewer started; read-only subscriptions active")

    def on_raw(self, msg: Image) -> None:
        self.raw_image = image_to_bgr(msg)

    def on_debug(self, msg: Image) -> None:
        self.debug_image = image_to_bgr(msg)

    def on_bbox(self, msg: Float32MultiArray) -> None:
        self.bbox = [float(v) for v in msg.data]

    def on_label(self, msg: String) -> None:
        self.label = msg.data

    def on_confidence(self, msg: Float32) -> None:
        self.confidence = float(msg.data)

    def on_point_camera(self, msg: PointStamped) -> None:
        self.point_camera = msg

    def on_point_base(self, msg: PointStamped) -> None:
        self.point_base = msg

    def on_axis(self, msg: Vector3Stamped) -> None:
        self.long_axis_base = msg

    def render(self) -> None:
        frame = self.debug_image if self.debug_image is not None else self.raw_image
        if frame is None:
            now = time.monotonic()
            if now - self.last_log > 2.0:
                self.get_logger().info("waiting for arm camera/debug image")
                self.last_log = now
            return

        view = frame.copy()
        if self.debug_image is None and self.bbox is not None and len(self.bbox) >= 4:
            u, v, w, h = self.bbox[:4]
            x1, y1 = int(round(u - w * 0.5)), int(round(v - h * 0.5))
            x2, y2 = int(round(u + w * 0.5)), int(round(v + h * 0.5))
            cv2.rectangle(view, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.drawMarker(view, (int(round(u)), int(round(v))), (0, 0, 255), cv2.MARKER_CROSS, 12, 2)

        lines = [
            f"label={self.label or 'none'} conf={self.confidence:.3f}" if self.confidence is not None else f"label={self.label or 'none'} conf=none",
            f"bbox uvwh_area_conf={','.join(f'{v:.1f}' for v in self.bbox)}" if self.bbox else "bbox=none",
            f"point_camera={fmt_point(self.point_camera)}",
            f"point_base={fmt_point(self.point_base)}",
            f"long_axis_base={fmt_vec(self.long_axis_base)}",
        ]
        y = 22
        for line in lines:
            cv2.putText(view, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(view, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            y += 20

        now = time.monotonic()
        save_period = float(self.get_parameter("save_period_sec").value)
        if now - self.last_save >= save_period:
            path = Path(str(self.get_parameter("save_path").value)).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), view)
            self.last_save = now

        if self.display_enabled:
            try:
                cv2.imshow(str(self.get_parameter("window_name").value), view)
                cv2.waitKey(1)
            except cv2.error as exc:
                self.display_enabled = False
                self.get_logger().warning(f"OpenCV window disabled: {exc}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BobacGraspPerceptionViewer()
    try:
        rclpy.spin(node)
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
