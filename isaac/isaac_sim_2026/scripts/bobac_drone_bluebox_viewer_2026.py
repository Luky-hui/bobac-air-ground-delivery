#!/usr/bin/env python3
"""Down-camera blue target viewer for the Bobac drone stage.

This node mirrors the blue-box detection evidence used by the mission node,
but it is display-only and does not command the drone.
"""

from __future__ import annotations

import math
import re
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String


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


def parse_visual_status(text: str) -> dict[str, float | bool]:
    out: dict[str, float | bool] = {"detected": "detected=true" in text}
    for key in ("u", "v", "area", "fill", "confidence"):
        match = re.search(rf"{key}=([-+0-9.]+)", text)
        if match:
            out[key] = float(match.group(1))
    return out


class BobacDroneBlueboxViewer(Node):
    def __init__(self) -> None:
        super().__init__("bobac_drone_bluebox_viewer_2026")
        self.declare_parameter("image_topic", "/drone/down_camera/rgb")
        self.declare_parameter("visual_target_topic", "/drone/visual_target")
        self.declare_parameter("save_path", "/tmp/bobac_drone_bluebox_view.png")
        self.declare_parameter("window_name", "Bobac drone blue-box alignment")
        self.declare_parameter("show_window", True)
        self.declare_parameter("save_period_sec", 0.4)
        self.declare_parameter("blue_h_min", 96)
        self.declare_parameter("blue_h_max", 125)
        self.declare_parameter("blue_s_min", 10)
        self.declare_parameter("blue_v_min", 120)
        self.declare_parameter("min_blue_area", 40.0)
        self.declare_parameter("max_blue_area_ratio", 0.35)
        self.declare_parameter("min_fill_ratio", 0.015)
        self.declare_parameter("expected_area_ratio", 0.012)
        self.declare_parameter("drop_visual_tolerance", 0.080)
        self.declare_parameter("min_visual_confidence", 0.60)

        self.image: Optional[np.ndarray] = None
        self.status_text = ""
        self.status_values: dict[str, float | bool] = {}
        self.display_enabled = bool(self.get_parameter("show_window").value)
        self.last_save = 0.0
        self.last_log = 0.0

        self.create_subscription(Image, str(self.get_parameter("image_topic").value), self.on_image, 10)
        self.create_subscription(String, str(self.get_parameter("visual_target_topic").value), self.on_status, 10)
        self.create_timer(0.1, self.render)
        self.get_logger().info("drone blue-box viewer started; read-only subscriptions active")

    def on_image(self, msg: Image) -> None:
        self.image = image_to_bgr(msg)

    def on_status(self, msg: String) -> None:
        self.status_text = msg.data
        self.status_values = parse_visual_status(msg.data)

    def detect(self, image: np.ndarray) -> tuple[Optional[dict[str, float]], np.ndarray]:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower = np.array(
            [
                int(self.get_parameter("blue_h_min").value),
                int(self.get_parameter("blue_s_min").value),
                int(self.get_parameter("blue_v_min").value),
            ],
            dtype=np.uint8,
        )
        upper = np.array([int(self.get_parameter("blue_h_max").value), 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        valid_mask = np.zeros(mask.shape, dtype=np.uint8)
        min_blue_area = float(self.get_parameter("min_blue_area").value)
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area >= min_blue_area * 0.15:
                cv2.drawContours(valid_mask, [contour], -1, 255, thickness=cv2.FILLED)

        pixels = cv2.findNonZero(valid_mask)
        if pixels is None:
            return None, valid_mask

        height, width = image.shape[:2]
        image_area = float(width * height)
        x, y, w, h = cv2.boundingRect(pixels)
        area = float(cv2.countNonZero(valid_mask))
        max_area = image_area * float(self.get_parameter("max_blue_area_ratio").value)
        if area < min_blue_area or area > max_area or w <= 2 or h <= 2:
            return None, valid_mask
        fill = area / max(float(w * h), 1.0)
        if fill < float(self.get_parameter("min_fill_ratio").value):
            return None, valid_mask

        moments = cv2.moments(valid_mask, binaryImage=True)
        if abs(moments["m00"]) > 1.0e-6:
            cx = float(moments["m10"] / moments["m00"])
            cy = float(moments["m01"] / moments["m00"])
        else:
            cx = float(x + w * 0.5)
            cy = float(y + h * 0.5)
        u = (cx - width * 0.5) / max(width * 0.5, 1.0)
        v = (cy - height * 0.5) / max(height * 0.5, 1.0)
        pixel_err = math.hypot(u, v)
        area_ratio = area / max(image_area, 1.0)
        confidence = max(
            0.05,
            min(
                0.99,
                0.35
                + min(area_ratio / max(float(self.get_parameter("expected_area_ratio").value), 1.0e-6), 1.0)
                * 0.35
                + max(0.0, 1.0 - pixel_err) * 0.30,
            ),
        )
        return {
            "x": float(x),
            "y": float(y),
            "w": float(w),
            "h": float(h),
            "cx": cx,
            "cy": cy,
            "u": u,
            "v": v,
            "area": area,
            "fill": fill,
            "confidence": confidence,
            "pixel_err": pixel_err,
        }, valid_mask

    def render(self) -> None:
        if self.image is None:
            now = time.monotonic()
            if now - self.last_log > 2.0:
                self.get_logger().info("waiting for down camera image")
                self.last_log = now
            return

        view = self.image.copy()
        detection, mask = self.detect(view)
        if cv2.countNonZero(mask) > 0:
            blue = np.zeros_like(view)
            blue[:, :] = (255, 0, 0)
            view = np.where(mask[..., None] > 0, (view * 0.70 + blue * 0.30).astype(np.uint8), view)

        height, width = view.shape[:2]
        center = (width // 2, height // 2)
        cv2.drawMarker(view, center, (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
        cv2.circle(view, center, int(min(width, height) * float(self.get_parameter("drop_visual_tolerance").value) * 0.5), (0, 0, 255), 1)

        lines: list[str]
        if detection is not None:
            x, y, w, h = [int(round(detection[k])) for k in ("x", "y", "w", "h")]
            cx, cy = int(round(detection["cx"])), int(round(detection["cy"]))
            ok = (
                detection["pixel_err"] <= float(self.get_parameter("drop_visual_tolerance").value)
                and detection["confidence"] >= float(self.get_parameter("min_visual_confidence").value)
            )
            color = (0, 255, 0) if ok else (0, 255, 255)
            cv2.rectangle(view, (x, y), (x + w, y + h), color, 2)
            cv2.drawMarker(view, (cx, cy), color, cv2.MARKER_TILTED_CROSS, 16, 2)
            cv2.arrowedLine(view, center, (cx, cy), color, 2, tipLength=0.15)
            lines = [
                "detected=true" + (" aligned=true" if ok else " aligned=false"),
                f"u={detection['u']:+.4f} v={detection['v']:+.4f} pixel_err={detection['pixel_err']:.4f}",
                f"confidence={detection['confidence']:.3f} area={detection['area']:.1f} fill={detection['fill']:.3f}",
            ]
        else:
            lines = ["detected=false", "waiting for blue target contour"]

        if self.status_text:
            status = self.status_text[:110]
            lines.append(f"mission: {status}")

        y0 = 22
        for line in lines:
            cv2.putText(view, line, (8, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(view, line, (8, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
            y0 += 21

        now = time.monotonic()
        if now - self.last_save >= float(self.get_parameter("save_period_sec").value):
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
    node = BobacDroneBlueboxViewer()
    try:
        rclpy.spin(node)
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
