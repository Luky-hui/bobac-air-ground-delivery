#!/usr/bin/env python3
"""Downward-camera visual alignment and drop mission for the Bobac scene drone.

The coarse route still uses known map waypoints, but the final target alignment
uses the drone's downward RGB camera and OpenCV blue-rectangle detection instead
of target-prim ground truth.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String


CUBE_WORLD = (4.790769, -4.700881, 0.747550)


class BobacDroneVisualAlignDrop(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("bobac_drone_visual_align_drop_2026")
        self.args = args
        self.drone_pub = self.create_publisher(String, "/drone/command", 10)
        self.cargo_pub = self.create_publisher(String, "/cargo_bay/command", 10)
        self.visual_pub = self.create_publisher(String, "/drone/visual_target", 10)
        self.position = None
        self.latest_image = None
        self.latest_image_stamp = 0.0
        self.last_detection = None
        self.last_alignment = None
        self.drone_status = ""
        self.cargo_status = ""
        self.create_subscription(PointStamped, "/drone/world_position", self._on_position, 10)
        self.create_subscription(Image, self.args.image_topic, self._on_image, 10)
        self.create_subscription(String, "/drone/status", self._on_drone_status, 10)
        self.create_subscription(String, "/cargo_bay/status", self._on_cargo_status, 10)

    def _on_position(self, msg: PointStamped) -> None:
        self.position = msg.point

    def _on_image(self, msg: Image) -> None:
        try:
            self.latest_image = self._image_to_bgr(msg)
            self.latest_image_stamp = time.time()
        except Exception as exc:
            self.get_logger().warn(f"failed to decode down camera image: {exc}")

    def _on_drone_status(self, msg: String) -> None:
        self.drone_status = msg.data
        if self.args.verbose_status:
            self.get_logger().info(f"drone: {msg.data}")

    def _on_cargo_status(self, msg: String) -> None:
        self.cargo_status = msg.data
        self.get_logger().info(f"cargo: {msg.data}")

    def wait_for_links(self) -> None:
        deadline = time.time() + self.args.link_timeout
        while rclpy.ok() and time.time() < deadline:
            if (
                self.drone_pub.get_subscription_count()
                and self.cargo_pub.get_subscription_count()
                and self.position is not None
                and self.latest_image is not None
            ):
                return
            rclpy.spin_once(self, timeout_sec=0.05)
        raise RuntimeError(
            f"missing /drone, /cargo_bay, or {self.args.image_topic}; is Isaac Sim playing?"
        )

    def publish_string(self, pub, text: str) -> None:
        msg = String()
        msg.data = text
        pub.publish(msg)

    def cargo(self, command: str, wait_sec: float = 1.5) -> None:
        self.get_logger().info(f"cargo command: {command}")
        self.publish_string(self.cargo_pub, command)
        end = time.time() + wait_sec
        while rclpy.ok() and time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.1)

    def rise_in_place(self, name: str, height: float) -> None:
        if height <= 0.0:
            return
        if self.position is None:
            raise RuntimeError(f"{name}: no /drone/world_position")
        x = float(self.position.x)
        y = float(self.position.y)
        z = float(self.position.z) + height
        self.goto(name, x, y, z, self.args.yaw, self.args.coarse_tolerance, self.args.goto_timeout)

    def rise_to_at_current_xy(self, name: str, target_z: float) -> None:
        if self.position is None:
            raise RuntimeError(f"{name}: no /drone/world_position")
        x = float(self.position.x)
        y = float(self.position.y)
        current_z = float(self.position.z)
        z = max(current_z, float(target_z))
        if z <= current_z + 0.02:
            self.get_logger().info(
                f"{name}: already clear at pos=[{x:.3f},{y:.3f},{current_z:.3f}]"
            )
            return
        self.goto(name, x, y, z, self.args.yaw, self.args.coarse_tolerance, self.args.goto_timeout)

    def goto(self, name: str, x: float, y: float, z: float, yaw: float, tolerance: float, timeout_sec: float) -> None:
        command = f"goto {x:.6f} {y:.6f} {z:.6f} {yaw:.6f}"
        self.get_logger().info(f"{name}: {command}")
        self.publish_string(self.drone_pub, command)
        deadline = time.time() + timeout_sec
        last_log = 0.0
        while rclpy.ok() and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.position is None:
                continue
            dist = self.distance_to(x, y, z)
            now = time.time()
            if now - last_log >= 1.0:
                p = self.position
                self.get_logger().info(f"{name}: pos=[{p.x:.3f},{p.y:.3f},{p.z:.3f}], dist={dist:.3f}")
                last_log = now
            if dist <= tolerance:
                self.get_logger().info(f"{name}: reached")
                return
        raise RuntimeError(f"{name}: timeout before reaching target")

    def distance_to(self, x: float, y: float, z: float) -> float:
        if self.position is None:
            return float("inf")
        dx = x - self.position.x
        dy = y - self.position.y
        dz = z - self.position.z
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def horizontal_error(self, target_x: float, target_y: float) -> tuple[float, float, float]:
        if self.position is None:
            return float("inf"), float("inf"), float("inf")
        dx = target_x - self.position.x
        dy = target_y - self.position.y
        return dx, dy, math.hypot(dx, dy)

    @staticmethod
    def _image_to_bgr(msg: Image):
        dtype = np.uint8 if msg.encoding.lower() not in {"16uc1", "mono16"} else np.uint16
        channels_by_encoding = {
            "rgb8": 3,
            "bgr8": 3,
            "rgba8": 4,
            "bgra8": 4,
            "mono8": 1,
            "8uc1": 1,
        }
        encoding = msg.encoding.lower()
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
            if encoding == "rgba8":
                return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
            return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
        if encoding == "rgb8":
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        return arr.copy()

    def _detect_blue_target_in_image(self):
        image = self.latest_image
        if image is None:
            return None
        if time.time() - self.latest_image_stamp > self.args.image_stale_sec:
            return None

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower = np.array([self.args.blue_h_min, self.args.blue_s_min, self.args.blue_v_min], dtype=np.uint8)
        upper = np.array([self.args.blue_h_max, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        height, width = image.shape[:2]
        image_area = float(width * height)
        best = None
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.args.min_blue_area:
                continue
            if area > image_area * self.args.max_blue_area_ratio:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if w <= 2 or h <= 2:
                continue
            aspect = float(w) / float(h)
            if aspect < self.args.min_aspect or aspect > self.args.max_aspect:
                continue
            rect_area = float(w * h)
            fill = area / max(rect_area, 1.0)
            if fill < self.args.min_fill_ratio:
                continue
            score = area * min(fill, 1.0)
            if best is None or score > best["score"]:
                m = cv2.moments(contour)
                if abs(m["m00"]) > 1.0e-6:
                    cx = float(m["m10"] / m["m00"])
                    cy = float(m["m01"] / m["m00"])
                else:
                    cx = float(x + w * 0.5)
                    cy = float(y + h * 0.5)
                best = {
                    "score": score,
                    "cx": cx,
                    "cy": cy,
                    "area": area,
                    "bbox": (x, y, w, h),
                    "fill": fill,
                    "width": width,
                    "height": height,
                }
        return best

    def detect_target(self, _target_x: float, _target_y: float, _target_z: float) -> tuple[float, float, float]:
        detection = self._detect_blue_target_in_image()
        if detection is None:
            msg = String()
            msg.data = "target=blue_square source=down_camera detected=false confidence=0.000"
            self.visual_pub.publish(msg)
            self.last_detection = None
            return float("nan"), float("nan"), 0.0

        width = float(detection["width"])
        height = float(detection["height"])
        u = (float(detection["cx"]) - width * 0.5) / max(width * 0.5, 1.0)
        v = (float(detection["cy"]) - height * 0.5) / max(height * 0.5, 1.0)
        area_ratio = float(detection["area"]) / max(width * height, 1.0)
        center_error = math.hypot(u, v)
        confidence = max(
            0.05,
            min(
                0.99,
                0.35
                + min(area_ratio / max(self.args.expected_area_ratio, 1.0e-6), 1.0) * 0.35
                + max(0.0, 1.0 - center_error) * 0.30,
            ),
        )
        self.last_detection = detection
        msg = String()
        msg.data = (
            f"target=blue_square source=down_camera detected=true u={u:.4f} v={v:.4f} "
            f"area={detection['area']:.1f} fill={detection['fill']:.3f} "
            f"confidence={confidence:.3f}"
        )
        self.visual_pub.publish(msg)
        return u, v, confidence

    def _image_error_to_world_step(self, u: float, v: float, z: float, target_z: float):
        altitude = max(0.30, float(z) - float(target_z))
        gain = max(0.01, float(self.args.pixel_to_world_gain))
        # Camera image x is world/drone lateral x, image y grows down. For a
        # downward camera, a positive v means the target is lower in the image,
        # so move in negative world y with the current yaw convention.
        dx = float(u) * altitude * gain
        dy = -float(v) * altitude * gain
        step_norm = math.hypot(dx, dy)
        max_step = max(0.02, float(self.args.align_step))
        if step_norm > max_step:
            scale = max_step / step_norm
            dx *= scale
            dy *= scale
        return dx, dy

    def align_at_altitude(
        self,
        altitude_name: str,
        z: float,
        target_x: float,
        target_y: float,
        target_z: float,
        *,
        goto_tolerance: float | None = None,
        z_tolerance: float | None = None,
        visual_tolerance: float | None = None,
    ) -> None:
        yaw = self.args.yaw
        last_command_time = 0.0
        self.goto(
            altitude_name,
            target_x,
            target_y,
            z,
            yaw,
            self.args.coarse_tolerance if goto_tolerance is None else goto_tolerance,
            self.args.goto_timeout,
        )
        last_log = 0.0
        deadline = time.time() + self.args.align_timeout
        while rclpy.ok() and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            u, v, confidence = self.detect_target(target_x, target_y, target_z)
            pixel_err = math.hypot(u, v) if math.isfinite(u) and math.isfinite(v) else float("inf")
            dx, dy, map_err = self.horizontal_error(target_x, target_y)
            z_err = abs((self.position.z if self.position is not None else float("inf")) - z)
            now = time.time()
            if now - last_log >= 0.5:
                self.get_logger().info(
                    f"{altitude_name}: visual u={u:.4f}, v={v:.4f}, conf={confidence:.3f}, "
                    f"pixel_err={pixel_err:.4f}, map_xy_err={map_err:.4f}, z_err={z_err:.4f}"
                )
                last_log = now
            z_ok = z_tolerance is None or z_err <= z_tolerance
            required_visual_tolerance = (
                self.args.visual_tolerance
                if visual_tolerance is None
                else float(visual_tolerance)
            )
            if pixel_err <= required_visual_tolerance and confidence >= self.args.min_confidence and z_ok:
                self.last_alignment = {
                    "name": altitude_name,
                    "u": u,
                    "v": v,
                    "pixel_err": pixel_err,
                    "confidence": confidence,
                    "z": float(self.position.z if self.position is not None else z),
                }
                self.get_logger().info(f"{altitude_name}: aligned")
                return
            if self.position is None:
                continue
            if not math.isfinite(pixel_err):
                # If target is not visible, use the known map target as a search
                # fallback until the real camera sees the blue region.
                search_err = max(math.hypot(dx, dy), 1.0e-6)
                step = min(self.args.align_step, max(0.02, search_err))
                x = self.position.x + dx / search_err * step
                y = self.position.y + dy / search_err * step
            else:
                step_x, step_y = self._image_error_to_world_step(u, v, z, target_z)
                x = self.position.x + step_x
                y = self.position.y + step_y
            if now - last_command_time >= self.args.align_command_period:
                self.publish_string(self.drone_pub, f"goto {x:.6f} {y:.6f} {z:.6f} {yaw:.6f}")
                last_command_time = now
        raise RuntimeError(f"{altitude_name}: visual alignment timeout")

    def calibrate_target_samples(self) -> None:
        folder = Path(self.args.target_image_dir).expanduser()
        if not folder.exists():
            self.get_logger().warn(f"target image folder not found: {folder}")
            return
        images = [p for p in folder.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
        self.get_logger().info(f"target image samples: {len(images)} files in {folder}")

    def run(self) -> None:
        target_x = float(self.args.target_x)
        target_y = float(self.args.target_y)
        target_z = float(self.args.target_z)
        drop_z = float(self.args.drop_body_z)
        drop_visual_z = float(self.args.drop_visual_z)
        search_z = (
            float(self.args.search_z)
            if self.args.search_z is not None
            else max(target_z + float(self.args.search_above), drop_visual_z)
        )

        self.wait_for_links()
        self.calibrate_target_samples()
        self.cargo("left_close", wait_sec=1.0)
        safe_start_z = None
        if self.args.start_with_waypoints:
            safe_start_z = max(1.80, float(self.position.z) + self.args.initial_rise)
        elif self.args.post_load_rise > 0.0:
            safe_start_z = float(self.position.z) + self.args.post_load_rise
        if safe_start_z is not None:
            self.rise_to_at_current_xy("initial_vertical_clearance", safe_start_z)

        if self.args.start_with_waypoints:
            for name, x, y, z in [
                ("wall_left_entry", 2.60, -1.70, 1.80),
                ("wall_left_exit", 2.60, -4.70, 1.80),
            ]:
                self.goto(name, x, y, z, self.args.yaw, self.args.coarse_tolerance, self.args.goto_timeout)

        self.align_at_altitude("visual_search_8m", search_z, target_x, target_y, target_z)
        for z in self.args.descent_z:
            self.align_at_altitude(f"visual_descend_{z:.2f}", float(z), target_x, target_y, target_z)
        self.align_at_altitude(
            "drop_visual_alignment",
            drop_visual_z,
            target_x,
            target_y,
            target_z,
            goto_tolerance=min(self.args.coarse_tolerance, self.args.drop_z_tolerance),
            z_tolerance=self.args.drop_z_tolerance,
            visual_tolerance=self.args.drop_visual_tolerance,
        )
        alignment = dict(self.last_alignment or {})
        if self.position is None:
            raise RuntimeError("drop_descent: no /drone/world_position")
        drop_x = float(self.position.x)
        drop_y = float(self.position.y)
        self.goto(
            "drop_descent",
            drop_x,
            drop_y,
            drop_z,
            self.args.yaw,
            min(self.args.coarse_tolerance, self.args.drop_z_tolerance),
            self.args.goto_timeout,
        )
        settle_end = time.time() + self.args.final_settle_sec
        while rclpy.ok() and time.time() < settle_end:
            rclpy.spin_once(self, timeout_sec=0.1)

        u = float(alignment.get("u", float("nan")))
        v = float(alignment.get("v", float("nan")))
        confidence = float(alignment.get("confidence", 0.0))
        visual_err = float(alignment.get("pixel_err", float("inf")))
        _dx, _dy, map_err = self.horizontal_error(target_x, target_y)
        z_err = abs((self.position.z if self.position is not None else float("inf")) - drop_z)
        if (
            visual_err > self.args.drop_visual_tolerance
            or confidence < self.args.min_confidence
            or z_err > self.args.drop_z_tolerance
        ):
            raise RuntimeError(
                f"drop gate failed: visual_err={visual_err:.3f}, conf={confidence:.3f}, "
                f"map_xy_err={map_err:.3f}, z_err={z_err:.3f}"
            )

        self.get_logger().info(
            f"drop gate passed: visual_err={visual_err:.3f}, conf={confidence:.3f}, "
            f"map_xy_err={map_err:.3f}, z_err={z_err:.3f}"
        )
        if self.args.drop:
            self.get_logger().info("opening bottom cargo bay")
            self.cargo("bottom_open", wait_sec=self.args.bottom_open_wait)
            self.cargo("bottom_close", wait_sec=1.0)
        if self.args.land_after_drop:
            land_z = float(self.args.land_z)
            self.get_logger().info(f"landing after drop: land {land_z:.3f}")
            self.publish_string(self.drone_pub, f"land {land_z:.6f}")
            end = time.time() + self.args.land_wait_sec
            while rclpy.ok() and time.time() < end:
                rclpy.spin_once(self, timeout_sec=0.1)
            return
        if self.position is not None:
            self.publish_string(
                self.drone_pub,
                f"goto {self.position.x:.6f} {self.position.y:.6f} {self.position.z:.6f} {self.args.yaw:.6f}",
            )
        else:
            self.publish_string(self.drone_pub, "hold")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-x", type=float, default=CUBE_WORLD[0])
    parser.add_argument("--target-y", type=float, default=CUBE_WORLD[1])
    parser.add_argument("--target-z", type=float, default=CUBE_WORLD[2])
    parser.add_argument("--target-image-dir", default="/home/u-zhuang/桌面/无人机投放")
    parser.add_argument("--image-topic", default="/drone/down_camera/rgb")
    parser.add_argument("--search-above", type=float, default=0.5)
    parser.add_argument("--search-z", type=float, default=None)
    parser.add_argument("--drop-body-z", type=float, default=1.23)
    parser.add_argument("--drop-visual-z", type=float, default=2.2)
    parser.add_argument("--descent-z", type=float, nargs="*", default=[2.2])
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--align-tolerance", type=float, default=0.045)
    parser.add_argument("--visual-tolerance", type=float, default=0.08)
    parser.add_argument("--drop-tolerance", type=float, default=0.055)
    parser.add_argument("--drop-visual-tolerance", type=float, default=0.055)
    parser.add_argument("--drop-z-tolerance", type=float, default=0.08)
    parser.add_argument("--coarse-tolerance", type=float, default=0.10)
    parser.add_argument("--align-step", type=float, default=0.18)
    parser.add_argument("--align-command-period", type=float, default=0.35)
    parser.add_argument("--pixel-to-world-gain", type=float, default=0.55)
    parser.add_argument("--vision-range", type=float, default=1.2)
    parser.add_argument("--min-confidence", type=float, default=0.65)
    parser.add_argument("--image-stale-sec", type=float, default=1.0)
    parser.add_argument("--blue-h-min", type=int, default=96)
    parser.add_argument("--blue-h-max", type=int, default=125)
    parser.add_argument("--blue-s-min", type=int, default=10)
    parser.add_argument("--blue-v-min", type=int, default=120)
    parser.add_argument("--min-blue-area", type=float, default=40.0)
    parser.add_argument("--max-blue-area-ratio", type=float, default=0.35)
    parser.add_argument("--min-aspect", type=float, default=0.35)
    parser.add_argument("--max-aspect", type=float, default=2.80)
    parser.add_argument("--min-fill-ratio", type=float, default=0.015)
    parser.add_argument("--expected-area-ratio", type=float, default=0.012)
    parser.add_argument("--goto-timeout", type=float, default=35.0)
    parser.add_argument("--align-timeout", type=float, default=18.0)
    parser.add_argument("--link-timeout", type=float, default=5.0)
    parser.add_argument("--bottom-open-wait", type=float, default=3.0)
    parser.add_argument("--final-settle-sec", type=float, default=1.5)
    parser.add_argument("--land-after-drop", action="store_true")
    parser.add_argument("--land-z", type=float, default=1.02)
    parser.add_argument("--land-wait-sec", type=float, default=8.0)
    parser.add_argument("--initial-rise", type=float, default=0.30)
    parser.add_argument("--post-load-rise", type=float, default=0.0)
    parser.add_argument("--start-with-waypoints", action="store_true")
    parser.add_argument("--drop", action="store_true", default=True)
    parser.add_argument("--no-drop", dest="drop", action="store_false")
    parser.add_argument("--verbose-status", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = BobacDroneVisualAlignDrop(args)
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
