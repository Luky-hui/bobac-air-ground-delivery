#!/usr/bin/env python3
"""PX4 Offboard mission helper for the Bobac drone stage.

This node follows the official PX4 ROS 2 Offboard path:
- publish /fmu/in/offboard_control_mode continuously
- publish /fmu/in/trajectory_setpoint continuously
- switch PX4 to Offboard, arm, fly setpoints, then land

It only uses ROS 2 topics. It does not edit USD files or move Isaac Sim prims.
Cargo bay commands are optional String messages on /cargo_bay/command.
"""

from __future__ import annotations

import math
import subprocess
import time
from typing import Iterable, Optional

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import TrajectorySetpoint
from px4_msgs.msg import VehicleCommand
from px4_msgs.msg import VehicleLocalPosition
from px4_msgs.msg import VehicleStatus
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
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


def arm_state_name(value: int) -> str:
    disarmed = getattr(
        VehicleStatus,
        "ARMING_STATE_DISARMED",
        getattr(VehicleStatus, "ARMING_STATE_STANDBY", 1),
    )
    if int(value) == int(disarmed):
        return "DISARMED"
    if int(value) == int(VehicleStatus.ARMING_STATE_ARMED):
        return "ARMED"
    return f"UNKNOWN({int(value)})"


def nav_state_name(value: int) -> str:
    names = {
        VehicleStatus.NAVIGATION_STATE_MANUAL: "MANUAL",
        VehicleStatus.NAVIGATION_STATE_ALTCTL: "ALTCTL",
        VehicleStatus.NAVIGATION_STATE_POSCTL: "POSCTL",
        VehicleStatus.NAVIGATION_STATE_AUTO_MISSION: "AUTO_MISSION",
        VehicleStatus.NAVIGATION_STATE_AUTO_LOITER: "AUTO_LOITER",
        VehicleStatus.NAVIGATION_STATE_AUTO_RTL: "AUTO_RTL",
        VehicleStatus.NAVIGATION_STATE_AUTO_TAKEOFF: "AUTO_TAKEOFF",
        VehicleStatus.NAVIGATION_STATE_AUTO_LAND: "AUTO_LAND",
        VehicleStatus.NAVIGATION_STATE_OFFBOARD: "OFFBOARD",
    }
    return names.get(int(value), f"UNKNOWN({int(value)})")


def world_enu_delta_to_px4_ned_delta(delta: Iterable[float]) -> list[float]:
    dx, dy, dz = [float(v) for v in delta]
    return [dy, dx, -dz]


def finite_position(msg: VehicleLocalPosition) -> bool:
    return math.isfinite(msg.x) and math.isfinite(msg.y) and math.isfinite(msg.z)


def parse_target_ned(value: str) -> Optional[list[float]]:
    text = str(value).strip()
    if not text:
        return None
    text = text.strip("[]()")
    parts = [p.strip() for p in text.replace(",", " ").split() if p.strip()]
    if len(parts) != 3:
        raise ValueError("target_ned must contain exactly three values: x,y,z")
    return [float(p) for p in parts]


def parse_vector3_list(value: str, label: str) -> list[list[float]]:
    text = str(value).strip()
    if not text:
        return []
    text = text.replace(";", "|")
    chunks = [chunk.strip().strip("[]()") for chunk in text.split("|") if chunk.strip()]
    points: list[list[float]] = []
    for chunk in chunks:
        parts = [p.strip() for p in chunk.replace(",", " ").split() if p.strip()]
        if len(parts) != 3:
            raise ValueError(f"{label} point must contain exactly three values: {chunk!r}")
        points.append([float(p) for p in parts])
    return points


def world_enu_point_to_px4_ned(point: Iterable[float]) -> list[float]:
    x, y, z = [float(v) for v in point]
    return [y, x, -z]


def world_enu_point_to_px4_ned_from_origin(
    point: Iterable[float],
    world_origin: Iterable[float],
    ned_origin: Iterable[float],
) -> list[float]:
    px, py, pz = [float(v) for v in point]
    ox, oy, oz = [float(v) for v in world_origin]
    nx, ny, nz = [float(v) for v in ned_origin]
    dx, dy, dz = px - ox, py - oy, pz - oz
    ndx, ndy, ndz = world_enu_delta_to_px4_ned_delta([dx, dy, dz])
    return [nx + ndx, ny + ndy, nz + ndz]


class BobacPx4OffboardMission(Node):
    def __init__(self) -> None:
        super().__init__("bobac_px4_offboard_mission_2026")

        self.declare_parameter("takeoff_height", 0.8)
        self.declare_parameter("move_enu", [0.5, 0.0, 0.0])
        self.declare_parameter("waypoints_ned", "")
        self.declare_parameter("waypoints_enu", "")
        self.declare_parameter("waypoints_world_enu", "")
        self.declare_parameter("target_ned", "")
        self.declare_parameter("target_enu", "")
        self.declare_parameter("target_world_enu", "")
        self.declare_parameter("visual_target_plane_world_z", 0.74755)
        self.declare_parameter("drop_world_enu", "")
        self.declare_parameter("drop_descend_timeout", 35.0)
        self.declare_parameter("drop_height_tolerance", 0.13)
        self.declare_parameter("hold_before_drop", 1.2)
        self.declare_parameter("takeoff_timeout", 15.0)
        self.declare_parameter("move_timeout", 15.0)
        self.declare_parameter("land_timeout", 25.0)
        self.declare_parameter("position_tolerance", 0.12)
        self.declare_parameter("takeoff_position_tolerance", 0.08)
        self.declare_parameter("hold_after_takeoff", 2.0)
        self.declare_parameter("hold_between_waypoints", 0.5)
        self.declare_parameter("hold_after_move", 2.0)
        self.declare_parameter("hold_after_visual_alignment", 1.5)
        self.declare_parameter("post_visual_drop_offset_world_enu", "0,0,0")
        self.declare_parameter("prestream_sec", 1.5)
        self.declare_parameter("require_local_position_valid", True)
        self.declare_parameter("preflight_timeout", 35.0)
        self.declare_parameter("close_side_before_takeoff", True)
        self.declare_parameter("close_bottom_before_takeoff", True)
        self.declare_parameter("open_bottom_at_target", False)
        self.declare_parameter("close_bottom_after_drop", True)
        self.declare_parameter("post_drop_bottom_close_delay", 1.2)
        self.declare_parameter("post_drop_land_world_enu", "")
        self.declare_parameter("post_drop_land_move_timeout", 35.0)
        self.declare_parameter("post_drop_land_hold", 1.5)
        self.declare_parameter("cargo_command_topic", "/cargo_bay/command")
        self.declare_parameter("cargo_status_topic", "/cargo_bay/status")
        self.declare_parameter("cargo_wait_sec", 1.5)
        self.declare_parameter("cargo_close_settle_sec", 2.2)
        self.declare_parameter("require_cargo_confirmation", False)
        self.declare_parameter("side_close_command", "left_close")
        self.declare_parameter("bottom_close_command", "bottom_close")
        self.declare_parameter("bottom_open_command", "bottom_open")
        self.declare_parameter("land_after_mission", True)
        self.declare_parameter("hold_after_mission_sec", 0.0)
        self.declare_parameter("disarm_after_landed", True)
        self.declare_parameter("force_disarm_after_land", False)
        self.declare_parameter("accept_landed_without_disarm", True)
        self.declare_parameter("land_on_failure", True)
        self.declare_parameter("landed_z_tolerance", 0.06)
        self.declare_parameter("landed_vz_tolerance", 0.08)
        self.declare_parameter("landed_stable_sec", 3.0)
        self.declare_parameter("pre_disarm_settle_sec", 5.0)
        self.declare_parameter("configure_px4_rc_input", True)
        self.declare_parameter("configure_px4_rc_loss_exception", True)
        self.declare_parameter(
            "px4_param_tool",
            "/home/u-zhuang/PX4-Autopilot/build/px4_sitl_default/bin/px4-param",
        )
        self.declare_parameter("enable_visual_alignment", False)
        self.declare_parameter("image_topic", "/drone/down_camera/rgb")
        self.declare_parameter("visual_target_topic", "/drone/visual_target")
        self.declare_parameter("visual_align_timeout", 25.0)
        self.declare_parameter("visual_tolerance", 0.060)
        self.declare_parameter("drop_visual_tolerance", 0.055)
        self.declare_parameter("visual_stable_sec", 1.5)
        self.declare_parameter("min_visual_confidence", 0.60)
        self.declare_parameter("visual_align_step", 0.008)
        self.declare_parameter("visual_align_command_period", 1.50)
        self.declare_parameter("visual_pixel_to_world_gain", 0.08)
        self.declare_parameter("visual_u_to_world_x_sign", 1.0)
        self.declare_parameter("visual_v_to_world_y_sign", -1.0)
        self.declare_parameter("image_stale_sec", 1.0)
        self.declare_parameter("blue_h_min", 96)
        self.declare_parameter("blue_h_max", 125)
        self.declare_parameter("blue_s_min", 10)
        self.declare_parameter("blue_v_min", 120)
        self.declare_parameter("min_blue_area", 40.0)
        self.declare_parameter("max_blue_area_ratio", 0.35)
        self.declare_parameter("min_fill_ratio", 0.015)
        self.declare_parameter("expected_area_ratio", 0.012)
        self.declare_parameter("reject_clipped_blue_target", True)
        self.declare_parameter("blue_target_clip_margin_px", 4)
        self.declare_parameter("require_visual_alignment_for_drop", True)

        self.offboard_pub = self.create_publisher(
            OffboardControlMode, "/fmu/in/offboard_control_mode", PX4_IN_QOS
        )
        self.setpoint_pub = self.create_publisher(
            TrajectorySetpoint, "/fmu/in/trajectory_setpoint", PX4_IN_QOS
        )
        self.command_pub = self.create_publisher(
            VehicleCommand, "/fmu/in/vehicle_command", PX4_IN_QOS
        )
        self.cargo_pub = self.create_publisher(
            String, str(self.get_parameter("cargo_command_topic").value), 10
        )
        self.visual_pub = self.create_publisher(
            String, str(self.get_parameter("visual_target_topic").value), 10
        )

        self.local_position: Optional[VehicleLocalPosition] = None
        self.world_position: Optional[PointStamped] = None
        self.vehicle_status: Optional[VehicleStatus] = None
        self.cargo_status = ""
        self.latest_image: Optional[np.ndarray] = None
        self.latest_image_stamp = 0.0
        self.last_visual_alignment: Optional[dict[str, float]] = None
        self.target: Optional[list[float]] = None
        self.target_yaw: Optional[float] = None
        self.stream_enabled = False

        self.create_subscription(
            VehicleLocalPosition,
            "/fmu/out/vehicle_local_position",
            self._on_local_position,
            PX4_OUT_QOS,
        )
        self.create_subscription(
            PointStamped,
            "/drone/world_position",
            self._on_world_position,
            10,
        )
        self.create_subscription(
            VehicleStatus,
            "/fmu/out/vehicle_status",
            self._on_vehicle_status,
            PX4_OUT_QOS,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("cargo_status_topic").value),
            self._on_cargo_status,
            10,
        )
        self.create_subscription(
            Image,
            str(self.get_parameter("image_topic").value),
            self._on_image,
            10,
        )
        self.create_timer(0.05, self._publish_stream)

    def timestamp_us(self) -> int:
        return int(self.get_clock().now().nanoseconds / 1000)

    def _on_local_position(self, msg: VehicleLocalPosition) -> None:
        if finite_position(msg):
            self.local_position = msg

    def _on_world_position(self, msg: PointStamped) -> None:
        self.world_position = msg

    def _on_vehicle_status(self, msg: VehicleStatus) -> None:
        self.vehicle_status = msg

    def _on_cargo_status(self, msg: String) -> None:
        self.cargo_status = msg.data
        self.get_logger().info(f"cargo status: {msg.data}")

    def _on_image(self, msg: Image) -> None:
        try:
            self.latest_image = self._image_to_bgr(msg)
            self.latest_image_stamp = time.monotonic()
        except Exception as exc:
            self.get_logger().warning(f"failed to decode down camera image: {exc}")

    def _publish_stream(self) -> None:
        if not self.stream_enabled or self.target is None:
            return
        now = self.timestamp_us()

        mode = OffboardControlMode()
        mode.timestamp = now
        mode.position = True
        self.offboard_pub.publish(mode)

        setpoint = TrajectorySetpoint()
        setpoint.timestamp = now
        setpoint.position = [float(self.target[0]), float(self.target[1]), float(self.target[2])]
        if self.target_yaw is not None:
            setpoint.yaw = float(self.target_yaw)
        self.setpoint_pub.publish(setpoint)

    def spin_for(self, seconds: float) -> None:
        end = time.monotonic() + max(0.0, float(seconds))
        while rclpy.ok() and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

    def wait_for_px4(self, timeout: float = 10.0) -> bool:
        end = time.monotonic() + timeout
        require_valid = bool(self.get_parameter("require_local_position_valid").value)
        while rclpy.ok() and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.local_position is None or self.vehicle_status is None:
                continue
            if require_valid:
                valid = (
                    bool(getattr(self.local_position, "xy_valid", False))
                    and bool(getattr(self.local_position, "z_valid", False))
                    and bool(getattr(self.local_position, "v_xy_valid", False))
                    and bool(getattr(self.local_position, "v_z_valid", False))
                    and bool(getattr(self.vehicle_status, "pre_flight_checks_pass", False))
                )
                if not valid:
                    continue
                return True
            return True
        self.get_logger().error(
            f"PX4 not ready: local_position={self.local_position is not None}, "
            f"vehicle_status={self.vehicle_status is not None}"
        )
        return False

    def configure_px4_params(self) -> bool:
        tool = str(self.get_parameter("px4_param_tool").value)

        params: list[tuple[str, str]] = []
        if bool(self.get_parameter("configure_px4_rc_input").value):
            params.append(("COM_RC_IN_MODE", "4"))
        if bool(self.get_parameter("configure_px4_rc_loss_exception").value):
            params.append(("COM_RCL_EXCEPT", "4"))

        for name, value in params:
            try:
                result = subprocess.run(
                    [tool, "set", name, value],
                    check=False,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=5.0,
                )
            except Exception as exc:
                self.get_logger().error(f"failed to run px4-param for {name}: {exc}")
                return False
            if result.stdout.strip():
                self.get_logger().info(result.stdout.strip())
            if result.returncode != 0:
                self.get_logger().error(f"px4-param set {name} returned {result.returncode}")
                return False
        return True

    def send_vehicle_command(
        self, command: int, param1: float = 0.0, param2: float = 0.0, count: int = 3
    ) -> None:
        for _ in range(max(1, int(count))):
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
            self.spin_for(0.1)

    def set_target(self, target: Iterable[float], yaw: Optional[float] = None) -> None:
        values = [float(v) for v in target]
        if len(values) != 3:
            raise ValueError("target must contain three values")
        self.target = values
        self.target_yaw = yaw
        self.stream_enabled = True
        self.get_logger().info(
            f"target NED: x={values[0]:.3f}, y={values[1]:.3f}, z={values[2]:.3f}"
        )

    @staticmethod
    def _image_to_bgr(msg: Image) -> np.ndarray:
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
            if encoding == "rgba8":
                return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
            return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
        if encoding == "rgb8":
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        return arr.copy()

    def _publish_visual_result(
        self,
        detected: bool,
        *,
        u: float = float("nan"),
        v: float = float("nan"),
        area: float = 0.0,
        fill: float = 0.0,
        confidence: float = 0.0,
    ) -> None:
        msg = String()
        if detected:
            msg.data = (
                f"target=blue_square source=down_camera detected=true "
                f"u={u:.4f} v={v:.4f} area={area:.1f} fill={fill:.3f} "
                f"confidence={confidence:.3f}"
            )
        else:
            msg.data = "target=blue_square source=down_camera detected=false confidence=0.000"
        self.visual_pub.publish(msg)

    def detect_blue_target(self) -> Optional[dict[str, float]]:
        image = self.latest_image
        if image is None:
            self._publish_visual_result(False)
            return None
        if time.monotonic() - self.latest_image_stamp > float(
            self.get_parameter("image_stale_sec").value
        ):
            self._publish_visual_result(False)
            return None

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
        if not contours:
            self._publish_visual_result(False)
            return None

        height, width = image.shape[:2]
        image_area = float(width * height)
        min_blue_area = float(self.get_parameter("min_blue_area").value)
        max_area = image_area * float(self.get_parameter("max_blue_area_ratio").value)
        min_fill = float(self.get_parameter("min_fill_ratio").value)
        valid_mask = np.zeros(mask.shape, dtype=np.uint8)
        for contour in contours:
            area = float(cv2.contourArea(contour))
            # The target is a hollow blue rectangle. Merge its separated edges
            # into one measured target instead of chasing the largest edge.
            if area < min_blue_area * 0.15:
                continue
            cv2.drawContours(valid_mask, [contour], -1, 255, thickness=cv2.FILLED)

        blue_pixels = cv2.findNonZero(valid_mask)
        if blue_pixels is None:
            self._publish_visual_result(False)
            return None

        x, y, w, h = cv2.boundingRect(blue_pixels)
        if w <= 2 or h <= 2:
            self._publish_visual_result(False)
            return None
        area = float(cv2.countNonZero(valid_mask))
        if area < min_blue_area or area > max_area:
            self._publish_visual_result(False)
            return None
        rect_area = float(w * h)
        fill = area / max(rect_area, 1.0)
        if fill < min_fill:
            self._publish_visual_result(False)
            return None

        moments = cv2.moments(valid_mask, binaryImage=True)
        if abs(moments["m00"]) > 1.0e-6:
            cx = float(moments["m10"] / moments["m00"])
            cy = float(moments["m01"] / moments["m00"])
        else:
            cx = float(x + w * 0.5)
            cy = float(y + h * 0.5)

        margin = max(1, int(self.get_parameter("blue_target_clip_margin_px").value))
        clipped = (
            x <= margin
            or y <= margin
            or x + w >= width - margin
            or y + h >= height - margin
        )
        if clipped and bool(self.get_parameter("reject_clipped_blue_target").value):
            self._publish_visual_result(False)
            return None

        best = {
            "score": area * min(fill, 1.0),
            "cx": cx,
            "cy": cy,
            "area": area,
            "fill": fill,
            "width": float(width),
            "height": float(height),
        }

        u = (best["cx"] - best["width"] * 0.5) / max(best["width"] * 0.5, 1.0)
        v = (best["cy"] - best["height"] * 0.5) / max(best["height"] * 0.5, 1.0)
        area_ratio = best["area"] / max(image_area, 1.0)
        center_error = math.hypot(u, v)
        confidence = max(
            0.05,
            min(
                0.99,
                0.35
                + min(
                    area_ratio / max(float(self.get_parameter("expected_area_ratio").value), 1.0e-6),
                    1.0,
                )
                * 0.35
                + max(0.0, 1.0 - center_error) * 0.30,
            ),
        )
        result = {
            "u": u,
            "v": v,
            "area": best["area"],
            "fill": best["fill"],
            "confidence": confidence,
            "pixel_err": center_error,
        }
        self._publish_visual_result(
            True,
            u=u,
            v=v,
            area=best["area"],
            fill=best["fill"],
            confidence=confidence,
        )
        return result

    def wait_for_visual_image(self, timeout: float = 5.0) -> bool:
        end = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.latest_image is not None:
                return True
        self.get_logger().error(
            f"no image on {str(self.get_parameter('image_topic').value)}"
        )
        return False

    def _image_error_to_ned_step(self, u: float, v: float, target_world_z: Optional[float]) -> list[float]:
        altitude = 1.0
        if self.world_position is not None and target_world_z is not None:
            altitude = max(0.30, float(self.world_position.point.z) - float(target_world_z))
        gain = max(0.01, float(self.get_parameter("visual_pixel_to_world_gain").value))
        # Same default camera convention used by the legacy visual script:
        # image +u -> world +x, image +v -> world -y. The signs are parameters
        # because a camera-prim yaw flip is easier to verify in sim than to
        # infer from USD by eye.
        dx_world = (
            float(u)
            * altitude
            * gain
            * float(self.get_parameter("visual_u_to_world_x_sign").value)
        )
        dy_world = (
            float(v)
            * altitude
            * gain
            * float(self.get_parameter("visual_v_to_world_y_sign").value)
        )
        step_norm = math.hypot(dx_world, dy_world)
        max_step = max(0.002, float(self.get_parameter("visual_align_step").value))
        if step_norm > max_step:
            scale = max_step / step_norm
            dx_world *= scale
            dy_world *= scale
        return world_enu_delta_to_px4_ned_delta([dx_world, dy_world, 0.0])

    def visual_align_before_drop(self, target_world_z: Optional[float]) -> bool:
        if not self.wait_for_visual_image(timeout=5.0):
            return False
        timeout = float(self.get_parameter("visual_align_timeout").value)
        tolerance = float(self.get_parameter("drop_visual_tolerance").value)
        min_conf = float(self.get_parameter("min_visual_confidence").value)
        command_period = float(self.get_parameter("visual_align_command_period").value)
        stable_required = max(0.0, float(self.get_parameter("visual_stable_sec").value))
        deadline = time.monotonic() + timeout
        last_command = 0.0
        last_log = 0.0
        stable_since: Optional[float] = None
        history: list[tuple[float, float, float, float]] = []
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            detection = self.detect_blue_target()
            now = time.monotonic()
            if detection is None:
                stable_since = None
                if now - last_log > 1.0:
                    self.get_logger().info("visual_align: blue target not detected yet")
                    last_log = now
                continue

            u = float(detection["u"])
            v = float(detection["v"])
            pixel_err = float(detection["pixel_err"])
            confidence = float(detection["confidence"])
            if confidence < min_conf:
                stable_since = None
                if now - last_log > 0.5:
                    self.get_logger().info(
                        "visual_align: low confidence "
                        f"u={u:.4f}, v={v:.4f}, pixel_err={pixel_err:.4f}, "
                        f"confidence={confidence:.3f}"
                    )
                    last_log = now
                continue

            history.append((u, v, pixel_err, confidence))
            history = history[-7:]
            filt_u = float(np.median([item[0] for item in history]))
            filt_v = float(np.median([item[1] for item in history]))
            filt_err = math.hypot(filt_u, filt_v)
            filt_confidence = float(np.median([item[3] for item in history]))
            if now - last_log > 0.5:
                self.get_logger().info(
                    "visual_align: "
                    f"u={u:.4f}, v={v:.4f}, pixel_err={pixel_err:.4f}, "
                    f"confidence={confidence:.3f}, "
                    f"filtered_u={filt_u:.4f}, filtered_v={filt_v:.4f}, "
                    f"filtered_err={filt_err:.4f}"
                )
                last_log = now
            stable = filt_err <= tolerance and filt_confidence >= min_conf
            if stable:
                if stable_since is None:
                    stable_since = now
                stable_elapsed = now - stable_since
                if now - last_log > 0.5:
                    self.get_logger().info(
                        "visual_align: stable hold "
                        f"{stable_elapsed:.2f}/{stable_required:.2f}s"
                    )
                    last_log = now
                if stable_elapsed >= stable_required:
                    self.last_visual_alignment = detection
                    self.get_logger().info(
                        "visual alignment passed: "
                        f"pixel_err={filt_err:.4f}, confidence={filt_confidence:.3f}, "
                        f"stable={stable_elapsed:.2f}s"
                    )
                    return True
                continue
            else:
                stable_since = None

            if self.target is None:
                continue
            if now - last_command >= command_period:
                step = self._image_error_to_ned_step(filt_u, filt_v, target_world_z)
                # Use the last commanded setpoint as the correction base. Using
                # the current vehicle position here makes the target chase PX4
                # lateral inertia, which can carry the drone past the blue-box
                # center and toward the visible edge.
                base = list(self.target)
                new_target = [
                    float(base[0]) + float(step[0]),
                    float(base[1]) + float(step[1]),
                    float(base[2]),
                ]
                self.set_target(new_target, self.target_yaw)
                last_command = now

        self.get_logger().error("visual alignment timeout")
        return False

    def arm_and_offboard(self) -> bool:
        deadline = time.monotonic() + 12.0
        next_command = 0.0
        last_log = 0.0
        while rclpy.ok() and time.monotonic() < deadline:
            now = time.monotonic()
            rclpy.spin_once(self, timeout_sec=0.05)
            status = self.vehicle_status
            if (
                status is not None
                and status.arming_state == VehicleStatus.ARMING_STATE_ARMED
                and status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD
            ):
                return True
            if now >= next_command:
                self.send_vehicle_command(
                    VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
                    1.0,
                    6.0,
                    count=2,
                )
                self.send_vehicle_command(
                    VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
                    1.0,
                    0.0,
                    count=2,
                )
                next_command = now + 0.5
            if status is not None and now - last_log >= 1.0:
                self.get_logger().info(
                    "arm/offboard wait: "
                    f"arming={arm_state_name(status.arming_state)}, "
                    f"nav={nav_state_name(status.nav_state)}, "
                    f"preflight={getattr(status, 'pre_flight_checks_pass', 'unknown')}"
                )
                last_log = now
        return self.wait_state(armed=True, offboard=True, timeout=1.0)

    def disarm_and_wait(self, timeout: float = 5.0) -> bool:
        def _wait(seconds: float) -> bool:
            end = time.monotonic() + max(0.0, seconds)
            while rclpy.ok() and time.monotonic() < end:
                rclpy.spin_once(self, timeout_sec=0.05)
                status = self.vehicle_status
                if status is not None and status.arming_state != VehicleStatus.ARMING_STATE_ARMED:
                    self.stream_enabled = False
                    return True
            return False

        self.get_logger().info("landed stable; sending disarm command")
        self.send_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
            0.0,
            0.0,
            count=5,
        )
        if _wait(timeout):
            return True

        status = self.vehicle_status
        state = (
            f": arming={arm_state_name(status.arming_state)}, nav={nav_state_name(status.nav_state)}"
            if status is not None
            else ""
        )
        if not bool(self.get_parameter("force_disarm_after_land").value):
            self.get_logger().warning(
                "normal disarm not confirmed"
                + state
                + "; keeping PX4 armed instead of force-disarming to avoid tip-over"
            )
            return bool(self.get_parameter("accept_landed_without_disarm").value)

        self.get_logger().warning(
            "normal disarm not confirmed" + state + "; sending PX4 force-disarm"
        )
        self.send_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
            0.0,
            21196.0,
            count=5,
        )
        return _wait(timeout)

    def hold_landed_before_disarm(self, seconds: float) -> None:
        seconds = max(0.0, float(seconds))
        if seconds <= 0.0:
            return
        self.get_logger().info(
            f"landed stable; holding armed for {seconds:.1f}s before normal disarm"
        )
        end = time.monotonic() + seconds
        last_log = 0.0
        while rclpy.ok() and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
            status = self.vehicle_status
            pos = self.local_position
            if status is not None and status.arming_state != VehicleStatus.ARMING_STATE_ARMED:
                self.stream_enabled = False
                return
            now = time.monotonic()
            if pos is not None and now - last_log >= 1.0:
                self.get_logger().info(
                    f"landed settle: z={pos.z:.3f}, vz={pos.vz:.3f}"
                )
                last_log = now

    def wait_state(self, armed: bool, offboard: bool, timeout: float) -> bool:
        end = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
            status = self.vehicle_status
            if status is None:
                continue
            armed_ok = status.arming_state == VehicleStatus.ARMING_STATE_ARMED if armed else True
            offboard_ok = (
                status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD if offboard else True
            )
            if armed_ok and offboard_ok:
                return True
        status = self.vehicle_status
        if status is not None:
            self.get_logger().error(
                "state wait failed: "
                f"arming={arm_state_name(status.arming_state)}, "
                f"nav={nav_state_name(status.nav_state)}, "
                f"preflight={getattr(status, 'pre_flight_checks_pass', 'unknown')}"
            )
        return False

    def wait_reached(
        self, label: str, timeout: float, tolerance: Optional[float] = None
    ) -> bool:
        if self.target is None:
            return False
        tol = (
            float(tolerance)
            if tolerance is not None
            else float(self.get_parameter("position_tolerance").value)
        )
        end = time.monotonic() + timeout
        last_print = 0.0
        while rclpy.ok() and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
            pos = self.local_position
            if pos is None:
                continue
            err = math.sqrt(
                (pos.x - self.target[0]) ** 2
                + (pos.y - self.target[1]) ** 2
                + (pos.z - self.target[2]) ** 2
            )
            now = time.monotonic()
            if now - last_print > 1.0:
                self.get_logger().info(
                    f"{label}: pos=[{pos.x:.3f},{pos.y:.3f},{pos.z:.3f}], err={err:.3f}"
                )
                last_print = now
            if err <= tol:
                return True
        return False

    def send_cargo(self, command: str, expected: str = "") -> bool:
        command = str(command)
        self.cargo_status = ""
        msg = String()
        msg.data = command
        wait_sec = float(self.get_parameter("cargo_wait_sec").value)
        require = bool(self.get_parameter("require_cargo_confirmation").value)

        self.get_logger().info(f"cargo command: {command}")
        for _ in range(3):
            self.cargo_pub.publish(msg)
            self.spin_for(0.08)

        if wait_sec <= 0.0:
            return True
        end = time.monotonic() + wait_sec
        while rclpy.ok() and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
            if expected and expected in self.cargo_status:
                if "close" in command:
                    self.spin_for(float(self.get_parameter("cargo_close_settle_sec").value))
                return True
        if require:
            self.get_logger().error(f"cargo command {command} was not confirmed as {expected}")
            return False
        self.get_logger().warning(f"cargo command {command} had no confirmation; continuing")
        return True

    def land_on_failure(self, reason: str) -> None:
        if not bool(self.get_parameter("land_on_failure").value):
            return
        self.get_logger().error(f"{reason}; sending safety land")
        self.stream_enabled = False
        self.send_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND, count=3)
        self.spin_for(2.0)

    def mission(self) -> bool:
        if not self.wait_for_px4(timeout=float(self.get_parameter("preflight_timeout").value)):
            return False
        if not self.configure_px4_params():
            return False

        if bool(self.get_parameter("close_side_before_takeoff").value):
            if not self.send_cargo(
                str(self.get_parameter("side_close_command").value), "left_closed"
            ):
                return False
        if bool(self.get_parameter("close_bottom_before_takeoff").value):
            if not self.send_cargo(
                str(self.get_parameter("bottom_close_command").value), "bottom_closed"
            ):
                return False

        start = self.local_position
        assert start is not None
        start_world = self.world_position
        world_origin = None
        ned_origin = [float(start.x), float(start.y), float(start.z)]
        if start_world is not None:
            world_origin = [
                float(start_world.point.x),
                float(start_world.point.y),
                float(start_world.point.z),
            ]
            self.get_logger().info(
                "world/local origin: "
                f"world=[{world_origin[0]:.3f},{world_origin[1]:.3f},{world_origin[2]:.3f}], "
                f"ned=[{ned_origin[0]:.3f},{ned_origin[1]:.3f},{ned_origin[2]:.3f}]"
            )
        height = max(0.05, float(self.get_parameter("takeoff_height").value))
        takeoff_target = [float(start.x), float(start.y), float(start.z) - height]
        yaw = float(start.heading) if math.isfinite(start.heading) else None
        self.set_target(takeoff_target, yaw)
        self.get_logger().info(
            f"streaming setpoint for {float(self.get_parameter('prestream_sec').value):.1f}s"
        )
        self.spin_for(float(self.get_parameter("prestream_sec").value))

        if not self.arm_and_offboard():
            self.land_on_failure("arm/offboard failed")
            return False
        if not self.wait_reached(
            "takeoff",
            float(self.get_parameter("takeoff_timeout").value),
            float(self.get_parameter("takeoff_position_tolerance").value),
        ):
            self.get_logger().error("takeoff target not reached")
            self.land_on_failure("takeoff target not reached")
            return False
        self.spin_for(float(self.get_parameter("hold_after_takeoff").value))

        waypoints = parse_vector3_list(str(self.get_parameter("waypoints_ned").value), "waypoints_ned")
        waypoints += [
            world_enu_point_to_px4_ned(point)
            for point in parse_vector3_list(str(self.get_parameter("waypoints_enu").value), "waypoints_enu")
        ]
        world_waypoints = parse_vector3_list(
            str(self.get_parameter("waypoints_world_enu").value), "waypoints_world_enu"
        )
        if world_waypoints:
            if world_origin is None:
                self.get_logger().error(
                    "waypoints_world_enu requested but /drone/world_position is unavailable"
                )
                self.land_on_failure("missing /drone/world_position")
                return False
            waypoints += [
                world_enu_point_to_px4_ned_from_origin(point, world_origin, ned_origin)
                for point in world_waypoints
            ]

        fixed_target = parse_target_ned(str(self.get_parameter("target_ned").value))
        fixed_target_enu = parse_target_ned(str(self.get_parameter("target_enu").value))
        if fixed_target is None and fixed_target_enu is not None:
            fixed_target = world_enu_point_to_px4_ned(fixed_target_enu)
        fixed_target_world_enu = parse_target_ned(
            str(self.get_parameter("target_world_enu").value)
        )
        target_world_z_for_visual: Optional[float] = float(
            self.get_parameter("visual_target_plane_world_z").value
        )
        drop_target: Optional[list[float]] = None
        if fixed_target_world_enu is not None:
            if world_origin is None:
                self.get_logger().error(
                    "target_world_enu requested but /drone/world_position is unavailable"
                )
                self.land_on_failure("missing /drone/world_position")
                return False
            fixed_target = world_enu_point_to_px4_ned_from_origin(
                fixed_target_world_enu, world_origin, ned_origin
            )
        drop_world_enu = parse_target_ned(str(self.get_parameter("drop_world_enu").value))
        if drop_world_enu is not None:
            if world_origin is None:
                self.get_logger().error(
                    "drop_world_enu requested but /drone/world_position is unavailable"
                )
                self.land_on_failure("missing /drone/world_position")
                return False
            drop_target = world_enu_point_to_px4_ned_from_origin(
                drop_world_enu, world_origin, ned_origin
            )

        if fixed_target is None and not waypoints:
            move_enu = list(self.get_parameter("move_enu").value)
            ned_delta = world_enu_delta_to_px4_ned_delta(move_enu)
            move_target = [
                takeoff_target[0] + ned_delta[0],
                takeoff_target[1] + ned_delta[1],
                takeoff_target[2] + ned_delta[2],
            ]
        else:
            if fixed_target is not None:
                waypoints.append(fixed_target)
            move_target = waypoints[-1]

        waypoint_timeout = float(self.get_parameter("move_timeout").value)
        hold_between = float(self.get_parameter("hold_between_waypoints").value)
        for index, waypoint in enumerate(waypoints or [move_target], start=1):
            label = "target" if index == len(waypoints or [move_target]) else f"waypoint_{index:02d}"
            self.set_target(waypoint, yaw)
            if not self.wait_reached(label, waypoint_timeout):
                self.get_logger().error(f"{label} not reached")
                self.land_on_failure(f"{label} not reached")
                return False
            if index != len(waypoints or [move_target]) and hold_between > 0.0:
                self.spin_for(hold_between)
        self.spin_for(float(self.get_parameter("hold_after_move").value))

        if bool(self.get_parameter("enable_visual_alignment").value):
            self.get_logger().info(
                "starting down-camera visual alignment before bottom-door release"
            )
            if not self.visual_align_before_drop(target_world_z_for_visual):
                if bool(self.get_parameter("require_visual_alignment_for_drop").value):
                    self.land_on_failure("visual alignment failed")
                    return False
                self.get_logger().warning(
                    "visual alignment failed but require_visual_alignment_for_drop=false; continuing"
                )
            else:
                offset_world = parse_target_ned(
                    str(self.get_parameter("post_visual_drop_offset_world_enu").value)
                )
                if offset_world is not None and any(abs(v) > 1.0e-6 for v in offset_world):
                    offset_ned = world_enu_delta_to_px4_ned_delta(offset_world)
                    self.set_target(
                        [
                            float(self.target[0]) + float(offset_ned[0]),
                            float(self.target[1]) + float(offset_ned[1]),
                            float(self.target[2]) + float(offset_ned[2]),
                        ],
                        self.target_yaw,
                    )
                    self.get_logger().info(
                        "applying post-visual payload release offset "
                        f"world_enu=[{offset_world[0]:.3f},{offset_world[1]:.3f},{offset_world[2]:.3f}]"
                    )
                settle = float(self.get_parameter("hold_after_visual_alignment").value)
                if settle > 0.0:
                    self.get_logger().info(
                        f"visual alignment settled; holding {settle:.1f}s before drop"
                    )
                    self.spin_for(settle)

        if drop_target is not None:
            self.get_logger().info(
                "descending vertically to drop height after visual alignment"
            )
            current = self.local_position
            if current is not None:
                drop_target = [float(current.x), float(current.y), float(drop_target[2])]
            self.set_target(drop_target, self.target_yaw)
            if not self.wait_reached(
                "drop_height",
                float(self.get_parameter("drop_descend_timeout").value),
                float(self.get_parameter("drop_height_tolerance").value),
            ):
                self.land_on_failure("drop height not reached")
                return False
            self.spin_for(float(self.get_parameter("hold_before_drop").value))

        if bool(self.get_parameter("open_bottom_at_target").value):
            if not self.send_cargo(
                str(self.get_parameter("bottom_open_command").value),
                "bottom_opened",
            ):
                return False
            self.spin_for(max(0.0, float(self.get_parameter("post_drop_bottom_close_delay").value)))
            if bool(self.get_parameter("close_bottom_after_drop").value):
                if not self.send_cargo(
                    str(self.get_parameter("bottom_close_command").value),
                    "bottom_closed",
                ):
                    return False

        post_drop_land_world_enu = parse_target_ned(
            str(self.get_parameter("post_drop_land_world_enu").value)
        )
        if post_drop_land_world_enu is not None:
            if world_origin is None:
                self.get_logger().error(
                    "post_drop_land_world_enu requested but /drone/world_position is unavailable"
                )
                self.land_on_failure("missing /drone/world_position")
                return False
            post_drop_land_target = world_enu_point_to_px4_ned_from_origin(
                post_drop_land_world_enu, world_origin, ned_origin
            )
            self.get_logger().info(
                "moving away from drop target before landing: "
                f"world_enu=[{post_drop_land_world_enu[0]:.3f},"
                f"{post_drop_land_world_enu[1]:.3f},{post_drop_land_world_enu[2]:.3f}]"
            )
            self.set_target(post_drop_land_target, self.target_yaw)
            if not self.wait_reached(
                "post_drop_land_site",
                float(self.get_parameter("post_drop_land_move_timeout").value),
            ):
                self.land_on_failure("post-drop landing site not reached")
                return False
            self.spin_for(float(self.get_parameter("post_drop_land_hold").value))

        if bool(self.get_parameter("land_after_mission").value):
            self.get_logger().info("sending land command")
            self.send_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND, count=1)
            end = time.monotonic() + float(self.get_parameter("land_timeout").value)
            land_start = time.monotonic()
            stable_since: Optional[float] = None
            start_z = float(start.z)
            z_tol = float(self.get_parameter("landed_z_tolerance").value)
            vz_tol = float(self.get_parameter("landed_vz_tolerance").value)
            stable_sec = float(self.get_parameter("landed_stable_sec").value)
            while rclpy.ok() and time.monotonic() < end:
                rclpy.spin_once(self, timeout_sec=0.1)
                status = self.vehicle_status
                pos = self.local_position
                if status is not None and pos is not None:
                    self.get_logger().info(
                        "land: "
                        f"arming={arm_state_name(status.arming_state)}, "
                        f"nav={nav_state_name(status.nav_state)}, z={pos.z:.3f}"
                    )
                    if status.arming_state != VehicleStatus.ARMING_STATE_ARMED:
                        self.stream_enabled = False
                        return True
                    if (time.monotonic() - land_start) > 3.0:
                        z = float(pos.z)
                        near_start_ground = abs(z - start_z) <= z_tol
                        near_local_ground = abs(z) <= z_tol
                        landed = (near_start_ground or near_local_ground) and abs(float(pos.vz)) <= vz_tol
                        if landed:
                            if stable_since is None:
                                stable_since = time.monotonic()
                            elif time.monotonic() - stable_since >= stable_sec:
                                if not bool(self.get_parameter("disarm_after_landed").value):
                                    self.hold_landed_before_disarm(
                                        float(self.get_parameter("pre_disarm_settle_sec").value)
                                    )
                                    self.get_logger().info(
                                        "landed stable; keeping PX4 armed for attitude hold"
                                    )
                                    return True
                                self.hold_landed_before_disarm(
                                    float(self.get_parameter("pre_disarm_settle_sec").value)
                                )
                                if self.disarm_and_wait(timeout=5.0):
                                    return True
                                self.get_logger().error("landed but disarm was not confirmed")
                                return False
                        else:
                            stable_since = None
                    self.spin_for(1.0)
            self.get_logger().error("land did not finish before timeout")
            return False

        final_hold = float(self.get_parameter("hold_after_mission_sec").value)
        if final_hold > 0.0:
            self.get_logger().info(
                f"mission target reached; holding current Offboard setpoint for {final_hold:.1f}s"
            )
            self.spin_for(final_hold)

        return True


def main() -> None:
    rclpy.init()
    node = BobacPx4OffboardMission()
    try:
        ok = node.mission()
        if ok:
            node.get_logger().info("mission completed")
        else:
            node.get_logger().error("mission failed")
            raise SystemExit(1)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
