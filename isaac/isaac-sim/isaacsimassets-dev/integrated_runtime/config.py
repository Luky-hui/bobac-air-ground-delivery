#!/usr/bin/env python3
"""Shared configuration for the cargo delivery demo."""

from __future__ import annotations

import os
from pathlib import Path
import re


DEMO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = DEMO_ROOT.parents[1]
ASSET_ROOT = DEMO_ROOT / "assets"
LOCAL_WORLDS_DIR = ASSET_ROOT / "full_worlds"
SINGLE_WORLDS_DIR = ASSET_ROOT / "single_worlds"
LOCAL_ROBOTS_DIR = ASSET_ROOT / "robots"

PEGASUS_EXTENSION = Path(
    os.environ.get(
        "PEGASUS_EXTENSION",
        str(
            PROJECT_ROOT
            / "reference_projects/PegasusSimulator/extensions/pegasus.simulator"
        ),
    )
)

CARGO_BAY_PRIM = "/World/quadrotor/mounted_cargo_bay/transparent_cargo_bay"
CARGO_BODY_PRIM = "/World/quadrotor/mounted_cargo_bay/transparent_cargo_bay/cargo_body"
LEFT_DOOR_JOINT = "/World/quadrotor/mounted_cargo_bay/transparent_cargo_bay/joints/left_door_joint"
BOTTOM_DOOR_JOINT = "/World/quadrotor/mounted_cargo_bay/transparent_cargo_bay/joints/bottom_door_joint"

DEFAULT_X1_ASSEMBLED_SCENE_USD = (
    LOCAL_WORLDS_DIR / "Collected_X1_race_scene/X1_race_scene.usd"
)
X1_RACE_SCENE_NEW3_USD = PROJECT_ROOT / "isaac_sim_2026/isaac-sim/X1_race_scene_new3.usd"
X1_RACE_SCENE_NEW2_USD = PROJECT_ROOT / "isaac_sim_2026/isaac-sim/X1_race_scene_new2.usd"
X1_RACE_SCENE_NEW_USD = PROJECT_ROOT / "isaac_sim_2026/isaac-sim/X1_race_scene_new.usd"
X1_ASSEMBLED_SCENE_USD = Path(
    os.environ.get(
        "X1_ASSEMBLED_SCENE_USD",
        str(
            X1_RACE_SCENE_NEW3_USD
            if X1_RACE_SCENE_NEW3_USD.exists()
            else (
                X1_RACE_SCENE_NEW2_USD
                if X1_RACE_SCENE_NEW2_USD.exists()
                else (
                    X1_RACE_SCENE_NEW_USD
                    if X1_RACE_SCENE_NEW_USD.exists()
                    else DEFAULT_X1_ASSEMBLED_SCENE_USD
                )
            )
        ),
    )
)
BOBAC_ASSEMBLED_SCENE_USD = (
    LOCAL_WORLDS_DIR
    / "Collected_bobac_race_scene/bobac_race_scene.usd"
)
BOBAC_ENABLE_PX4 = os.environ.get("BOBAC_ENABLE_PX4", "0").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
BOBAC_PX4_ATTACH_EXISTING = os.environ.get(
    "BOBAC_PX4_ATTACH_EXISTING", "1"
).strip().lower() in ("1", "true", "yes", "on")
BOBAC_PX4_PRIM = os.environ.get("BOBAC_PX4_PRIM", "/World/quadrotor")
BOBAC_PX4_SPAWN = os.environ.get("BOBAC_PX4_SPAWN", "").strip()
BOBAC_PX4_SANITIZE_CHILD_MASS = os.environ.get(
    "BOBAC_PX4_SANITIZE_CHILD_MASS", "1"
).strip().lower() in ("1", "true", "yes", "on")
BOBAC_PX4_CHILD_COLLIDER_MASS = float(
    os.environ.get("BOBAC_PX4_CHILD_COLLIDER_MASS", "0.001")
)
BOBAC_PX4_ROTOR_CONSTANT_MULTIPLIER = float(
    os.environ.get("BOBAC_PX4_ROTOR_CONSTANT_MULTIPLIER", "2.0")
)
BOBAC_PX4_INPUT_SCALING = float(os.environ.get("BOBAC_PX4_INPUT_SCALING", "1000.0"))
BOBAC_PX4_DRONE_USD = Path(
    os.environ.get(
        "BOBAC_PX4_DRONE_USD",
        str(
            PROJECT_ROOT
            / "official_sources/drone-training-package/assets/robots/"
            "sunray150_with_mid360_cargo/sunray150_with_mid360_cargo.usda"
        ),
    )
)
CREATE_MAP_SCENE_USD = (
    SINGLE_WORLDS_DIR
    / "Collected_race_single_scene/race_single_scene.usd"
)

PENCIL_PRIM = "/World/layout/caughting/SM_Pencil_White3_129"
PENCIL_CHILD_PRIM = "/World/layout/caughting/SM_Pencil_White3_129/SM_Pencil_White3_129"
PENCIL_MESH_PRIM = (
    "/World/layout/caughting/SM_Pencil_White3_129/SM_Pencil_White3_129/SM_Pencil_White"
)
PENCIL_PROXY_PRIM = "/World/layout/caughting/SM_Pencil_White3_129/cargo_pencil_collision_proxy"

RUNTIME_SCOPE = "/World/cargo_delivery_runtime"
MOUNT_JOINT_PATH = f"{RUNTIME_SCOPE}/cargo_mount_joint"
PAYLOAD_LOCK_JOINT_PATH = f"{RUNTIME_SCOPE}/payload_lock_joint"

CARGO_BAY_REFERENCE_HEIGHT = 0.146834
CARGO_NAME_HINTS = ("transparent_cargo_bay", "air_fpv_box")
CARGO_BODY_NAME_HINTS = ("cargo_body", "body")
LEFT_DOOR_NAME_HINTS = ("left_door", "left")
BOTTOM_DOOR_NAME_HINTS = ("bottom_door", "bottom")
TABLE_TOP_PRIM = "/World/layout/scene/SM_TableB3_32612/Cube"
CARGO_TABLE_CLEARANCE = 0.02
ENABLE_PENCIL_PAYLOAD = False
CARGO_LOCAL_PAYLOAD_OFFSET = (0.0, 0.0, 0.072)
PENCIL_PROXY_SCALE = (0.008, 0.008, 0.132)
PENCIL_PROXY_OFFSET = (0.0, 0.0, 0.0)
PENCIL_CHILD_ROTATION_XYZ_DEG = (0.0, 90.0, 0.0)

LIDAR_CONFIG = "OS0_REV7_128ch10hz512res"
MAP_FRAME = "map"
BASE_FRAME = "avoidance_base_link"
LIDAR_FRAME = "avoidance_lidar"
LIDAR_ROOT_PRIM = "/World/quadrotor/body/avoidance_lidar"
LIDAR_PRIM = "/World/quadrotor/body/avoidance_lidar/sensor"
LIDAR_TRANSLATION = (0.0, 0.0, 0.12)
# Keep the LiDAR frame aligned with the USD prim; no extra ROS TF pitch correction.
LIDAR_TF_ROTATION_XYZW = (0.0, 0.0, 0.70710678, 0.70710678)
# Keep EGO/RViz map aligned with Isaac Sim world ENU so planned paths match
# the scene axes. The LiDAR sensor axis correction is handled separately.
WORLD_TO_EGO_MAP_ROTATION_XYZW = (0.0, 0.0, 0.0, 1.0)
POINTCLOUD_TOPIC = "/avoidance/lidar/pointcloud"
EGO_ODOM_TOPIC = "/drone_0_ego_odom"


def world_to_ego_map_position(values) -> tuple[float, float, float]:
    return (float(values[0]), float(values[1]), float(values[2]))


def world_to_ego_map_vector(values) -> tuple[float, float, float]:
    return world_to_ego_map_position(values)


def ego_map_delta_to_world_delta(values) -> tuple[float, float, float]:
    return (float(values[0]), float(values[1]), float(values[2]))


def xyzw_quat_multiply(left, right) -> tuple[float, float, float, float]:
    lx, ly, lz, lw = (float(value) for value in left)
    rx, ry, rz, rw = (float(value) for value in right)
    return (
        (lw * rx) + (lx * rw) + (ly * rz) - (lz * ry),
        (lw * ry) - (lx * rz) + (ly * rw) + (lz * rx),
        (lw * rz) + (lx * ry) - (ly * rx) + (lz * rw),
        (lw * rw) - (lx * rx) - (ly * ry) - (lz * rz),
    )


def world_to_ego_map_quaternion(quat_xyzw) -> tuple[float, float, float, float]:
    return xyzw_quat_multiply(WORLD_TO_EGO_MAP_ROTATION_XYZW, quat_xyzw)


def enable_demo_imports() -> None:
    """Make the Pegasus extension and demo root importable in Isaac --exec runs."""
    import sys

    for path in (DEMO_ROOT, PEGASUS_EXTENSION):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
