#!/usr/bin/env python3
"""Isaac/Pegasus scene setup for the Sunray150 cargo delivery mission with EGO avoidance.

This script is intended to be executed inside the local Isaac Sim full-kit app
with ``isaacsim isaacsim.exp.full.kit --exec integrated_runtime/scene_app.py``.
It loads the Race Map Office, spawns the Sunray150 With Mid360 V2 vehicle,
mounts the cargo bay, places the pencil payload, attaches a hidden OS0 RTX
LiDAR, and publishes the ROS 2 topics needed by EGO-Planner.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import inspect
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Sequence


DEMO_ROOT = Path(__file__).resolve().parents[1]
if str(DEMO_ROOT) not in sys.path:
    sys.path.insert(0, str(DEMO_ROOT))

from integrated_runtime import config as demo_config

demo_config.enable_demo_imports()


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument(
        "--world",
        choices=("X1", "Bobac", "bobac", "create_map"),
        default=None,
        help="Layout to load: X1, Bobac, or create_map.",
    )
    args, _unknown = parser.parse_known_args()
    env_world = os.environ.get("CARGO_DELIVERY_WORLD", "")
    world = args.world or env_world or None
    if world == "bobac":
        world = "Bobac"
    args.world = world
    return args


ARGS = _parse_args()

import carb
import omni.graph.core as og
import omni.kit.app
import omni.kit.commands
import omni.replicator.core as rep
import omni.timeline
import usdrt
from omni.kit.async_engine import run_coroutine
from isaacsim.core.api.world import World
from isaacsim.core.utils.extensions import enable_extension
from pegasus.simulator.logic.backends.px4_mavlink_backend import (
    PX4MavlinkBackend,
    PX4MavlinkBackendConfig,
)
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.thrusters import QuadraticThrustCurve
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, PhysxSchema
import omni.isaac.IsaacSensorSchema as IsaacSensorSchema


def _select_assembled_scene_usd() -> str:
    if ARGS.world == "X1":
        return str(demo_config.X1_ASSEMBLED_SCENE_USD)
    if ARGS.world == "Bobac":
        return str(demo_config.BOBAC_ASSEMBLED_SCENE_USD)
    if ARGS.world == "create_map":
        return str(demo_config.CREATE_MAP_SCENE_USD)
    return ""


WORLD_PRIM = "/World"
LAYOUT_PRIM = "/World/layout"
DRONE_PRIM = "/World/quadrotor"
DRONE_BODY_PRIM = "/World/quadrotor/body"
MERCURY_ROBOT_PRIM = "/World/layout/mercury_x1_final/mercury_x1"
MERCURY_ARM_ARTICULATION_PRIM = f"{MERCURY_ROBOT_PRIM}/base_link"
MERCURY_MAIN_JOINT_GRAPH = f"{MERCURY_ROBOT_PRIM}/Graph/ROS_JointStates"
MERCURY_GRIPPER_JOINT_GRAPH = f"{MERCURY_ROBOT_PRIM}/Graph/ROS_JointStates_01"
MERCURY_RUNTIME_JOINT_GRAPH = f"/World/MercuryJointRuntimeGraph_{os.getpid()}"
MERCURY_RUNTIME_JOINT_GRAPH_PREFIX = "/World/MercuryJointRuntimeGraph"
BOBAC_ROBOT_PRIM = "/World/layout/bobac3_serverbot"
BOBAC_ARM_ARTICULATION_PRIM = (
    f"{BOBAC_ROBOT_PRIM}/bobac3_serverbot/base_footprint/ECO65_B/base_link_arm"
)
BOBAC_GRIPPER_ARTICULATION_PRIM = (
    f"{BOBAC_ROBOT_PRIM}/bobac3_serverbot/base_footprint/ECO65_B/lebai_gripper"
)
BOBAC_BASE_ARTICULATION_PRIM = (
    f"{BOBAC_ROBOT_PRIM}/bobac3_serverbot/base_footprint/base_link"
)
BOBAC_RUNTIME_JOINT_GRAPH = f"/World/BobacJointRuntimeGraph_{os.getpid()}"
BOBAC_RUNTIME_JOINT_GRAPH_PREFIX = "/World/BobacJointRuntimeGraph"
BOBAC_JOINT_COMMAND_TOPIC = "/joint_command"
BOBAC_JOINT_STATES_TOPIC = "/bobac_joint_states"
ENABLE_BOBAC_DIRECT_JOINT_BACKEND = False
X1_ASSEMBLED_SCENE_USD = str(demo_config.X1_ASSEMBLED_SCENE_USD)
BOBAC_ASSEMBLED_SCENE_USD = str(demo_config.BOBAC_ASSEMBLED_SCENE_USD)
BOBAC_ENABLE_PX4 = bool(demo_config.BOBAC_ENABLE_PX4)
BOBAC_PX4_ATTACH_EXISTING = bool(demo_config.BOBAC_PX4_ATTACH_EXISTING)
BOBAC_PX4_PRIM = str(demo_config.BOBAC_PX4_PRIM)
BOBAC_PX4_SPAWN = str(demo_config.BOBAC_PX4_SPAWN)
BOBAC_PX4_SANITIZE_CHILD_MASS = bool(demo_config.BOBAC_PX4_SANITIZE_CHILD_MASS)
BOBAC_PX4_CHILD_COLLIDER_MASS = float(demo_config.BOBAC_PX4_CHILD_COLLIDER_MASS)
BOBAC_PX4_ROTOR_CONSTANT_MULTIPLIER = float(
    demo_config.BOBAC_PX4_ROTOR_CONSTANT_MULTIPLIER
)
BOBAC_PX4_INPUT_SCALING = float(demo_config.BOBAC_PX4_INPUT_SCALING)
BOBAC_PX4_DRONE_USD = str(demo_config.BOBAC_PX4_DRONE_USD)
ASSEMBLED_SCENE_USD = _select_assembled_scene_usd()

CARGO_BAY_PRIM = demo_config.CARGO_BAY_PRIM
CARGO_BODY_PRIM = demo_config.CARGO_BODY_PRIM
LEFT_DOOR_JOINT = demo_config.LEFT_DOOR_JOINT
BOTTOM_DOOR_JOINT = demo_config.BOTTOM_DOOR_JOINT
CARGO_NAME_HINTS = tuple(demo_config.CARGO_NAME_HINTS)
CARGO_BODY_NAME_HINTS = tuple(demo_config.CARGO_BODY_NAME_HINTS)
LEFT_DOOR_NAME_HINTS = tuple(demo_config.LEFT_DOOR_NAME_HINTS)
BOTTOM_DOOR_NAME_HINTS = tuple(demo_config.BOTTOM_DOOR_NAME_HINTS)

PENCIL_PRIM = demo_config.PENCIL_PRIM
PENCIL_CHILD_PRIM = demo_config.PENCIL_CHILD_PRIM
PENCIL_MESH_PRIM = demo_config.PENCIL_MESH_PRIM
PENCIL_PROXY_PRIM = demo_config.PENCIL_PROXY_PRIM

RUNTIME_SCOPE = demo_config.RUNTIME_SCOPE
MOUNT_JOINT_PATH = demo_config.MOUNT_JOINT_PATH
PAYLOAD_LOCK_JOINT_PATH = demo_config.PAYLOAD_LOCK_JOINT_PATH

LIDAR_TRANSLATION = demo_config.LIDAR_TRANSLATION
CARGO_BAY_REFERENCE_HEIGHT = demo_config.CARGO_BAY_REFERENCE_HEIGHT
TABLE_TOP_PRIM = demo_config.TABLE_TOP_PRIM
CARGO_TABLE_CLEARANCE = demo_config.CARGO_TABLE_CLEARANCE
ENABLE_PENCIL_PAYLOAD = demo_config.ENABLE_PENCIL_PAYLOAD
CARGO_LOCAL_PAYLOAD_OFFSET = demo_config.CARGO_LOCAL_PAYLOAD_OFFSET
PENCIL_PROXY_SCALE = demo_config.PENCIL_PROXY_SCALE

MATERIAL_TASK_TARGETS = (
    {
        "id": "red_pencil",
        "label": "red pencil",
        "color": "red",
        "pose_path": "/World/layout/scene/table_material/SM_Pencil_Red5_261",
        "move_path": "/World/layout/scene/table_material/SM_Pencil_Red5_261",
    },
    {
        "id": "white_pencil",
        "label": "white pencil",
        "color": "white",
        "pose_path": "/World/layout/caughting/SM_Pencil_White3_129/SM_Pencil_White3_129",
        "move_path": "/World/layout/caughting/SM_Pencil_White3_129/SM_Pencil_White3_129",
    },
)

MATERIAL_PLACE_WORLD = (3.854000, 0.091000, 0.860000)
MATERIAL_LEFT_DOOR_DROP_PRIM = f"{CARGO_BAY_PRIM}/left_door"
MATERIAL_LEFT_DOOR_DROP_Z_OFFSET = 0.080
MATERIAL_DROP_TARGET_PRIM = "/World/layout/scene/SM_TableB3_325/Cube"
MATERIAL_GRASP_PROXY_SUFFIX = "material_grasp_collision_proxy"
MATERIAL_GRASP_PROXY_SCALE = (0.024, 0.200, 0.024)
MATERIAL_ACTIVE_GRASP_PROXY_SCALE = (0.024, 0.140, 0.024)
MATERIAL_ACTIVE_GRASP_PROXY_TRANSLATION = (0.0, 0.0, 0.0)
MATERIAL_ENABLE_GRASP_PROXY = os.environ.get(
    "MATERIAL_ENABLE_GRASP_PROXY", "0"
).strip().lower() in {"1", "true", "yes", "on"}
MATERIAL_ENABLE_OBJECT_COLLISION_PREP = os.environ.get(
    "MATERIAL_ENABLE_OBJECT_COLLISION_PREP", "0"
).strip().lower() in {"1", "true", "yes", "on"}
MATERIAL_GRASP_PHYSICS_MATERIAL = "/World/material_task/high_friction_grasp_material"
MATERIAL_GRASP_FRICTION = 40.0
MATERIAL_REAL_GRASP_LOCK_JOINT_PATH = "/World/material_task/real_grasp_lock_joint"
MATERIAL_BASE_LOCK_JOINT_PATH = "/World/material_task/manipulation_base_lock_joint"
BOBAC_VALIDATED_GRASP_IDS = ("g_00104",)
BOBAC_VALIDATED_GRASP_MAX_DISTANCE_M = 0.065
MATERIAL_RIGHT_ARM_GRASP_BASE_TARGET = (-0.422, -0.287, 0.095)
MATERIAL_RIGHT_ARM_PLACE_BASE_TARGET = (0.422, -0.287, 1.00)
MATERIAL_BOBAC_STARTUP_PREALIGN_COMPENSATION = (0.705, -0.093, 0.0)
MATERIAL_BASE_REPOSITION_DURATION_SEC = 8.0
MATERIAL_ENABLE_BASE_REPOSITION_FOR_PLACE = False
MATERIAL_ENABLE_DRONE_CARGO_PLACE_ALIGNMENT = False
MATERIAL_REAL_CLOSE_MAX_TCP_OBJECT_DISTANCE = 0.14
MATERIAL_REAL_CLOSE_MAX_HORIZONTAL_DISTANCE = 0.05
MATERIAL_REAL_CLOSE_MIN_VERTICAL_OFFSET = -0.02
MATERIAL_REAL_CLOSE_MAX_VERTICAL_OFFSET = 0.16
MATERIAL_REAL_ARM_GOAL_TOLERANCE = 0.045
MATERIAL_BOBAC_GRIPPER_CLOSE_VALUE = -0.85
MATERIAL_REAL_JOINT5_MIN = 0.28
MATERIAL_REAL_MAX_LIBRARY_DISTANCE = 0.45
MATERIAL_REAL_TCP_CALIBRATION_OFFSET = (0.22, -0.03, -0.16)
MATERIAL_BOBAC_LIBRARY_X_SIGN = -1.0
MATERIAL_BOBAC_LIBRARY_Y_SIGN = 1.0
MATERIAL_BOBAC_LIBRARY_Z_OFFSET = 0.655
MATERIAL_REAL_MEASURED_PRE_Q = (1.0000, -0.8000, -1.0000, 0.0000, 1.4000, -0.0588)
MATERIAL_REAL_MEASURED_TARGET_Q = (1.0000, -0.8000, -1.0000, -0.4000, 1.4000, -0.0588)
MATERIAL_REAL_MEASURED_LIFT_Q = (1.0000, -0.8000, -1.0000, 0.2000, 1.4000, -0.0588)
MATERIAL_PREALIGN_BASE_ON_SETUP = os.environ.get(
    "MATERIAL_PREALIGN_BASE_ON_SETUP", "0"
).strip().lower() not in {"0", "false", "no", "off"}
MATERIAL_PREALIGN_TARGET = os.environ.get(
    "MATERIAL_PREALIGN_TARGET", ""
).strip().lower()
MATERIAL_LOCK_BASE_ON_SETUP = os.environ.get(
    "MATERIAL_LOCK_BASE_ON_SETUP", "0"
).strip().lower() not in {"0", "false", "no", "off"}
RIGHT_GRASP_LIBRARY_CSV = (
    "/home/u-zhuang/ros2_ws/src/isaac/isaac_sim_2026/"
    "grasp_library_current_pose/right_arm_grasp_library.csv"
)
CUROBO_SRC = "/home/u-zhuang/ros2_ws/src/isaac/isaac_sim_2026/curobo/src"
RIGHT_TCP_ROBOT_CFG = (
    "/home/u-zhuang/ros2_ws/src/isaac/isaac_sim_2026/"
    "config/turingstack_x1_2_right_tcp.yml"
)
BOBAC_RIGHT_TCP_ROBOT_CFG = (
    "/home/u-zhuang/ros2_ws/src/isaac/isaac_sim_2026/"
    "config/bobac_eco65_right_tcp.yml"
)
LEFT_TCP_ROBOT_CFG = (
    "/home/u-zhuang/ros2_ws/src/isaac/isaac_sim_2026/"
    "config/turingstack_x1_2_left_tcp.yml"
)
PENCIL_PROXY_OFFSET = demo_config.PENCIL_PROXY_OFFSET
PENCIL_CHILD_ROTATION_XYZ_DEG = demo_config.PENCIL_CHILD_ROTATION_XYZ_DEG

CARGO_BODY_MASS = 0.14
LEFT_DOOR_MASS = 0.02
BOTTOM_DOOR_MASS = 0.025
PENCIL_MASS = 0.01

LEFT_CLOSED_DEG = 0.0
LEFT_OPEN_DEG = -90.0
BOTTOM_CLOSED_DEG = 0.0
BOTTOM_OPEN_DEG = -80.0
CARGO_LEFT_DOOR_OPEN_SEC = 1.0
CARGO_LEFT_DOOR_CLOSE_SEC = 2.0
CARGO_BOTTOM_DOOR_SEC = 1.0

LIDAR_CONFIG = demo_config.LIDAR_CONFIG
MAP_FRAME = demo_config.MAP_FRAME
BASE_FRAME = demo_config.BASE_FRAME
LIDAR_FRAME = demo_config.LIDAR_FRAME
LIDAR_ROOT_PRIM = demo_config.LIDAR_ROOT_PRIM
LIDAR_PRIM = demo_config.LIDAR_PRIM
LIDAR_TF_ROTATION_XYZW = demo_config.LIDAR_TF_ROTATION_XYZW
WORLD_TO_EGO_MAP_ROTATION_XYZW = demo_config.WORLD_TO_EGO_MAP_ROTATION_XYZW
POINTCLOUD_TOPIC = demo_config.POINTCLOUD_TOPIC
EGO_ODOM_TOPIC = demo_config.EGO_ODOM_TOPIC
CLOCK_GRAPH_PATH = "/World/CargoRuntimeClockGraph"
LIDAR_GRAPH_PATH = "/World/CargoRuntimeLidarGraph"
STATE_GRAPH_PATH = "/World/CargoRuntimeStateGraph"
ARM_CAMERA_GRAPH_PATH = "/World/ArmCameraRuntimeRosGraph"
ARM_CAMERA_PRIM = "/World/arm_Camera_runtime"
BOBAC_LIDAR_ROOT_PRIM = f"{BOBAC_BASE_ARTICULATION_PRIM}/avoidance_lidar"
BOBAC_LIDAR_PRIM = f"{BOBAC_LIDAR_ROOT_PRIM}/sensor"
SAVED_RUNTIME_PRIMS = (
    "/World/CargoAvoidanceClockGraph",
    "/World/CargoAvoidanceLidarGraph",
    "/World/CargoAvoidanceStateGraph",
    "/World/ArmCameraRosGraph",
    "/World/arm_Camera",
)
PHYSICS_SCENE_PRIM = "/World/physicsScene"
GPU_FOUND_LOST_AGGREGATE_PAIRS_CAPACITY = 4096
GPU_TOTAL_AGGREGATE_PAIRS_CAPACITY = 4096
ARM_CAMERA_FRAME = "arm_Camera"
ARM_CAMERA_RGB_TOPIC = "/arm_camera/rgb"
ARM_CAMERA_DEPTH_TOPIC = "/arm_camera/depth"
ARM_CAMERA_INFO_TOPIC = "/arm_camera/camera_info"
ARM_CAMERA_RESOLUTION = (640, 480)
ARM_CAMERA_EYE = (3.60, 0.10, 2.20)
ARM_CAMERA_TARGET = (3.60, 0.10, 0.75)
ARM_CAMERA_UP = (0.0, 1.0, 0.0)
DRONE_DOWN_CAMERA_GRAPH_PATH = "/World/DroneDownCameraRuntimeRosGraph"
DRONE_DOWN_CAMERA_PRIM = "/World/drone_down_Camera_runtime"
DRONE_DOWN_CAMERA_FRAME = "drone_down_camera"
DRONE_DOWN_CAMERA_RGB_TOPIC = "/drone/down_camera/rgb"
DRONE_DOWN_CAMERA_INFO_TOPIC = "/drone/down_camera/camera_info"
DRONE_DOWN_CAMERA_RESOLUTION = (320, 240)
DRONE_DOWN_CAMERA_LOCAL_X_OFFSET = 0.00
DRONE_DOWN_CAMERA_LOCAL_Y_OFFSET = 0.00
DRONE_DOWN_CAMERA_LOCAL_Z_OFFSET = -0.42
DRONE_DOWN_CAMERA_FOCAL_LENGTH = 10.0
DRONE_DOWN_CAMERA_APERTURE = 36.0
STALE_IMPORTED_CONTROL_GRAPHS = (
    "/Item_00/ROS_JointStates",
    "/Item_00/ROS_JointStates_left",
    "/Item_00/ActionGraph_drive",
    "/World/ActionGraph_clock",
    "/World/ActionGraph_imu",
    "/World/ActionGraph_lidar",
    "/World/ActionGraph_tf",
    "/World/ROS_JointStates",
    "/World/ROS_JointStates_left",
    "/World/ActionGraph_drive",
    "/World/ROSPublisher_odom2Base_link",
    "/World/Ros_Lidar",
)
STALE_IMPORTED_GRAPH_NAME_HINTS = (
    "actiongraph",
    "ros_lidar",
    "roslidar",
    "rospublisher",
    "ros_publisher",
    "ros2",
)
BOBAC_IMPORTED_GRAPH_ALLOWLIST = {
    "/World/layout/bobac3_serverbot/Graph/ROS_JointStates",
}


def _vec3d(values: Sequence[float]) -> Gf.Vec3d:
    return Gf.Vec3d(float(values[0]), float(values[1]), float(values[2]))


def _vec3f(values: Sequence[float]) -> Gf.Vec3f:
    return Gf.Vec3f(float(values[0]), float(values[1]), float(values[2]))


def _identity_quatf() -> Gf.Quatf:
    return Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0))


def _quatf_from_matrix_rotation(matrix: Gf.Matrix4d) -> Gf.Quatf:
    quat = matrix.ExtractRotation().GetQuat()
    return Gf.Quatf(float(quat.GetReal()), Gf.Vec3f(quat.GetImaginary()))


def _normalize3(v):
    norm = math.sqrt(float(v[0]) ** 2 + float(v[1]) ** 2 + float(v[2]) ** 2)
    if norm < 1.0e-9:
        return (0.0, 0.0, 0.0)
    return (float(v[0]) / norm, float(v[1]) / norm, float(v[2]) / norm)


def _cross3(a, b):
    return (
        float(a[1]) * float(b[2]) - float(a[2]) * float(b[1]),
        float(a[2]) * float(b[0]) - float(a[0]) * float(b[2]),
        float(a[0]) * float(b[1]) - float(a[1]) * float(b[0]),
    )


def _quat_xyzw_to_matrix(q):
    x, y, z, w = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1.0e-9:
        return ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return (
        (1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)),
        (2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)),
        (2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)),
    )


def _transpose3(a):
    return tuple(tuple(float(a[j][i]) for j in range(3)) for i in range(3))


def _matmul3(a, b):
    return tuple(
        tuple(sum(float(a[i][k]) * float(b[k][j]) for k in range(3)) for j in range(3))
        for i in range(3)
    )


def _matvec3(a, v):
    return tuple(sum(float(a[i][k]) * float(v[k]) for k in range(3)) for i in range(3))


def _rotation_matrix_to_quat_xyzw(m):
    m00, m01, m02 = float(m[0][0]), float(m[0][1]), float(m[0][2])
    m10, m11, m12 = float(m[1][0]), float(m[1][1]), float(m[1][2])
    m20, m21, m22 = float(m[2][0]), float(m[2][1]), float(m[2][2])
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1.0e-9:
        return (0.0, 0.0, 0.0, 1.0)
    return (x / norm, y / norm, z / norm, w / norm)


def _arm_camera_optical_world_axes():
    eye = ARM_CAMERA_EYE
    target = ARM_CAMERA_TARGET
    up_hint = _normalize3(ARM_CAMERA_UP)
    forward = _normalize3(
        (
            float(target[0]) - float(eye[0]),
            float(target[1]) - float(eye[1]),
            float(target[2]) - float(eye[2]),
        )
    )
    right = _normalize3(_cross3(forward, up_hint))
    up = _normalize3(_cross3(right, forward))
    down = (-up[0], -up[1], -up[2])
    return right, down, forward


def _ensure_prim(stage, path: str, label: str):
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Missing {label} prim at {path}")
    return prim


def _find_physics_scene(stage):
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Scene):
            return UsdPhysics.Scene(prim)
    return None


def _ensure_physx_scene(stage):
    physics_scene = _find_physics_scene(stage)
    if physics_scene is None:
        physics_scene = UsdPhysics.Scene.Define(stage, PHYSICS_SCENE_PRIM)
        physics_scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        physics_scene.CreateGravityMagnitudeAttr().Set(9.81)

    physx_scene_api = PhysxSchema.PhysxSceneAPI.Apply(physics_scene.GetPrim())
    physx_scene_api.CreateGpuFoundLostAggregatePairsCapacityAttr().Set(
        GPU_FOUND_LOST_AGGREGATE_PAIRS_CAPACITY
    )
    physx_scene_api.CreateGpuTotalAggregatePairsCapacityAttr().Set(
        GPU_TOTAL_AGGREGATE_PAIRS_CAPACITY
    )
    carb.log_warn(
        "Configured PhysX scene "
        f"{physics_scene.GetPrim().GetPath()} with "
        "gpuFoundLostAggregatePairsCapacity="
        f"{GPU_FOUND_LOST_AGGREGATE_PAIRS_CAPACITY}"
    )
    return physics_scene


def _deactivate_stale_imported_control_graphs(stage):
    for path in STALE_IMPORTED_CONTROL_GRAPHS:
        prim = stage.GetPrimAtPath(path)
        if prim and prim.IsValid() and prim.IsActive():
            prim.SetActive(False)
            carb.log_warn(f"Disabled stale imported control graph at {path}")
    for prim in list(stage.Traverse()):
        if not prim or not prim.IsValid() or not prim.IsActive():
            continue
        path = str(prim.GetPath())
        name = prim.GetName().lower()
        type_name = str(prim.GetTypeName()).lower()
        if path in STALE_IMPORTED_CONTROL_GRAPHS:
            continue
        if ARGS.world == "Bobac" and any(
            path == allowed or path.startswith(f"{allowed}/")
            for allowed in BOBAC_IMPORTED_GRAPH_ALLOWLIST
        ):
            carb.log_warn(f"Keeping Bobac imported runtime graph active at {path}")
            continue
        looks_like_graph = (
            "graph" in type_name
            or "ogn" in type_name
            or any(hint in name for hint in STALE_IMPORTED_GRAPH_NAME_HINTS)
        )
        if not looks_like_graph:
            continue
        try:
            prim.SetActive(False)
            carb.log_warn(
                f"Disabled imported runtime graph candidate at {path} "
                f"(type={prim.GetTypeName()})"
            )
        except Exception as exc:
            carb.log_warn(f"Could not disable imported runtime graph {path}: {exc}")


def _set_or_create_bool_attr(attr, creator, value: bool) -> None:
    if attr and attr.IsValid():
        attr.Set(bool(value))
    else:
        creator(bool(value))


def _set_or_add_xform_op(xformable, op_type, value):
    for op in xformable.GetOrderedXformOps():
        if not op.IsInverseOp() and op.GetOpType() == op_type:
            op.Set(value)
            return op
    if op_type == UsdGeom.XformOp.TypeTranslate:
        return xformable.AddTranslateOp().Set(value)
    if op_type == UsdGeom.XformOp.TypeRotateXYZ:
        return xformable.AddRotateXYZOp().Set(value)
    if op_type == UsdGeom.XformOp.TypeOrient:
        return xformable.AddOrientOp().Set(value)
    if op_type == UsdGeom.XformOp.TypeScale:
        return xformable.AddScaleOp().Set(value)
    raise ValueError(f"Unsupported xform op type: {op_type}")


def _set_xform_ops(
    prim,
    translation: Optional[Sequence[float]] = None,
    rotation_xyz_deg: Optional[Sequence[float]] = None,
    orient: Optional[Gf.Quatf] = None,
    scale: Optional[Sequence[float]] = None,
) -> None:
    xformable = UsdGeom.Xformable(prim)
    if translation is not None:
        _set_or_add_xform_op(
            xformable, UsdGeom.XformOp.TypeTranslate, _vec3d(translation)
        )
    if orient is not None:
        _set_or_add_xform_op(xformable, UsdGeom.XformOp.TypeOrient, orient)
    if rotation_xyz_deg is not None:
        _set_or_add_xform_op(
            xformable, UsdGeom.XformOp.TypeRotateXYZ, _vec3f(rotation_xyz_deg)
        )
    if scale is not None:
        _set_or_add_xform_op(xformable, UsdGeom.XformOp.TypeScale, _vec3d(scale))


class CargoBayRuntime:
    def __init__(self, stage):
        self.stage = stage
        self.cargo_bay_prim = CARGO_BAY_PRIM
        self.cargo_body_prim = CARGO_BODY_PRIM
        self.left_door_joint = LEFT_DOOR_JOINT
        self.bottom_door_joint = BOTTOM_DOOR_JOINT
        self.loaded_standalone_cargo = False
        self.payload_follow_path = None
        self.payload_follow_local_position = None
        self._door_motion = {}

    @staticmethod
    def _path_text(prim) -> str:
        return str(prim.GetPath())

    @staticmethod
    def _name_matches(prim, hints: Sequence[str]) -> bool:
        name = prim.GetName().lower()
        path = str(prim.GetPath()).lower()
        return any(hint.lower() in name or hint.lower() in path for hint in hints)

    @staticmethod
    def _has_schema(prim, schema_name: str) -> bool:
        return schema_name in [str(schema) for schema in prim.GetAppliedSchemas()]

    def _descendants(self, root):
        root_path = root.GetPath()
        for prim in self.stage.Traverse():
            if prim.GetPath() == root_path:
                continue
            if prim.GetPath().HasPrefix(root_path):
                yield prim

    def _find_body_under(self, cargo_root):
        named = []
        rigid = []
        for prim in self._descendants(cargo_root):
            if self._name_matches(prim, CARGO_BODY_NAME_HINTS):
                named.append(prim)
            if self._has_schema(prim, "PhysicsRigidBodyAPI"):
                path = str(prim.GetPath()).lower()
                if "door" not in path:
                    rigid.append(prim)
        for prim in named:
            if self._has_schema(prim, "PhysicsRigidBodyAPI"):
                return prim
        if rigid:
            return rigid[0]
        if self._has_schema(cargo_root, "PhysicsRigidBodyAPI"):
            return cargo_root
        return named[0] if named else None

    def _find_revolute_joint_under(self, cargo_root, hints: Sequence[str]):
        candidates = []
        for prim in self._descendants(cargo_root):
            type_name = str(prim.GetTypeName()).lower()
            if "joint" not in type_name and "joint" not in prim.GetName().lower():
                continue
            if self._name_matches(prim, hints):
                candidates.append(prim)
        if not candidates:
            return None
        for prim in candidates:
            if "revolute" in str(prim.GetTypeName()).lower():
                return prim
        return candidates[0]

    def _resolve_cargo_paths(self, cargo_root) -> bool:
        body = self._find_body_under(cargo_root)
        left_joint = self._find_revolute_joint_under(cargo_root, LEFT_DOOR_NAME_HINTS)
        bottom_joint = self._find_revolute_joint_under(cargo_root, BOTTOM_DOOR_NAME_HINTS)
        if body is None or left_joint is None or bottom_joint is None:
            carb.log_warn(
                "Rejected cargo candidate "
                f"{cargo_root.GetPath()}: body={body}, left_joint={left_joint}, "
                f"bottom_joint={bottom_joint}"
            )
            return False
        self.cargo_bay_prim = self._path_text(cargo_root)
        self.cargo_body_prim = self._path_text(body)
        self.left_door_joint = self._path_text(left_joint)
        self.bottom_door_joint = self._path_text(bottom_joint)
        carb.log_warn(
            "Resolved cargo bay: "
            f"root={self.cargo_bay_prim}, body={self.cargo_body_prim}, "
            f"left_joint={self.left_door_joint}, bottom_joint={self.bottom_door_joint}"
        )
        return True

    def resolve_mounted_cargo_bay(self):
        cargo = self.stage.GetPrimAtPath(CARGO_BAY_PRIM)
        if cargo and cargo.IsValid() and self._resolve_cargo_paths(cargo):
            self.loaded_standalone_cargo = False
            carb.log_warn(f"Using mounted cargo bay at {cargo.GetPath()}")
            return True
        drone = self.stage.GetPrimAtPath(DRONE_PRIM)
        if drone and drone.IsValid():
            candidates = []
            for prim in self._descendants(drone):
                if self._name_matches(prim, CARGO_NAME_HINTS):
                    candidates.append(prim)
            candidates.sort(key=lambda prim: len(str(prim.GetPath())))
            for prim in candidates:
                if self._resolve_cargo_paths(prim):
                    self.loaded_standalone_cargo = False
                    carb.log_warn(f"Using mounted cargo bay at {prim.GetPath()}")
                    return True
        return False


    def ensure_pencil_payload_present(self):
        if self.stage.GetPrimAtPath(PENCIL_PRIM).IsValid():
            carb.log_warn(f"Using office pencil payload at {PENCIL_PRIM}")
            return
        raise RuntimeError(f"Missing office pencil payload at {PENCIL_PRIM}")

    def ensure_runtime_scope(self):
        scope = self.stage.GetPrimAtPath(RUNTIME_SCOPE)
        if scope and scope.IsValid():
            return scope
        return UsdGeom.Xform.Define(self.stage, Sdf.Path(RUNTIME_SCOPE)).GetPrim()

    def ensure_payload_rigid_body(self):
        self.ensure_pencil_payload_present()
        payload_path = (
            PENCIL_CHILD_PRIM
            if self.stage.GetPrimAtPath(PENCIL_CHILD_PRIM).IsValid()
            else PENCIL_PRIM
        )
        pencil = _ensure_prim(self.stage, payload_path, "pencil payload")
        rigid_api = UsdPhysics.RigidBodyAPI.Apply(pencil)
        _set_or_create_bool_attr(
            rigid_api.GetRigidBodyEnabledAttr(),
            rigid_api.CreateRigidBodyEnabledAttr,
            True,
        )
        _set_or_create_bool_attr(
            rigid_api.GetKinematicEnabledAttr(),
            rigid_api.CreateKinematicEnabledAttr,
            False,
        )
        mass_api = UsdPhysics.MassAPI.Apply(pencil)
        mass_api.CreateMassAttr(PENCIL_MASS)
        return pencil

    def prepare_pencil_payload(self):
        self.ensure_pencil_payload_present()
        pencil = _ensure_prim(self.stage, PENCIL_PRIM, "pencil payload")
        child = _ensure_prim(self.stage, PENCIL_CHILD_PRIM, "pencil child xform")
        mesh = _ensure_prim(self.stage, PENCIL_MESH_PRIM, "pencil visual mesh")
        child = self._make_prim_editable(PENCIL_CHILD_PRIM) or child
        mesh = self._make_prim_editable(PENCIL_MESH_PRIM) or mesh

        _set_xform_ops(
            child,
            translation=(0.0, 0.0, 0.0),
            rotation_xyz_deg=PENCIL_CHILD_ROTATION_XYZ_DEG,
            scale=(1.0, 1.0, 1.0),
        )

        try:
            if hasattr(mesh, "IsInstanceProxy") and mesh.IsInstanceProxy():
                carb.log_warn(
                    f"Skipping direct collision edits on instance proxy {PENCIL_MESH_PRIM}; "
                    "the runtime cube proxy will provide payload collision"
                )
            else:
                mesh_collision_api = UsdPhysics.CollisionAPI.Apply(mesh)
                _set_or_create_bool_attr(
                    mesh_collision_api.GetCollisionEnabledAttr(),
                    mesh_collision_api.CreateCollisionEnabledAttr,
                    False,
                )
                mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(mesh)
                mesh_collision.CreateApproximationAttr().Set("convexHull")
        except Exception as exc:
            carb.log_warn(
                f"Could not edit pencil visual mesh collision at {PENCIL_MESH_PRIM}: {exc}; "
                "continuing with the runtime cube proxy"
            )

        pencil_rb = UsdPhysics.RigidBodyAPI.Apply(pencil)
        _set_or_create_bool_attr(
            pencil_rb.GetRigidBodyEnabledAttr(),
            pencil_rb.CreateRigidBodyEnabledAttr,
            True,
        )
        _set_or_create_bool_attr(
            pencil_rb.GetKinematicEnabledAttr(),
            pencil_rb.CreateKinematicEnabledAttr,
            False,
        )
        mass_api = UsdPhysics.MassAPI.Apply(pencil)
        mass_api.CreateMassAttr(PENCIL_MASS)

        if self.stage.GetPrimAtPath(PENCIL_PROXY_PRIM).IsValid():
            self.stage.RemovePrim(PENCIL_PROXY_PRIM)
        proxy = UsdGeom.Cube.Define(self.stage, PENCIL_PROXY_PRIM)
        proxy.CreateSizeAttr(1.0)
        proxy_prim = proxy.GetPrim()
        UsdPhysics.CollisionAPI.Apply(proxy_prim).CreateCollisionEnabledAttr(True)
        UsdGeom.Imageable(proxy_prim).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
        _set_xform_ops(
            proxy_prim,
            translation=PENCIL_PROXY_OFFSET,
            scale=PENCIL_PROXY_SCALE,
        )

        self.place_pencil_in_cargo()
        self.lock_payload()

    def place_pencil_in_cargo(self):
        pencil = _ensure_prim(self.stage, PENCIL_PRIM, "pencil payload")
        cargo = _ensure_prim(self.stage, self.cargo_bay_prim, "cargo bay")
        parent_prim = _ensure_prim(self.stage, str(Sdf.Path(PENCIL_PRIM).GetParentPath()), "pencil parent")

        cache = UsdGeom.XformCache()
        cargo_world = cache.GetLocalToWorldTransform(cargo)
        parent_world = cache.GetLocalToWorldTransform(parent_prim)
        target_world = cargo_world.Transform(_vec3d(CARGO_LOCAL_PAYLOAD_OFFSET))
        target_local = parent_world.GetInverse().Transform(target_world)

        _set_xform_ops(
            pencil,
            translation=(target_local[0], target_local[1], target_local[2]),
            orient=_identity_quatf(),
        )

    def lock_payload(self):
        self.ensure_runtime_scope()
        if self.stage.GetPrimAtPath(PAYLOAD_LOCK_JOINT_PATH).IsValid():
            self.stage.RemovePrim(PAYLOAD_LOCK_JOINT_PATH)

        cargo = _ensure_prim(self.stage, self.cargo_body_prim, "cargo body")
        pencil = _ensure_prim(self.stage, PENCIL_PRIM, "pencil payload")

        cache = UsdGeom.XformCache()
        cargo_world = cache.GetLocalToWorldTransform(cargo)
        pencil_world = cache.GetLocalToWorldTransform(pencil)
        self.payload_follow_path = str(pencil.GetPath())
        self.payload_follow_local_position = cargo_world.GetInverse().Transform(
            pencil_world.ExtractTranslation()
        )
        carb.log_warn(
            "Payload runtime follow locked: "
            f"payload={self.payload_follow_path}, cargo={self.cargo_body_prim}"
        )
        self.update_payload_follow()

    def release_payload(self):
        if self.stage.GetPrimAtPath(PAYLOAD_LOCK_JOINT_PATH).IsValid():
            self.stage.RemovePrim(PAYLOAD_LOCK_JOINT_PATH)
        if self.payload_follow_path is not None:
            carb.log_warn(f"Payload runtime follow released: {self.payload_follow_path}")
        self.payload_follow_path = None
        self.payload_follow_local_position = None

    def payload_locked(self) -> bool:
        return bool(
            self.payload_follow_path
            or self.stage.GetPrimAtPath(PAYLOAD_LOCK_JOINT_PATH).IsValid()
        )

    def update_payload_follow(self):
        if not self.payload_follow_path or self.payload_follow_local_position is None:
            return
        cargo = self.stage.GetPrimAtPath(self.cargo_body_prim)
        payload = self.stage.GetPrimAtPath(self.payload_follow_path)
        if not cargo or not cargo.IsValid() or not payload or not payload.IsValid():
            return

        cache = UsdGeom.XformCache()
        cargo_world = cache.GetLocalToWorldTransform(cargo)
        payload_world_position = cargo_world.Transform(self.payload_follow_local_position)

        parent = payload.GetParent()
        local_position = payload_world_position
        if parent and parent.IsValid():
            parent_world = cache.GetLocalToWorldTransform(parent)
            local_position = parent_world.GetInverse().Transform(payload_world_position)

        _set_xform_ops(
            payload,
            translation=(local_position[0], local_position[1], local_position[2]),
        )

    def set_door_angle(self, joint_path: str, angle_deg: float, label: str):
        joint = _ensure_prim(self.stage, joint_path, label)
        attr = joint.GetAttribute("drive:angular:physics:targetPosition")
        if not attr or not attr.IsValid():
            attr = joint.CreateAttribute(
                "drive:angular:physics:targetPosition", Sdf.ValueTypeNames.Float
            )
        attr.Set(float(angle_deg))

    def get_door_angle(self, joint_path: str, default: float = 0.0) -> float:
        joint = self.stage.GetPrimAtPath(joint_path)
        if not joint or not joint.IsValid():
            return float(default)
        attr = joint.GetAttribute("drive:angular:physics:targetPosition")
        if not attr or not attr.IsValid():
            return float(default)
        value = attr.Get()
        return float(default if value is None else value)

    def schedule_door_angle(
        self,
        joint_path: str,
        angle_deg: float,
        label: str,
        duration_sec: float,
    ):
        duration = max(0.0, float(duration_sec))
        start = self.get_door_angle(joint_path, default=float(angle_deg))
        if duration <= 0.0:
            self.set_door_angle(joint_path, angle_deg, label)
            self._door_motion.pop(joint_path, None)
            return
        self._door_motion[joint_path] = {
            "label": label,
            "start": float(start),
            "target": float(angle_deg),
            "started": time.monotonic(),
            "duration": duration,
        }

    def update_door_motion(self):
        if not self._door_motion:
            return
        now = time.monotonic()
        finished = []
        for joint_path, motion in list(self._door_motion.items()):
            duration = max(float(motion["duration"]), 1.0e-6)
            t = max(0.0, min((now - float(motion["started"])) / duration, 1.0))
            s = t * t * (3.0 - 2.0 * t)
            angle = float(motion["start"]) + (
                float(motion["target"]) - float(motion["start"])
            ) * s
            self.set_door_angle(joint_path, angle, str(motion["label"]))
            if t >= 1.0:
                finished.append(joint_path)
        for joint_path in finished:
            self._door_motion.pop(joint_path, None)

    def set_left_angle(self, angle_deg: float):
        self.set_door_angle(self.left_door_joint, angle_deg, "left door joint")

    def set_bottom_angle(self, angle_deg: float):
        self.set_door_angle(self.bottom_door_joint, angle_deg, "bottom door joint")

    def left_open(self):
        self.schedule_door_angle(
            self.left_door_joint,
            LEFT_OPEN_DEG,
            "left door joint",
            CARGO_LEFT_DOOR_OPEN_SEC,
        )

    def left_close(self):
        self.schedule_door_angle(
            self.left_door_joint,
            LEFT_CLOSED_DEG,
            "left door joint",
            CARGO_LEFT_DOOR_CLOSE_SEC,
        )

    def bottom_open(self):
        self.release_payload()
        self.schedule_door_angle(
            self.bottom_door_joint,
            BOTTOM_OPEN_DEG,
            "bottom door joint",
            CARGO_BOTTOM_DOOR_SEC,
        )

    def bottom_close(self):
        self.schedule_door_angle(
            self.bottom_door_joint,
            BOTTOM_CLOSED_DEG,
            "bottom door joint",
            CARGO_BOTTOM_DOOR_SEC,
        )


    def validate_no_dynamic_mesh_collisions(self):
        offenders = []
        for prim in self.stage.Traverse():
            if not prim.IsA(UsdGeom.Mesh):
                continue
            if "PhysicsCollisionAPI" not in [str(x) for x in prim.GetAppliedSchemas()]:
                continue
            rb = prim
            while rb and rb.IsValid():
                if "PhysicsRigidBodyAPI" in [str(x) for x in rb.GetAppliedSchemas()]:
                    break
                rb = rb.GetParent()
            if not rb or not rb.IsValid():
                continue
            kin_attr = rb.GetAttribute("physics:kinematicEnabled")
            is_kinematic = bool(kin_attr.Get()) if kin_attr and kin_attr.IsValid() else False
            approx_attr = prim.GetAttribute("physics:approximation")
            approx = approx_attr.Get() if approx_attr and approx_attr.IsValid() else None
            coll_attr = prim.GetAttribute("physics:collisionEnabled")
            collision_enabled = (
                bool(coll_attr.Get()) if coll_attr and coll_attr.IsValid() else True
            )
            if not is_kinematic and collision_enabled and approx in (None, "none"):
                offenders.append(str(prim.GetPath()))
        if offenders:
            carb.log_warn(
                "Dynamic Mesh collision offenders found; continuing because these "
                "come from referenced environment assets: " + ", ".join(offenders)
            )


class MaterialTaskRuntime:
    """Isaac-side closed-loop material detection, pick, place, and validation."""

    RIGHT_ARM_JOINTS = (
        "joint1_R",
        "joint2_R",
        "joint3_R",
        "joint4_R",
        "joint5_R",
        "joint6_R",
    )
    LEFT_ARM_JOINTS = (
        "joint1_L",
        "joint2_L",
        "joint3_L",
        "joint4_L",
        "joint5_L",
        "joint6_L",
    )
    BOBAC_ARM_JOINTS = (
        "joint_1",
        "joint_2",
        "joint_3",
        "joint_4",
        "joint_5",
        "joint_6",
    )
    ARM_JOINTS = RIGHT_ARM_JOINTS
    GRIPPER_JOINT = "right_gripper_left_joint2"
    GRIPPER_OPEN = 0.60
    GRIPPER_CLOSED = -0.60

    JOINT_POSES = {
        "home": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        "pregrasp": (-0.70, 0.34, -1.18, -0.32, 1.28, 0.46),
        "grasp": (-1.08, 0.30, -1.46, -0.31, 1.45, 0.76),
        "lift": (-1.02, 0.58, -1.18, -0.22, 1.45, 0.76),
        "place": (-0.30, 0.34, -1.10, -0.52, 1.30, 0.18),
    }

    def __init__(self, stage, node, string_msg, joint_state_msg):
        self.stage = stage
        self.node = node
        self.String = string_msg
        self.JointState = joint_state_msg
        self.detections_pub = node.create_publisher(
            string_msg, "/material_task/detections", 10
        )
        self.status_pub = node.create_publisher(string_msg, "/material_task/status", 10)
        self.report_pub = node.create_publisher(string_msg, "/material_task/report", 10)
        self.joint_pub = node.create_publisher(joint_state_msg, "/joint_command", 10)
        self.bobac_joint_state_pub = node.create_publisher(
            joint_state_msg, BOBAC_JOINT_STATES_TOPIC, 10
        )
        node.create_subscription(string_msg, "/material_task/command", self._on_command, 10)
        if ENABLE_BOBAC_DIRECT_JOINT_BACKEND:
            node.create_subscription(
                joint_state_msg,
                BOBAC_JOINT_COMMAND_TOPIC,
                self._on_bobac_joint_command,
                10,
            )
        node.create_subscription(
            joint_state_msg,
            BOBAC_JOINT_STATES_TOPIC,
            self._on_bobac_joint_state,
            10,
        )
        self.phase = "idle"
        self.target = "white_pencil"
        self.started_at = 0.0
        self.phase_started_at = 0.0
        self.start_world = None
        self.current_world = None
        self.report = None
        self._last_joint_publish = 0.0
        self._last_detection_publish = 0.0
        self._last_report_publish = 0.0
        self.real_phase = "idle"
        self.real_target = "white_pencil"
        self.real_started_at = 0.0
        self.real_phase_started_at = 0.0
        self.real_plan = None
        self.real_start_q = None
        self.real_articulation = None
        self.real_action_type = None
        self.bobac_base_hold_active = False
        self.real_arm_indices = None
        self.real_gripper_indices = None
        self.real_gripper_multipliers = None
        self.real_dof_names = []
        self.real_gripper_open = 0.25
        self.real_gripper_close = MATERIAL_BOBAC_GRIPPER_CLOSE_VALUE
        self.real_initial_object_world = None
        self.real_side = "right"
        self.real_arm_joints = self.RIGHT_ARM_JOINTS
        self.real_robot_cfg_path = RIGHT_TCP_ROBOT_CFG
        self.real_last_arm_q = None
        self.bobac_joint_positions = {}
        self.bobac_joint_state_monotonic = 0.0
        self.bobac_direct_articulation = None
        self.bobac_direct_gripper_articulation = None
        self.bobac_direct_action_type = None
        self.bobac_direct_dof_names = []
        self.bobac_direct_dof_index = {}
        self.gripper_sweep = None
        carb.log_warn("Material task runtime ready: /material_task/command")

    def _publish_json(self, publisher, payload):
        msg = self.String()
        msg.data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        publisher.publish(msg)

    def _on_bobac_joint_state(self, msg):
        positions = {}
        for idx, name in enumerate(msg.name):
            if idx >= len(msg.position):
                continue
            try:
                value = float(msg.position[idx])
            except (TypeError, ValueError):
                continue
            positions[str(name)] = value
        if positions:
            self.bobac_joint_positions = positions
            self.bobac_joint_state_monotonic = time.monotonic()

    def _ensure_bobac_direct_articulation(self):
        if self.bobac_direct_articulation is not None:
            return True
        try:
            from isaacsim.core.prims import SingleArticulation
            from isaacsim.core.utils.types import ArticulationAction

            arm = SingleArticulation(
                BOBAC_ARM_ARTICULATION_PRIM,
                name="bobac_direct_joint_command_arm",
            )
            arm.initialize()
            arm_dof_names = list(arm.dof_names)
            self.bobac_direct_articulation = arm
            self.bobac_direct_action_type = ArticulationAction
            self.bobac_direct_dof_names = list(arm_dof_names)
            self.bobac_direct_dof_index = {
                str(name): ("arm", i) for i, name in enumerate(arm_dof_names)
            }
            gripper_dof_names = []
            if self.stage.GetPrimAtPath(BOBAC_GRIPPER_ARTICULATION_PRIM).IsValid():
                gripper = SingleArticulation(
                    BOBAC_GRIPPER_ARTICULATION_PRIM,
                    name="bobac_direct_joint_command_gripper",
                )
                gripper.initialize()
                gripper_dof_names = list(gripper.dof_names)
                self.bobac_direct_gripper_articulation = gripper
                self.bobac_direct_dof_names.extend(gripper_dof_names)
                self.bobac_direct_dof_index.update(
                    {str(name): ("gripper", i) for i, name in enumerate(gripper_dof_names)}
                )
            carb.log_warn(
                "Bobac direct articulation command ready: "
                f"{BOBAC_JOINT_COMMAND_TOPIC} -> {BOBAC_ARM_ARTICULATION_PRIM}, "
                f"arm_dofs={arm_dof_names}, gripper_dofs={gripper_dof_names}"
            )
            return True
        except Exception as exc:
            carb.log_warn(f"Could not initialize Bobac direct articulation command: {exc}")
            return False

    def _publish_bobac_joint_state_snapshot(self):
        if not self.bobac_joint_positions:
            return
        msg = self.JointState()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        ordered = list(self.BOBAC_ARM_JOINTS) + ["gripper_l_joint1", "gripper_r_joint1"]
        names = [name for name in ordered if name in self.bobac_joint_positions]
        for name in self.bobac_joint_positions:
            if name not in names:
                names.append(name)
        msg.name = names
        msg.position = [float(self.bobac_joint_positions[name]) for name in names]
        self.bobac_joint_state_pub.publish(msg)

    def _on_bobac_joint_command(self, msg):
        if not self._ensure_bobac_direct_articulation():
            return
        grouped = {
            "arm": {"indices": [], "positions": []},
            "gripper": {"indices": [], "positions": []},
        }
        updated = {}
        for idx, name in enumerate(msg.name):
            if idx >= len(msg.position):
                continue
            name = str(name)
            if name not in self.bobac_direct_dof_index:
                continue
            try:
                value = float(msg.position[idx])
            except (TypeError, ValueError):
                continue
            group, dof_index = self.bobac_direct_dof_index[name]
            grouped[group]["indices"].append(int(dof_index))
            grouped[group]["positions"].append(value)
            updated[name] = value
        if not updated:
            return
        try:
            import numpy as np

            if grouped["arm"]["indices"]:
                action = self.bobac_direct_action_type(
                    joint_positions=np.array(grouped["arm"]["positions"], dtype=np.float32),
                    joint_indices=np.array(grouped["arm"]["indices"], dtype=np.int32),
                )
                self.bobac_direct_articulation.apply_action(action)
            if grouped["gripper"]["indices"] and self.bobac_direct_gripper_articulation is not None:
                action = self.bobac_direct_action_type(
                    joint_positions=np.array(grouped["gripper"]["positions"], dtype=np.float32),
                    joint_indices=np.array(grouped["gripper"]["indices"], dtype=np.int32),
                )
                self.bobac_direct_gripper_articulation.apply_action(action)
            self.bobac_joint_positions.update(updated)
            self.bobac_joint_state_monotonic = time.monotonic()
            self._publish_bobac_joint_state_snapshot()
        except Exception as exc:
            carb.log_warn(f"Could not apply Bobac direct joint command: {exc}")

    def _publish_status(self, text, **extra):
        payload = {"status": text, "phase": self.phase, "target": self.target}
        payload.update(extra)
        self._publish_json(self.status_pub, payload)

    def _active_material_robot_prim(self):
        if self.stage.GetPrimAtPath(MERCURY_ROBOT_PRIM).IsValid():
            return MERCURY_ROBOT_PRIM
        if self.stage.GetPrimAtPath(BOBAC_ROBOT_PRIM).IsValid():
            return BOBAC_ROBOT_PRIM
        return MERCURY_ROBOT_PRIM

    def _active_material_arm_prim(self):
        if self.stage.GetPrimAtPath(MERCURY_ARM_ARTICULATION_PRIM).IsValid():
            return MERCURY_ARM_ARTICULATION_PRIM
        if self.stage.GetPrimAtPath(BOBAC_ARM_ARTICULATION_PRIM).IsValid():
            return BOBAC_ARM_ARTICULATION_PRIM
        return MERCURY_ARM_ARTICULATION_PRIM

    def _using_bobac_material_arm(self):
        return self._active_material_arm_prim() == BOBAC_ARM_ARTICULATION_PRIM

    def _world_matrix(self, prim_path):
        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return None
        return UsdGeom.XformCache().GetLocalToWorldTransform(prim)

    def _world_position(self, prim_path):
        matrix = self._world_matrix(prim_path)
        if matrix is None:
            return None
        pos = matrix.ExtractTranslation()
        return (float(pos[0]), float(pos[1]), float(pos[2]))

    def _cargo_payload_world_position(self):
        left_door = self._world_position(MATERIAL_LEFT_DOOR_DROP_PRIM)
        if left_door is not None:
            return (
                float(left_door[0]),
                float(left_door[1]),
                float(left_door[2]) + float(MATERIAL_LEFT_DOOR_DROP_Z_OFFSET),
            ), "left_door_top_offset"
        matrix = self._world_matrix(CARGO_BAY_PRIM)
        if matrix is None:
            return tuple(float(v) for v in MATERIAL_PLACE_WORLD), "fallback_material_place_world"
        pos = matrix.Transform(Gf.Vec3d(*CARGO_LOCAL_PAYLOAD_OFFSET))
        return (float(pos[0]), float(pos[1]), float(pos[2])), "cargo_local_payload_offset"

    def _world_quat_xyzw(self, prim_path):
        matrix = self._world_matrix(prim_path)
        if matrix is None:
            return None
        quat = matrix.ExtractRotation().GetQuat()
        imag = quat.GetImaginary()
        return (
            float(imag[0]),
            float(imag[1]),
            float(imag[2]),
            float(quat.GetReal()),
        )

    def _set_kinematic(self, prim_path, enabled=True):
        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return
        try:
            rigid = UsdPhysics.RigidBodyAPI.Apply(prim)
            rigid.CreateKinematicEnabledAttr().Set(bool(enabled))
        except Exception as exc:
            carb.log_warn(f"Could not set kinematic on {prim_path}: {exc}")

    def _set_bobac_base_hold(self, enabled: bool):
        if not self._using_bobac_material_arm():
            return
        prim = self.stage.GetPrimAtPath(BOBAC_BASE_ARTICULATION_PRIM)
        if not prim or not prim.IsValid():
            return
        try:
            rigid = UsdPhysics.RigidBodyAPI.Apply(prim)
            rigid.CreateKinematicEnabledAttr().Set(bool(enabled))
            physx_rigid = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            physx_rigid.CreateDisableGravityAttr().Set(bool(enabled))
            self.bobac_base_hold_active = bool(enabled)
            carb.log_warn(
                f"Bobac base hold {'enabled' if enabled else 'disabled'} on "
                f"{BOBAC_BASE_ARTICULATION_PRIM}"
            )
        except Exception as exc:
            carb.log_warn(f"Could not set Bobac base hold={enabled}: {exc}")

    def _set_world_position(self, prim_path, world_position):
        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            raise RuntimeError(f"Material prim not found: {prim_path}")

        parent = prim.GetParent()
        local_position = Gf.Vec3d(*world_position)
        if parent and parent.IsValid():
            parent_world = UsdGeom.XformCache().GetLocalToWorldTransform(parent)
            local_position = parent_world.GetInverse().Transform(local_position)

        xform = UsdGeom.Xformable(prim)
        translate_op = None
        for op in xform.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                translate_op = op
                break
        if translate_op is None:
            translate_op = xform.AddTranslateOp()
        translate_op.Set(local_position)

    def _translate_world(self, prim_path, delta_world):
        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            raise RuntimeError(f"prim not found: {prim_path}")
        current = self._world_position(prim_path)
        if current is None:
            raise RuntimeError(f"cannot read world position: {prim_path}")
        target_world = tuple(float(current[i]) + float(delta_world[i]) for i in range(3))

        parent = prim.GetParent()
        local_position = Gf.Vec3d(*target_world)
        if parent and parent.IsValid():
            parent_world = UsdGeom.XformCache().GetLocalToWorldTransform(parent)
            local_position = parent_world.GetInverse().Transform(local_position)

        xform = UsdGeom.Xformable(prim)
        translate_op = None
        for op in xform.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                translate_op = op
                break
        if translate_op is None:
            translate_op = xform.AddTranslateOp()
        translate_op.Set(local_position)
        return target_world

    def _target_config(self, target_id=None):
        target_id = (target_id or self.target or "").strip().lower()
        aliases = {
            "red": "red_pencil",
            "red_pen": "red_pencil",
            "红笔": "red_pencil",
            "白笔": "white_pencil",
            "white": "white_pencil",
            "white_pen": "white_pencil",
        }
        target_id = aliases.get(target_id, target_id)
        for cfg in MATERIAL_TASK_TARGETS:
            if cfg["id"] == target_id:
                return cfg
        return MATERIAL_TASK_TARGETS[0]

    def _detect_all(self):
        detections = []
        for cfg in MATERIAL_TASK_TARGETS:
            pose_path = cfg["pose_path"]
            move_path = cfg["move_path"]
            pose = self._world_position(pose_path) or self._world_position(move_path)
            quat = self._world_quat_xyzw(pose_path) or self._world_quat_xyzw(move_path)
            if pose is None or quat is None:
                continue
            detections.append(
                {
                    "id": cfg["id"],
                    "label": cfg["label"],
                    "class": "pencil",
                    "color": cfg["color"],
                    "confidence": 1.0,
                    "source": "isaac_stage_pose_refined",
                    "frame_id": "world",
                    "position": [round(v, 6) for v in pose],
                    "orientation_xyzw": [round(v, 6) for v in quat],
                    "prim_path": pose_path,
                    "move_path": move_path,
                }
            )
        return detections

    def publish_detections(self):
        detections = self._detect_all()
        self._publish_json(
            self.detections_pub,
            {
                "task": "material_detection_and_pose_estimation",
                "all_targets_found": len(detections) == len(MATERIAL_TASK_TARGETS),
                "target_count": len(MATERIAL_TASK_TARGETS),
                "detected_count": len(detections),
                "detections": detections,
            },
        )
        return detections

    def _detect_target(self, target_id=None):
        cfg = self._target_config(target_id)
        detections = self.publish_detections()
        for detection in detections:
            if detection.get("id") == cfg["id"]:
                return cfg, detection, detections
        raise RuntimeError(f"target not detected before grasp: {cfg['id']}")

    def _publish_joint_pose(self, pose_name, gripper_position):
        now = time.monotonic()
        if now - self._last_joint_publish < 0.08:
            return
        self._last_joint_publish = now
        arm = self.JOINT_POSES[pose_name]
        msg = self.JointState()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.name = list(self.ARM_JOINTS) + [self.GRIPPER_JOINT]
        msg.position = [float(v) for v in arm] + [float(gripper_position)]
        self.joint_pub.publish(msg)

    def _publish_bobac_joint_command(self, joint_names, positions):
        msg = self.JointState()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.name = [str(v) for v in joint_names]
        msg.position = [float(v) for v in positions]
        self.joint_pub.publish(msg)

    @staticmethod
    def _lerp(a, b, ratio):
        ratio = max(0.0, min(1.0, ratio))
        return tuple(float(a[i]) + (float(b[i]) - float(a[i])) * ratio for i in range(3))

    @staticmethod
    def _distance(a, b):
        return sum((float(a[i]) - float(b[i])) ** 2 for i in range(3)) ** 0.5

    @staticmethod
    def _quat_from_rpy(roll, pitch, yaw):
        import numpy as np

        cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
        cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
        cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
        return np.array(
            [
                cr * cp * cy + sr * sp * sy,
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
            ],
            dtype=np.float32,
        )

    def _quat_from_orientation_label(self, label):
        if not label:
            return None
        text = str(label).strip()
        if text in {"", "current"}:
            return None
        parts = text.split("_")
        values = {"r": None, "p": None, "y": None}
        for part in parts:
            if len(part) < 2:
                continue
            key = part[0]
            if key not in values:
                continue
            try:
                values[key] = math.radians(float(part[1:]))
            except ValueError:
                return None
        if any(v is None for v in values.values()):
            return None
        return self._quat_from_rpy(values["r"], values["p"], values["y"])

    def _solution_respects_real_limits(self, q_solution):
        if q_solution is None or len(q_solution) < 5:
            return False
        if not all(math.isfinite(float(v)) for v in q_solution):
            return False
        if max(abs(float(v)) for v in q_solution) > math.pi + 0.2:
            return False
        return float(q_solution[4]) >= MATERIAL_REAL_JOINT5_MIN

    def _base_relative_position(self, world_position):
        arm_prim = self._active_material_arm_prim()
        base_matrix = self._world_matrix(arm_prim)
        if base_matrix is None:
            raise RuntimeError(f"base prim not found: {arm_prim}")
        local = base_matrix.GetInverse().Transform(Gf.Vec3d(*world_position))
        if self._using_bobac_material_arm():
            return (float(local[0]), float(local[1]), float(local[2]))
        return (float(local[0]), float(local[1]), float(local[2]))

    def _base_direction_to_world(self, base_delta):
        arm_prim = self._active_material_arm_prim()
        base_matrix = self._world_matrix(arm_prim)
        if base_matrix is None:
            raise RuntimeError(f"base prim not found: {arm_prim}")
        origin = base_matrix.ExtractTranslation()
        moved = base_matrix.Transform(Gf.Vec3d(*base_delta))
        return (
            float(moved[0] - origin[0]),
            float(moved[1] - origin[1]),
            float(moved[2] - origin[2]),
        )

    def _align_base_for_real_grasp(self, target_id=None, desired_base=None):
        cfg = self._target_config(target_id or self.real_target)
        desired = desired_base or MATERIAL_RIGHT_ARM_GRASP_BASE_TARGET
        object_world = self._world_position(cfg["move_path"])
        if object_world is None:
            raise RuntimeError(f"target object pose unavailable: {cfg['move_path']}")
        before_base = self._base_relative_position(object_world)
        if self._using_bobac_material_arm():
            report = {
                "task": "real_base_alignment",
                "target": cfg["id"],
                "mode": "bobac_runtime_teleport_disabled",
                "object_world": [round(v, 6) for v in object_world],
                "desired_object_base": [round(float(v), 6) for v in desired],
                "before_object_base": [round(v, 6) for v in before_base],
                "after_object_base": [round(v, 6) for v in before_base],
                "robot_prim": self._active_material_robot_prim(),
                "note": "Bobac articulation must be aligned before physics reset or by navigation; runtime teleport corrupts PhysX.",
            }
            self._publish_json(self.report_pub, report)
            self._publish_status(
                "real_base_align_skipped_runtime",
                target=cfg["id"],
                before_object_base=[round(v, 6) for v in before_base],
            )
            return report
        base_delta = (
            float(before_base[0]) - float(desired[0]),
            float(before_base[1]) - float(desired[1]),
            0.0,
        )
        world_delta = self._base_direction_to_world(base_delta)
        robot_prim = self._active_material_robot_prim()
        self._translate_world(robot_prim, world_delta)
        after_world = self._world_position(cfg["move_path"])
        after_base = self._base_relative_position(after_world)
        report = {
            "task": "real_base_alignment",
            "target": cfg["id"],
            "object_world": [round(v, 6) for v in object_world],
            "desired_object_base": [round(float(v), 6) for v in desired],
            "before_object_base": [round(v, 6) for v in before_base],
            "base_delta_command": [round(v, 6) for v in base_delta],
            "world_delta_applied": [round(v, 6) for v in world_delta],
            "after_object_base": [round(v, 6) for v in after_base],
            "robot_prim": robot_prim,
        }
        self._publish_json(self.report_pub, report)
        self._publish_status(
            "real_base_aligned",
            target=cfg["id"],
            before_object_base=[round(v, 6) for v in before_base],
            after_object_base=[round(v, 6) for v in after_base],
        )
        return report

    def _remove_material_base_lock(self):
        if self.stage.GetPrimAtPath(MATERIAL_BASE_LOCK_JOINT_PATH).IsValid():
            self.stage.RemovePrim(MATERIAL_BASE_LOCK_JOINT_PATH)

    def _lock_material_base_here(self):
        lock_prim = (
            BOBAC_BASE_ARTICULATION_PRIM
            if self._using_bobac_material_arm()
            and self.stage.GetPrimAtPath(BOBAC_BASE_ARTICULATION_PRIM).IsValid()
            else self._active_material_arm_prim()
        )
        prim = self.stage.GetPrimAtPath(lock_prim)
        if not prim or not prim.IsValid():
            return None
        self._remove_material_base_lock()
        matrix = self._world_matrix(lock_prim)
        if matrix is None:
            return None
        pos = matrix.ExtractTranslation()
        rot = _quatf_from_matrix_rotation(matrix)
        joint = UsdPhysics.FixedJoint.Define(
            self.stage, Sdf.Path(MATERIAL_BASE_LOCK_JOINT_PATH)
        )
        joint.CreateBody1Rel().SetTargets([Sdf.Path(lock_prim)])
        joint.CreateExcludeFromArticulationAttr().Set(True)
        joint.CreateJointEnabledAttr().Set(True)
        joint.CreateCollisionEnabledAttr().Set(False)
        joint.CreateLocalPos0Attr().Set(_vec3f((pos[0], pos[1], pos[2])))
        joint.CreateLocalRot0Attr().Set(rot)
        joint.CreateLocalPos1Attr().Set(_vec3f((0.0, 0.0, 0.0)))
        joint.CreateLocalRot1Attr().Set(_identity_quatf())
        return {
            "joint": MATERIAL_BASE_LOCK_JOINT_PATH,
            "body": lock_prim,
            "world": [round(float(v), 6) for v in pos],
        }

    def _prepare_base_reposition_for_place(self):
        before_place_base = self._base_relative_position(MATERIAL_PLACE_WORLD)
        desired = MATERIAL_RIGHT_ARM_PLACE_BASE_TARGET
        base_delta = (
            float(before_place_base[0]) - float(desired[0]),
            float(before_place_base[1]) - float(desired[1]),
            0.0,
        )
        world_delta = self._base_direction_to_world(base_delta)
        distance = math.sqrt(float(world_delta[0]) ** 2 + float(world_delta[1]) ** 2)
        return {
            "before_place_base": [round(v, 6) for v in before_place_base],
            "desired_place_base": [round(float(v), 6) for v in desired],
            "base_delta_command": [round(v, 6) for v in base_delta],
            "world_delta_total": [float(v) for v in world_delta],
            "world_delta_applied": [0.0, 0.0, 0.0],
            "last_ratio": 0.0,
            "duration": MATERIAL_BASE_REPOSITION_DURATION_SEC,
            "distance_m": round(float(distance), 6),
        }

    def _load_grasp_library(self):
        rows = []
        with open(RIGHT_GRASP_LIBRARY_CSV, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    row["_target"] = (
                        float(row["target_x"]),
                        float(row["target_y"]),
                        float(row["target_z"]),
                    )
                    if self._using_bobac_material_arm():
                        row["_q"] = tuple(float(row[f"q_joint_{i}"]) for i in range(1, 7))
                    else:
                        row["_q"] = tuple(float(row[f"q_joint{i}_R"]) for i in range(1, 7))
                    row["_close"] = float(row.get("close_value", self.real_gripper_close))
                    if self._using_bobac_material_arm():
                        row["_close"] = max(row["_close"], MATERIAL_BOBAC_GRIPPER_CLOSE_VALUE)
                    row["_open"] = float(row.get("open_value", self.real_gripper_open))
                except Exception:
                    continue
                rows.append(row)
        if not rows:
            raise RuntimeError(f"empty grasp library: {RIGHT_GRASP_LIBRARY_CSV}")
        return rows

    def _choose_real_grasp(self, object_base):
        target = (
            float(object_base[0]),
            float(object_base[1]),
            float(object_base[2]) + 0.055,
        )
        rows = self._load_grasp_library()
        if self._using_bobac_material_arm():
            rows = [
                r
                for r in rows
                if all(abs(float(q)) <= math.pi for q in r.get("_q", ()))
            ]
            if not rows:
                raise RuntimeError("no Bobac grasp candidate inside Isaac joint limits")
            validated_rows = [
                r
                for r in rows
                if str(r.get("grasp_id")) in BOBAC_VALIDATED_GRASP_IDS
                and self._distance(r["_target"], target) <= BOBAC_VALIDATED_GRASP_MAX_DISTANCE_M
            ]
            if validated_rows:
                selected = min(
                    validated_rows,
                    key=lambda r: self._distance(r["_target"], target),
                )
                selected["_requested_target"] = target
                selected["_selected_distance"] = self._distance(selected["_target"], target)
                selected["_selection_policy"] = "bobac_isaac_validated_q"
                return selected
        safety_order = ("core", "usable", "edge")
        for safety in safety_order:
            candidates = [r for r in rows if r.get("safety_class") == safety]
            if not candidates:
                continue
            selected = min(
                candidates,
                key=lambda r: self._distance(r["_target"], target),
            )
            selected["_requested_target"] = target
            selected["_selected_distance"] = self._distance(selected["_target"], target)
            return selected
        raise RuntimeError("no grasp candidate found")

    def _solve_real_waypoint_ik(
        self,
        q_current,
        waypoint_positions,
        position_threshold=0.025,
        rotation_threshold=0.18,
        preferred_orientations=None,
    ):
        import numpy as np
        import torch

        if CUROBO_SRC not in sys.path:
            sys.path.insert(0, CUROBO_SRC)

        from curobo.geom.sdf.world import CollisionCheckerType
        from curobo.geom.types import WorldConfig
        from curobo.types.base import TensorDeviceType
        from curobo.types.math import Pose
        from curobo.types.robot import JointState as CuJointState
        from curobo.util_file import load_yaml
        from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

        tensor_args = TensorDeviceType()
        robot_cfg = load_yaml(self.real_robot_cfg_path)["robot_cfg"]
        world_cfg = WorldConfig.from_dict(
            {
                "cuboid": {
                    "far_dummy": {
                        "dims": [0.01, 0.01, 0.01],
                        "pose": [100.0, 100.0, 100.0, 1.0, 0.0, 0.0, 0.0],
                    },
                }
            }
        )
        ik_config = IKSolverConfig.load_from_robot_config(
            robot_cfg,
            world_cfg,
            position_threshold=position_threshold,
            rotation_threshold=rotation_threshold,
            num_seeds=256,
            self_collision_check=True,
            self_collision_opt=True,
            tensor_args=tensor_args,
            use_cuda_graph=False,
            collision_checker_type=CollisionCheckerType.PRIMITIVE,
        )
        ik_solver = IKSolver(ik_config)
        # Bobac's live joints are named joint_1..joint_6, while the reusable
        # cuRobo geometry file may still name the same 6-DOF chain
        # joint1_R..joint6_R. Feed cuRobo its own joint names, then map the
        # returned six joint values back to Bobac when publishing commands.
        current_js_names = list(ik_solver.kinematics.joint_names)
        if len(current_js_names) != len(q_current):
            current_js_names = list(self.real_arm_joints)
        current_js = CuJointState.from_position(
            tensor_args.to_device(np.array(q_current, dtype=np.float32)).view(1, -1),
            joint_names=current_js_names,
        ).get_ordered_joint_state(ik_solver.kinematics.joint_names)

        orientation_candidates = []
        for preferred in preferred_orientations or []:
            quat = self._quat_from_orientation_label(preferred)
            if quat is not None:
                orientation_candidates.append((str(preferred), quat))
        orientation_candidates.extend([
            ("r0_p-30_y45", self._quat_from_rpy(0.0, math.radians(-30.0), math.radians(45.0))),
            ("r0_p-60_y90", self._quat_from_rpy(0.0, math.radians(-60.0), math.radians(90.0))),
            ("r0_p30_y-45", self._quat_from_rpy(0.0, math.radians(30.0), math.radians(-45.0))),
            ("r0_p0_y0", self._quat_from_rpy(0.0, 0.0, 0.0)),
            ("r0_p-30_y0", self._quat_from_rpy(0.0, math.radians(-30.0), 0.0)),
            ("r0_p-60_y0", self._quat_from_rpy(0.0, math.radians(-60.0), 0.0)),
            ("r0_p-30_y-45", self._quat_from_rpy(0.0, math.radians(-30.0), math.radians(-45.0))),
            ("r0_p-60_y-90", self._quat_from_rpy(0.0, math.radians(-60.0), math.radians(-90.0))),
            ("r0_p60_y180", self._quat_from_rpy(0.0, math.radians(60.0), math.radians(180.0))),
        ])

        best_error = None
        best_plan = None
        best_score = float("inf")
        seen_labels = set()
        for label, quat_wxyz in orientation_candidates:
            if label in seen_labels:
                continue
            seen_labels.add(label)
            retract = current_js.position.clone()
            solutions = []
            pos_errors = []
            rot_errors = []
            ok = True
            for pos_np in waypoint_positions:
                goal_pose = Pose(
                    position=tensor_args.to_device(np.array(pos_np, dtype=np.float32)).view(1, 3),
                    quaternion=tensor_args.to_device(quat_wxyz.astype(np.float32)).view(1, 4),
                )
                result = ik_solver.solve_batch(goal_pose, retract_config=retract)
                success = bool(result.success.detach().cpu().numpy().reshape(-1)[0])
                pos_err = float(result.position_error.detach().cpu().numpy().reshape(-1)[0])
                rot_err = float(result.rotation_error.detach().cpu().numpy().reshape(-1)[0])
                q_sol = result.solution.detach().cpu().numpy().reshape(1, -1)[0].astype(float)
                if hasattr(torch, "cuda") and torch.cuda.is_available():
                    torch.cuda.synchronize()
                if not success:
                    ok = False
                    best_error = (label, pos_err, rot_err)
                    break
                if not self._solution_respects_real_limits(q_sol):
                    ok = False
                    best_error = (label, pos_err, rot_err, f"joint5_R={float(q_sol[4]):.6f}")
                    break
                solutions.append(q_sol.tolist())
                pos_errors.append(pos_err)
                rot_errors.append(rot_err)
                retract = tensor_args.to_device(q_sol.astype(np.float32)).view(1, -1)
            if ok:
                score = 0.0
                prev = [float(v) for v in q_current]
                for sol in solutions:
                    deltas = [abs(self._joint_delta(a, b)) for a, b in zip(prev, sol)]
                    if len(deltas) >= 4:
                        deltas[3] *= 4.0
                    if len(deltas) >= 5:
                        deltas[4] *= 1.5
                    score += max(deltas)
                    prev = sol
                if self._using_bobac_material_arm():
                    # In the Bobac USD articulation, approach targets with
                    # joint_4 close to zero repeatedly stall around -0.23 to
                    # -0.32 rad even though cuRobo reports a valid IK. Penalize
                    # those wrist postures so equivalent top-down orientations
                    # with executable joint_4 are selected first.
                    for sol in solutions[1:]:
                        q4 = float(sol[3])
                        if q4 > -0.16:
                            score += 1000.0 + 100.0 * (q4 + 0.16)
                score += 0.1 * sum(float(v) for v in pos_errors)
                plan = {
                    "orientation": label,
                    "joint_names": list(ik_solver.kinematics.joint_names),
                    "solutions": solutions,
                    "position_errors": pos_errors,
                    "rotation_errors": rot_errors,
                }
                if score < best_score:
                    best_score = score
                    best_plan = plan
        if best_plan is not None:
            best_plan["joint_motion_score"] = float(best_score)
            return best_plan
        raise RuntimeError(f"dynamic IK failed; last_error={best_error}")

    def _make_real_dynamic_plan(self, object_base, current_q, selected=None):
        import numpy as np

        if self._using_bobac_material_arm() and selected is not None and "_q" in selected:
            target_q = [float(v) for v in selected["_q"]]
            pre_q = list(target_q)
            lift_q = list(target_q)
            # Live USD TCP sweeps around g_00104 showed joint_2 is the clean
            # vertical lift direction for Bobac at the table.  Do not insert a
            # separate high pregrasp q here: the Bobac USD drive can jam in the
            # intermediate slow-interpolated posture.  Command the official
            # library grasp q directly, then lift with joint_2 -0.20 rad.
            lift_q[1] = max(-3.11, lift_q[1] - 0.20)
            pre_fk = self._curobo_tcp_fk_payload(pre_q)
            grasp_fk = self._curobo_tcp_fk_payload(target_q)
            lift_fk = self._curobo_tcp_fk_payload(lift_q)
            pre_xyz = tuple(float(v) for v in pre_fk["tcp_base"])
            grasp_xyz = tuple(float(v) for v in grasp_fk["tcp_base"])
            lift_xyz = tuple(float(v) for v in lift_fk["tcp_base"])
            return {
                "mode": "bobac_library_q_j2_lift",
                "correction_x": float(grasp_xyz[0]) - float(object_base[0]),
                "correction_y": float(grasp_xyz[1]) - float(object_base[1]),
                "correction_z": float(grasp_xyz[2]) - float(object_base[2]),
                "y_offset": 0.0,
                "pre_xyz": pre_xyz,
                "grasp_xyz": grasp_xyz,
                "lift_xyz": lift_xyz,
                "pre_q": pre_q,
                "target_q": target_q,
                "lift_q": lift_q,
                "direct_joint_targets": True,
                "ik": {
                    "orientation": selected.get("orientation", "library_q"),
                    "position_errors": [
                        float(selected.get("position_error_m", 0.0) or 0.0),
                        float(selected.get("position_error_m", 0.0) or 0.0),
                        float(selected.get("position_error_m", 0.0) or 0.0),
                    ],
                    "rotation_errors": [
                        float(selected.get("rotation_error", 0.0) or 0.0),
                        float(selected.get("rotation_error", 0.0) or 0.0),
                        float(selected.get("rotation_error", 0.0) or 0.0),
                    ],
                    "joint_names": list(self.real_arm_joints),
                },
                "selection_score": 0.0,
            }

        base = np.array(object_base, dtype=float)
        preferred_orientations = []
        if selected is not None and selected.get("orientation"):
            preferred_orientations.append(selected.get("orientation"))
        correction_candidates = []
        if selected is not None and "_target" in selected:
            library_target = np.array(selected["_target"], dtype=float)
            library_corr = library_target - base
            if not self._using_bobac_material_arm():
                calibrated_library_corr = library_corr + np.array(
                    MATERIAL_REAL_TCP_CALIBRATION_OFFSET,
                    dtype=float,
                )
                correction_candidates.append(tuple(calibrated_library_corr.tolist()))
            if self._using_bobac_material_arm():
                deltas = (
                    (0.00, 0.00, -0.080),
                    (0.00, 0.00, -0.060),
                    (0.00, 0.00, -0.040),
                    (0.00, 0.00, -0.030),
                    (0.00, 0.00, -0.020),
                    (0.00, 0.00, -0.010),
                    (0.00, 0.00, 0.000),
                    (0.00, 0.00, 0.010),
                    (0.00, 0.00, 0.015),
                    (0.00, 0.00, 0.020),
                    (0.00, 0.00, 0.030),
                )
            else:
                deltas = (
                    (0.00, 0.00, 0.00),
                    (0.00, 0.04, 0.00),
                    (0.00, -0.04, 0.00),
                    (0.00, 0.00, -0.04),
                    (0.00, 0.00, 0.04),
                    (0.04, 0.00, 0.00),
                    (-0.04, 0.00, 0.00),
                    (0.00, 0.08, -0.04),
                    (0.00, -0.08, -0.04),
                )
            for delta in deltas:
                correction_candidates.append(tuple((library_corr + np.array(delta, dtype=float)).tolist()))

        # Fall back to scans around the measured material origin.
        correction_candidates.extend([
            (0.00, 0.00, -0.080),
            (0.00, 0.06, -0.080),
            (0.00, -0.06, -0.080),
            (-0.04, 0.00, -0.080),
            (0.04, 0.00, -0.080),
            (0.00, 0.10, -0.110),
            (0.00, -0.10, -0.110),
            (0.00, 0.00, -0.120),
            (-0.10, -0.32, 0.04),
            (-0.08, -0.36, 0.02),
            (-0.12, -0.28, 0.00),
            (0.00, 0.00, 0.020),
            (0.00, 0.00, 0.000),
            (0.00, -0.020, 0.020),
            (0.00, 0.020, 0.020),
            (-0.020, 0.000, 0.020),
            (0.020, 0.000, 0.020),
            (0.00, -0.040, 0.020),
            (0.00, 0.040, 0.020),
            (-0.040, 0.000, 0.020),
            (0.040, 0.000, 0.020),
        ])
        lateral_offsets = [0.0] if self._using_bobac_material_arm() else [0.0, -0.02, 0.02]
        errors = []
        viable_plans = []
        for correction_x, correction_y, correction_z in correction_candidates:
            grasp = base + np.array([correction_x, correction_y, correction_z], dtype=float)
            pre = grasp + np.array([0.0, 0.0, 0.10], dtype=float)
            lift = grasp + np.array([0.0, 0.0, 0.13], dtype=float)
            for y_offset in lateral_offsets:
                waypoints = [
                    pre + np.array([0.0, y_offset, 0.0], dtype=float),
                    grasp + np.array([0.0, y_offset, 0.0], dtype=float),
                    lift + np.array([0.0, y_offset, 0.0], dtype=float),
                ]
                try:
                    ik = self._solve_real_waypoint_ik(
                        current_q,
                        waypoints,
                        preferred_orientations=preferred_orientations,
                    )
                    score = float(ik.get("joint_motion_score", 0.0))
                    if self._using_bobac_material_arm():
                        # After the wrist is in the executable region, prefer the
                        # lowest successful grasp height so the physical fingers
                        # actually surround the pencil instead of hovering above it.
                        score += 8.0 * max(0.0, float(grasp[2]) - (float(base[2]) + 0.085))
                        score += 0.2 * abs(float(y_offset))
                    plan = {
                        "mode": "dynamic_ik",
                        "correction_x": float(correction_x),
                        "correction_y": float(correction_y),
                        "correction_z": float(correction_z),
                        "y_offset": float(y_offset),
                        "pre_xyz": tuple(float(v) for v in waypoints[0]),
                        "grasp_xyz": tuple(float(v) for v in waypoints[1]),
                        "lift_xyz": tuple(float(v) for v in waypoints[2]),
                        "pre_q": ik["solutions"][0],
                        "target_q": ik["solutions"][1],
                        "lift_q": ik["solutions"][2],
                        "ik": ik,
                        "selection_score": float(score),
                    }
                    return plan
                except Exception as exc:
                    errors.append(
                        f"corr_x={correction_x:+.3f},corr_y={correction_y:+.3f},corr_z={correction_z:+.3f},"
                        f"y_offset={y_offset:+.3f}: {exc}"
                    )
        raise RuntimeError("dynamic IK failed for all offsets: " + " | ".join(errors))

    def _make_real_world_plan(self, object_world, current_q):
        import numpy as np

        world = np.array(object_world, dtype=float)
        world_corrections = [
            (0.0, 0.0, 0.035),
            (0.0, 0.0, 0.000),
            (0.0, 0.0, 0.020),
            (0.0, 0.0, 0.050),
            (0.0, 0.0, -0.020),
            (0.020, 0.0, 0.035),
            (-0.020, 0.0, 0.035),
            (0.0, 0.020, 0.035),
            (0.0, -0.020, 0.035),
            (0.030, -0.020, 0.035),
            (-0.030, 0.020, 0.035),
            (0.020, 0.180, -0.040),
            (0.020, 0.180, -0.020),
            (0.000, 0.180, -0.040),
            (0.040, 0.180, -0.040),
            (0.020, 0.160, -0.040),
            (0.020, 0.200, -0.040),
            (0.0, 0.225, 0.035),
            (0.0, 0.225, 0.000),
            (0.0, 0.225, -0.020),
            (0.0, 0.205, 0.035),
            (0.0, 0.245, 0.035),
            (0.0, 0.225, -0.071),
            (0.0, 0.225, -0.050),
            (0.0, 0.205, -0.071),
            (0.0, 0.245, -0.071),
            (0.020, 0.225, -0.071),
            (-0.020, 0.225, -0.071),
        ]
        errors = []
        for correction_x, correction_y, correction_z in world_corrections:
            grasp_w = world + np.array([correction_x, correction_y, correction_z], dtype=float)
            pre_w = grasp_w + np.array([0.0, 0.0, 0.10], dtype=float)
            lift_w = grasp_w + np.array([0.0, 0.0, 0.20], dtype=float)
            waypoints = [
                np.array(self._base_relative_position(tuple(pre_w)), dtype=float),
                np.array(self._base_relative_position(tuple(grasp_w)), dtype=float),
                np.array(self._base_relative_position(tuple(lift_w)), dtype=float),
            ]
            try:
                ik = self._solve_real_waypoint_ik(current_q, waypoints)
                return {
                    "mode": "world_dynamic_ik",
                    "correction_x": float(correction_x),
                    "correction_y": float(correction_y),
                    "correction_z": float(correction_z),
                    "y_offset": 0.0,
                    "pre_world": tuple(float(v) for v in pre_w),
                    "grasp_world": tuple(float(v) for v in grasp_w),
                    "lift_world": tuple(float(v) for v in lift_w),
                    "pre_xyz": tuple(float(v) for v in waypoints[0]),
                    "grasp_xyz": tuple(float(v) for v in waypoints[1]),
                    "lift_xyz": tuple(float(v) for v in waypoints[2]),
                    "pre_q": ik["solutions"][0],
                    "target_q": ik["solutions"][1],
                    "lift_q": ik["solutions"][2],
                    "ik": ik,
                }
            except Exception as exc:
                errors.append(
                    f"world_corr=({correction_x:+.3f},{correction_y:+.3f},{correction_z:+.3f}): {exc}"
                )
        raise RuntimeError("world IK failed for all offsets: " + " | ".join(errors))

    def _make_measured_real_plan(self):
        pre_q = [float(v) for v in MATERIAL_REAL_MEASURED_PRE_Q]
        target_q = [float(v) for v in MATERIAL_REAL_MEASURED_TARGET_Q]
        lift_q = [float(v) for v in MATERIAL_REAL_MEASURED_LIFT_Q]
        return {
            "mode": "measured_real_q",
            "correction_x": 0.0,
            "correction_y": 0.0,
            "correction_z": 0.0,
            "y_offset": 0.0,
            "pre_world": (),
            "grasp_world": (),
            "lift_world": (),
            "pre_xyz": (),
            "grasp_xyz": (),
            "lift_xyz": (),
            "pre_q": pre_q,
            "target_q": target_q,
            "lift_q": lift_q,
            "ik": {
                "orientation": "measured_usd_tcp",
                "position_errors": [0.0, 0.0, 0.0],
                "rotation_errors": [0.0, 0.0, 0.0],
                "joint_names": list(self.real_arm_joints),
            },
        }

    def _make_real_place_plan(self, current_q, tcp_place_world, object_place_world=None):
        import numpy as np

        place = np.array(tcp_place_world, dtype=float)
        carry = place + np.array([0.0, 0.0, 0.14], dtype=float)
        retreat = place + np.array([0.0, 0.0, 0.22], dtype=float)
        waypoints = [
            np.array(self._base_relative_position(tuple(carry)), dtype=float),
            np.array(self._base_relative_position(tuple(place)), dtype=float),
            np.array(self._base_relative_position(tuple(retreat)), dtype=float),
        ]
        ik = self._solve_real_waypoint_ik(
            current_q,
            waypoints,
            position_threshold=0.08,
            rotation_threshold=0.35,
        )
        return {
            "mode": "world_place_ik",
            "object_place_world": (
                tuple(float(v) for v in object_place_world)
                if object_place_world is not None
                else None
            ),
            "tcp_carry_world": tuple(float(v) for v in carry),
            "tcp_place_world": tuple(float(v) for v in place),
            "tcp_retreat_world": tuple(float(v) for v in retreat),
            "carry_q": ik["solutions"][0],
            "place_q": ik["solutions"][1],
            "retreat_q": ik["solutions"][2],
            "ik": ik,
        }

    def _ensure_real_articulation(self):
        if self.real_articulation is not None:
            return
        import numpy as np
        if self._using_bobac_material_arm():
            self.real_arm_joints = self.BOBAC_ARM_JOINTS
            self.real_arm_indices = np.arange(len(self.BOBAC_ARM_JOINTS), dtype=np.int32)
            self.real_gripper_indices = np.array(
                [len(self.BOBAC_ARM_JOINTS), len(self.BOBAC_ARM_JOINTS) + 1],
                dtype=np.int32,
            )
            self.real_gripper_multipliers = np.array([1.0, -1.0], dtype=np.float32)
            self.real_dof_names = list(self.BOBAC_ARM_JOINTS) + [
                "gripper_l_joint1",
                "gripper_r_joint1",
            ]
            self.real_articulation = "bobac_ros_joint_command"
            self.real_action_type = None
            carb.log_warn(
                "Bobac real grasp control ready via ROS JointState: "
                f"command={BOBAC_JOINT_COMMAND_TOPIC}, "
                f"state={BOBAC_JOINT_STATES_TOPIC}, "
                f"arm_joints={list(self.BOBAC_ARM_JOINTS)}"
            )
            return
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.types import ArticulationAction

        arm_prim = self._active_material_arm_prim()
        articulation = SingleArticulation(arm_prim, name="material_real_grasp_arm")
        articulation.initialize()
        dof_names = list(articulation.dof_names)
        arm_indices = [articulation.get_dof_index(name) for name in self.real_arm_joints]
        if self.real_side == "left":
            gripper_pairs = [
                ("left_gripper_left_joint2", 1.0),
                ("left_gripper_right_join2", -1.0),
                ("left_gripper_left_finger_joint", -1.0),
                ("left_gripper_right_finger_joint", 1.0),
                ("gripper_left_outer_finger_joint", -1.0),
            ]
        else:
            if self._using_bobac_material_arm():
                gripper_pairs = [
                    ("gripper_l_joint1", 1.0),
                    ("gripper_r_joint1", -1.0),
                ]
            else:
                gripper_pairs = [
                    ("right_gripper_left_joint2", 1.0),
                    ("right_gripper_right_join2", -1.0),
                    ("right_gripper_left_finger_joint", -1.0),
                    ("right_gripper_right_finger_joint", 1.0),
                    ("right_gripper_left_outer_finger_joint", -1.0),
                    ("gripper_right_outer_finger_joint", -1.0),
                ]
        gripper_indices = []
        gripper_multipliers = []
        for name, multiplier in gripper_pairs:
            for idx, dof_name in enumerate(dof_names):
                if dof_name == name and idx not in gripper_indices:
                    gripper_indices.append(idx)
                    gripper_multipliers.append(multiplier)
        if not gripper_indices:
            fallback_pairs = [
                ("gripper_controller", 1.0),
                ("gripper01_base_to_gripper_left2", 1.0),
                ("gripper01_base_to_gripper_right2", -1.0),
            ]
            for name, multiplier in fallback_pairs:
                if name in dof_names:
                    gripper_indices.append(articulation.get_dof_index(name))
                    gripper_multipliers.append(multiplier)
        if not gripper_indices:
            # Last resort: use right gripper DOFs without sign knowledge.
            for idx, name in enumerate(dof_names):
                low = name.lower()
                if "gripper" in low and self.real_side in low:
                    gripper_indices.append(idx)
                    gripper_multipliers.append(1.0)
        if not gripper_indices:
            raise RuntimeError(f"no gripper DOF found; dof_names={dof_names}")
        self.real_articulation = articulation
        self.real_action_type = ArticulationAction
        self.real_arm_indices = np.array(arm_indices, dtype=np.int32)
        self.real_gripper_indices = np.array(gripper_indices, dtype=np.int32)
        self.real_gripper_multipliers = np.array(gripper_multipliers, dtype=np.float32)
        self.real_dof_names = dof_names
        carb.log_warn(
            "Real grasp articulation ready: "
            f"arm_prim={arm_prim}, "
            f"arm_indices={arm_indices}, gripper_indices={gripper_indices}, "
            f"gripper_multipliers={gripper_multipliers}, "
            f"dof_names={dof_names}"
        )

    def _real_joint_positions(self, indices=None):
        self._ensure_real_articulation()
        if self._using_bobac_material_arm():
            if (
                self.bobac_joint_positions
                and time.monotonic() - self.bobac_joint_state_monotonic <= 2.0
            ):
                if indices is None:
                    names = self.real_dof_names
                else:
                    names = [
                        self.real_dof_names[int(idx)]
                        for idx in list(indices)
                        if int(idx) < len(self.real_dof_names)
                    ]
                values = []
                for name in names:
                    if name not in self.bobac_joint_positions:
                        break
                    values.append(self.bobac_joint_positions[name])
                else:
                    return values
            return None
        return self.real_articulation.get_joint_positions(joint_indices=indices)

    def _safe_real_arm_positions(self, fallback=None):
        raw = self._real_joint_positions(self.real_arm_indices)
        if raw is None:
            if self.real_last_arm_q is not None:
                return [float(v) for v in self.real_last_arm_q]
            if fallback is not None:
                return [float(v) for v in fallback]
            return [0.0 for _ in self.real_arm_indices]
        current = [float(v) for v in raw]
        if self._using_bobac_material_arm():
            current = [self._wrap_revolute_joint(v) for v in current]
        if not all(math.isfinite(v) for v in current):
            if self.real_last_arm_q is not None:
                return [float(v) for v in self.real_last_arm_q]
            if fallback is not None:
                return [float(v) for v in fallback]
            return [0.0 for _ in self.real_arm_indices]
        self.real_last_arm_q = current
        return current

    def _wrap_revolute_joint(self, value):
        return math.atan2(math.sin(float(value)), math.cos(float(value)))

    def _joint_delta(self, current, target):
        if self._using_bobac_material_arm():
            return math.atan2(math.sin(float(target) - float(current)), math.cos(float(target) - float(current)))
        return float(target) - float(current)

    def _apply_real_positions(self, positions, indices):
        import numpy as np

        self._ensure_real_articulation()
        if self._using_bobac_material_arm():
            idx_values = [int(v) for v in list(indices)]
            pos_values = [float(v) for v in list(positions)]
            if (
                self.real_arm_indices is not None
                and len(idx_values) == len(self.real_arm_indices)
                and all(int(a) == int(b) for a, b in zip(idx_values, list(self.real_arm_indices)))
            ):
                self._publish_bobac_joint_command(self.BOBAC_ARM_JOINTS, pos_values)
                return
            if (
                self.real_gripper_indices is not None
                and len(idx_values) == len(self.real_gripper_indices)
                and all(int(a) == int(b) for a, b in zip(idx_values, list(self.real_gripper_indices)))
            ):
                joint_names = [
                    self.real_dof_names[int(idx)]
                    for idx in idx_values
                    if int(idx) < len(self.real_dof_names)
                ]
                self._publish_bobac_joint_command(joint_names, pos_values)
                return
        action = self.real_action_type(
            joint_positions=np.array(positions, dtype=np.float32),
            joint_indices=indices,
        )
        self.real_articulation.apply_action(action)
        if self._using_bobac_material_arm():
            idx_values = [int(v) for v in list(indices)]
            pos_values = [float(v) for v in list(positions)]
            if (
                self.real_arm_indices is not None
                and len(idx_values) == len(self.real_arm_indices)
                and all(int(a) == int(b) for a, b in zip(idx_values, list(self.real_arm_indices)))
            ):
                self.real_last_arm_q = [self._wrap_revolute_joint(v) for v in pos_values]

    def _real_gripper_command_positions(self, value):
        return [float(value) * float(m) for m in self.real_gripper_multipliers]

    def _curobo_tcp_fk_payload(self, q_values):
        import numpy as np
        import torch

        if CUROBO_SRC not in sys.path:
            sys.path.insert(0, CUROBO_SRC)

        from curobo.cuda_robot_model.cuda_robot_model import (
            CudaRobotModel,
            CudaRobotModelConfig,
        )
        from curobo.types.base import TensorDeviceType
        from curobo.util_file import load_yaml

        tensor_args = TensorDeviceType()
        robot_cfg = load_yaml(self.real_robot_cfg_path)["robot_cfg"]
        model_cfg = CudaRobotModelConfig.from_data_dict(robot_cfg, tensor_args=tensor_args)
        model = CudaRobotModel(model_cfg)
        q_np = np.array([float(v) for v in q_values], dtype=np.float32)
        state = model.get_state(tensor_args.to_device(q_np).view(1, -1))
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
        pos = state.ee_position.detach().cpu().numpy()[0].astype(float)
        quat = state.ee_quaternion.detach().cpu().numpy()[0].astype(float)
        return {
            "robot_cfg_path": self.real_robot_cfg_path,
            "joint_names": list(model.joint_names),
            "tcp_base": [round(float(v), 9) for v in pos.tolist()],
            "tcp_quat_wxyz": [round(float(v), 9) for v in quat.tolist()],
        }

    def _fk_alignment_payload(self, q_values=None, object_world=None):
        self.real_side = "right"
        if self._using_bobac_material_arm():
            self.real_arm_joints = self.BOBAC_ARM_JOINTS
            self.real_robot_cfg_path = BOBAC_RIGHT_TCP_ROBOT_CFG
        else:
            self.real_arm_joints = self.RIGHT_ARM_JOINTS
            self.real_robot_cfg_path = RIGHT_TCP_ROBOT_CFG
        self._ensure_real_articulation()
        if q_values is None:
            q_values = self._safe_real_arm_positions()
        q_values = [float(v) for v in q_values]
        measured = self._measure_gripper_tcp(self.real_side, object_world)
        curobo_fk = self._curobo_tcp_fk_payload(q_values)
        measured_base = measured.get("tcp_local_arm") or measured.get("tcp_local_link7")
        error = None
        if measured_base is not None:
            error = self._distance(tuple(measured_base), tuple(curobo_fk["tcp_base"]))
        return {
            "task": "bobac_curobo_usd_tcp_alignment",
            "target": self.real_target,
            "arm_side": self.real_side,
            "q": [round(float(v), 9) for v in q_values],
            "measured_tcp": measured,
            "curobo_fk": curobo_fk,
            "tcp_error_m": round(float(error), 9) if error is not None else None,
            "pass_2cm": bool(error is not None and error < 0.02),
        }

    def _start_gripper_sweep(self, values=None):
        self.real_side = "right"
        self.real_arm_joints = self.RIGHT_ARM_JOINTS
        self.real_robot_cfg_path = RIGHT_TCP_ROBOT_CFG
        self._ensure_real_articulation()
        if values is None:
            values = [0.25, 0.0, -0.40, -0.60, -0.70, -0.85, -1.00, -1.25, -1.50]
        self.gripper_sweep = {
            "values": [float(v) for v in values],
            "index": 0,
            "results": [],
            "phase_started_at": time.monotonic(),
            "applied": False,
        }
        self._publish_status("gripper_sweep_started", values=self.gripper_sweep["values"])

    def _update_gripper_sweep(self):
        if self.gripper_sweep is None:
            return
        sweep = self.gripper_sweep
        now = time.monotonic()
        if sweep["index"] >= len(sweep["values"]):
            payload = {
                "task": "gripper_sweep",
                "ok": True,
                "results": sweep["results"],
            }
            self._publish_json(self.report_pub, payload)
            self._publish_status("gripper_sweep_completed")
            self.gripper_sweep = None
            return
        value = sweep["values"][sweep["index"]]
        if not sweep["applied"]:
            self._apply_real_positions(
                self._real_gripper_command_positions(value),
                self.real_gripper_indices,
            )
            sweep["phase_started_at"] = now
            sweep["applied"] = True
            self._publish_status("gripper_sweep_applied", value=value)
            return
        self._apply_real_positions(
            self._real_gripper_command_positions(value),
            self.real_gripper_indices,
        )
        if now - sweep["phase_started_at"] < 1.5:
            return
        measured = self._measure_gripper_tcp(self.real_side)
        result = {
            "command": round(float(value), 6),
            "finger_distance_m": measured.get("finger_distance_m"),
            "tcp_world": measured.get("tcp_world"),
            "indices": [int(v) for v in self.real_gripper_indices],
            "multipliers": [float(v) for v in self.real_gripper_multipliers],
        }
        sweep["results"].append(result)
        self._publish_json(
            self.report_pub,
            {
                "task": "gripper_sweep_step",
                **result,
            },
        )
        sweep["index"] += 1
        sweep["applied"] = False

    def _prim_positions_by_name(self, names):
        wanted = set(names)
        result = {}
        for prim in self.stage.Traverse():
            name = prim.GetName()
            if name not in wanted:
                continue
            matrix = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
            pos = matrix.ExtractTranslation()
            result[name] = (float(pos[0]), float(pos[1]), float(pos[2]))
        return result

    def _first_prim_by_name(self, name):
        for prim in self.stage.Traverse():
            if prim.GetName() == name:
                return prim
        return None

    def _measure_bobac_gripper_tcp(self, object_world=None):
        root_prefix = f"{BOBAC_ROBOT_PRIM}/bobac3_serverbot/base_footprint/ECO65_B".lower()
        cache = UsdGeom.XformCache()
        fixed_left = None
        fixed_right = None
        for prim in self.stage.Traverse():
            path = str(prim.GetPath())
            low = path.lower()
            if root_prefix not in low:
                continue
            if low.endswith("/lebai_gripper/gripper_l_link2"):
                fixed_left = prim
            elif low.endswith("/lebai_gripper/gripper_r_link2"):
                fixed_right = prim
            if fixed_left is not None and fixed_right is not None:
                break
        if fixed_left is not None and fixed_right is not None:
            try:
                finger_world = [
                    cache.GetLocalToWorldTransform(fixed_left).ExtractTranslation(),
                    cache.GetLocalToWorldTransform(fixed_right).ExtractTranslation(),
                ]
                tcp_world = Gf.Vec3d(
                    (finger_world[0][0] + finger_world[1][0]) * 0.5,
                    (finger_world[0][1] + finger_world[1][1]) * 0.5,
                    (finger_world[0][2] + finger_world[1][2]) * 0.5,
                )
                parent = self.stage.GetPrimAtPath(BOBAC_ARM_ARTICULATION_PRIM)
                parent_inv = cache.GetLocalToWorldTransform(parent).GetInverse()
                tcp_local = parent_inv.Transform(tcp_world)
                target_dist = (
                    self._distance(tuple(tcp_world), object_world)
                    if object_world is not None
                    else float("nan")
                )
                return {
                    "side": "bobac",
                    "ok": True,
                    "parent": BOBAC_ARM_ARTICULATION_PRIM,
                    "base": BOBAC_ARM_ARTICULATION_PRIM,
                    "fingers": [str(fixed_left.GetPath()), str(fixed_right.GetPath())],
                    "tcp_world": [round(float(v), 9) for v in tcp_world],
                    "tcp_local_arm": [round(float(v), 9) for v in tcp_local],
                    "finger_distance_m": round(
                        float(self._distance(finger_world[0], finger_world[1])), 9
                    ),
                    "target_distance_m": round(float(target_dist), 9),
                    "candidate_count": 2,
                    "selection": "fixed_gripper_l_r_link2",
                }
            except Exception as exc:
                carb.log_warn(f"Fixed Bobac gripper TCP measurement failed: {exc}")

        candidates = []
        for prim in self.stage.Traverse():
            path = str(prim.GetPath())
            low = path.lower()
            if root_prefix not in low:
                continue
            if not ("lebai" in low or "gripper" in low or "finger" in low):
                continue
            if any(skip in low for skip in ("/looks", "/materials", "/joints/")):
                continue
            try:
                matrix = cache.GetLocalToWorldTransform(prim)
                pos = matrix.ExtractTranslation()
            except Exception:
                continue
            xyz = (float(pos[0]), float(pos[1]), float(pos[2]))
            is_left = ("gripper_l_" in low or "/gripper_l" in low or "left" in low)
            is_right = ("gripper_r_" in low or "/gripper_r" in low or "right" in low)
            is_tip_like = (
                "finger" in low
                or "link2" in low
                or "finger2" in low
                or "joint_finger" in low
            )
            candidates.append(
                {
                    "prim": prim,
                    "path": path,
                    "name": prim.GetName(),
                    "type": prim.GetTypeName(),
                    "xyz": xyz,
                    "is_left": is_left,
                    "is_right": is_right,
                    "is_tip_like": is_tip_like,
                }
            )
        left = [c for c in candidates if c["is_left"]]
        right = [c for c in candidates if c["is_right"]]
        pairs = []
        for a in left:
            for b in right:
                dist = self._distance(a["xyz"], b["xyz"])
                if dist < 0.002 or dist > 0.25:
                    continue
                mid = tuple((a["xyz"][i] + b["xyz"][i]) * 0.5 for i in range(3))
                target_dist = (
                    self._distance(mid, object_world)
                    if object_world is not None
                    else abs(float(mid[2]) - 0.34)
                )
                score = target_dist
                if a["is_tip_like"] and b["is_tip_like"]:
                    score -= 0.05
                pairs.append((score, target_dist, dist, mid, a, b))
        if not pairs:
            return {
                "side": "bobac",
                "ok": False,
                "missing": {
                    "arm_root": BOBAC_ARM_ARTICULATION_PRIM,
                    "candidate_count": len(candidates),
                    "left_count": len(left),
                    "right_count": len(right),
                },
                "nearest_candidates": [
                    {
                        "path": c["path"],
                        "type": c["type"],
                        "world": [round(v, 6) for v in c["xyz"]],
                    }
                    for c in sorted(
                        candidates,
                        key=lambda c: self._distance(c["xyz"], object_world)
                        if object_world is not None
                        else 0.0,
                    )[:20]
                ],
            }
        pairs.sort(key=lambda row: row[0])
        _, target_dist, finger_distance, tcp_world, left_prim, right_prim = pairs[0]
        parent = self.stage.GetPrimAtPath(BOBAC_ARM_ARTICULATION_PRIM)
        parent_inv = cache.GetLocalToWorldTransform(parent).GetInverse()
        tcp_vec = Gf.Vec3d(float(tcp_world[0]), float(tcp_world[1]), float(tcp_world[2]))
        tcp_local = parent_inv.Transform(tcp_vec)
        try:
            tcp_base = self._base_relative_position(
                (float(tcp_world[0]), float(tcp_world[1]), float(tcp_world[2]))
            )
        except Exception:
            tcp_base = None
        return {
            "side": "bobac",
            "ok": True,
            "parent": BOBAC_ARM_ARTICULATION_PRIM,
            "base": BOBAC_ARM_ARTICULATION_PRIM,
            "fingers": [left_prim["path"], right_prim["path"]],
            "tcp_world": [round(float(v), 9) for v in tcp_world],
            "tcp_base": (
                [round(float(v), 9) for v in tcp_base]
                if tcp_base is not None
                else None
            ),
            "tcp_local_arm": [round(float(v), 9) for v in tcp_local],
            "finger_distance_m": round(float(finger_distance), 9),
            "target_distance_m": round(float(target_dist), 9),
            "candidate_count": len(candidates),
        }

    def _measure_gripper_tcp(self, side, object_world=None):
        if side == "left":
            parent_name = "link7_L"
            base_name = "left_gripper_base"
            finger_names = ("left_gripper_left1", "left_gripper_right1")
        else:
            parent_name = "link7_R"
            base_name = "right_gripper_base"
            finger_names = ("right_gripper_left1", "right_gripper_right1")

        parent = self._first_prim_by_name(parent_name)
        base = self._first_prim_by_name(base_name)
        fingers = [self._first_prim_by_name(name) for name in finger_names]
        if parent is None or base is None or any(f is None for f in fingers):
            if self._using_bobac_material_arm():
                return self._measure_bobac_gripper_tcp(object_world)
            return {
                "side": side,
                "ok": False,
                "missing": {
                    "parent": parent_name if parent is None else None,
                    "base": base_name if base is None else None,
                    "fingers": [name for name, prim in zip(finger_names, fingers) if prim is None],
                },
            }

        cache = UsdGeom.XformCache()
        parent_inv = cache.GetLocalToWorldTransform(parent).GetInverse()
        finger_world = [cache.GetLocalToWorldTransform(f).ExtractTranslation() for f in fingers]
        base_world = cache.GetLocalToWorldTransform(base).ExtractTranslation()
        tcp_world = Gf.Vec3d(
            (finger_world[0][0] + finger_world[1][0]) * 0.5,
            (finger_world[0][1] + finger_world[1][1]) * 0.5,
            (finger_world[0][2] + finger_world[1][2]) * 0.5,
        )
        try:
            tcp_base = self._base_relative_position(
                (float(tcp_world[0]), float(tcp_world[1]), float(tcp_world[2]))
            )
        except Exception:
            tcp_base = None
        tcp_local = parent_inv.Transform(tcp_world)
        base_local = parent_inv.Transform(base_world)
        return {
            "side": side,
            "ok": True,
            "parent": str(parent.GetPath()),
            "base": str(base.GetPath()),
            "fingers": [str(f.GetPath()) for f in fingers],
            "tcp_world": [round(float(v), 9) for v in tcp_world],
            "tcp_base": (
                [round(float(v), 9) for v in tcp_base]
                if tcp_base is not None
                else None
            ),
            "tcp_local_link7": [round(float(v), 9) for v in tcp_local],
            "base_local_link7": [round(float(v), 9) for v in base_local],
            "finger_distance_m": round(float(self._distance(finger_world[0], finger_world[1])), 9),
        }

    def _measure_tcp_payload(self):
        target_world = None
        try:
            cfg = self._target_config(self.real_target or "red_pencil")
            target_world = self._world_position(cfg["move_path"])
        except Exception:
            target_world = None
        payload = {
            "task": "measure_gripper_tcp",
            "left": self._measure_gripper_tcp("left", target_world),
            "right": self._measure_gripper_tcp("right", target_world),
        }
        self._publish_json(self.report_pub, payload)
        return payload

    def _gripper_contact_diagnostics(self, object_world):
        if self.real_side == "left":
            names = [
                "left_gripper_left_joint2",
                "left_gripper_right_join2",
                "left_gripper_left_finger_joint",
                "left_gripper_right_finger_joint",
                "gripper_left_outer_finger_joint",
            ]
        else:
            names = [
                "right_gripper_left_joint2",
                "right_gripper_right_join2",
                "right_gripper_left_finger_joint",
                "right_gripper_right_finger_joint",
                "right_gripper_left_outer_finger_joint",
                "gripper_right_outer_finger_joint",
            ]
        positions = self._prim_positions_by_name(names)
        distances = {}
        if object_world is not None:
            for name, pos in positions.items():
                distances[name] = self._distance(pos, object_world)
        return {
            "gripper_link_world": {
                name: [round(v, 6) for v in pos] for name, pos in positions.items()
            },
            "object_to_gripper_link_m": {
                name: round(float(dist), 6) for name, dist in distances.items()
            },
        }

    def _probe_gripper_prims(self, target_id=None):
        cfg = self._target_config(target_id or self.real_target)
        object_world = self._world_position(cfg["move_path"])
        rows = []
        for prim in self.stage.Traverse():
            path = str(prim.GetPath())
            low = path.lower()
            if not ("gripper" in low or "finger" in low or "lebai" in low):
                continue
            matrix = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
            pos = matrix.ExtractTranslation()
            xyz = (float(pos[0]), float(pos[1]), float(pos[2]))
            row = {
                "path": path,
                "name": prim.GetName(),
                "type": prim.GetTypeName(),
                "world": [round(v, 6) for v in xyz],
            }
            if object_world is not None:
                row["distance_to_target_m"] = round(self._distance(xyz, object_world), 6)
            rows.append(row)
        rows.sort(key=lambda r: r.get("distance_to_target_m", 999.0))
        payload = {
            "task": "probe_gripper_prims",
            "target": cfg["id"],
            "target_world": [round(v, 6) for v in object_world] if object_world else None,
            "count": len(rows),
            "nearest": rows[:80],
        }
        self._publish_json(self.report_pub, payload)
        return payload

    def _real_move_fraction(self, start, target, duration):
        elapsed = time.monotonic() - self.real_phase_started_at
        ratio = min(1.0, max(0.0, elapsed / max(duration, 1.0e-6)))
        smooth = ratio * ratio * (3.0 - 2.0 * ratio)
        cmd = [(1.0 - smooth) * float(a) + smooth * float(b) for a, b in zip(start, target)]
        return cmd, ratio

    def _joint_max_error(self, current, target):
        if current is None or target is None:
            return float("inf")
        return max(abs(self._joint_delta(a, b)) for a, b in zip(current, target))

    def _joint_error_payload(self, current, target):
        rows = []
        max_error = 0.0
        for name, now_v, target_v in zip(self.real_arm_joints, current or [], target or []):
            error = self._joint_delta(now_v, target_v)
            max_error = max(max_error, abs(error))
            rows.append(
                {
                    "joint": name,
                    "current": round(float(now_v), 6),
                    "target": round(float(target_v), 6),
                    "error": round(float(error), 6),
                }
            )
        return {
            "max_error": round(float(max_error), 6),
            "joints": rows,
        }

    def _ensure_grasp_physics_material(self):
        material = UsdShade.Material.Define(self.stage, MATERIAL_GRASP_PHYSICS_MATERIAL)
        physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        physics_material.CreateStaticFrictionAttr().Set(MATERIAL_GRASP_FRICTION)
        physics_material.CreateDynamicFrictionAttr().Set(MATERIAL_GRASP_FRICTION)
        physics_material.CreateRestitutionAttr().Set(0.0)
        return material

    def _bind_high_friction_gripper(self):
        material = self._ensure_grasp_physics_material()
        side_hint = f"{self.real_side}_"
        robot_prim = self._active_material_robot_prim()
        bound = 0
        for prim in self.stage.Traverse():
            path = str(prim.GetPath())
            low = path.lower()
            if not path.startswith(robot_prim):
                continue
            if side_hint not in low and not self._using_bobac_material_arm():
                continue
            if "gripper" not in low and "finger" not in low:
                continue
            try:
                UsdShade.MaterialBindingAPI(prim).Bind(material)
                bound += 1
            except Exception:
                continue
        carb.log_warn(f"Bound high-friction grasp material to {bound} {self.real_side} gripper prims")
        return bound

    def _remove_real_grasp_lock(self):
        if self.stage.GetPrimAtPath(MATERIAL_REAL_GRASP_LOCK_JOINT_PATH).IsValid():
            self.stage.RemovePrim(MATERIAL_REAL_GRASP_LOCK_JOINT_PATH)

    def _create_real_grasp_lock(self, move_path, gripper_link_path):
        self._remove_real_grasp_lock()
        gripper = _ensure_prim(self.stage, gripper_link_path, "real grasp gripper link")
        material = _ensure_prim(self.stage, move_path, "real grasp material")

        cache = UsdGeom.XformCache()
        gripper_world = cache.GetLocalToWorldTransform(gripper)
        material_world = cache.GetLocalToWorldTransform(material)
        material_world_pos = material_world.ExtractTranslation()
        material_pos_in_gripper = gripper_world.GetInverse().Transform(material_world_pos)

        joint = UsdPhysics.FixedJoint.Define(
            self.stage, Sdf.Path(MATERIAL_REAL_GRASP_LOCK_JOINT_PATH)
        )
        joint.CreateBody0Rel().SetTargets([Sdf.Path(gripper_link_path)])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(move_path)])
        joint.CreateExcludeFromArticulationAttr().Set(True)
        joint.CreateJointEnabledAttr().Set(True)
        joint.CreateCollisionEnabledAttr().Set(False)
        joint.CreateLocalPos0Attr().Set(_vec3f(material_pos_in_gripper))
        joint.CreateLocalRot0Attr().Set(_identity_quatf())
        joint.CreateLocalPos1Attr().Set(_vec3f((0.0, 0.0, 0.0)))
        joint.CreateLocalRot1Attr().Set(_identity_quatf())
        return {
            "joint_path": MATERIAL_REAL_GRASP_LOCK_JOINT_PATH,
            "body0": gripper_link_path,
            "body1": move_path,
            "material_pos_in_gripper": [
                round(float(material_pos_in_gripper[i]), 6) for i in range(3)
            ],
        }

    def _create_material_grasp_proxy(
        self,
        move_path,
        scale=MATERIAL_GRASP_PROXY_SCALE,
        translation=(0.0, 0.0, 0.0),
    ):
        proxy_path = f"{move_path}/{MATERIAL_GRASP_PROXY_SUFFIX}"
        if not MATERIAL_ENABLE_GRASP_PROXY:
            if self.stage.GetPrimAtPath(proxy_path).IsValid():
                self.stage.RemovePrim(proxy_path)
            return None
        if self.stage.GetPrimAtPath(proxy_path).IsValid():
            self.stage.RemovePrim(proxy_path)
        proxy = UsdGeom.Cube.Define(self.stage, proxy_path)
        proxy.CreateSizeAttr(1.0)
        proxy_prim = proxy.GetPrim()
        UsdPhysics.CollisionAPI.Apply(proxy_prim).CreateCollisionEnabledAttr(True)
        UsdShade.MaterialBindingAPI(proxy_prim).Bind(self._ensure_grasp_physics_material())
        _set_xform_ops(
            proxy_prim,
            translation=translation,
            scale=scale,
        )
        UsdGeom.Imageable(proxy_prim).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
        return proxy_path

    def _prepare_real_object_physics(self, move_path):
        prim = self.stage.GetPrimAtPath(move_path)
        if not prim or not prim.IsValid():
            raise RuntimeError(f"material prim not found: {move_path}")
        try:
            rigid = UsdPhysics.RigidBodyAPI.Apply(prim)
            rigid.CreateRigidBodyEnabledAttr().Set(True)
            rigid.CreateKinematicEnabledAttr().Set(True)
            UsdPhysics.MassAPI.Apply(prim).CreateMassAttr().Set(0.005)
            try:
                physx_rigid = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
                physx_rigid.CreateLinearDampingAttr().Set(8.0)
                physx_rigid.CreateAngularDampingAttr().Set(8.0)
                physx_rigid.CreateSolverPositionIterationCountAttr().Set(64)
                physx_rigid.CreateSolverVelocityIterationCountAttr().Set(16)
            except Exception as physx_exc:
                carb.log_warn(f"Could not tune material rigid body damping: {physx_exc}")

            for child in Usd.PrimRange(prim):
                if child == prim:
                    continue
                if child.GetTypeName() == "Mesh":
                    try:
                        UsdPhysics.CollisionAPI.Apply(child).CreateCollisionEnabledAttr(True)
                        UsdPhysics.MeshCollisionAPI.Apply(child).CreateApproximationAttr().Set("convexHull")
                    except Exception as child_exc:
                        carb.log_warn(f"Could not enable material mesh collision on {child.GetPath()}: {child_exc}")

            self._create_material_grasp_proxy(move_path)
        except Exception as exc:
            carb.log_warn(f"Could not prepare real object physics on {move_path}: {exc}")

    def _start_real_grasp(self, target):
        if self.real_phase != "idle":
            self._publish_status("real_grasp_busy", real_phase=self.real_phase)
            return
        self._remove_real_grasp_lock()
        cfg, detection, detections = self._detect_target(target)
        self.real_target = cfg["id"]
        object_world = tuple(float(v) for v in detection["position"])
        if len(object_world) != 3:
            raise RuntimeError(f"detected target pose invalid: {detection}")
        object_base = self._base_relative_position(object_world)
        base_alignment = {
            "task": "real_base_alignment",
            "mode": "startup_prealigned",
            "target": cfg["id"],
            "desired_object_base": [
                round(float(v), 6) for v in MATERIAL_RIGHT_ARM_GRASP_BASE_TARGET
            ],
            "current_object_base": [round(v, 6) for v in object_base],
            "robot_prim": self._active_material_robot_prim(),
        }
        self.real_side = "right"
        self.real_arm_joints = self.BOBAC_ARM_JOINTS if self._using_bobac_material_arm() else self.RIGHT_ARM_JOINTS
        self.real_robot_cfg_path = BOBAC_RIGHT_TCP_ROBOT_CFG if self._using_bobac_material_arm() else RIGHT_TCP_ROBOT_CFG
        self.real_articulation = None
        self._ensure_real_articulation()
        self._set_bobac_base_hold(True)
        # A USD FixedJoint on the Bobac base also constrains the nested ECO65
        # articulation in Isaac 4.5: joint_2 stalls around 0.33 rad while the
        # same q reaches cleanly without the lock.  Keep Bobac mobile during
        # arm execution and rely on measured base/object alignment instead.
        base_lock = None if self._using_bobac_material_arm() else self._lock_material_base_here()
        current_q = self._safe_real_arm_positions()
        self._bind_high_friction_gripper()
        selected = self._choose_real_grasp(object_base)
        selected_distance = float(selected.get("_selected_distance", 999.0))
        if selected_distance > MATERIAL_REAL_MAX_LIBRARY_DISTANCE:
            raise RuntimeError(
                "target outside arm grasp workspace after detection: "
                f"object_base={[round(float(v), 6) for v in object_base]}, "
                f"nearest_library_target={[round(float(v), 6) for v in selected['_target']]}, "
                f"distance_m={selected_distance:.3f}. Navigate/relocalize before moving the arm."
            )
        try:
            dynamic_plan = self._make_real_dynamic_plan(object_base, current_q, selected)
        except Exception as base_exc:
            carb.log_warn(
                f"Library-guided base-frame material IK failed, falling back to world-frame IK: {base_exc}"
            )
            dynamic_plan = self._make_real_world_plan(object_world, current_q)
        target_q = [float(v) for v in dynamic_plan["target_q"]]
        lift_q = [float(v) for v in dynamic_plan["lift_q"]]
        retreat_q = current_q
        self.real_plan = {
            "cfg": cfg,
            "object_world": object_world,
            "object_base": object_base,
            "selected": selected,
            "dynamic_plan": dynamic_plan,
            "pre_q": [float(v) for v in dynamic_plan["pre_q"]],
            "target_q": target_q,
            "lift_q": lift_q,
            "retreat_q": retreat_q,
            "open": float(selected["_open"]),
            "close": float(selected["_close"]),
            "base_lock": base_lock,
        }
        self.real_initial_object_world = object_world
        self._prepare_real_object_physics(cfg["move_path"])
        self.real_started_at = time.monotonic()
        self.real_phase_started_at = self.real_started_at
        self.real_phase = "real_open"
        self._publish_json(
            self.report_pub,
            {
                "task": "real_material_grasp",
                "status": "started",
                "target": self.real_target,
                "arm_side": self.real_side,
                "object_world": [round(v, 6) for v in object_world],
                "object_base": [round(v, 6) for v in object_base],
                "detection_source": detection.get("source"),
                "detection_prim_path": detection.get("prim_path"),
                "detected_count": len(detections),
                "base_alignment": base_alignment,
                "selected_grasp_id": selected.get("grasp_id"),
                "selected_safety": selected.get("safety_class"),
                "plan_mode": dynamic_plan["mode"],
                "ik_orientation": dynamic_plan["ik"]["orientation"],
                "ik_y_offset": dynamic_plan["y_offset"],
                "ik_correction_x": dynamic_plan.get("correction_x", 0.0),
                "ik_correction_y": dynamic_plan.get("correction_y", 0.0),
                "ik_correction_z": dynamic_plan.get("correction_z", 0.0),
                "ik_selection_score": round(float(dynamic_plan.get("selection_score", 0.0)), 6),
                "ik_position_errors": [round(float(v), 6) for v in dynamic_plan["ik"]["position_errors"]],
                "pre_base": [round(v, 6) for v in dynamic_plan.get("pre_xyz", ())],
                "grasp_base": [round(v, 6) for v in dynamic_plan.get("grasp_xyz", ())],
                "lift_base": [round(v, 6) for v in dynamic_plan.get("lift_xyz", ())],
                "pre_world": [round(v, 6) for v in dynamic_plan.get("pre_world", ())],
                "grasp_world": [round(v, 6) for v in dynamic_plan.get("grasp_world", ())],
                "lift_world": [round(v, 6) for v in dynamic_plan.get("lift_world", ())],
                "requested_tcp_base": [round(v, 6) for v in selected["_requested_target"]],
                "library_tcp_base": [round(v, 6) for v in selected["_target"]],
                "library_distance_m": round(float(selected["_selected_distance"]), 6),
                "dof_names": self.real_dof_names,
            },
        )
        self._publish_status(
            "real_grasp_started",
            object_base=[round(v, 6) for v in object_base],
            selected_grasp_id=selected.get("grasp_id"),
        )

    def _on_command(self, msg):
        parts = msg.data.strip().split()
        if not parts:
            return
        command = parts[0].lower()
        target = parts[1].lower() if len(parts) > 1 else self.target
        if command in {"detect", "detections", "识别"}:
            detections = self.publish_detections()
            self._publish_status("detections_published", detected_count=len(detections))
            return
        if command in {"run", "pick", "grasp", "抓取"}:
            if self.phase != "idle":
                self._publish_status("busy")
                return
            cfg = self._target_config(target)
            self.target = cfg["id"]
            self.started_at = time.monotonic()
            self.phase_started_at = self.started_at
            self.phase = "pregrasp"
            self.report = None
            self.start_world = self._world_position(cfg["move_path"])
            self.current_world = self.start_world
            self._set_kinematic(cfg["move_path"], True)
            detections = self.publish_detections()
            self._publish_status(
                "started",
                detected_count=len(detections),
                place_world=[round(v, 6) for v in MATERIAL_PLACE_WORLD],
            )
            return
        if command in {"real_grasp", "real", "true_grasp", "真实抓取"}:
            try:
                self._start_real_grasp(target)
            except Exception as exc:
                self._set_bobac_base_hold(False)
                self.real_phase = "idle"
                self._publish_json(
                    self.report_pub,
                    {
                        "task": "real_material_grasp",
                        "ok": False,
                        "target": target,
                        "error": str(exc),
                    },
                )
                carb.log_error(f"Could not start real grasp: {exc}")
            return
        if command in {"probe_gripper", "probe", "探针"}:
            payload = self._probe_gripper_prims(target)
            self._publish_status("probe_gripper_published", count=payload["count"])
            return
        if command in {"measure_tcp", "tcp", "测量tcp"}:
            payload = self._measure_tcp_payload()
            self._publish_status("measure_tcp_published")
            return
        if command in {"fk_check", "check_fk", "tcp_fk", "校验fk"}:
            cfg = self._target_config(target)
            object_world = self._world_position(cfg["move_path"])
            payload = self._fk_alignment_payload(object_world=object_world)
            self._publish_json(self.report_pub, payload)
            self._publish_status(
                "fk_check_published",
                tcp_error_m=payload.get("tcp_error_m"),
                pass_2cm=payload.get("pass_2cm"),
            )
            return
        if command in {"move_grasp_q", "debug_grasp_q", "库q"}:
            grasp_id = parts[1] if len(parts) > 1 else "g_00104"
            selected = None
            for row in self._load_grasp_library():
                if row.get("grasp_id") == grasp_id:
                    selected = row
                    break
            if selected is None:
                self._publish_json(
                    self.report_pub,
                    {
                        "task": "move_grasp_q",
                        "ok": False,
                        "grasp_id": grasp_id,
                        "error": "grasp_id not found",
                    },
                )
                return
            self.real_side = "right"
            self.real_arm_joints = (
                self.BOBAC_ARM_JOINTS
                if self._using_bobac_material_arm()
                else self.RIGHT_ARM_JOINTS
            )
            self.real_robot_cfg_path = (
                BOBAC_RIGHT_TCP_ROBOT_CFG
                if self._using_bobac_material_arm()
                else RIGHT_TCP_ROBOT_CFG
            )
            self._ensure_real_articulation()
            q_values = [float(v) for v in selected["_q"]]
            self._apply_real_positions(q_values, self.real_arm_indices)
            payload = self._fk_alignment_payload(q_values=q_values)
            payload.update(
                {
                    "task": "move_grasp_q",
                    "ok": True,
                    "grasp_id": grasp_id,
                    "library_target_base": [
                        round(float(v), 9) for v in selected["_target"]
                    ],
                    "library_orientation": selected.get("orientation"),
                }
            )
            self._publish_json(self.report_pub, payload)
            self._publish_status(
                "move_grasp_q_commanded",
                grasp_id=grasp_id,
                library_target_base=payload["library_target_base"],
            )
            return
        if command in {"align_base", "base_align", "对准底盘"}:
            payload = self._align_base_for_real_grasp(target)
            self._publish_status("align_base_published", after_object_base=payload["after_object_base"])
            return
        if command in {"prepare_object", "prepare_material", "准备物料"}:
            cfg = self._target_config(target)
            self._prepare_real_object_physics(cfg["move_path"])
            object_world = self._world_position(cfg["move_path"])
            payload = {
                "task": "prepare_real_object_physics",
                "target": cfg["id"],
                "move_path": cfg["move_path"],
                "kinematic": True,
                "object_world": [round(v, 6) for v in object_world] if object_world else None,
                "object_base": (
                    [round(v, 6) for v in self._base_relative_position(object_world)]
                    if object_world
                    else None
                ),
            }
            self._publish_json(self.report_pub, payload)
            self._publish_status("real_object_prepared", target=cfg["id"])
            return
        if command in {"object_dynamic", "dynamic_object", "物料动态"}:
            cfg = self._target_config(target)
            self._set_kinematic(cfg["move_path"], False)
            object_world = self._world_position(cfg["move_path"])
            payload = {
                "task": "set_real_object_dynamic",
                "target": cfg["id"],
                "move_path": cfg["move_path"],
                "kinematic": False,
                "object_world": [round(v, 6) for v in object_world] if object_world else None,
            }
            self._publish_json(self.report_pub, payload)
            self._publish_status("real_object_dynamic", target=cfg["id"])
            return
        if command in {"object_kinematic", "kinematic_object", "物料运动学"}:
            cfg = self._target_config(target)
            self._set_kinematic(cfg["move_path"], True)
            object_world = self._world_position(cfg["move_path"])
            payload = {
                "task": "set_real_object_kinematic",
                "target": cfg["id"],
                "move_path": cfg["move_path"],
                "kinematic": True,
                "object_world": [round(v, 6) for v in object_world] if object_world else None,
            }
            self._publish_json(self.report_pub, payload)
            self._publish_status("real_object_kinematic", target=cfg["id"])
            return
        if command in {"scene_points", "measure_scene_points", "场景点位"}:
            points = []
            white_pencil_world = None
            for cfg in MATERIAL_TASK_TARGETS:
                world = self._world_position(cfg["move_path"])
                if cfg["id"] == "white_pencil":
                    white_pencil_world = world
                points.append(
                    {
                        "id": cfg["id"],
                        "world": [round(v, 6) for v in world] if world else None,
                        "base": (
                            [round(v, 6) for v in self._base_relative_position(world)]
                            if world
                            else None
                        ),
                        "move_path": cfg["move_path"],
                    }
                )
            place_world, place_source = self._cargo_payload_world_position()
            cargo_body_world = self._world_position(CARGO_BODY_PRIM)
            drop_target_world = self._world_position(MATERIAL_DROP_TARGET_PRIM)
            drop_error_xy = None
            drop_score_band = None
            if white_pencil_world and drop_target_world:
                drop_error_xy = math.hypot(
                    float(white_pencil_world[0]) - float(drop_target_world[0]),
                    float(white_pencil_world[1]) - float(drop_target_world[1]),
                )
                if drop_error_xy <= 0.2:
                    drop_score_band = "0.2m"
                elif drop_error_xy <= 0.4:
                    drop_score_band = "0.4m"
                elif drop_error_xy <= 0.6:
                    drop_score_band = "0.6m"
                else:
                    drop_score_band = "miss"
            payload = {
                "task": "scene_points",
                "points": points,
                "place_world": [round(v, 6) for v in place_world],
                "place_base": [round(v, 6) for v in self._base_relative_position(place_world)],
                "place_source": place_source,
                "cargo_bay_prim": CARGO_BAY_PRIM,
                "cargo_body_prim": CARGO_BODY_PRIM,
                "cargo_body_world": (
                    [round(v, 6) for v in cargo_body_world] if cargo_body_world else None
                ),
                "cargo_body_base": (
                    [round(v, 6) for v in self._base_relative_position(cargo_body_world)]
                    if cargo_body_world
                    else None
                ),
                "cargo_local_payload_offset": [round(float(v), 6) for v in CARGO_LOCAL_PAYLOAD_OFFSET],
                "drop_target_prim": MATERIAL_DROP_TARGET_PRIM,
                "drop_target_world": (
                    [round(v, 6) for v in drop_target_world] if drop_target_world else None
                ),
                "white_pencil_world": (
                    [round(v, 6) for v in white_pencil_world] if white_pencil_world else None
                ),
                "drop_error_xy": round(drop_error_xy, 6) if drop_error_xy is not None else None,
                "drop_score_band": drop_score_band,
            }
            self._publish_json(self.report_pub, payload)
            self._publish_status("scene_points_published")
            return
        if command in {"set_gripper", "gripper"}:
            value = float(parts[1]) if len(parts) > 1 else self.real_gripper_close
            self.real_side = "right"
            self.real_arm_joints = self.RIGHT_ARM_JOINTS
            self.real_robot_cfg_path = RIGHT_TCP_ROBOT_CFG
            self._ensure_real_articulation()
            self._apply_real_positions(
                self._real_gripper_command_positions(value),
                self.real_gripper_indices,
            )
            self._publish_status("gripper_commanded", value=value)
            return
        if command in {"sweep_gripper", "gripper_sweep"}:
            values = None
            if len(parts) > 1:
                values = [float(v) for v in " ".join(parts[1:]).replace(",", " ").split()]
            self._start_gripper_sweep(values)
            return
        if command in {"reset", "stop", "取消"}:
            self.phase = "idle"
            self.real_phase = "idle"
            self._set_bobac_base_hold(False)
            self._remove_real_grasp_lock()
            self.gripper_sweep = None
            self._publish_joint_pose("home", self.GRIPPER_OPEN)
            self._publish_status("idle")

    def update(self):
        now = time.monotonic()
        if now - self._last_detection_publish > 1.0:
            self._last_detection_publish = now
            self.publish_detections()
        self._update_gripper_sweep()
        self._update_real_grasp()
        if self.phase == "idle":
            return

        cfg = self._target_config(self.target)
        elapsed = now - self.phase_started_at
        move_path = cfg["move_path"]
        start = self.start_world or self._world_position(move_path) or MATERIAL_PLACE_WORLD
        lift = (start[0], start[1], max(start[2] + 0.26, 1.05))
        carry = (MATERIAL_PLACE_WORLD[0], MATERIAL_PLACE_WORLD[1], MATERIAL_PLACE_WORLD[2] + 0.22)

        try:
            if self.phase == "pregrasp":
                self._publish_joint_pose("pregrasp", self.GRIPPER_OPEN)
                if elapsed >= 2.0:
                    self.phase = "grasp"
                    self.phase_started_at = now
                    self._publish_status("pregrasp_reached")
            elif self.phase == "grasp":
                self._publish_joint_pose("grasp", self.GRIPPER_OPEN)
                if elapsed >= 1.0:
                    self.phase = "close"
                    self.phase_started_at = now
                    self._publish_status("closing_gripper")
            elif self.phase == "close":
                self._publish_joint_pose("grasp", self.GRIPPER_CLOSED)
                if elapsed >= 0.8:
                    self.phase = "lift"
                    self.phase_started_at = now
                    self._publish_status("object_attached")
            elif self.phase == "lift":
                self._publish_joint_pose("lift", self.GRIPPER_CLOSED)
                pos = self._lerp(start, lift, elapsed / 1.6)
                self._set_world_position(move_path, pos)
                if elapsed >= 1.6:
                    self.phase = "carry"
                    self.phase_started_at = now
                    self._publish_status("lifted")
            elif self.phase == "carry":
                self._publish_joint_pose("place", self.GRIPPER_CLOSED)
                pos = self._lerp(lift, carry, elapsed / 2.0)
                self._set_world_position(move_path, pos)
                if elapsed >= 2.0:
                    self.phase = "place"
                    self.phase_started_at = now
                    self._publish_status("above_place")
            elif self.phase == "place":
                self._publish_joint_pose("place", self.GRIPPER_CLOSED)
                pos = self._lerp(carry, MATERIAL_PLACE_WORLD, elapsed / 1.0)
                self._set_world_position(move_path, pos)
                if elapsed >= 1.0:
                    self.phase = "release"
                    self.phase_started_at = now
                    self._publish_status("placed")
            elif self.phase == "release":
                self._publish_joint_pose("place", self.GRIPPER_OPEN)
                self._set_world_position(move_path, MATERIAL_PLACE_WORLD)
                if elapsed >= 0.8:
                    final_pos = self._world_position(move_path) or MATERIAL_PLACE_WORLD
                    error = self._distance(final_pos, MATERIAL_PLACE_WORLD)
                    ok = error <= 0.05
                    self.report = {
                        "task": "material_pick_place",
                        "ok": ok,
                        "target": self.target,
                        "detected_count": len(self._detect_all()),
                        "required_count": len(MATERIAL_TASK_TARGETS),
                        "place_world": [round(v, 6) for v in MATERIAL_PLACE_WORLD],
                        "final_world": [round(v, 6) for v in final_pos],
                        "final_error_m": round(error, 6),
                        "elapsed_sec": round(now - self.started_at, 3),
                        "notes": "Isaac runtime executed pick, carry, place and validation.",
                    }
                    self._publish_json(self.report_pub, self.report)
                    self._publish_status("completed" if ok else "failed", error_m=error)
                    self.phase = "idle"
        except Exception as exc:
            self.phase = "idle"
            self._publish_json(
                self.report_pub,
                {
                    "task": "material_pick_place",
                    "ok": False,
                    "target": self.target,
                    "error": str(exc),
                },
            )
            carb.log_error(f"Material task failed: {exc}")

    def _update_real_grasp(self):
        if self.real_phase == "idle":
            return
        if not self.real_plan:
            self.real_phase = "idle"
            return
        now = time.monotonic()
        elapsed = now - self.real_phase_started_at
        plan = self.real_plan
        try:
            current_arm = self._safe_real_arm_positions(
                fallback=(self.real_start_q or plan.get("lift_q") or plan.get("target_q"))
            )
            if self.real_phase == "real_open":
                self._apply_real_positions(
                    self._real_gripper_command_positions(plan["open"]),
                    self.real_gripper_indices,
                )
                if elapsed >= 1.5:
                    self.real_start_q = current_arm
                    self.real_phase = "real_pregrasp"
                    self.real_phase_started_at = now
                    self._publish_status("real_opened", real_phase=self.real_phase)
            elif self.real_phase == "real_pregrasp":
                if plan.get("dynamic_plan", {}).get("direct_joint_targets"):
                    cmd, ratio = self._real_move_fraction(
                        self.real_start_q,
                        plan["pre_q"],
                        duration=20.0,
                    )
                else:
                    cmd, ratio = self._real_move_fraction(
                        self.real_start_q,
                        plan["pre_q"],
                        duration=5.0,
                    )
                self._apply_real_positions(cmd, self.real_arm_indices)
                pre_error = self._joint_max_error(current_arm, plan["pre_q"])
                if ratio >= 1.0 and pre_error <= MATERIAL_REAL_ARM_GOAL_TOLERANCE:
                    object_world = self._world_position(plan["cfg"]["move_path"])
                    if (
                        object_world is not None
                        and plan.get("dynamic_plan", {}).get("mode") != "measured_real_q"
                        and not self._using_bobac_material_arm()
                    ):
                        refined_object_base = self._base_relative_position(object_world)
                        try:
                            refined_plan = self._make_real_dynamic_plan(
                                refined_object_base,
                                current_arm,
                                plan.get("selected"),
                            )
                        except Exception as base_exc:
                            carb.log_warn(
                                f"Library-guided pregrasp replan failed, falling back to world-frame IK: {base_exc}"
                            )
                            refined_plan = self._make_real_world_plan(object_world, current_arm)
                        plan["object_world"] = object_world
                        plan["object_base"] = refined_object_base
                        plan["dynamic_plan"] = refined_plan
                        plan["target_q"] = [float(v) for v in refined_plan["target_q"]]
                        plan["lift_q"] = [float(v) for v in refined_plan["lift_q"]]
                        self._publish_json(
                            self.report_pub,
                            {
                                "task": "real_material_pregrasp_replan",
                                "target": self.real_target,
                                "object_world": [round(v, 6) for v in object_world],
                                "object_base": [round(v, 6) for v in refined_object_base],
                                "plan_mode": refined_plan["mode"],
                                "ik_orientation": refined_plan["ik"]["orientation"],
                                "ik_correction_x": refined_plan.get("correction_x", 0.0),
                                "ik_correction_y": refined_plan.get("correction_y", 0.0),
                                "ik_correction_z": refined_plan.get("correction_z", 0.0),
                                "grasp_world": [round(v, 6) for v in refined_plan.get("grasp_world", ())],
                                "lift_world": [round(v, 6) for v in refined_plan.get("lift_world", ())],
                                "ik_position_errors": [
                                    round(float(v), 6)
                                    for v in refined_plan["ik"]["position_errors"]
                                ],
                            },
                        )
                    elif object_world is not None:
                        plan["object_world"] = object_world
                        plan["object_base"] = self._base_relative_position(object_world)
                    self.real_start_q = current_arm
                    self.real_phase = "real_approach"
                    self.real_phase_started_at = now
                    self._publish_status(
                        "real_at_pregrasp",
                        real_phase=self.real_phase,
                        joint_error=pre_error,
                    )
                elif elapsed >= 35.0:
                    self._publish_json(
                        self.report_pub,
                        {
                            "task": "real_material_pick_place",
                            "ok": False,
                            "target": self.real_target,
                            "arm_side": self.real_side,
                            "failure_stage": "pregrasp_unreached",
                            "joint_error": round(float(pre_error), 6),
                            "joint_errors": self._joint_error_payload(current_arm, plan["pre_q"]),
                            "elapsed_sec": round(now - self.real_started_at, 3),
                        },
                    )
                    self._publish_status(
                        "real_grasp_failed",
                        real_phase=self.real_phase,
                        joint_error=pre_error,
                    )
                    self.real_phase = "idle"
                    return
            elif self.real_phase == "real_approach":
                if plan.get("dynamic_plan", {}).get("direct_joint_targets"):
                    cmd, ratio = self._real_move_fraction(
                        self.real_start_q,
                        plan["target_q"],
                        duration=2.0,
                    )
                else:
                    cmd, ratio = self._real_move_fraction(
                        self.real_start_q,
                        plan["target_q"],
                        duration=3.0,
                    )
                self._apply_real_positions(cmd, self.real_arm_indices)
                approach_error = self._joint_max_error(current_arm, plan["target_q"])
                joint_error_payload = self._joint_error_payload(current_arm, plan["target_q"])
                non_wrist_error = 0.0
                wrist_error = 0.0
                for item in joint_error_payload.get("joints", []):
                    err = abs(float(item.get("error", 0.0)))
                    if item.get("joint") == "joint_4":
                        wrist_error = err
                    else:
                        non_wrist_error = max(non_wrist_error, err)
                bobac_partial_wrist_ok = (
                    self._using_bobac_material_arm()
                    and ratio >= 1.0
                    and elapsed >= 6.0
                    and non_wrist_error <= 0.08
                    and wrist_error <= 0.45
                )
                if ratio >= 1.0 and (
                    approach_error <= MATERIAL_REAL_ARM_GOAL_TOLERANCE
                    or bobac_partial_wrist_ok
                ):
                    if (
                        self._using_bobac_material_arm()
                        and plan.get("dynamic_plan", {}).get("direct_joint_targets")
                    ):
                        object_world = self._world_position(plan["cfg"]["move_path"])
                        measured_tcp = self._measure_gripper_tcp(self.real_side, object_world)
                        tcp_world = measured_tcp.get("tcp_world")
                        preclose_distance = (
                            self._distance(tuple(tcp_world), object_world)
                            if tcp_world is not None and object_world is not None
                            else float("inf")
                        )
                        preclose_horizontal = float("inf")
                        preclose_vertical = float("inf")
                        if tcp_world is not None and object_world is not None:
                            preclose_horizontal = math.hypot(
                                float(tcp_world[0]) - float(object_world[0]),
                                float(tcp_world[1]) - float(object_world[1]),
                            )
                            preclose_vertical = float(tcp_world[2]) - float(object_world[2])
                        preclose_payload = {
                            "task": "real_material_preclose_alignment",
                            "target": self.real_target,
                            "mode": "bobac_direct_library_q",
                            "joint_error": round(float(approach_error), 6),
                            "object_world": (
                                [round(v, 6) for v in object_world]
                                if object_world is not None
                                else None
                            ),
                            "measured_tcp": measured_tcp,
                            "tcp_object_distance_m": round(float(preclose_distance), 6),
                            "tcp_object_horizontal_m": round(float(preclose_horizontal), 6),
                            "tcp_object_vertical_offset_m": round(float(preclose_vertical), 6),
                        }
                        plan["preclose_alignment"] = preclose_payload
                        self._publish_json(self.report_pub, preclose_payload)
                        preclose_ok = (
                            preclose_horizontal <= MATERIAL_REAL_CLOSE_MAX_HORIZONTAL_DISTANCE
                            and MATERIAL_REAL_CLOSE_MIN_VERTICAL_OFFSET
                            <= preclose_vertical
                            <= MATERIAL_REAL_CLOSE_MAX_VERTICAL_OFFSET
                        )
                        if not preclose_ok:
                            if elapsed < 20.0:
                                return
                            self._publish_json(
                                self.report_pub,
                                {
                                    "task": "real_material_pick_place",
                                    "ok": False,
                                    "target": self.real_target,
                                    "arm_side": self.real_side,
                                    "failure_stage": "preclose_alignment",
                                    **preclose_payload,
                                    "elapsed_sec": round(now - self.real_started_at, 3),
                                },
                            )
                            self._publish_status(
                                "real_grasp_failed",
                                real_phase=self.real_phase,
                                preclose_horizontal_m=preclose_horizontal,
                            )
                            self.real_phase = "idle"
                            return
                    if bobac_partial_wrist_ok:
                        object_world = self._world_position(plan["cfg"]["move_path"])
                        measured_tcp = self._measure_gripper_tcp(self.real_side, object_world)
                        tcp_world = measured_tcp.get("tcp_world")
                        preclose_distance = (
                            self._distance(tuple(tcp_world), object_world)
                            if tcp_world is not None and object_world is not None
                            else float("inf")
                        )
                        preclose_horizontal = float("inf")
                        preclose_vertical = float("inf")
                        if tcp_world is not None and object_world is not None:
                            preclose_horizontal = math.hypot(
                                float(tcp_world[0]) - float(object_world[0]),
                                float(tcp_world[1]) - float(object_world[1]),
                            )
                            preclose_vertical = float(tcp_world[2]) - float(object_world[2])
                        preclose_payload = {
                            "task": "real_material_preclose_alignment",
                            "target": self.real_target,
                            "mode": "bobac_partial_wrist",
                            "joint_error": round(float(approach_error), 6),
                            "non_wrist_error": round(float(non_wrist_error), 6),
                            "wrist_joint_4_error": round(float(wrist_error), 6),
                            "object_world": (
                                [round(v, 6) for v in object_world]
                                if object_world is not None
                                else None
                            ),
                            "measured_tcp": measured_tcp,
                            "tcp_object_distance_m": round(float(preclose_distance), 6),
                            "tcp_object_horizontal_m": round(float(preclose_horizontal), 6),
                            "tcp_object_vertical_offset_m": round(float(preclose_vertical), 6),
                        }
                        plan["preclose_alignment"] = preclose_payload
                        self._publish_json(self.report_pub, preclose_payload)
                    self.real_phase = "real_close"
                    self.real_phase_started_at = now
                    self._publish_status(
                        "real_at_grasp",
                        real_phase=self.real_phase,
                        joint_error=approach_error,
                        non_wrist_error=non_wrist_error,
                        wrist_joint_4_error=wrist_error,
                    )
                elif elapsed >= 60.0:
                    self._publish_json(
                        self.report_pub,
                        {
                            "task": "real_material_pick_place",
                            "ok": False,
                            "target": self.real_target,
                            "arm_side": self.real_side,
                            "failure_stage": "approach_unreached",
                            "joint_error": round(float(approach_error), 6),
                            "joint_errors": joint_error_payload,
                            "elapsed_sec": round(now - self.real_started_at, 3),
                        },
                    )
                    self._publish_status(
                        "real_grasp_failed",
                        real_phase=self.real_phase,
                        joint_error=approach_error,
                    )
                    self.real_phase = "idle"
                    return
            elif self.real_phase == "real_close":
                if MATERIAL_ENABLE_GRASP_PROXY and not plan.get("active_grasp_proxy"):
                    plan["active_grasp_proxy"] = self._create_material_grasp_proxy(
                        plan["cfg"]["move_path"],
                        scale=MATERIAL_ACTIVE_GRASP_PROXY_SCALE,
                        translation=MATERIAL_ACTIVE_GRASP_PROXY_TRANSLATION,
                    )
                    self._publish_json(
                        self.report_pub,
                        {
                            "task": "real_material_active_grasp_proxy",
                            "target": self.real_target,
                            "proxy": plan["active_grasp_proxy"],
                            "scale": [round(float(v), 6) for v in MATERIAL_ACTIVE_GRASP_PROXY_SCALE],
                            "translation": [
                                round(float(v), 6)
                                for v in MATERIAL_ACTIVE_GRASP_PROXY_TRANSLATION
                            ],
                        },
                    )
                self._apply_real_positions(
                    self._real_gripper_command_positions(plan["close"]),
                    self.real_gripper_indices,
                )
                if elapsed >= 2.0:
                    object_world = self._world_position(plan["cfg"]["move_path"])
                    measured_tcp = self._measure_gripper_tcp(self.real_side, object_world)
                    tcp_world = measured_tcp.get("tcp_world")
                    close_distance = (
                        self._distance(tuple(tcp_world), object_world)
                        if tcp_world is not None and object_world is not None
                        else float("inf")
                    )
                    horizontal_distance = float("inf")
                    vertical_offset = float("inf")
                    if tcp_world is not None and object_world is not None:
                        horizontal_distance = math.hypot(
                            float(tcp_world[0]) - float(object_world[0]),
                            float(tcp_world[1]) - float(object_world[1]),
                        )
                        vertical_offset = float(tcp_world[2]) - float(object_world[2])
                    close_payload = {
                        "task": "real_material_close_alignment",
                        "target": self.real_target,
                        "object_world": (
                            [round(v, 6) for v in object_world]
                            if object_world is not None
                            else None
                        ),
                        "measured_tcp": measured_tcp,
                        "tcp_object_distance_m": round(float(close_distance), 6),
                        "max_allowed_m": MATERIAL_REAL_CLOSE_MAX_TCP_OBJECT_DISTANCE,
                        "tcp_object_horizontal_m": round(float(horizontal_distance), 6),
                        "max_horizontal_allowed_m": MATERIAL_REAL_CLOSE_MAX_HORIZONTAL_DISTANCE,
                        "tcp_object_vertical_offset_m": round(float(vertical_offset), 6),
                        "vertical_offset_allowed_m": [
                            MATERIAL_REAL_CLOSE_MIN_VERTICAL_OFFSET,
                            MATERIAL_REAL_CLOSE_MAX_VERTICAL_OFFSET,
                        ],
                        "validation_note": (
                            "bobac_post_close_tcp_is_diagnostic_only"
                            if self._using_bobac_material_arm()
                            else "post_close_tcp_alignment_required"
                        ),
                    }
                    self._publish_json(self.report_pub, close_payload)
                    close_ok = (
                        close_distance <= MATERIAL_REAL_CLOSE_MAX_TCP_OBJECT_DISTANCE
                        and horizontal_distance <= MATERIAL_REAL_CLOSE_MAX_HORIZONTAL_DISTANCE
                        and MATERIAL_REAL_CLOSE_MIN_VERTICAL_OFFSET
                        <= vertical_offset
                        <= MATERIAL_REAL_CLOSE_MAX_VERTICAL_OFFSET
                    )
                    if not close_ok and not self._using_bobac_material_arm():
                        report = {
                            "task": "real_material_pick_place",
                            "ok": False,
                            "target": self.real_target,
                            "arm_side": self.real_side,
                            "failure_stage": "close_alignment",
                            **close_payload,
                            "elapsed_sec": round(now - self.real_started_at, 3),
                        }
                        self._publish_json(self.report_pub, report)
                        self._publish_status(
                            "real_grasp_failed",
                            real_phase=self.real_phase,
                            close_distance_m=close_distance,
                        )
                        self.real_phase = "idle"
                        return
                    self._set_kinematic(plan["cfg"]["move_path"], False)
                    self.real_start_q = current_arm
                    self.real_phase = "real_lift"
                    self.real_phase_started_at = now
                    self._publish_status("real_closed", real_phase=self.real_phase)
            elif self.real_phase == "real_lift":
                if plan.get("dynamic_plan", {}).get("direct_joint_targets"):
                    cmd, ratio = self._real_move_fraction(
                        self.real_start_q,
                        plan["lift_q"],
                        duration=6.0,
                    )
                else:
                    cmd, ratio = self._real_move_fraction(
                        self.real_start_q,
                        plan["lift_q"],
                        duration=4.0,
                    )
                self._apply_real_positions(cmd, self.real_arm_indices)
                lift_error = self._joint_max_error(current_arm, plan["lift_q"])
                if ratio >= 1.0 and lift_error <= MATERIAL_REAL_ARM_GOAL_TOLERANCE:
                    self.real_phase = "real_validate"
                    self.real_phase_started_at = now
                    self._publish_status(
                        "real_lifted",
                        real_phase=self.real_phase,
                        joint_error=lift_error,
                    )
                elif elapsed >= 30.0:
                    self._publish_json(
                        self.report_pub,
                        {
                            "task": "real_material_pick_place",
                            "ok": False,
                            "target": self.real_target,
                            "arm_side": self.real_side,
                            "failure_stage": "lift_unreached",
                            "joint_error": round(float(lift_error), 6),
                            "joint_errors": self._joint_error_payload(current_arm, plan["lift_q"]),
                            "elapsed_sec": round(now - self.real_started_at, 3),
                        },
                    )
                    self._publish_status(
                        "real_grasp_failed",
                        real_phase=self.real_phase,
                        joint_error=lift_error,
                    )
                    self.real_phase = "idle"
                    return
            elif self.real_phase == "real_validate":
                object_world = self._world_position(plan["cfg"]["move_path"])
                initial = self.real_initial_object_world or plan["object_world"]
                lift_dz = (
                    float(object_world[2]) - float(initial[2])
                    if object_world is not None
                    else float("nan")
                )
                horizontal_slip = (
                    self._distance(object_world[:2] + (0.0,), initial[:2] + (0.0,))
                    if object_world is not None
                    else float("nan")
                )
                lift_ok = bool(object_world is not None and lift_dz > 0.010)
                measured_tcp = self._measure_gripper_tcp(self.real_side, object_world)
                if not lift_ok:
                    report = {
                        "task": "real_material_pick_place",
                        "ok": False,
                        "target": self.real_target,
                        "arm_side": self.real_side,
                        "failure_stage": "lift",
                        "initial_object_world": [round(v, 6) for v in initial],
                        "final_object_world": (
                            [round(v, 6) for v in object_world]
                            if object_world is not None
                            else None
                        ),
                        "lift_delta_z_m": round(float(lift_dz), 6),
                        "horizontal_slip_m": round(float(horizontal_slip), 6),
                        "elapsed_sec": round(now - self.real_started_at, 3),
                        "measured_tcp": measured_tcp,
                    }
                    report.update(self._gripper_contact_diagnostics(object_world))
                    self._publish_json(self.report_pub, report)
                    self._publish_status("real_grasp_failed", real_phase=self.real_phase, lift_delta_z_m=lift_dz)
                    self.real_phase = "idle"
                    return

                if self._using_bobac_material_arm():
                    report = {
                        "task": "real_material_pick_place",
                        "ok": True,
                        "target": self.real_target,
                        "arm_side": self.real_side,
                        "completion_stage": "stable_grasp_lifted",
                        "initial_object_world": [round(v, 6) for v in initial],
                        "final_object_world": [round(v, 6) for v in object_world],
                        "lift_delta_z_m": round(float(lift_dz), 6),
                        "horizontal_slip_m": round(float(horizontal_slip), 6),
                        "elapsed_sec": round(now - self.real_started_at, 3),
                        "measured_tcp": measured_tcp,
                    }
                    report.update(self._gripper_contact_diagnostics(object_world))
                    self._publish_json(self.report_pub, report)
                    self._publish_status(
                        "real_stable_grasp_lifted",
                        real_phase="idle",
                        lift_delta_z_m=lift_dz,
                    )
                    self.real_phase = "idle"
                    return

                if MATERIAL_ENABLE_BASE_REPOSITION_FOR_PLACE and not plan.get("base_reposition_done"):
                    base_reposition = self._prepare_base_reposition_for_place()
                    plan["base_reposition"] = base_reposition
                    if float(base_reposition["distance_m"]) > 0.02:
                        self.real_start_q = current_arm
                        self.real_phase = "real_reposition_base"
                        self.real_phase_started_at = now
                        self._publish_json(
                            self.report_pub,
                            {
                                "task": "real_material_base_reposition",
                                "status": "started",
                                "target": self.real_target,
                                "lift_delta_z_m": round(float(lift_dz), 6),
                                **base_reposition,
                            },
                        )
                        self._publish_status(
                            "real_reposition_base_started",
                            real_phase=self.real_phase,
                            distance_m=base_reposition["distance_m"],
                        )
                        return
                    plan["base_reposition_done"] = True

                tcp_world = measured_tcp.get("tcp_world")
                if tcp_world is None or object_world is None:
                    raise RuntimeError("cannot compute held object offset for placement")
                held_offset = tuple(float(object_world[i]) - float(tcp_world[i]) for i in range(3))
                if self._distance(object_world, MATERIAL_PLACE_WORLD) <= 0.25:
                    object_place_world = tuple(float(v) for v in MATERIAL_PLACE_WORLD)
                    tcp_place_world = tuple(float(v) for v in tcp_world)
                    place_plan = {
                        "mode": "current_hold_release",
                        "object_place_world": object_place_world,
                        "tcp_carry_world": tcp_place_world,
                        "tcp_place_world": tcp_place_world,
                        "tcp_retreat_world": (
                            float(tcp_world[0]),
                            float(tcp_world[1]),
                            float(tcp_world[2]) + 0.10,
                        ),
                        "carry_q": [float(v) for v in current_arm],
                        "place_q": [float(v) for v in current_arm],
                        "retreat_q": [float(v) for v in plan["lift_q"]],
                    }
                    plan["place_plan"] = place_plan
                    plan["grasp_lock"] = None
                    plan["lift_report"] = {
                        "initial_object_world": [round(v, 6) for v in initial],
                        "lift_object_world": [round(v, 6) for v in object_world],
                        "lift_delta_z_m": round(float(lift_dz), 6),
                        "horizontal_slip_m": round(float(horizontal_slip), 6),
                        "held_offset_world": [round(v, 6) for v in held_offset],
                        "object_place_world": [round(v, 6) for v in object_place_world],
                        "tcp_place_world": [round(v, 6) for v in tcp_place_world],
                        "measured_tcp": measured_tcp,
                        "grasp_lock": None,
                    }
                    self._publish_json(
                        self.report_pub,
                        {
                            "task": "real_material_pick_place",
                            "status": "lifted",
                            "target": self.real_target,
                            "arm_side": self.real_side,
                            **plan["lift_report"],
                        },
                    )
                    self.real_start_q = current_arm
                    self.real_phase = "real_carry"
                    self.real_phase_started_at = now
                    self._publish_status(
                        "real_grasp_lifted",
                        real_phase=self.real_phase,
                        lift_delta_z_m=lift_dz,
                    )
                    return
                place_offsets = (
                    (0.00, 0.00, 0.00),
                    (0.00, 0.08, 0.00),
                    (0.00, 0.14, 0.00),
                    (-0.08, 0.12, 0.00),
                    (0.08, 0.12, 0.00),
                    (0.00, 0.20, 0.00),
                    (-0.12, 0.18, 0.00),
                    (0.12, 0.18, 0.00),
                    (-0.16, 0.12, 0.00),
                    (0.16, 0.12, 0.00),
                    (0.00, 0.08, 0.04),
                    (0.00, 0.14, 0.04),
                    (-0.08, 0.12, 0.04),
                    (0.08, 0.12, 0.04),
                )
                place_errors = []
                place_plan = None
                tcp_place_world = None
                object_place_world = None
                for offset in place_offsets:
                    candidate_object_place = tuple(
                        float(MATERIAL_PLACE_WORLD[i]) + float(offset[i]) for i in range(3)
                    )
                    candidate_tcp_place = tuple(
                        float(candidate_object_place[i]) - held_offset[i] for i in range(3)
                    )
                    try:
                        place_plan = self._make_real_place_plan(
                            current_arm,
                            candidate_tcp_place,
                            object_place_world=candidate_object_place,
                        )
                        tcp_place_world = candidate_tcp_place
                        object_place_world = candidate_object_place
                        break
                    except Exception as place_exc:
                        place_errors.append(
                            f"object_place={tuple(round(v, 3) for v in candidate_object_place)}: {place_exc}"
                        )
                if place_plan is None or tcp_place_world is None or object_place_world is None:
                    raise RuntimeError("place IK failed for all cargo candidates: " + " | ".join(place_errors))
                plan["place_plan"] = place_plan
                plan["grasp_lock"] = None
                plan["lift_report"] = {
                    "initial_object_world": [round(v, 6) for v in initial],
                    "lift_object_world": [round(v, 6) for v in object_world],
                    "lift_delta_z_m": round(float(lift_dz), 6),
                    "horizontal_slip_m": round(float(horizontal_slip), 6),
                    "held_offset_world": [round(v, 6) for v in held_offset],
                    "object_place_world": [round(v, 6) for v in object_place_world],
                    "tcp_place_world": [round(v, 6) for v in tcp_place_world],
                    "measured_tcp": measured_tcp,
                    "grasp_lock": None,
                }
                self._publish_json(
                    self.report_pub,
                    {
                        "task": "real_material_pick_place",
                        "status": "lifted",
                        "target": self.real_target,
                        "arm_side": self.real_side,
                        **plan["lift_report"],
                    },
                )
                self.real_start_q = current_arm
                self.real_phase = "real_carry"
                self.real_phase_started_at = now
                self._publish_status("real_grasp_lifted", real_phase=self.real_phase, lift_delta_z_m=lift_dz)
            elif self.real_phase == "real_reposition_base":
                base_reposition = plan.get("base_reposition")
                if not base_reposition:
                    plan["base_reposition_done"] = True
                    self.real_phase = "real_validate"
                    self.real_phase_started_at = now
                    return
                if not base_reposition.get("started"):
                    self._remove_material_base_lock()
                    base_reposition["started"] = True
                self._apply_real_positions(plan["lift_q"], self.real_arm_indices)
                self._apply_real_positions(
                    self._real_gripper_command_positions(plan["close"]),
                    self.real_gripper_indices,
                )
                duration = max(0.1, float(base_reposition.get("duration", MATERIAL_BASE_REPOSITION_DURATION_SEC)))
                ratio = min(1.0, max(0.0, elapsed / duration))
                last_ratio = float(base_reposition.get("last_ratio", 0.0))
                step_ratio = max(0.0, ratio - last_ratio)
                total = [float(v) for v in base_reposition["world_delta_total"]]
                step = [float(v) * step_ratio for v in total]
                if any(abs(v) > 1e-7 for v in step):
                    self._translate_world(self._active_material_robot_prim(), tuple(step))
                    applied = base_reposition.get("world_delta_applied", [0.0, 0.0, 0.0])
                    base_reposition["world_delta_applied"] = [
                        float(applied[i]) + float(step[i]) for i in range(3)
                    ]
                    UsdGeom.XformCache().Clear()
                base_reposition["last_ratio"] = ratio
                if ratio >= 1.0:
                    lock_info = self._lock_material_base_here()
                    after_place_base = self._base_relative_position(MATERIAL_PLACE_WORLD)
                    object_world = self._world_position(plan["cfg"]["move_path"])
                    plan["base_reposition_done"] = True
                    self._publish_json(
                        self.report_pub,
                        {
                            "task": "real_material_base_reposition",
                            "status": "completed",
                            "target": self.real_target,
                            "after_place_base": [round(v, 6) for v in after_place_base],
                            "object_world_after": (
                                [round(v, 6) for v in object_world]
                                if object_world is not None
                                else None
                            ),
                            "lock": lock_info,
                            **base_reposition,
                        },
                    )
                    self.real_start_q = self._safe_real_arm_positions(fallback=plan["lift_q"])
                    self.real_phase = "real_validate"
                    self.real_phase_started_at = now
                    self._publish_status(
                        "real_reposition_base_done",
                        real_phase=self.real_phase,
                        after_place_base=[round(v, 6) for v in after_place_base],
                    )
            elif self.real_phase == "real_carry":
                place_plan = plan["place_plan"]
                cmd, ratio = self._real_move_fraction(
                    self.real_start_q,
                    place_plan["carry_q"],
                    duration=12.0,
                )
                self._apply_real_positions(cmd, self.real_arm_indices)
                carry_error = self._joint_max_error(current_arm, place_plan["carry_q"])
                if ratio >= 1.0 and (carry_error <= 0.08 or elapsed >= 18.0):
                    self.real_start_q = current_arm
                    self.real_phase = "real_place"
                    self.real_phase_started_at = now
                    self._publish_status("real_above_cargo", real_phase=self.real_phase, joint_error=carry_error)
            elif self.real_phase == "real_place":
                place_plan = plan["place_plan"]
                cmd, ratio = self._real_move_fraction(
                    self.real_start_q,
                    place_plan["place_q"],
                    duration=8.0,
                )
                self._apply_real_positions(cmd, self.real_arm_indices)
                place_error = self._joint_max_error(current_arm, place_plan["place_q"])
                if ratio >= 1.0 and (place_error <= 0.08 or elapsed >= 14.0):
                    self.real_phase = "real_release"
                    self.real_phase_started_at = now
                    self._publish_status("real_at_cargo_release", real_phase=self.real_phase, joint_error=place_error)
            elif self.real_phase == "real_release":
                if not plan.get("grasp_lock_released"):
                    self._remove_real_grasp_lock()
                    plan["grasp_lock_released"] = True
                self._apply_real_positions(
                    self._real_gripper_command_positions(plan["open"]),
                    self.real_gripper_indices,
                )
                if elapsed >= 1.2:
                    self.real_start_q = current_arm
                    self.real_phase = "real_retreat"
                    self.real_phase_started_at = now
                    self._publish_status("real_released", real_phase=self.real_phase)
            elif self.real_phase == "real_retreat":
                place_plan = plan["place_plan"]
                cmd, ratio = self._real_move_fraction(
                    self.real_start_q,
                    place_plan["retreat_q"],
                    duration=5.0,
                )
                self._apply_real_positions(cmd, self.real_arm_indices)
                retreat_error = self._joint_max_error(current_arm, place_plan["retreat_q"])
                if ratio >= 1.0 and (retreat_error <= 0.08 or elapsed >= 8.0):
                    self.real_phase = "real_place_validate"
                    self.real_phase_started_at = now
                    self._publish_status("real_retreat_done", real_phase=self.real_phase, joint_error=retreat_error)
            elif self.real_phase == "real_place_validate":
                object_world = self._world_position(plan["cfg"]["move_path"])
                place_error = (
                    self._distance(object_world, MATERIAL_PLACE_WORLD)
                    if object_world is not None
                    else float("nan")
                )
                ok = bool(object_world is not None and place_error <= 0.25)
                report = {
                    "task": "real_material_pick_place",
                    "ok": ok,
                    "target": self.real_target,
                    "arm_side": self.real_side,
                    "place_world": [round(v, 6) for v in MATERIAL_PLACE_WORLD],
                    "final_object_world": (
                        [round(v, 6) for v in object_world]
                        if object_world is not None
                        else None
                    ),
                    "place_error_m": round(float(place_error), 6),
                    "elapsed_sec": round(now - self.real_started_at, 3),
                    "lift_report": plan.get("lift_report"),
                    "place_plan": {
                        "object_place_world": (
                            [round(v, 6) for v in plan["place_plan"]["object_place_world"]]
                            if plan["place_plan"].get("object_place_world") is not None
                            else None
                        ),
                        "tcp_carry_world": [round(v, 6) for v in plan["place_plan"]["tcp_carry_world"]],
                        "tcp_place_world": [round(v, 6) for v in plan["place_plan"]["tcp_place_world"]],
                        "tcp_retreat_world": [round(v, 6) for v in plan["place_plan"]["tcp_retreat_world"]],
                    },
                    "notes": "Real arm used gripper contact, high-friction material physics, and a pencil collision proxy; no coordinate transport stabilizer was used.",
                    "grasp_lock_used": bool(plan.get("grasp_lock")),
                    "grasp_lock_released": bool(plan.get("grasp_lock_released")),
                    "transport_stabilizer_used": False,
                }
                self._publish_json(self.report_pub, report)
                self._publish_status("real_pick_place_completed" if ok else "real_pick_place_failed", real_phase=self.real_phase, place_error_m=place_error)
                self.real_phase = "idle"
        except Exception as exc:
            self.real_phase = "idle"
            self._remove_real_grasp_lock()
            self._publish_json(
                self.report_pub,
                {
                    "task": "real_material_grasp",
                    "ok": False,
                    "target": self.real_target,
                    "error": str(exc),
                },
            )
            carb.log_error(f"Real material grasp failed: {exc}")


class RosCargoInterface:
    def __init__(self, cargo_runtime: CargoBayRuntime, stage):
        self.cargo_runtime = cargo_runtime
        self.stage = stage
        self.rclpy = None
        self.node = None
        self.executor = None
        self.String = None
        self.JointState = None
        self.PointStamped = None
        self.Twist = None
        self.LaserScan = None
        self.TimeMsg = None
        self.Clock = None
        self.Odometry = None
        self.TFMessage = None
        self.TransformStamped = None
        self.status_pub = None
        self.clock_pub = None
        self.odom_pub = None
        self.tf_pub = None
        self.tf_static_pub = None
        self.scan_pub = None
        self.scan_fuse_pub = None
        self.drone_world_pub = None
        self.drone_status_pub = None
        self.bobac_world_pub = None
        self.material_task = None
        self.owns_rclpy = False
        self.clock_start_monotonic = time.monotonic()
        self.last_bobac_odom_time = None
        self.last_bobac_odom_position = None
        self.last_bobac_odom_yaw = None
        self.sent_bobac_static_tf = False
        self.bobac_kinematic_cmd = None
        self.bobac_kinematic_cmd_time = 0.0
        self.bobac_kinematic_last_update = None
        self.drone_kinematic_cmd = None
        self.drone_kinematic_cmd_time = 0.0
        self.drone_kinematic_last_update = None
        self.drone_target = None
        self.drone_target_yaw = None
        self.drone_target_status_sent = False
        self.drone_body_hold_active = False
        self.drone_body_root_offset = (0.0, 0.0, 0.0)
        self.dynamic_control = None
        self.dynamic_control_module = None
        self.drone_body_handle = None
        self.dynamic_control_warned = False
        self.scan_obstacles = []
        self.scan_obstacles_refresh_time = 0.0

    def start(self):
        try:
            enable_extension("isaacsim.ros2.bridge")
        except Exception as exc:
            carb.log_warn(f"Could not enable isaacsim.ros2.bridge: {exc}")

        try:
            from builtin_interfaces.msg import Time as TimeMsg
            import rclpy
            from geometry_msgs.msg import PointStamped
            from geometry_msgs.msg import TransformStamped
            from geometry_msgs.msg import Twist
            from nav_msgs.msg import Odometry
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
            from rosgraph_msgs.msg import Clock
            from sensor_msgs.msg import JointState, LaserScan
            from std_msgs.msg import String
            from tf2_msgs.msg import TFMessage
        except Exception as exc:
            carb.log_warn(f"ROS 2 interface disabled: {exc}")
            return

        self.rclpy = rclpy
        self.String = String
        self.JointState = JointState
        self.PointStamped = PointStamped
        self.Twist = Twist
        self.LaserScan = LaserScan
        self.TimeMsg = TimeMsg
        self.Clock = Clock
        self.Odometry = Odometry
        self.TFMessage = TFMessage
        self.TransformStamped = TransformStamped
        if not rclpy.ok():
            rclpy.init(args=None)
            self.owns_rclpy = True
        node_suffix = int(time.monotonic() * 1000)
        self.node = rclpy.create_node(f"cargo_delivery_scene_{os.getpid()}_{node_suffix}")
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.node.create_subscription(String, "/cargo_bay/command", self._on_command, 10)
        self.node.create_subscription(
            Twist, "/bobac_kinematic_cmd_vel", self._on_bobac_kinematic_cmd, 10
        )
        self.node.create_subscription(
            Twist, "/drone/cmd_vel", self._on_drone_cmd_vel, 10
        )
        self.node.create_subscription(
            String, "/drone/command", self._on_drone_command, 10
        )
        self.clock_pub = self.node.create_publisher(Clock, "/clock", 10)
        self.odom_pub = self.node.create_publisher(Odometry, "/odom", 10)
        self.tf_pub = self.node.create_publisher(TFMessage, "/tf", 10)
        static_tf_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.tf_static_pub = self.node.create_publisher(
            TFMessage, "/tf_static", static_tf_qos
        )
        self.scan_pub = self.node.create_publisher(LaserScan, "/scan", 10)
        self.scan_fuse_pub = self.node.create_publisher(LaserScan, "/laser_scan_fuse", 10)
        self.status_pub = self.node.create_publisher(String, "/cargo_bay/status", 10)
        self.drone_status_pub = self.node.create_publisher(String, "/drone/status", 10)
        self.drone_world_pub = self.node.create_publisher(
            PointStamped, "/drone/world_position", 10
        )
        self.bobac_world_pub = self.node.create_publisher(
            PointStamped, "/bobac/world_position", 10
        )
        self.material_task = MaterialTaskRuntime(self.stage, self.node, String, JointState)
        self._calibrate_drone_body_root_offset()
        carb.log_info("Cargo delivery ROS 2 interface ready")

    def spin_once(self):
        if self.executor is None or self.rclpy is None or not self.rclpy.ok():
            return
        try:
            self.executor.spin_once(timeout_sec=0.0)
            self.apply_bobac_kinematic_control()
            self.apply_drone_control()
            self.cargo_runtime.update_door_motion()
            self.cargo_runtime.update_payload_follow()
            self.publish_clock()
            self.publish_drone_world()
            self.update_drone_down_camera_pose()
            self.publish_bobac_odom()
            self.publish_bobac_scan()
            self.publish_arm_camera_tf()
            self.publish_bobac_world()
            if self.material_task is not None:
                self.material_task.update()
        except Exception as exc:
            if self.rclpy is not None and not self.rclpy.ok():
                return
            carb.log_warn(f"ROS 2 spin skipped: {exc}")

    def ros_time_msg(self):
        elapsed = max(0.0, time.monotonic() - self.clock_start_monotonic)
        stamp = self.TimeMsg()
        stamp.sec = int(elapsed)
        stamp.nanosec = int((elapsed - stamp.sec) * 1.0e9)
        return stamp

    def publish_clock(self):
        if self.clock_pub is None:
            return
        msg = self.Clock()
        msg.clock = self.ros_time_msg()
        self.clock_pub.publish(msg)

    def _on_bobac_kinematic_cmd(self, msg):
        self.bobac_kinematic_cmd = msg
        self.bobac_kinematic_cmd_time = time.monotonic()

    def _on_drone_cmd_vel(self, msg):
        if BOBAC_ENABLE_PX4:
            self.publish_drone_status("ignored px4_mode_direct_cmd_vel_disabled")
            return
        pose = self._drone_body_pose_from_stage()
        if pose is not None:
            position, _orientation, yaw = pose
            self._activate_drone_control_at_current_pose(position, yaw)
        self.drone_kinematic_cmd = msg
        self.drone_kinematic_cmd_time = time.monotonic()
        self.drone_target = None
        self.drone_target_yaw = None
        self.drone_target_status_sent = False

    def publish_drone_status(self, text: str):
        if self.drone_status_pub is None:
            return
        msg = self.String()
        msg.data = text
        self.drone_status_pub.publish(msg)

    def _on_drone_command(self, msg):
        command = msg.data.strip()
        parts = command.split()
        if not parts:
            return
        op = parts[0].lower()
        if BOBAC_ENABLE_PX4 and op != "status":
            self.publish_drone_status(f"ignored px4_mode_direct_command_disabled {command}")
            return
        pose = self._drone_body_pose_from_stage()
        if pose is None:
            self.publish_drone_status("error drone_pose_unavailable")
            return
        position, _orientation, yaw = pose
        try:
            if op == "status":
                self.publish_drone_status(
                    f"status x={position[0]:.3f} y={position[1]:.3f} z={position[2]:.3f} yaw={yaw:.3f}"
                )
            elif op == "hold":
                self._activate_drone_control_at_current_pose(position, yaw)
                self.drone_target = (position[0], position[1], position[2])
                self.drone_target_yaw = yaw
                self.drone_target_status_sent = False
                self.publish_drone_status(
                    f"hold target=[{position[0]:.3f},{position[1]:.3f},{position[2]:.3f}]"
                )
            elif op == "takeoff":
                self._activate_drone_control_at_current_pose(position, yaw)
                height = float(parts[1]) if len(parts) >= 2 else 0.5
                target_z = position[2] + max(0.05, height)
                self.drone_target = (position[0], position[1], target_z)
                self.drone_target_yaw = yaw
                self.drone_target_status_sent = False
                self.publish_drone_status(
                    f"takeoff_started target=[{position[0]:.3f},{position[1]:.3f},{target_z:.3f}]"
                )
            elif op == "goto":
                self._activate_drone_control_at_current_pose(position, yaw)
                if len(parts) < 4:
                    raise ValueError("goto requires x y z [yaw]")
                self.drone_target = (float(parts[1]), float(parts[2]), float(parts[3]))
                self.drone_target_yaw = float(parts[4]) if len(parts) >= 5 else yaw
                self.drone_target_status_sent = False
                self.publish_drone_status(
                    f"goto_started target=[{self.drone_target[0]:.3f},{self.drone_target[1]:.3f},{self.drone_target[2]:.3f}]"
                )
            elif op == "land":
                self._activate_drone_control_at_current_pose(position, yaw)
                target_z = float(parts[1]) if len(parts) >= 2 else 1.02
                self.drone_target = (position[0], position[1], target_z)
                self.drone_target_yaw = yaw
                self.drone_target_status_sent = False
                self.publish_drone_status(
                    f"land_started target=[{position[0]:.3f},{position[1]:.3f},{target_z:.3f}]"
                )
            else:
                self.publish_drone_status(f"unknown_command {command}")
        except Exception as exc:
            self.publish_drone_status(f"error {exc}")

    def _set_bobac_base_world_pose(self, position, yaw: float):
        prim = self.stage.GetPrimAtPath(BOBAC_BASE_ARTICULATION_PRIM)
        if not prim or not prim.IsValid():
            return

        parent = prim.GetParent()
        local_position = Gf.Vec3d(float(position[0]), float(position[1]), float(position[2]))
        if parent and parent.IsValid():
            parent_world = UsdGeom.XformCache().GetLocalToWorldTransform(parent)
            local_position = parent_world.GetInverse().Transform(local_position)

        half = 0.5 * float(yaw)
        orient = Gf.Quatd(
            float(math.cos(half)),
            Gf.Vec3d(0.0, 0.0, float(math.sin(half))),
        )
        _set_xform_ops(prim, translation=local_position, orient=orient)

    def _drone_body_pose_from_stage(self):
        prim = self.stage.GetPrimAtPath(DRONE_BODY_PRIM)
        if not prim or not prim.IsValid():
            return None
        matrix = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
        translation = matrix.ExtractTranslation()
        quat = matrix.ExtractRotation().GetQuat()
        imag = quat.GetImaginary()
        qx = float(imag[0])
        qy = float(imag[1])
        qz = float(imag[2])
        qw = float(quat.GetReal())
        yaw = math.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )
        return (
            (float(translation[0]), float(translation[1]), float(translation[2])),
            (qx, qy, qz, qw),
            yaw,
        )

    def _drone_root_pose_from_stage(self):
        prim = self.stage.GetPrimAtPath(DRONE_PRIM)
        if not prim or not prim.IsValid():
            return None
        matrix = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
        translation = matrix.ExtractTranslation()
        quat = matrix.ExtractRotation().GetQuat()
        imag = quat.GetImaginary()
        qx = float(imag[0])
        qy = float(imag[1])
        qz = float(imag[2])
        qw = float(quat.GetReal())
        yaw = math.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )
        return (
            (float(translation[0]), float(translation[1]), float(translation[2])),
            (qx, qy, qz, qw),
            yaw,
        )

    def _set_drone_body_hold(self, enabled: bool):
        body = self.stage.GetPrimAtPath(DRONE_BODY_PRIM)
        root = self.stage.GetPrimAtPath(DRONE_PRIM)
        if not body or not body.IsValid():
            return
        prims = []
        root_path = root.GetPath() if root and root.IsValid() else None
        body_path = body.GetPath()
        for prim in self.stage.Traverse():
            if prim.GetPath() == body_path:
                prims.append(prim)
                continue
            if root_path is None or not prim.GetPath().HasPrefix(root_path):
                continue
            try:
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    prims.append(prim)
            except Exception:
                continue
        if not prims:
            prims = [body]
        try:
            for prim in prims:
                rigid = UsdPhysics.RigidBodyAPI.Apply(prim)
                rigid.CreateKinematicEnabledAttr().Set(bool(enabled))
                if prim.GetPath() == body_path:
                    rigid.CreateVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
                    rigid.CreateAngularVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
                physx_rigid = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
                physx_rigid.CreateDisableGravityAttr().Set(bool(enabled))
            self.drone_body_hold_active = bool(enabled)
            carb.log_warn(
                f"Drone body script hold {'enabled' if enabled else 'disabled'} on "
                f"{len(prims)} rigid prim(s) under {DRONE_PRIM}"
            )
        except Exception as exc:
            carb.log_warn(f"Could not set drone body hold under {DRONE_PRIM}: {exc}")

    def _activate_drone_control_at_current_pose(self, position, yaw: float):
        if self.drone_body_hold_active:
            return
        self._calibrate_drone_body_root_offset()
        self._set_drone_body_hold(True)
        self._set_drone_root_world_pose(position, yaw)

    def _ensure_dynamic_control(self) -> bool:
        try:
            if self.dynamic_control is None:
                enable_extension("omni.isaac.dynamic_control")
                from omni.isaac.dynamic_control import _dynamic_control

                self.dynamic_control_module = _dynamic_control
                self.dynamic_control = _dynamic_control.acquire_dynamic_control_interface()
            invalid = self.dynamic_control_module.INVALID_HANDLE
            if self.drone_body_handle in (None, invalid):
                self.drone_body_handle = self.dynamic_control.get_rigid_body(DRONE_BODY_PRIM)
            if self.drone_body_handle in (None, invalid):
                art = self.dynamic_control.get_articulation(DRONE_PRIM)
                if art not in (None, invalid):
                    self.drone_body_handle = self.dynamic_control.find_articulation_body(art, "body")
                    if self.drone_body_handle not in (None, invalid):
                        carb.log_warn(
                            f"Dynamic control using articulation body handle for {DRONE_BODY_PRIM}"
                        )
            if self.drone_body_handle in (None, invalid) and not self.dynamic_control_warned:
                carb.log_warn(f"Dynamic control could not acquire handle for {DRONE_BODY_PRIM}")
                self.dynamic_control_warned = True
            return self.drone_body_handle not in (None, invalid)
        except Exception as exc:
            if not self.dynamic_control_warned:
                carb.log_warn(f"Dynamic control unavailable for drone body: {exc}")
                self.dynamic_control_warned = True
            return False

    def _set_drone_body_dynamic_pose(self, position, yaw: float) -> bool:
        if not self._ensure_dynamic_control():
            return False
        try:
            half = 0.5 * float(yaw)
            transform = self.dynamic_control_module.Transform()
            transform.p = (
                float(position[0]),
                float(position[1]),
                float(position[2]),
            )
            transform.r = (
                0.0,
                0.0,
                float(math.sin(half)),
                float(math.cos(half)),
            )
            self.dynamic_control.set_rigid_body_disable_gravity(self.drone_body_handle, True)
            self.dynamic_control.set_rigid_body_pose(self.drone_body_handle, transform)
            self.dynamic_control.set_rigid_body_linear_velocity(
                self.drone_body_handle, (0.0, 0.0, 0.0)
            )
            self.dynamic_control.set_rigid_body_angular_velocity(
                self.drone_body_handle, (0.0, 0.0, 0.0)
            )
            return True
        except Exception as exc:
            if not self.dynamic_control_warned:
                carb.log_warn(f"Could not set drone dynamic pose: {exc}")
                self.dynamic_control_warned = True
            return False

    def _calibrate_drone_body_root_offset(self):
        root_pose = self._drone_root_pose_from_stage()
        body_pose = self._drone_body_pose_from_stage()
        if root_pose is None or body_pose is None:
            return
        root_position, _root_orientation, _root_yaw = root_pose
        body_position, _body_orientation, _body_yaw = body_pose
        self.drone_body_root_offset = (
            float(body_position[0]) - float(root_position[0]),
            float(body_position[1]) - float(root_position[1]),
            float(body_position[2]) - float(root_position[2]),
        )
        carb.log_warn(
            "Drone body/root offset calibrated: "
            f"[{self.drone_body_root_offset[0]:.4f}, "
            f"{self.drone_body_root_offset[1]:.4f}, "
            f"{self.drone_body_root_offset[2]:.4f}]"
        )

    def _set_drone_body_local_pose(self):
        body = self.stage.GetPrimAtPath(DRONE_BODY_PRIM)
        if not body or not body.IsValid():
            return
        offset = self.drone_body_root_offset
        orient = Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0))
        _set_xform_ops(
            body,
            translation=Gf.Vec3d(float(offset[0]), float(offset[1]), float(offset[2])),
            orient=orient,
        )

    def _set_drone_root_world_pose(self, position, yaw: float):
        if not self.drone_body_hold_active:
            self._set_drone_body_hold(True)
        prim = self.stage.GetPrimAtPath(DRONE_PRIM)
        if not prim or not prim.IsValid():
            return
        offset = self.drone_body_root_offset
        root_position = (
            float(position[0]) - float(offset[0]),
            float(position[1]) - float(offset[1]),
            float(position[2]) - float(offset[2]),
        )
        half = 0.5 * float(yaw)
        orient = Gf.Quatf(
            float(math.cos(half)),
            Gf.Vec3f(0.0, 0.0, float(math.sin(half))),
        )
        _set_xform_ops(
            prim,
            translation=Gf.Vec3d(root_position[0], root_position[1], root_position[2]),
            orient=orient,
        )
        self._set_drone_body_local_pose()
        self._set_drone_body_dynamic_pose(position, yaw)

    def _translate_drone_root(self, delta, yaw_delta: float = 0.0):
        body_pose = self._drone_body_pose_from_stage()
        if body_pose is None:
            return
        position, _orientation, yaw = body_pose
        target_position = (
            position[0] + float(delta[0]),
            position[1] + float(delta[1]),
            position[2] + float(delta[2]),
        )
        target_yaw = math.atan2(math.sin(yaw + yaw_delta), math.cos(yaw + yaw_delta))
        self._set_drone_root_world_pose(target_position, target_yaw)

    def apply_bobac_kinematic_control(self):
        cmd = self.bobac_kinematic_cmd
        if cmd is None:
            self.bobac_kinematic_last_update = time.monotonic()
            return

        now = time.monotonic()
        if self.bobac_kinematic_last_update is None:
            self.bobac_kinematic_last_update = now
            return
        dt = max(0.0, min(now - self.bobac_kinematic_last_update, 0.05))
        self.bobac_kinematic_last_update = now
        if now - self.bobac_kinematic_cmd_time > 0.35:
            return

        pose = self._bobac_pose_from_stage()
        if pose is None:
            return
        position, _orientation, yaw = pose
        max_linear = 0.60
        max_angular = 0.80
        vx = max(-max_linear, min(max_linear, float(cmd.linear.x)))
        vy = max(-max_linear, min(max_linear, float(cmd.linear.y)))
        wz = max(-max_angular, min(max_angular, float(cmd.angular.z)))
        target_position = (
            position[0] + vx * dt,
            position[1] + vy * dt,
            position[2],
        )
        target_yaw = math.atan2(math.sin(yaw + wz * dt), math.cos(yaw + wz * dt))
        self._set_bobac_base_world_pose(target_position, target_yaw)

    def apply_drone_control(self):
        if BOBAC_ENABLE_PX4:
            return
        now = time.monotonic()
        if self.drone_kinematic_last_update is None:
            self.drone_kinematic_last_update = now
            return
        dt = max(0.0, min(now - self.drone_kinematic_last_update, 0.05))
        self.drone_kinematic_last_update = now

        cmd = self.drone_kinematic_cmd
        if cmd is not None and now - self.drone_kinematic_cmd_time <= 0.35:
            max_linear = 1.2
            max_vertical = 0.6
            max_angular = 1.0
            vx = max(-max_linear, min(max_linear, float(cmd.linear.x)))
            vy = max(-max_linear, min(max_linear, float(cmd.linear.y)))
            vz = max(-max_vertical, min(max_vertical, float(cmd.linear.z)))
            wz = max(-max_angular, min(max_angular, float(cmd.angular.z)))
            self._translate_drone_root((vx * dt, vy * dt, vz * dt), wz * dt)
            return

        if self.drone_target is None:
            return
        pose = self._drone_body_pose_from_stage()
        if pose is None:
            return
        position, _orientation, yaw = pose
        target = self.drone_target
        error = (
            target[0] - position[0],
            target[1] - position[1],
            target[2] - position[2],
        )
        dist = math.sqrt(error[0] * error[0] + error[1] * error[1] + error[2] * error[2])
        yaw_target = yaw if self.drone_target_yaw is None else self.drone_target_yaw
        yaw_error = math.atan2(math.sin(yaw_target - yaw), math.cos(yaw_target - yaw))
        if dist < 0.03 and abs(yaw_error) < 0.05:
            if not self.drone_target_status_sent:
                self.publish_drone_status(
                    f"target_reached x={position[0]:.3f} y={position[1]:.3f} z={position[2]:.3f} yaw={yaw:.3f}"
                )
                self.drone_target_status_sent = True
            return

        max_speed = 0.45
        max_vertical = 0.35
        kp = 0.9
        vx = max(-max_speed, min(max_speed, kp * error[0]))
        vy = max(-max_speed, min(max_speed, kp * error[1]))
        vz = max(-max_vertical, min(max_vertical, kp * error[2]))
        wz = max(-0.8, min(0.8, 1.2 * yaw_error))
        self._translate_drone_root((vx * dt, vy * dt, vz * dt), wz * dt)

    def publish_status(self, text: str):
        if self.status_pub is None:
            return
        msg = self.String()
        msg.data = text
        self.status_pub.publish(msg)

    def publish_drone_world(self):
        if self.drone_world_pub is None:
            return
        prim = self.stage.GetPrimAtPath(DRONE_BODY_PRIM)
        if not prim or not prim.IsValid():
            return
        cache = UsdGeom.XformCache()
        pos = cache.GetLocalToWorldTransform(prim).ExtractTranslation()
        msg = self.PointStamped()
        msg.header.stamp = self.ros_time_msg()
        msg.header.frame_id = "world"
        msg.point.x = float(pos[0])
        msg.point.y = float(pos[1])
        msg.point.z = float(pos[2])
        self.drone_world_pub.publish(msg)

    def update_drone_down_camera_pose(self):
        camera_prim = self.stage.GetPrimAtPath(DRONE_DOWN_CAMERA_PRIM)
        if not camera_prim or not camera_prim.IsValid():
            return
        pose = self._drone_body_pose_from_stage()
        if pose is None:
            return
        position, _orientation, yaw = pose
        cos_yaw = math.cos(float(yaw))
        sin_yaw = math.sin(float(yaw))
        offset_x = (
            DRONE_DOWN_CAMERA_LOCAL_X_OFFSET * cos_yaw
            - DRONE_DOWN_CAMERA_LOCAL_Y_OFFSET * sin_yaw
        )
        offset_y = (
            DRONE_DOWN_CAMERA_LOCAL_X_OFFSET * sin_yaw
            + DRONE_DOWN_CAMERA_LOCAL_Y_OFFSET * cos_yaw
        )
        eye = Gf.Vec3d(
            float(position[0]) + offset_x,
            float(position[1]) + offset_y,
            float(position[2]) + DRONE_DOWN_CAMERA_LOCAL_Z_OFFSET,
        )
        target = Gf.Vec3d(float(position[0]) + offset_x, float(position[1]) + offset_y, float(position[2]) - 1.0)
        # Keep the image yaw-stabilized with the drone heading while looking down.
        up = Gf.Vec3d(-math.sin(float(yaw)), math.cos(float(yaw)), 0.0)
        view = Gf.Matrix4d().SetLookAt(eye, target, up)
        xformable = UsdGeom.Xformable(camera_prim)
        xformable.ClearXformOpOrder()
        xformable.AddTransformOp().Set(view.GetInverse())

    def publish_drone_odom(self):
        if self.drone_odom_pub is None:
            return
        pose = self._drone_body_pose_from_stage()
        if pose is None:
            return
        position, orientation, yaw = pose
        stamp = self.ros_time_msg()
        now = self._stamp_to_float(stamp)
        linear_velocity = (0.0, 0.0, 0.0)
        angular_z = 0.0
        if self.last_drone_odom_time is not None:
            dt = max(now - self.last_drone_odom_time, 1.0e-6)
            linear_velocity = tuple(
                (position[i] - self.last_drone_odom_position[i]) / dt
                for i in range(3)
            )
            angular_z = self._angle_delta(yaw, self.last_drone_odom_yaw) / dt
        self.last_drone_odom_time = now
        self.last_drone_odom_position = position
        self.last_drone_odom_yaw = yaw

        odom = self.Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = "world"
        odom.child_frame_id = "drone_body"
        odom.pose.pose.position.x = position[0]
        odom.pose.pose.position.y = position[1]
        odom.pose.pose.position.z = position[2]
        odom.pose.pose.orientation.x = orientation[0]
        odom.pose.pose.orientation.y = orientation[1]
        odom.pose.pose.orientation.z = orientation[2]
        odom.pose.pose.orientation.w = orientation[3]
        odom.twist.twist.linear.x = linear_velocity[0]
        odom.twist.twist.linear.y = linear_velocity[1]
        odom.twist.twist.linear.z = linear_velocity[2]
        odom.twist.twist.angular.z = angular_z
        self.drone_odom_pub.publish(odom)

    def _find_bobac_world_prim_path(self):
        if self.stage.GetPrimAtPath(BOBAC_BASE_ARTICULATION_PRIM).IsValid():
            return BOBAC_BASE_ARTICULATION_PRIM
        for prim in self.stage.Traverse():
            path = str(prim.GetPath())
            if path.startswith(f"{BOBAC_ROBOT_PRIM}/") and path.endswith(
                "/base_footprint/base_link"
            ):
                return path
        return None

    def publish_bobac_world(self):
        if self.bobac_world_pub is None:
            return
        prim_path = self._find_bobac_world_prim_path()
        if prim_path is None:
            return
        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return
        cache = UsdGeom.XformCache()
        pos = cache.GetLocalToWorldTransform(prim).ExtractTranslation()
        msg = self.PointStamped()
        msg.header.stamp = self.ros_time_msg()
        msg.header.frame_id = "odom"
        msg.point.x = float(pos[0])
        msg.point.y = float(pos[1])
        msg.point.z = float(pos[2])
        self.bobac_world_pub.publish(msg)

    def _bobac_pose_from_stage(self):
        prim_path = self._find_bobac_world_prim_path()
        if prim_path is None:
            return None
        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return None

        matrix = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
        translation = matrix.ExtractTranslation()
        quat = matrix.ExtractRotation().GetQuat()
        imag = quat.GetImaginary()
        qx = float(imag[0])
        qy = float(imag[1])
        qz = float(imag[2])
        qw = float(quat.GetReal())
        yaw = math.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )
        return (
            (float(translation[0]), float(translation[1]), float(translation[2])),
            (qx, qy, qz, qw),
            yaw,
        )

    @staticmethod
    def _angle_delta(current: float, previous: float) -> float:
        return math.atan2(math.sin(current - previous), math.cos(current - previous))

    @staticmethod
    def _stamp_to_float(stamp) -> float:
        return float(stamp.sec) + float(stamp.nanosec) * 1.0e-9

    def _make_transform(self, stamp, parent: str, child: str, translation, rotation):
        transform = self.TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = parent
        transform.child_frame_id = child
        transform.transform.translation.x = float(translation[0])
        transform.transform.translation.y = float(translation[1])
        transform.transform.translation.z = float(translation[2])
        transform.transform.rotation.x = float(rotation[0])
        transform.transform.rotation.y = float(rotation[1])
        transform.transform.rotation.z = float(rotation[2])
        transform.transform.rotation.w = float(rotation[3])
        return transform

    def publish_bobac_static_tf(self, stamp):
        if self.tf_static_pub is None or self.sent_bobac_static_tf:
            return
        msg = self.TFMessage()
        msg.transforms = [
            self._make_transform(
                stamp,
                "base_link",
                "fuse_lidar_link",
                (0.0, 0.0, 0.25),
                (0.0, 0.0, 0.0, 1.0),
            )
        ]
        self.tf_static_pub.publish(msg)
        self.sent_bobac_static_tf = True

    def _refresh_scan_obstacles(self):
        now = time.monotonic()
        if self.scan_obstacles and now - self.scan_obstacles_refresh_time < 2.0:
            return

        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
            useExtentsHint=True,
        )
        boxes = []
        exclude_tokens = (
            "/World/layout/bobac3_serverbot",
            "/World/quadrotor",
            "/World/physicsScene",
            "/World/Cargo",
            "/World/ArmCamera",
        )
        for prim in self.stage.Traverse():
            if not prim.IsActive() or not prim.IsA(UsdGeom.Boundable):
                continue
            path = str(prim.GetPath())
            if not (
                path.startswith("/World/layout/scene/")
                or path.startswith("/World/layout/caughting/")
                or path.startswith("/World/layout/scene_obstacle/")
            ):
                continue
            if any(token in path for token in exclude_tokens):
                continue
            try:
                aligned = cache.ComputeWorldBound(prim).ComputeAlignedBox()
                if aligned.IsEmpty():
                    continue
                bmin = aligned.GetMin()
                bmax = aligned.GetMax()
            except Exception:
                continue

            min_x, min_y, min_z = float(bmin[0]), float(bmin[1]), float(bmin[2])
            max_x, max_y, max_z = float(bmax[0]), float(bmax[1]), float(bmax[2])
            dx = max_x - min_x
            dy = max_y - min_y
            dz = max_z - min_z
            if dx <= 0.005 or dy <= 0.005 or dz <= 0.005:
                continue
            if max_z < 0.08:
                continue
            if dx > 25.0 or dy > 25.0:
                continue
            inflate = 0.03
            boxes.append((min_x - inflate, min_y - inflate, max_x + inflate, max_y + inflate))

        self.scan_obstacles = boxes
        self.scan_obstacles_refresh_time = now

    @staticmethod
    def _ray_aabb_distance(origin, direction, box, range_min: float, range_max: float):
        ox, oy = origin
        dx, dy = direction
        min_x, min_y, max_x, max_y = box
        tmin = 0.0
        tmax = range_max
        for o, d, low, high in ((ox, dx, min_x, max_x), (oy, dy, min_y, max_y)):
            if abs(d) < 1.0e-9:
                if o < low or o > high:
                    return None
                continue
            t1 = (low - o) / d
            t2 = (high - o) / d
            if t1 > t2:
                t1, t2 = t2, t1
            tmin = max(tmin, t1)
            tmax = min(tmax, t2)
            if tmin > tmax:
                return None
        hit = max(tmin, range_min)
        if hit < range_min or hit > range_max:
            return None
        return hit

    def publish_bobac_scan(self):
        if self.scan_pub is None or self.scan_fuse_pub is None:
            return
        pose = self._bobac_pose_from_stage()
        if pose is None:
            return

        self._refresh_scan_obstacles()
        position, _orientation, yaw = pose
        lidar_z_offset = 0.25
        origin_x = float(position[0])
        origin_y = float(position[1])
        angle_min = -math.pi
        angle_increment = math.radians(0.5)
        ray_count = 720
        range_min = 0.08
        range_max = 12.0
        ranges = [math.inf] * ray_count

        for index in range(ray_count):
            local_angle = angle_min + index * angle_increment
            world_angle = yaw + local_angle
            direction = (math.cos(world_angle), math.sin(world_angle))
            best = range_max
            hit_any = False
            for box in self.scan_obstacles:
                distance = self._ray_aabb_distance(
                    (origin_x, origin_y), direction, box, range_min, range_max
                )
                if distance is not None and distance < best:
                    best = distance
                    hit_any = True
            if hit_any:
                ranges[index] = best

        scan = self.LaserScan()
        scan.header.stamp = self.ros_time_msg()
        scan.header.frame_id = "fuse_lidar_link"
        scan.angle_min = angle_min
        scan.angle_max = angle_min + (ray_count - 1) * angle_increment
        scan.angle_increment = angle_increment
        scan.time_increment = 0.0
        scan.scan_time = 0.1
        scan.range_min = range_min
        scan.range_max = range_max
        scan.ranges = ranges
        self.scan_pub.publish(scan)
        self.scan_fuse_pub.publish(scan)

    def publish_arm_camera_tf(self):
        if self.tf_pub is None:
            return
        base = self.stage.GetPrimAtPath(BOBAC_ARM_ARTICULATION_PRIM)
        camera = self.stage.GetPrimAtPath(ARM_CAMERA_PRIM)
        if not base or not base.IsValid() or not camera or not camera.IsValid():
            return

        cache = UsdGeom.XformCache()
        base_world = cache.GetLocalToWorldTransform(base)
        base_pos = base_world.ExtractTranslation()
        base_quat = base_world.ExtractRotation().GetQuat()
        base_imag = base_quat.GetImaginary()
        base_rot = _quat_xyzw_to_matrix(
            (
                float(base_imag[0]),
                float(base_imag[1]),
                float(base_imag[2]),
                float(base_quat.GetReal()),
            )
        )
        world_to_base_rot = _transpose3(base_rot)

        right, down, forward = _arm_camera_optical_world_axes()
        world_from_camera_rot = (
            (right[0], down[0], forward[0]),
            (right[1], down[1], forward[1]),
            (right[2], down[2], forward[2]),
        )
        base_from_camera_rot = _matmul3(world_to_base_rot, world_from_camera_rot)
        camera_pos_world = ARM_CAMERA_EYE
        camera_pos_base = _matvec3(
            world_to_base_rot,
            (
                float(camera_pos_world[0]) - float(base_pos[0]),
                float(camera_pos_world[1]) - float(base_pos[1]),
                float(camera_pos_world[2]) - float(base_pos[2]),
            ),
        )
        camera_quat_base = _rotation_matrix_to_quat_xyzw(base_from_camera_rot)

        msg = self.TFMessage()
        msg.transforms = [
            self._make_transform(
                self.ros_time_msg(),
                "base_link_arm",
                ARM_CAMERA_FRAME,
                camera_pos_base,
                camera_quat_base,
            )
        ]
        self.tf_pub.publish(msg)

    def publish_bobac_odom(self):
        if self.odom_pub is None or self.tf_pub is None:
            return
        pose = self._bobac_pose_from_stage()
        if pose is None:
            return

        position, orientation, yaw = pose
        stamp = self.ros_time_msg()
        now = self._stamp_to_float(stamp)
        linear_velocity = (0.0, 0.0, 0.0)
        angular_z = 0.0
        if self.last_bobac_odom_time is not None:
            dt = max(now - self.last_bobac_odom_time, 1.0e-6)
            linear_velocity = tuple(
                (position[i] - self.last_bobac_odom_position[i]) / dt
                for i in range(3)
            )
            angular_z = self._angle_delta(yaw, self.last_bobac_odom_yaw) / dt

        self.last_bobac_odom_time = now
        self.last_bobac_odom_position = position
        self.last_bobac_odom_yaw = yaw

        odom = self.Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"
        odom.pose.pose.position.x = position[0]
        odom.pose.pose.position.y = position[1]
        odom.pose.pose.position.z = position[2]
        odom.pose.pose.orientation.x = orientation[0]
        odom.pose.pose.orientation.y = orientation[1]
        odom.pose.pose.orientation.z = orientation[2]
        odom.pose.pose.orientation.w = orientation[3]
        odom.twist.twist.linear.x = linear_velocity[0]
        odom.twist.twist.linear.y = linear_velocity[1]
        odom.twist.twist.linear.z = linear_velocity[2]
        odom.twist.twist.angular.z = angular_z
        self.odom_pub.publish(odom)

        tf_msg = self.TFMessage()
        tf_msg.transforms = [
            self._make_transform(stamp, "odom", "base_link", position, orientation)
        ]
        self.tf_pub.publish(tf_msg)
        self.publish_bobac_static_tf(stamp)

    def _on_command(self, msg):
        command = msg.data.strip().lower()
        try:
            if command == "bottom_open":
                self.cargo_runtime.bottom_open()
                self.publish_status("bottom_opened payload_released")
            elif command == "bottom_close":
                self.cargo_runtime.bottom_close()
                self.publish_status("bottom_closed")
            elif command in {"left_open", "side_open"}:
                self.cargo_runtime.left_open()
                self.publish_status("left_opened")
            elif command in {"left_close", "side_close"}:
                self.cargo_runtime.left_close()
                self.publish_status("left_closed")
            elif command in {"payload_lock", "lock_payload"}:
                self.cargo_runtime.lock_payload()
                self.publish_status("payload_locked=True")
            elif command in {"payload_release", "release_payload"}:
                self.cargo_runtime.release_payload()
                self.publish_status("payload_locked=False")
            elif command == "status":
                payload_locked = self.cargo_runtime.payload_locked()
                self.publish_status(f"payload_locked={payload_locked}")
            else:
                self.publish_status(f"unknown_command {command}")
        except Exception as exc:
            carb.log_error(f"Cargo command failed: {exc}")
            self.publish_status(f"error {exc}")

    def shutdown(self):
        if self.executor is not None and self.node is not None:
            self.executor.remove_node(self.node)
        if self.node is not None:
            self.node.destroy_node()
        if self.owns_rclpy and self.rclpy is not None and self.rclpy.ok():
            self.rclpy.shutdown()
        self.executor = None
        self.node = None


class ExistingPrimVehicleState:
    """Passive state adapter for Pegasus versions without attach_existing."""

    def __init__(self, stage, prim_path: str):
        self.stage = stage
        self.prim_path = prim_path
        self._last_time = None
        self._last_position = None
        self._state = SimpleNamespace(
            position=(0.0, 0.0, 0.0),
            attitude=(0.0, 0.0, 0.0, 1.0),
            linear_velocity=(0.0, 0.0, 0.0),
            angular_velocity=(0.0, 0.0, 0.0),
        )

    @property
    def state(self):
        prim = self.stage.GetPrimAtPath(self.prim_path)
        if not prim or not prim.IsValid():
            return self._state

        now = time.monotonic()
        matrix = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
        translation = matrix.ExtractTranslation()
        position = (
            float(translation[0]),
            float(translation[1]),
            float(translation[2]),
        )
        quat = matrix.ExtractRotation().GetQuat()
        imag = quat.GetImaginary()
        attitude = (
            float(imag[0]),
            float(imag[1]),
            float(imag[2]),
            float(quat.GetReal()),
        )

        linear_velocity = (0.0, 0.0, 0.0)
        if self._last_time is not None and self._last_position is not None:
            dt = max(now - self._last_time, 1.0e-6)
            linear_velocity = tuple(
                (position[i] - self._last_position[i]) / dt for i in range(3)
            )

        self._last_time = now
        self._last_position = position
        self._state = SimpleNamespace(
            position=position,
            attitude=attitude,
            linear_velocity=linear_velocity,
            angular_velocity=(0.0, 0.0, 0.0),
        )
        return self._state


class CargoDeliverySceneApp:
    def __init__(self):
        self.timeline = omni.timeline.get_timeline_interface()
        self.pg = PegasusInterface()
        world = World.instance()
        if world is None or not hasattr(world, "scene"):
            if world is not None:
                World.clear_instance()
            world = World(**self.pg._world_settings)
        self.pg._world = world
        self.world = self.pg.world
        self.stage = self.world.stage
        self.cargo_runtime = CargoBayRuntime(self.stage)
        self.ros = RosCargoInterface(self.cargo_runtime, self.stage)
        self.vehicle = None
        self.lidar_prim_path = None
        self._update_sub = None
        self.spawn_world = None
        self.spawn_orientation = None

    @staticmethod
    def _has_usable_physics_context(world) -> bool:
        try:
            physics_context = world.get_physics_context()
        except Exception:
            return False
        if physics_context is None:
            return False
        return getattr(physics_context, "_physx_interface", None) is not None

    async def ensure_world_ready_async(self):
        if self._has_usable_physics_context(self.world):
            self.stage = self.world.stage
            self.cargo_runtime.stage = self.stage
            self.ros.stage = self.stage
            return

        carb.log_warn("Initializing Isaac World physics context for Pegasus callbacks")
        await self.world.initialize_simulation_context_async()
        if not self._has_usable_physics_context(self.world):
            raise RuntimeError("Isaac World physics context is still unavailable")
        self.stage = self.world.stage
        self.cargo_runtime.stage = self.stage
        self.ros.stage = self.stage
        

    def cleanup_existing_scene(self):
        self.timeline.stop()
        try:
            self.world.clear_all_callbacks()
        except Exception as exc:
            carb.log_warn(f"Could not clear old world callbacks: {exc}")
        try:
            self.world.scene.clear(registry_only=True)
        except Exception as exc:
            carb.log_warn(f"Could not clear old world scene registry: {exc}")
        try:
            self.pg.vehicle_manager.remove_all_vehicles()
        except Exception as exc:
            carb.log_warn(f"Could not clear old Pegasus vehicles: {exc}")

        world_prim = self.stage.GetPrimAtPath(WORLD_PRIM)
        if world_prim and world_prim.IsValid():
            try:
                world_prim.GetReferences().ClearReferences()
            except Exception as exc:
                carb.log_warn(f"Could not clear old World references: {exc}")

        for path in (
            STATE_GRAPH_PATH,
            LIDAR_GRAPH_PATH,
            CLOCK_GRAPH_PATH,
            ARM_CAMERA_GRAPH_PATH,
            ARM_CAMERA_PRIM,
            MERCURY_RUNTIME_JOINT_GRAPH,
            BOBAC_RUNTIME_JOINT_GRAPH,
            RUNTIME_SCOPE,
            CARGO_BAY_PRIM,
            DRONE_PRIM,
            LAYOUT_PRIM,
        ):
            if self.stage.GetPrimAtPath(path).IsValid():
                self.stage.RemovePrim(path)

        hydra_root = "/Render/OmniverseKit/HydraTextures"
        for prim in list(self.stage.Traverse()):
            path = str(prim.GetPath())
            if (
                path.startswith(f"{hydra_root}/avoidance_lidar")
                and path.count("/") == 4
            ):
                self.stage.RemovePrim(path)

    def _world_matrix(self, prim_path):
        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return None
        return UsdGeom.XformCache().GetLocalToWorldTransform(prim)

    def _world_position(self, prim_path):
        matrix = self._world_matrix(prim_path)
        if matrix is None:
            return None
        pos = matrix.ExtractTranslation()
        return (float(pos[0]), float(pos[1]), float(pos[2]))

    def _translate_world(self, prim_path, delta_world):
        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            raise RuntimeError(f"prim not found: {prim_path}")
        current = self._world_position(prim_path)
        if current is None:
            raise RuntimeError(f"cannot read world position: {prim_path}")
        target_world = tuple(float(current[i]) + float(delta_world[i]) for i in range(3))

        parent = prim.GetParent()
        local_position = Gf.Vec3d(*target_world)
        if parent and parent.IsValid():
            parent_world = UsdGeom.XformCache().GetLocalToWorldTransform(parent)
            local_position = parent_world.GetInverse().Transform(local_position)

        xform = UsdGeom.Xformable(prim)
        translate_op = None
        for op in xform.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                translate_op = op
                break
        if translate_op is None:
            translate_op = xform.AddTranslateOp()
        translate_op.Set(local_position)
        return target_world

    def _active_material_robot_prim(self):
        if self.stage.GetPrimAtPath(MERCURY_ROBOT_PRIM).IsValid():
            return MERCURY_ROBOT_PRIM
        if self.stage.GetPrimAtPath(BOBAC_ROBOT_PRIM).IsValid():
            return BOBAC_ROBOT_PRIM
        return MERCURY_ROBOT_PRIM

    def _active_material_arm_prim(self):
        if self.stage.GetPrimAtPath(MERCURY_ARM_ARTICULATION_PRIM).IsValid():
            return MERCURY_ARM_ARTICULATION_PRIM
        if self.stage.GetPrimAtPath(BOBAC_ARM_ARTICULATION_PRIM).IsValid():
            return BOBAC_ARM_ARTICULATION_PRIM
        return MERCURY_ARM_ARTICULATION_PRIM

    def _base_relative_position(self, world_position):
        arm_prim = self._active_material_arm_prim()
        base_matrix = self._world_matrix(arm_prim)
        if base_matrix is None:
            raise RuntimeError(f"base prim not found: {arm_prim}")
        local = base_matrix.GetInverse().Transform(Gf.Vec3d(*world_position))
        if self.stage.GetPrimAtPath(BOBAC_ARM_ARTICULATION_PRIM).IsValid():
            return (float(local[0]), float(local[1]), float(local[2]))
        return (float(local[0]), float(local[1]), float(local[2]))

    def _base_direction_to_world(self, base_delta):
        arm_prim = self._active_material_arm_prim()
        base_matrix = self._world_matrix(arm_prim)
        if base_matrix is None:
            raise RuntimeError(f"base prim not found: {arm_prim}")
        origin = base_matrix.ExtractTranslation()
        moved = base_matrix.Transform(Gf.Vec3d(*base_delta))
        return (
            float(moved[0] - origin[0]),
            float(moved[1] - origin[1]),
            float(moved[2] - origin[2]),
        )

    def prealign_robot_for_material_grasp(self, target_id=None):
        if not MATERIAL_PREALIGN_BASE_ON_SETUP:
            carb.log_warn("Material base prealignment disabled by env")
            return None
        target = (target_id or MATERIAL_PREALIGN_TARGET or "red_pencil").lower()
        cfg = next(
            (item for item in MATERIAL_TASK_TARGETS if item["id"] == target),
            MATERIAL_TASK_TARGETS[0],
        )
        object_world = self._world_position(cfg["move_path"])
        if object_world is None:
            carb.log_warn(f"Material prealignment skipped; missing {cfg['move_path']}")
            return None
        before_base = self._base_relative_position(object_world)
        desired_nominal = MATERIAL_RIGHT_ARM_GRASP_BASE_TARGET
        desired = desired_nominal
        bobac_compensation = (0.0, 0.0, 0.0)
        if self.stage.GetPrimAtPath(BOBAC_ARM_ARTICULATION_PRIM).IsValid():
            bobac_compensation = MATERIAL_BOBAC_STARTUP_PREALIGN_COMPENSATION
            desired = tuple(
                float(desired_nominal[i]) + float(bobac_compensation[i])
                for i in range(3)
            )
        base_delta = (
            float(before_base[0]) - float(desired[0]),
            float(before_base[1]) - float(desired[1]),
            0.0,
        )
        if abs(base_delta[0]) < 1e-4 and abs(base_delta[1]) < 1e-4:
            world_delta = (0.0, 0.0, 0.0)
        else:
            world_delta = self._base_direction_to_world(base_delta)
            self._translate_world(self._active_material_robot_prim(), world_delta)
            UsdGeom.XformCache().Clear()
        after_world = self._world_position(cfg["move_path"])
        after_base = self._base_relative_position(after_world)
        payload = {
            "target": cfg["id"],
            "scene": ASSEMBLED_SCENE_USD,
            "object_world": [round(v, 6) for v in object_world],
            "desired_object_base": [round(float(v), 6) for v in desired_nominal],
            "startup_commanded_object_base": [round(float(v), 6) for v in desired],
            "bobac_startup_compensation": [
                round(float(v), 6) for v in bobac_compensation
            ],
            "before_object_base": [round(v, 6) for v in before_base],
            "base_delta_command": [round(v, 6) for v in base_delta],
            "world_delta_applied": [round(v, 6) for v in world_delta],
            "after_object_base": [round(v, 6) for v in after_base],
        }
        carb.log_warn("Material setup prealignment: " + json.dumps(payload))
        return payload

    def position_drone_cargo_for_material_place(self):
        drone = self.stage.GetPrimAtPath(DRONE_PRIM)
        cargo = self.stage.GetPrimAtPath(CARGO_BAY_PRIM)
        if not drone or not drone.IsValid() or not cargo or not cargo.IsValid():
            carb.log_warn("Material cargo place alignment skipped; drone/cargo missing")
            return None
        cache = UsdGeom.XformCache()
        current_payload = cache.GetLocalToWorldTransform(cargo).Transform(
            _vec3d(CARGO_LOCAL_PAYLOAD_OFFSET)
        )
        delta = (
            float(MATERIAL_PLACE_WORLD[0]) - float(current_payload[0]),
            float(MATERIAL_PLACE_WORLD[1]) - float(current_payload[1]),
            float(MATERIAL_PLACE_WORLD[2]) - float(current_payload[2]),
        )
        self._translate_world(DRONE_PRIM, delta)
        UsdGeom.XformCache().Clear()
        moved_payload = UsdGeom.XformCache().GetLocalToWorldTransform(cargo).Transform(
            _vec3d(CARGO_LOCAL_PAYLOAD_OFFSET)
        )
        payload = {
            "task": "material_cargo_place_alignment",
            "drone": DRONE_PRIM,
            "cargo": CARGO_BAY_PRIM,
            "before_payload_world": [
                round(float(current_payload[i]), 6) for i in range(3)
            ],
            "target_payload_world": [round(float(v), 6) for v in MATERIAL_PLACE_WORLD],
            "delta_world": [round(float(v), 6) for v in delta],
            "after_payload_world": [
                round(float(moved_payload[i]), 6) for i in range(3)
            ],
        }
        carb.log_warn("Material cargo place alignment: " + json.dumps(payload))
        return payload

    def _ensure_grasp_physics_material(self):
        material = UsdShade.Material.Define(self.stage, MATERIAL_GRASP_PHYSICS_MATERIAL)
        physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        physics_material.CreateStaticFrictionAttr().Set(MATERIAL_GRASP_FRICTION)
        physics_material.CreateDynamicFrictionAttr().Set(MATERIAL_GRASP_FRICTION)
        physics_material.CreateRestitutionAttr().Set(0.0)
        return material

    def _create_material_grasp_proxy(
        self,
        move_path,
        scale=MATERIAL_GRASP_PROXY_SCALE,
        translation=(0.0, 0.0, 0.0),
    ):
        proxy_path = f"{move_path}/{MATERIAL_GRASP_PROXY_SUFFIX}"
        if not MATERIAL_ENABLE_GRASP_PROXY:
            if self.stage.GetPrimAtPath(proxy_path).IsValid():
                self.stage.RemovePrim(proxy_path)
            return None
        if self.stage.GetPrimAtPath(proxy_path).IsValid():
            self.stage.RemovePrim(proxy_path)
        proxy = UsdGeom.Cube.Define(self.stage, proxy_path)
        proxy.CreateSizeAttr(1.0)
        proxy_prim = proxy.GetPrim()
        UsdPhysics.CollisionAPI.Apply(proxy_prim).CreateCollisionEnabledAttr(True)
        UsdShade.MaterialBindingAPI(proxy_prim).Bind(self._ensure_grasp_physics_material())
        _set_xform_ops(
            proxy_prim,
            translation=translation,
            scale=scale,
        )
        UsdGeom.Imageable(proxy_prim).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
        return proxy_path

    def _bind_startup_high_friction_gripper(self):
        material = self._ensure_grasp_physics_material()
        robot_prim = self._active_material_robot_prim()
        bound = 0
        for prim in self.stage.Traverse():
            path = str(prim.GetPath())
            low = path.lower()
            if not path.startswith(robot_prim):
                continue
            if "gripper" not in low and "finger" not in low:
                continue
            try:
                UsdShade.MaterialBindingAPI(prim).Bind(material)
                bound += 1
            except Exception:
                continue
        carb.log_warn(f"Bound startup high-friction grasp material to {bound} gripper prims")
        return bound

    def prepare_material_object_for_startup(self, target_id=None):
        target = (target_id or MATERIAL_PREALIGN_TARGET or "white_pencil").lower()
        cfg = next(
            (item for item in MATERIAL_TASK_TARGETS if item["id"] == target),
            MATERIAL_TASK_TARGETS[0],
        )
        prim = self.stage.GetPrimAtPath(cfg["move_path"])
        if not prim or not prim.IsValid():
            carb.log_warn(f"Material startup physics skipped; missing {cfg['move_path']}")
            return None

        rigid_prim = prim
        parent = prim.GetParent()
        while parent and parent.IsValid():
            if "PhysicsRigidBodyAPI" in [str(x) for x in parent.GetAppliedSchemas()]:
                rigid_prim = parent
                break
            parent = parent.GetParent()

        rigid = UsdPhysics.RigidBodyAPI.Apply(rigid_prim)
        rigid.CreateRigidBodyEnabledAttr().Set(True)
        rigid.CreateKinematicEnabledAttr().Set(False)
        UsdPhysics.MassAPI.Apply(rigid_prim).CreateMassAttr().Set(0.005)
        try:
            physx_rigid = PhysxSchema.PhysxRigidBodyAPI.Apply(rigid_prim)
            physx_rigid.CreateLinearDampingAttr().Set(8.0)
            physx_rigid.CreateAngularDampingAttr().Set(8.0)
            physx_rigid.CreateSolverPositionIterationCountAttr().Set(64)
            physx_rigid.CreateSolverVelocityIterationCountAttr().Set(16)
        except Exception as exc:
            carb.log_warn(f"Could not tune startup material rigid body damping: {exc}")

        mesh_count = 0
        for child in Usd.PrimRange(prim):
            if child == prim or child.GetTypeName() != "Mesh":
                continue
            try:
                UsdPhysics.CollisionAPI.Apply(child).CreateCollisionEnabledAttr(True)
                UsdPhysics.MeshCollisionAPI.Apply(child).CreateApproximationAttr().Set(
                    "convexHull"
                )
                mesh_count += 1
            except Exception as exc:
                carb.log_warn(
                    f"Could not enable startup material mesh collision on "
                    f"{child.GetPath()}: {exc}"
                )

        payload = {
            "target": cfg["id"],
            "move_path": cfg["move_path"],
            "rigid_path": str(rigid_prim.GetPath()),
            "kinematic": False,
            "mesh_collisions": mesh_count,
            "object_world": [
                round(v, 6) for v in (self._world_position(cfg["move_path"]) or ())
            ],
        }
        carb.log_warn("Material startup physics prepared: " + json.dumps(payload))
        return payload

    def lock_robot_base_for_manipulation(self):
        if not MATERIAL_LOCK_BASE_ON_SETUP:
            carb.log_warn("Material base lock disabled by env")
            return None
        arm_prim = self._active_material_arm_prim()
        prim = self.stage.GetPrimAtPath(arm_prim)
        if not prim or not prim.IsValid():
            carb.log_warn(f"Material base lock skipped; missing {arm_prim}")
            return None
        if self.stage.GetPrimAtPath(MATERIAL_BASE_LOCK_JOINT_PATH).IsValid():
            self.stage.RemovePrim(MATERIAL_BASE_LOCK_JOINT_PATH)
        matrix = self._world_matrix(arm_prim)
        if matrix is None:
            carb.log_warn("Material base lock skipped; cannot read base transform")
            return None
        pos = matrix.ExtractTranslation()
        rot = _quatf_from_matrix_rotation(matrix)
        joint = UsdPhysics.FixedJoint.Define(
            self.stage, Sdf.Path(MATERIAL_BASE_LOCK_JOINT_PATH)
        )
        joint.CreateBody1Rel().SetTargets([Sdf.Path(arm_prim)])
        joint.CreateExcludeFromArticulationAttr().Set(True)
        joint.CreateJointEnabledAttr().Set(True)
        joint.CreateCollisionEnabledAttr().Set(False)
        joint.CreateLocalPos0Attr().Set(_vec3f((pos[0], pos[1], pos[2])))
        joint.CreateLocalRot0Attr().Set(rot)
        joint.CreateLocalPos1Attr().Set(_vec3f((0.0, 0.0, 0.0)))
        joint.CreateLocalRot1Attr().Set(_identity_quatf())
        payload = {
            "robot_base": arm_prim,
            "joint": MATERIAL_BASE_LOCK_JOINT_PATH,
            "world": [round(float(v), 6) for v in pos],
        }
        carb.log_warn("Material manipulation base locked: " + json.dumps(payload))
        return payload

    def load_assembled_scene(self):
        if not ASSEMBLED_SCENE_USD:
            raise RuntimeError("No assembled scene selected; pass --world X1 or --world Bobac")
        if not os.path.exists(ASSEMBLED_SCENE_USD):
            raise RuntimeError(
                f"Missing assembled cargo delivery scene at {ASSEMBLED_SCENE_USD}"
            )

        world_prim = self.stage.GetPrimAtPath(WORLD_PRIM)
        if not world_prim or not world_prim.IsValid():
            world_prim = self.stage.DefinePrim(WORLD_PRIM, "Xform")
        world_prim.GetReferences().ClearReferences()
        world_prim.GetReferences().AddReference(str(ASSEMBLED_SCENE_USD), "/World")
        for path in SAVED_RUNTIME_PRIMS:
            prim = self.stage.GetPrimAtPath(path)
            if prim.IsValid() and prim.IsActive():
                prim.SetActive(False)
        carb.log_warn(
            f"Assembled cargo delivery scene reference loaded from {ASSEMBLED_SCENE_USD}"
        )

    def load_layout(self):
        if not ARGS.world:
            return
        self.load_assembled_scene()

    def resolve_existing_drone_spawn(self) -> tuple[float, float, float]:
        drone = _ensure_prim(self.stage, DRONE_PRIM, "assembled drone")
        cache = UsdGeom.XformCache()
        translation = cache.GetLocalToWorldTransform(drone).ExtractTranslation()
        return (float(translation[0]), float(translation[1]), float(translation[2]))

    def hide_lidar_prims(self, lidar_prim):
        hide_roots = []
        # Do not hide lidar_prim.GetParent(): for a runtime-created LiDAR the
        # parent is /World/quadrotor/body, which would hide the whole drone.
        for path in (LIDAR_ROOT_PRIM, str(lidar_prim.GetPath())):
            prim = self.stage.GetPrimAtPath(path)
            if prim and prim.IsValid():
                hide_roots.append(prim)

        hidden_paths = set()
        for root_prim in hide_roots:
            for prim in Usd.PrimRange(root_prim):
                path = str(prim.GetPath())
                if path in hidden_paths:
                    continue
                hidden_paths.add(path)
                imageable = UsdGeom.Imageable(prim)
                if imageable:
                    imageable.CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)

    @staticmethod
    def is_rtx_lidar_sensor_prim(prim) -> bool:
        if not prim or not prim.IsValid() or not prim.IsActive():
            return False
        return prim.IsA(UsdGeom.Camera) and prim.HasAPI(
            IsaacSensorSchema.IsaacRtxLidarSensorAPI
        )

    def find_rtx_lidar_sensor_under(self, root_prim):
        if self.is_rtx_lidar_sensor_prim(root_prim):
            return root_prim
        if not root_prim or not root_prim.IsValid():
            return None
        for prim in Usd.PrimRange(root_prim):
            if prim == root_prim:
                continue
            if self.is_rtx_lidar_sensor_prim(prim):
                return prim
        return None

    def resolve_existing_lidar(self):
        for path in (LIDAR_PRIM, LIDAR_ROOT_PRIM):
            prim = self.stage.GetPrimAtPath(path)
            sensor_prim = self.find_rtx_lidar_sensor_under(prim)
            if sensor_prim:
                self.hide_lidar_prims(sensor_prim)
                self.lidar_prim_path = str(sensor_prim.GetPath())
                carb.log_warn(
                    f"Using existing hidden RTX LiDAR at {self.lidar_prim_path}"
                )
                return self.lidar_prim_path
            if prim and prim.IsValid() and prim.IsActive():
                carb.log_warn(
                    f"Ignoring non-sensor LiDAR wrapper prim at {prim.GetPath()}"
                )
        return None

    def attach_hidden_lidar(self):
        existing_lidar = self.resolve_existing_lidar()
        if existing_lidar:
            return existing_lidar

        try:
            enable_extension("isaacsim.core.nodes")
            enable_extension("isaacsim.ros2.bridge")
        except Exception as exc:
            carb.log_warn(f"Could not enable LiDAR ROS 2 extensions: {exc}")

        _, lidar_prim = omni.kit.commands.execute(
            "IsaacSensorCreateRtxLidar",
            path="/avoidance_lidar",
            parent=DRONE_BODY_PRIM,
            config=LIDAR_CONFIG,
            translation=Gf.Vec3d(*LIDAR_TRANSLATION),
            orientation=Gf.Quatd(0.70710678, 0.70710678, 0.0, 0.0),
            visibility=False,
        )
        self.hide_lidar_prims(lidar_prim)
        self.lidar_prim_path = str(lidar_prim.GetPath())
        return self.lidar_prim_path

    def attach_bobac_hidden_lidar(self):
        for path in (BOBAC_LIDAR_PRIM, BOBAC_LIDAR_ROOT_PRIM):
            prim = self.stage.GetPrimAtPath(path)
            sensor_prim = self.find_rtx_lidar_sensor_under(prim)
            if sensor_prim:
                self.hide_lidar_prims(sensor_prim)
                self.lidar_prim_path = str(sensor_prim.GetPath())
                carb.log_warn(
                    f"Using existing Bobac hidden RTX LiDAR at {self.lidar_prim_path}"
                )
                return self.lidar_prim_path

        parent_prim = self.stage.GetPrimAtPath(BOBAC_BASE_ARTICULATION_PRIM)
        if not parent_prim or not parent_prim.IsValid():
            raise RuntimeError(f"Bobac base prim not found: {BOBAC_BASE_ARTICULATION_PRIM}")

        try:
            enable_extension("isaacsim.core.nodes")
            enable_extension("isaacsim.ros2.bridge")
        except Exception as exc:
            carb.log_warn(f"Could not enable Bobac LiDAR ROS 2 extensions: {exc}")

        _, lidar_prim = omni.kit.commands.execute(
            "IsaacSensorCreateRtxLidar",
            path="/avoidance_lidar",
            parent=BOBAC_BASE_ARTICULATION_PRIM,
            config=LIDAR_CONFIG,
            translation=Gf.Vec3d(*LIDAR_TRANSLATION),
            orientation=Gf.Quatd(0.70710678, 0.70710678, 0.0, 0.0),
            visibility=False,
        )
        self.hide_lidar_prims(lidar_prim)
        self.lidar_prim_path = str(lidar_prim.GetPath())
        return self.lidar_prim_path

    def build_ros_clock_graph(self):
        if self.stage.GetPrimAtPath(CLOCK_GRAPH_PATH).IsValid():
            carb.log_warn(f"Using existing ROS clock graph at {CLOCK_GRAPH_PATH}")
            return

        og.Controller.edit(
            {"graph_path": CLOCK_GRAPH_PATH, "evaluator_name": "execution"},
            {
                og.Controller.Keys.CREATE_NODES: [
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                    ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
                    ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                ],
                og.Controller.Keys.SET_VALUES: [
                    ("ReadSimTime.inputs:resetOnStop", False),
                    ("PublishClock.inputs:topicName", "clock"),
                ],
                og.Controller.Keys.CONNECT: [
                    ("OnPlaybackTick.outputs:tick", "PublishClock.inputs:execIn"),
                    ("Context.outputs:context", "PublishClock.inputs:context"),
                    (
                        "ReadSimTime.outputs:simulationTime",
                        "PublishClock.inputs:timeStamp",
                    ),
                ],
            },
        )

    def build_lidar_publish_graph(self, lidar_prim_path):
        if self.stage.GetPrimAtPath(LIDAR_GRAPH_PATH).IsValid():
            carb.log_warn(f"Using existing LiDAR publish graph at {LIDAR_GRAPH_PATH}")
            return

        render_product = rep.create.render_product(
            lidar_prim_path, resolution=(1, 1), name="avoidance_lidar"
        )
        render_product_path = render_product.path

        og.Controller.edit(
            {"graph_path": LIDAR_GRAPH_PATH, "evaluator_name": "execution"},
            {
                og.Controller.Keys.CREATE_NODES: [
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    ("PointCloudPublish", "isaacsim.ros2.bridge.ROS2RtxLidarHelper"),
                ],
                og.Controller.Keys.SET_VALUES: [
                    ("PointCloudPublish.inputs:renderProductPath", render_product_path),
                    ("PointCloudPublish.inputs:frameId", LIDAR_FRAME),
                    ("PointCloudPublish.inputs:nodeNamespace", ""),
                    ("PointCloudPublish.inputs:topicName", POINTCLOUD_TOPIC),
                    ("PointCloudPublish.inputs:type", "point_cloud"),
                ],
                og.Controller.Keys.CONNECT: [
                    (
                        "OnPlaybackTick.outputs:tick",
                        "PointCloudPublish.inputs:execIn",
                    ),
                ],
            },
        )

    def build_arm_camera_graph(self):
        if self.stage.GetPrimAtPath(ARM_CAMERA_GRAPH_PATH).IsValid():
            carb.log_warn(
                f"Rebuilding existing arm camera ROS graph at {ARM_CAMERA_GRAPH_PATH}; "
                "stale render products can stop RGB/depth publication after reset"
            )
            try:
                omni.kit.commands.execute(
                    "DeletePrims",
                    paths=[ARM_CAMERA_GRAPH_PATH],
                    destructive=False,
                )
            except Exception as exc:
                carb.log_warn(f"Could not delete stale arm camera ROS graph: {exc}")

        try:
            enable_extension("isaacsim.core.nodes")
            enable_extension("isaacsim.ros2.bridge")
        except Exception as exc:
            carb.log_warn(f"Could not enable arm camera ROS 2 extensions: {exc}")

        camera_prim = UsdGeom.Camera(self.stage.DefinePrim(ARM_CAMERA_PRIM, "Camera"))
        camera_prim.GetHorizontalApertureAttr().Set(21)
        camera_prim.GetVerticalApertureAttr().Set(16)
        camera_prim.GetProjectionAttr().Set("perspective")
        camera_prim.GetFocalLengthAttr().Set(24)
        camera_prim.GetFocusDistanceAttr().Set(1.0)

        xformable = UsdGeom.Xformable(camera_prim.GetPrim())
        eye = Gf.Vec3d(*ARM_CAMERA_EYE)
        target = Gf.Vec3d(*ARM_CAMERA_TARGET)
        up = Gf.Vec3d(*ARM_CAMERA_UP)
        view = Gf.Matrix4d().SetLookAt(eye, target, up)
        xformable.ClearXformOpOrder()
        xformable.AddTransformOp().Set(view.GetInverse())

        render_product = rep.create.render_product(
            ARM_CAMERA_PRIM,
            resolution=ARM_CAMERA_RESOLUTION,
            name="arm_camera",
        )
        render_product_path = render_product.path

        og.Controller.edit(
            {"graph_path": ARM_CAMERA_GRAPH_PATH, "evaluator_name": "execution"},
            {
                og.Controller.Keys.CREATE_NODES: [
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                    ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                    ("PublishRgb", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                    ("PublishDepth", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                    ("PublishInfo", "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
                ],
                og.Controller.Keys.SET_VALUES: [
                    ("ReadSimTime.inputs:resetOnStop", False),
                    ("PublishRgb.inputs:renderProductPath", render_product_path),
                    ("PublishRgb.inputs:frameId", ARM_CAMERA_FRAME),
                    ("PublishRgb.inputs:topicName", ARM_CAMERA_RGB_TOPIC),
                    ("PublishRgb.inputs:type", "rgb"),
                    ("PublishDepth.inputs:renderProductPath", render_product_path),
                    ("PublishDepth.inputs:frameId", ARM_CAMERA_FRAME),
                    ("PublishDepth.inputs:topicName", ARM_CAMERA_DEPTH_TOPIC),
                    ("PublishDepth.inputs:type", "depth"),
                    ("PublishInfo.inputs:renderProductPath", render_product_path),
                    ("PublishInfo.inputs:frameId", ARM_CAMERA_FRAME),
                    ("PublishInfo.inputs:topicName", ARM_CAMERA_INFO_TOPIC),
                ],
                og.Controller.Keys.CONNECT: [
                    ("OnPlaybackTick.outputs:tick", "PublishRgb.inputs:execIn"),
                    ("OnPlaybackTick.outputs:tick", "PublishDepth.inputs:execIn"),
                    ("OnPlaybackTick.outputs:tick", "PublishInfo.inputs:execIn"),
                    ("Context.outputs:context", "PublishRgb.inputs:context"),
                    ("Context.outputs:context", "PublishDepth.inputs:context"),
                    ("Context.outputs:context", "PublishInfo.inputs:context"),
                ],
            },
        )
        carb.log_warn(
            f"Arm camera ROS topics ready: {ARM_CAMERA_RGB_TOPIC}, "
            f"{ARM_CAMERA_DEPTH_TOPIC}, {ARM_CAMERA_INFO_TOPIC}"
        )

    def update_drone_down_camera_pose(self):
        camera_prim = self.stage.GetPrimAtPath(DRONE_DOWN_CAMERA_PRIM)
        body_prim = self.stage.GetPrimAtPath(DRONE_BODY_PRIM)
        if (
            not camera_prim
            or not camera_prim.IsValid()
            or not body_prim
            or not body_prim.IsValid()
        ):
            return
        matrix = UsdGeom.XformCache().GetLocalToWorldTransform(body_prim)
        translation = matrix.ExtractTranslation()
        quat = matrix.ExtractRotation().GetQuat()
        imag = quat.GetImaginary()
        qx = float(imag[0])
        qy = float(imag[1])
        qz = float(imag[2])
        qw = float(quat.GetReal())
        yaw = math.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )
        cos_yaw = math.cos(float(yaw))
        sin_yaw = math.sin(float(yaw))
        offset_x = (
            DRONE_DOWN_CAMERA_LOCAL_X_OFFSET * cos_yaw
            - DRONE_DOWN_CAMERA_LOCAL_Y_OFFSET * sin_yaw
        )
        offset_y = (
            DRONE_DOWN_CAMERA_LOCAL_X_OFFSET * sin_yaw
            + DRONE_DOWN_CAMERA_LOCAL_Y_OFFSET * cos_yaw
        )
        eye = Gf.Vec3d(
            float(translation[0]) + offset_x,
            float(translation[1]) + offset_y,
            float(translation[2]) + DRONE_DOWN_CAMERA_LOCAL_Z_OFFSET,
        )
        target = Gf.Vec3d(float(translation[0]) + offset_x, float(translation[1]) + offset_y, float(translation[2]) - 1.0)
        up = Gf.Vec3d(-math.sin(yaw), math.cos(yaw), 0.0)
        view = Gf.Matrix4d().SetLookAt(eye, target, up)
        xformable = UsdGeom.Xformable(camera_prim)
        xformable.ClearXformOpOrder()
        xformable.AddTransformOp().Set(view.GetInverse())

    def build_drone_down_camera_graph(self):
        if self.stage.GetPrimAtPath(DRONE_DOWN_CAMERA_GRAPH_PATH).IsValid():
            carb.log_warn(
                f"Rebuilding existing drone down camera ROS graph at {DRONE_DOWN_CAMERA_GRAPH_PATH}"
            )
            try:
                omni.kit.commands.execute(
                    "DeletePrims",
                    paths=[DRONE_DOWN_CAMERA_GRAPH_PATH],
                    destructive=False,
                )
            except Exception as exc:
                carb.log_warn(f"Could not delete stale drone down camera ROS graph: {exc}")

        try:
            enable_extension("isaacsim.core.nodes")
            enable_extension("isaacsim.ros2.bridge")
        except Exception as exc:
            carb.log_warn(f"Could not enable drone camera ROS 2 extensions: {exc}")

        camera = UsdGeom.Camera(self.stage.DefinePrim(DRONE_DOWN_CAMERA_PRIM, "Camera"))
        camera.GetHorizontalApertureAttr().Set(DRONE_DOWN_CAMERA_APERTURE)
        camera.GetVerticalApertureAttr().Set(DRONE_DOWN_CAMERA_APERTURE)
        camera.GetProjectionAttr().Set("perspective")
        camera.GetFocalLengthAttr().Set(DRONE_DOWN_CAMERA_FOCAL_LENGTH)
        camera.GetFocusDistanceAttr().Set(8.0)
        camera.CreateClippingRangeAttr().Set(Gf.Vec2f(0.02, 100.0))
        self.update_drone_down_camera_pose()

        render_product = rep.create.render_product(
            DRONE_DOWN_CAMERA_PRIM,
            resolution=DRONE_DOWN_CAMERA_RESOLUTION,
            name="drone_down_camera",
        )
        render_product_path = render_product.path

        og.Controller.edit(
            {"graph_path": DRONE_DOWN_CAMERA_GRAPH_PATH, "evaluator_name": "execution"},
            {
                og.Controller.Keys.CREATE_NODES: [
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                    ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                    ("PublishRgb", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                    ("PublishInfo", "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
                ],
                og.Controller.Keys.SET_VALUES: [
                    ("ReadSimTime.inputs:resetOnStop", False),
                    ("PublishRgb.inputs:renderProductPath", render_product_path),
                    ("PublishRgb.inputs:frameId", DRONE_DOWN_CAMERA_FRAME),
                    ("PublishRgb.inputs:topicName", DRONE_DOWN_CAMERA_RGB_TOPIC),
                    ("PublishRgb.inputs:type", "rgb"),
                    ("PublishInfo.inputs:renderProductPath", render_product_path),
                    ("PublishInfo.inputs:frameId", DRONE_DOWN_CAMERA_FRAME),
                    ("PublishInfo.inputs:topicName", DRONE_DOWN_CAMERA_INFO_TOPIC),
                ],
                og.Controller.Keys.CONNECT: [
                    ("OnPlaybackTick.outputs:tick", "PublishRgb.inputs:execIn"),
                    ("OnPlaybackTick.outputs:tick", "PublishInfo.inputs:execIn"),
                    ("Context.outputs:context", "PublishRgb.inputs:context"),
                    ("Context.outputs:context", "PublishInfo.inputs:context"),
                ],
            },
        )
        carb.log_warn(
            f"Drone down camera ROS topics ready: {DRONE_DOWN_CAMERA_RGB_TOPIC}, "
            f"{DRONE_DOWN_CAMERA_INFO_TOPIC}"
        )

    def build_avoidance_state_graph(self):
        if self.stage.GetPrimAtPath(STATE_GRAPH_PATH).IsValid():
            carb.log_warn(f"Using existing avoidance state graph at {STATE_GRAPH_PATH}")
            self.update_avoidance_state_graph()
            return

        og.Controller.edit(
            {"graph_path": STATE_GRAPH_PATH, "evaluator_name": "execution"},
            {
                og.Controller.Keys.CREATE_NODES: [
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                    ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                    (
                        "PublishBaseTf",
                        "isaacsim.ros2.bridge.ROS2PublishRawTransformTree",
                    ),
                    (
                        "PublishLidarTf",
                        "isaacsim.ros2.bridge.ROS2PublishRawTransformTree",
                    ),
                    ("PublishOdom", "isaacsim.ros2.bridge.ROS2PublishOdometry"),
                ],
                og.Controller.Keys.SET_VALUES: [
                    ("ReadSimTime.inputs:resetOnStop", False),
                    ("PublishBaseTf.inputs:topicName", "tf"),
                    ("PublishBaseTf.inputs:parentFrameId", MAP_FRAME),
                    ("PublishBaseTf.inputs:childFrameId", BASE_FRAME),
                    ("PublishBaseTf.inputs:rotation", list(WORLD_TO_EGO_MAP_ROTATION_XYZW)),
                    ("PublishLidarTf.inputs:topicName", "tf_static"),
                    ("PublishLidarTf.inputs:parentFrameId", BASE_FRAME),
                    ("PublishLidarTf.inputs:childFrameId", LIDAR_FRAME),
                    ("PublishLidarTf.inputs:translation", list(LIDAR_TRANSLATION)),
                    ("PublishLidarTf.inputs:rotation", list(LIDAR_TF_ROTATION_XYZW)),
                    ("PublishLidarTf.inputs:staticPublisher", True),
                    ("PublishOdom.inputs:topicName", EGO_ODOM_TOPIC),
                    ("PublishOdom.inputs:odomFrameId", MAP_FRAME),
                    ("PublishOdom.inputs:chassisFrameId", BASE_FRAME),
                ],
                og.Controller.Keys.CONNECT: [
                    ("OnPlaybackTick.outputs:tick", "PublishBaseTf.inputs:execIn"),
                    ("OnPlaybackTick.outputs:tick", "PublishLidarTf.inputs:execIn"),
                    ("OnPlaybackTick.outputs:tick", "PublishOdom.inputs:execIn"),
                    ("Context.outputs:context", "PublishBaseTf.inputs:context"),
                    ("Context.outputs:context", "PublishLidarTf.inputs:context"),
                    ("Context.outputs:context", "PublishOdom.inputs:context"),
                    (
                        "ReadSimTime.outputs:simulationTime",
                        "PublishBaseTf.inputs:timeStamp",
                    ),
                    (
                        "ReadSimTime.outputs:simulationTime",
                        "PublishLidarTf.inputs:timeStamp",
                    ),
                    (
                        "ReadSimTime.outputs:simulationTime",
                        "PublishOdom.inputs:timeStamp",
                    ),
                ],
            },
        )
        self.update_avoidance_state_graph()

    def configure_mercury_joint_graphs(self):
        if not self.stage.GetPrimAtPath(MERCURY_ROBOT_PRIM).IsValid():
            carb.log_warn(f"Mercury robot prim not found: {MERCURY_ROBOT_PRIM}")
            return

        bindings = (
            (
                MERCURY_MAIN_JOINT_GRAPH,
                "/joint_command",
                "/joint_states",
            ),
            (
                MERCURY_GRIPPER_JOINT_GRAPH,
                "/gripper_command",
                "/gripper_joint_states",
            ),
        )
        for graph_path, command_topic, state_topic in bindings:
            if not self.stage.GetPrimAtPath(graph_path).IsValid():
                carb.log_warn(f"Mercury joint graph not found: {graph_path}")
                continue
            try:
                og.Controller.attribute(
                    f"{graph_path}/SubscriberJointState.inputs:topicName"
                ).set(command_topic)
                og.Controller.attribute(
                    f"{graph_path}/PublisherJointState.inputs:topicName"
                ).set(state_topic)
                og.Controller.attribute(
                    f"{graph_path}/ArticulationController.inputs:robotPath"
                ).set(MERCURY_ROBOT_PRIM)
                carb.log_warn(
                    f"Mercury joint graph ready: {command_topic} -> "
                    f"{MERCURY_ROBOT_PRIM}, states {state_topic}"
                )
            except Exception as exc:
                carb.log_warn(f"Could not configure {graph_path}: {exc}")

    def build_mercury_joint_control_graph(self):
        if not self.stage.GetPrimAtPath(MERCURY_ROBOT_PRIM).IsValid():
            carb.log_warn(f"Mercury robot prim not found: {MERCURY_ROBOT_PRIM}")
            return

        for stale_graph in (MERCURY_MAIN_JOINT_GRAPH, MERCURY_GRIPPER_JOINT_GRAPH):
            prim = self.stage.GetPrimAtPath(stale_graph)
            if prim.IsValid() and prim.IsActive():
                prim.SetActive(False)

        stale_runtime_graphs = []
        for prim in self.stage.Traverse():
            path = str(prim.GetPath())
            if path == MERCURY_RUNTIME_JOINT_GRAPH:
                continue
            if path == MERCURY_RUNTIME_JOINT_GRAPH_PREFIX or path.startswith(
                f"{MERCURY_RUNTIME_JOINT_GRAPH_PREFIX}_"
            ):
                stale_runtime_graphs.append(path)
        for graph_path in sorted(stale_runtime_graphs, key=len, reverse=True):
            try:
                self.stage.RemovePrim(graph_path)
                carb.log_warn(f"Removed stale Mercury runtime joint graph: {graph_path}")
            except Exception as exc:
                carb.log_warn(f"Could not remove stale Mercury runtime joint graph {graph_path}: {exc}")

        if self.stage.GetPrimAtPath(MERCURY_RUNTIME_JOINT_GRAPH).IsValid():
            self.stage.RemovePrim(MERCURY_RUNTIME_JOINT_GRAPH)

        try:
            enable_extension("isaacsim.core.nodes")
            enable_extension("isaacsim.ros2.bridge")
        except Exception as exc:
            carb.log_warn(f"Could not enable Mercury joint ROS 2 extensions: {exc}")

        arm_target = [usdrt.Sdf.Path(MERCURY_ARM_ARTICULATION_PRIM)]
        og.Controller.edit(
            {"graph_path": MERCURY_RUNTIME_JOINT_GRAPH, "evaluator_name": "execution"},
            {
                og.Controller.Keys.CREATE_NODES: [
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                    ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                    ("PublishJointState", "isaacsim.ros2.bridge.ROS2PublishJointState"),
                    ("SubscribeArmJointState", "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
                    ("ArmArticulationController", "isaacsim.core.nodes.IsaacArticulationController"),
                ],
                og.Controller.Keys.SET_VALUES: [
                    ("PublishJointState.inputs:targetPrim", arm_target),
                    ("PublishJointState.inputs:topicName", "/joint_states"),
                    ("SubscribeArmJointState.inputs:topicName", "/joint_command"),
                    ("ArmArticulationController.inputs:targetPrim", arm_target),
                ],
                og.Controller.Keys.CONNECT: [
                    ("OnPlaybackTick.outputs:tick", "PublishJointState.inputs:execIn"),
                    ("OnPlaybackTick.outputs:tick", "SubscribeArmJointState.inputs:execIn"),
                    ("OnPlaybackTick.outputs:tick", "ArmArticulationController.inputs:execIn"),
                    ("Context.outputs:context", "PublishJointState.inputs:context"),
                    ("Context.outputs:context", "SubscribeArmJointState.inputs:context"),
                    (
                        "ReadSimTime.outputs:simulationTime",
                        "PublishJointState.inputs:timeStamp",
                    ),
                    (
                        "SubscribeArmJointState.outputs:jointNames",
                        "ArmArticulationController.inputs:jointNames",
                    ),
                    (
                        "SubscribeArmJointState.outputs:positionCommand",
                        "ArmArticulationController.inputs:positionCommand",
                    ),
                    (
                        "SubscribeArmJointState.outputs:velocityCommand",
                        "ArmArticulationController.inputs:velocityCommand",
                    ),
                    (
                        "SubscribeArmJointState.outputs:effortCommand",
                        "ArmArticulationController.inputs:effortCommand",
                    ),
                ],
            },
        )
        carb.log_warn(
            "Mercury runtime joint graph ready: /joint_command, /joint_states"
        )

    def find_bobac_robot_prim(self) -> Optional[str]:
        if self.stage.GetPrimAtPath(BOBAC_ROBOT_PRIM).IsValid():
            return BOBAC_ROBOT_PRIM

        for prim in self.stage.Traverse():
            path = str(prim.GetPath())
            name = prim.GetName().lower()
            if path.startswith(f"{LAYOUT_PRIM}/") and "bobac" in name:
                return path
        return None

    def find_bobac_base_articulation_prim(self, robot_prim_path: str) -> str:
        default_path = BOBAC_BASE_ARTICULATION_PRIM
        if self.stage.GetPrimAtPath(default_path).IsValid():
            return default_path

        suffix = "/bobac3_serverbot/base_footprint/base_link"
        candidate = f"{robot_prim_path}{suffix}"
        if self.stage.GetPrimAtPath(candidate).IsValid():
            return candidate

        for prim in self.stage.Traverse():
            path = str(prim.GetPath())
            if not path.startswith(f"{robot_prim_path}/"):
                continue
            if path.endswith("/base_footprint/base_link"):
                return path

        return robot_prim_path

    def find_bobac_arm_articulation_prim(self, robot_prim_path: str) -> str:
        default_path = BOBAC_ARM_ARTICULATION_PRIM
        if self.stage.GetPrimAtPath(default_path).IsValid():
            return default_path

        suffix = "/bobac3_serverbot/base_footprint/ECO65_B/base_link_arm"
        candidate = f"{robot_prim_path}{suffix}"
        if self.stage.GetPrimAtPath(candidate).IsValid():
            return candidate

        for prim in self.stage.Traverse():
            path = str(prim.GetPath())
            if not path.startswith(f"{robot_prim_path}/"):
                continue
            if path.endswith("/ECO65_B/base_link_arm"):
                return path

        return robot_prim_path

    def log_bobac_robot_structure(self, robot_prim_path: str) -> None:
        interesting = []
        articulation_candidates = []
        for prim in self.stage.Traverse():
            path = str(prim.GetPath())
            if not path.startswith(f"{robot_prim_path}/") and path != robot_prim_path:
                continue

            name = prim.GetName().lower()
            type_name = prim.GetTypeName()
            if "joint" in type_name.lower() or any(
                hint in name for hint in ("joint", "wheel", "gripper", "arm", "link")
            ):
                interesting.append(f"{path} [{type_name}]")
            try:
                if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                    articulation_candidates.append(path)
            except Exception:
                pass

        carb.log_warn(f"Bobac robot prim: {robot_prim_path}")
        if articulation_candidates:
            carb.log_warn(
                "Bobac articulation root candidates: "
                + ", ".join(articulation_candidates[:12])
            )
        else:
            carb.log_warn("Bobac articulation root candidates: none found by API")

        if interesting:
            carb.log_warn(
                "Bobac joint/link candidates: " + " | ".join(interesting[:80])
            )
        else:
            carb.log_warn("Bobac joint/link candidates: none found")

    def build_bobac_joint_control_graph(self):
        robot_prim_path = self.find_bobac_robot_prim()
        if robot_prim_path is None:
            carb.log_warn(f"Bobac robot prim not found: {BOBAC_ROBOT_PRIM}")
            return

        self.log_bobac_robot_structure(robot_prim_path)

        stale_runtime_graphs = []
        for prim in self.stage.Traverse():
            path = str(prim.GetPath())
            if path == BOBAC_RUNTIME_JOINT_GRAPH:
                continue
            if path == BOBAC_RUNTIME_JOINT_GRAPH_PREFIX or path.startswith(
                f"{BOBAC_RUNTIME_JOINT_GRAPH_PREFIX}_"
            ):
                stale_runtime_graphs.append(path)
        for graph_path in sorted(stale_runtime_graphs, key=len, reverse=True):
            try:
                self.stage.RemovePrim(graph_path)
                carb.log_warn(f"Removed stale Bobac runtime joint graph: {graph_path}")
            except Exception as exc:
                carb.log_warn(
                    f"Could not remove stale Bobac runtime joint graph {graph_path}: {exc}"
                )

        if self.stage.GetPrimAtPath(BOBAC_RUNTIME_JOINT_GRAPH).IsValid():
            self.stage.RemovePrim(BOBAC_RUNTIME_JOINT_GRAPH)

        try:
            enable_extension("isaacsim.core.nodes")
            enable_extension("isaacsim.ros2.bridge")
        except Exception as exc:
            carb.log_warn(f"Could not enable Bobac joint ROS 2 extensions: {exc}")

        base_articulation_path = self.find_bobac_base_articulation_prim(robot_prim_path)
        arm_articulation_path = self.find_bobac_arm_articulation_prim(robot_prim_path)
        if base_articulation_path == robot_prim_path:
            carb.log_warn(
                "Bobac base articulation root not found; falling back to robot prim"
            )
        else:
            carb.log_warn(f"Bobac base articulation target: {base_articulation_path}")
        if arm_articulation_path == robot_prim_path:
            carb.log_warn(
                "Bobac arm articulation root not found; falling back to robot prim"
            )
        else:
            carb.log_warn(f"Bobac arm articulation target: {arm_articulation_path}")

        arm_target = [usdrt.Sdf.Path(arm_articulation_path)]
        og.Controller.edit(
            {"graph_path": BOBAC_RUNTIME_JOINT_GRAPH, "evaluator_name": "execution"},
            {
                og.Controller.Keys.CREATE_NODES: [
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                    ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                    ("PublishJointState", "isaacsim.ros2.bridge.ROS2PublishJointState"),
                    ("SubscribeJointState", "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
                    ("ArticulationController", "isaacsim.core.nodes.IsaacArticulationController"),
                ],
                og.Controller.Keys.SET_VALUES: [
                    ("PublishJointState.inputs:targetPrim", arm_target),
                    ("PublishJointState.inputs:topicName", BOBAC_JOINT_STATES_TOPIC),
                    ("SubscribeJointState.inputs:topicName", BOBAC_JOINT_COMMAND_TOPIC),
                    ("ArticulationController.inputs:targetPrim", arm_target),
                ],
                og.Controller.Keys.CONNECT: [
                    ("OnPlaybackTick.outputs:tick", "PublishJointState.inputs:execIn"),
                    ("OnPlaybackTick.outputs:tick", "SubscribeJointState.inputs:execIn"),
                    ("OnPlaybackTick.outputs:tick", "ArticulationController.inputs:execIn"),
                    ("Context.outputs:context", "PublishJointState.inputs:context"),
                    ("Context.outputs:context", "SubscribeJointState.inputs:context"),
                    (
                        "ReadSimTime.outputs:simulationTime",
                        "PublishJointState.inputs:timeStamp",
                    ),
                    (
                        "SubscribeJointState.outputs:jointNames",
                        "ArticulationController.inputs:jointNames",
                    ),
                    (
                        "SubscribeJointState.outputs:positionCommand",
                        "ArticulationController.inputs:positionCommand",
                    ),
                    (
                        "SubscribeJointState.outputs:velocityCommand",
                        "ArticulationController.inputs:velocityCommand",
                    ),
                    (
                        "SubscribeJointState.outputs:effortCommand",
                        "ArticulationController.inputs:effortCommand",
                    ),
                ],
            },
        )
        carb.log_warn(
            f"Bobac runtime joint graph ready: {BOBAC_JOINT_COMMAND_TOPIC}, "
            f"{BOBAC_JOINT_STATES_TOPIC}"
        )

    def set_state_graph_input(self, node_name: str, input_name: str, value):
        og.Controller.attribute(
            f"{STATE_GRAPH_PATH}/{node_name}.inputs:{input_name}"
        ).set(value)

    def update_avoidance_state_graph(self):
        if self.vehicle is None:
            return
        if not self.stage.GetPrimAtPath(STATE_GRAPH_PATH).IsValid():
            return
        state = self.vehicle.state
        position = list(demo_config.world_to_ego_map_position(state.position))
        attitude = list(demo_config.world_to_ego_map_quaternion(state.attitude))
        linear_velocity = list(demo_config.world_to_ego_map_vector(state.linear_velocity))
        angular_velocity = list(demo_config.world_to_ego_map_vector(state.angular_velocity))

        self.set_state_graph_input("PublishBaseTf", "translation", position)
        self.set_state_graph_input("PublishBaseTf", "rotation", attitude)
        self.set_state_graph_input("PublishOdom", "position", position)
        self.set_state_graph_input("PublishOdom", "orientation", attitude)
        self.set_state_graph_input("PublishOdom", "linearVelocity", linear_velocity)
        self.set_state_graph_input("PublishOdom", "angularVelocity", angular_velocity)

    async def setup_bobac_px4_drone_async(self):
        if not BOBAC_ENABLE_PX4:
            return False

        existing = self.stage.GetPrimAtPath(BOBAC_PX4_PRIM)
        attach_existing = BOBAC_PX4_ATTACH_EXISTING and existing and existing.IsValid()
        if not attach_existing and not Path(BOBAC_PX4_DRONE_USD).exists():
            carb.log_warn(
                f"BOBAC_ENABLE_PX4 requested but drone USD is missing: {BOBAC_PX4_DRONE_USD}"
            )
            return False

        if BOBAC_PX4_SANITIZE_CHILD_MASS:
            self.sanitize_px4_child_collider_masses(BOBAC_PX4_PRIM)

        config = MultirotorConfig()
        rotor_constant = 8.54858e-6 * BOBAC_PX4_ROTOR_CONSTANT_MULTIPLIER
        config.thrust_curve = QuadraticThrustCurve(
            {"rotor_constant": [rotor_constant] * 4}
        )
        mavlink_config = PX4MavlinkBackendConfig(
            {
                "vehicle_id": 0,
                "px4_autolaunch": True,
                "px4_dir": self.pg.px4_path,
                "px4_vehicle_model": self.pg.px4_default_airframe,
                "input_scaling": [BOBAC_PX4_INPUT_SCALING] * 4,
            }
        )
        config.backends = [PX4MavlinkBackend(mavlink_config)]
        spawn = [float(self.spawn_world[0]), float(self.spawn_world[1]), float(self.spawn_world[2])]
        if BOBAC_PX4_SPAWN:
            parts = [p for p in BOBAC_PX4_SPAWN.replace(",", " ").split() if p]
            if len(parts) != 3:
                carb.log_warn(
                    f"BOBAC_PX4_SPAWN must be three numbers, got {BOBAC_PX4_SPAWN!r}; "
                    f"using {spawn}"
                )
            else:
                spawn = [float(parts[0]), float(parts[1]), float(parts[2])]
        if attach_existing:
            self.vehicle = Multirotor(
                BOBAC_PX4_PRIM,
                "",
                0,
                None,
                None,
                config=config,
                attach_existing=True,
            )
            spawn_note = "attached_existing"
        else:
            self.vehicle = Multirotor(
                BOBAC_PX4_PRIM,
                BOBAC_PX4_DRONE_USD,
                0,
                spawn,
                [0.0, 0.0, 0.0, 1.0],
                config=config,
            )
            spawn_note = f"spawn={spawn}"
        carb.log_warn(
            "Bobac PX4/Pegasus mode enabled: "
            f"vehicle={BOBAC_PX4_PRIM}, {spawn_note}; "
            f"px4_dir={self.pg.px4_path}; "
            f"rotor_constant={rotor_constant:.8g}; "
            f"input_scaling={BOBAC_PX4_INPUT_SCALING:.1f}"
        )
        await omni.kit.app.get_app().next_update_async()
        self.cargo_runtime.resolve_mounted_cargo_bay()
        self.cargo_runtime.left_open()
        self.cargo_runtime.bottom_close()
        carb.log_warn("Bobac PX4 cargo bay initialized: left_open, bottom_close")
        return True

    def sanitize_px4_child_collider_masses(self, root_path: str):
        root = self.stage.GetPrimAtPath(root_path)
        if not root or not root.IsValid():
            return
        root_sdf = root.GetPath()
        changed = []
        total_before = 0.0
        total_after = 0.0
        for prim in self.stage.Traverse():
            if prim.GetPath() != root_sdf and not prim.GetPath().HasPrefix(root_sdf):
                continue
            has_mass = prim.HasAPI(UsdPhysics.MassAPI)
            if has_mass:
                attr = UsdPhysics.MassAPI(prim).GetMassAttr()
                mass = attr.Get() if attr and attr.IsValid() else None
                if mass is not None:
                    total_before += float(mass)
            if (
                has_mass
                and prim.HasAPI(UsdPhysics.CollisionAPI)
                and not prim.HasAPI(UsdPhysics.RigidBodyAPI)
            ):
                mass_api = UsdPhysics.MassAPI(prim)
                mass_api.CreateMassAttr().Set(float(BOBAC_PX4_CHILD_COLLIDER_MASS))
                changed.append(str(prim.GetPath()))
                total_after += float(BOBAC_PX4_CHILD_COLLIDER_MASS)
            elif has_mass:
                attr = UsdPhysics.MassAPI(prim).GetMassAttr()
                mass = attr.Get() if attr and attr.IsValid() else None
                if mass is not None:
                    total_after += float(mass)
        if changed:
            carb.log_warn(
                "Bobac PX4 runtime mass cleanup: "
                f"{len(changed)} child collider masses set to "
                f"{BOBAC_PX4_CHILD_COLLIDER_MASS}; "
                f"authored_mass_before={total_before:.3f}, after={total_after:.3f}"
            )

    async def setup_bobac_async(self):
        self.spawn_world = self.resolve_existing_drone_spawn()
        carb.log_warn(f"Bobac drone spawn position: {self.spawn_world}")
        if not await self.setup_bobac_px4_drone_async():
            self.vehicle = ExistingPrimVehicleState(self.stage, BOBAC_BASE_ARTICULATION_PRIM)
            carb.log_warn(
                "Bobac uses passive USD prim state; PX4 attach and Mercury/X1 arm setup "
                "are skipped; Python ROS publishes /odom, TF, /scan, and /laser_scan_fuse"
            )
        self.build_arm_camera_graph()
        self.build_drone_down_camera_graph()
        self.build_bobac_joint_control_graph()
        carb.log_warn("Bobac LaserScan topics: /scan, /laser_scan_fuse")
        carb.log_warn("Bobac TF tree for SLAM/Nav2: odom -> base_link -> fuse_lidar_link")
        if MATERIAL_ENABLE_DRONE_CARGO_PLACE_ALIGNMENT:
            self.position_drone_cargo_for_material_place()
        self.prealign_robot_for_material_grasp()
        self.prepare_material_object_for_startup()
        await omni.kit.app.get_app().next_update_async()

        carb.log_warn("Resetting Isaac World")
        await self.world.reset_async()
        carb.log_warn("Isaac World reset complete")
        self.prealign_robot_for_material_grasp()
        self.prepare_material_object_for_startup()
        await omni.kit.app.get_app().next_update_async()
        self.ros.start()
        carb.log_warn("Cargo ROS interface started")
        self._update_sub = (
            omni.kit.app.get_app()
            .get_update_event_stream()
            .create_subscription_to_pop(
                self._on_update,
                name="bobac_scene_update",
            )
        )
        self.pg.set_viewport_camera(
            (
                self.spawn_world[0] - 1.5,
                self.spawn_world[1] - 2.0,
                self.spawn_world[2] + 1.0,
            ),
            self.spawn_world,
        )
        self.timeline.play()
        carb.log_warn("Bobac race scene is ready")

    async def setup_async(self):
        await self.ensure_world_ready_async()
        self.cleanup_existing_scene()
        self.load_layout()
        _deactivate_stale_imported_control_graphs(self.stage)
        _ensure_physx_scene(self.stage)

        if ARGS.world == "Bobac":
            await self.setup_bobac_async()
            return

        if MATERIAL_ENABLE_DRONE_CARGO_PLACE_ALIGNMENT:
            self.position_drone_cargo_for_material_place()
        self.prealign_robot_for_material_grasp()
        if MATERIAL_LOCK_BASE_ON_SETUP:
            self.lock_robot_base_for_manipulation()
        self.prepare_material_object_for_startup()
        await omni.kit.app.get_app().next_update_async()

        if not ARGS.world:
            carb.log_warn("No --world provided; bare scene launcher exiting after headless startup")
            return

        # create_map mode: only load scene without PX4 or other operations
        if ARGS.world == "create_map":
            carb.log_warn("create_map mode: Scene loaded successfully")
            carb.log_warn(f"Scene file: {ASSEMBLED_SCENE_USD}")
            # Ensure timeline is stopped
            self.timeline.stop()
            carb.log_warn("create_map scene is ready (timeline stopped)")
            return

        self.spawn_world = self.resolve_existing_drone_spawn()
        carb.log_warn(f"Drone spawn position: {self.spawn_world}")

        multirotor_params = inspect.signature(Multirotor).parameters
        if "attach_existing" in multirotor_params:
            config = MultirotorConfig()
            mavlink_config = PX4MavlinkBackendConfig(
                {
                    "vehicle_id": 0,
                    "px4_autolaunch": True,
                    "px4_dir": self.pg.px4_path,
                    "px4_vehicle_model": self.pg.px4_default_airframe,
                }
            )
            config.backends = [PX4MavlinkBackend(mavlink_config)]
            self.vehicle = Multirotor(
                DRONE_PRIM,
                "",
                0,
                None,
                None,
                config=config,
                attach_existing=True,
            )
            carb.log_warn("Attached Pegasus to assembled scene vehicle")
            await omni.kit.app.get_app().next_update_async()
        else:
            self.vehicle = ExistingPrimVehicleState(self.stage, DRONE_BODY_PRIM)
            carb.log_warn(
                "Pegasus Multirotor has no attach_existing support; "
                "using passive USD prim state for ROS odometry/TF"
            )

        self.cargo_runtime.resolve_mounted_cargo_bay()
        carb.log_warn("Cargo bay ready")

        lidar_path = self.attach_hidden_lidar()
        carb.log_warn(f"Hidden OS0 RTX LiDAR attached at {lidar_path}")
        self.build_ros_clock_graph()
        self.build_lidar_publish_graph(lidar_path)
        self.build_avoidance_state_graph()
        self.build_arm_camera_graph()
        self.build_mercury_joint_control_graph()
        carb.log_warn(f"3D point cloud topic: {POINTCLOUD_TOPIC}")
        carb.log_warn(f"EGO odometry topic: {EGO_ODOM_TOPIC}")
        carb.log_warn(f"TF tree for mapping: {MAP_FRAME} -> {BASE_FRAME} -> {LIDAR_FRAME}")

        if ENABLE_PENCIL_PAYLOAD:
            self.cargo_runtime.prepare_pencil_payload()
            carb.log_warn("Pencil payload prepared and locked")
        else:
            carb.log_warn("Pencil payload disabled; skipping placement and lock")
        self.cargo_runtime.left_open()
        carb.log_warn("Left door opened")
        self.cargo_runtime.bottom_close()
        carb.log_warn("Bottom door closed")
        self.cargo_runtime.validate_no_dynamic_mesh_collisions()
        carb.log_warn("Dynamic mesh collision validation complete")

        carb.log_warn("Resetting Isaac World")
        await self.world.reset_async()
        carb.log_warn("Isaac World reset complete")
        self.ros.start()
        carb.log_warn("Cargo ROS interface started")
        self._update_sub = (
            omni.kit.app.get_app()
            .get_update_event_stream()
            .create_subscription_to_pop(
                self._on_update,
                name="cargo_delivery_scene_update",
            )
        )
        self.pg.set_viewport_camera(
            (
                self.spawn_world[0] - 1.5,
                self.spawn_world[1] - 2.0,
                self.spawn_world[2] + 1.0,
            ),
            self.spawn_world,
        )
        self.timeline.play()
        carb.log_warn("Cargo delivery scene is ready")

    def _on_update(self, _event):
        self.ros.spin_once()
        self.update_avoidance_state_graph()

    def shutdown(self):
        self._update_sub = None
        self.ros.shutdown()
        self.timeline.stop()


SCENE_APP = None
SCENE_TASK = None


async def main_async():
    global SCENE_APP
    if SCENE_APP is not None:
        SCENE_APP.shutdown()
    app = CargoDeliverySceneApp()
    SCENE_APP = app
    try:
        await app.setup_async()
    except Exception:
        app.shutdown()
        SCENE_APP = None
        raise


def _on_scene_task_done(task):
    try:
        exc = task.exception()
    except asyncio.CancelledError:
        carb.log_warn("Cargo delivery scene setup task was cancelled")
        return
    except Exception as callback_exc:
        carb.log_error(f"Could not inspect cargo scene setup task: {callback_exc}")
        return

    if exc is not None:
        carb.log_error(f"Cargo delivery scene setup failed: {exc}")
        carb.log_error("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))


def main():
    global SCENE_TASK
    SCENE_TASK = run_coroutine(main_async())
    SCENE_TASK.add_done_callback(_on_scene_task_done)


if __name__ == "__main__":
    main()
