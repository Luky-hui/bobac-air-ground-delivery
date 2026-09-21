import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from geometry_msgs.msg import PointStamped, Vector3Stamped, PoseStamped


@dataclass
class Detection2D:
    label: str
    confidence: float
    u: float
    v: float
    width: float
    height: float
    mask_area: int


def normalize(vec):
    arr = np.array(vec, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if not np.isfinite(norm) or norm < 1e-9:
        return None
    return arr / norm


def point_stamped(frame_id, stamp, xyz):
    msg = PointStamped()
    msg.header.frame_id = frame_id
    msg.header.stamp = stamp
    msg.point.x = float(xyz[0])
    msg.point.y = float(xyz[1])
    msg.point.z = float(xyz[2])
    return msg


def vector_stamped(frame_id, stamp, xyz):
    msg = Vector3Stamped()
    msg.header.frame_id = frame_id
    msg.header.stamp = stamp
    msg.vector.x = float(xyz[0])
    msg.vector.y = float(xyz[1])
    msg.vector.z = float(xyz[2])
    return msg


def quaternion_from_axes(x_axis, y_axis, z_axis):
    r = np.array([x_axis, y_axis, z_axis], dtype=np.float64).T
    trace = float(np.trace(r))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (r[2, 1] - r[1, 2]) / s
        qy = (r[0, 2] - r[2, 0]) / s
        qz = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        qw = (r[2, 1] - r[1, 2]) / s
        qx = 0.25 * s
        qy = (r[0, 1] + r[1, 0]) / s
        qz = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        qw = (r[0, 2] - r[2, 0]) / s
        qx = (r[0, 1] + r[1, 0]) / s
        qy = 0.25 * s
        qz = (r[1, 2] + r[2, 1]) / s
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        qw = (r[1, 0] - r[0, 1]) / s
        qx = (r[0, 2] + r[2, 0]) / s
        qy = (r[1, 2] + r[2, 1]) / s
        qz = 0.25 * s
    return qx, qy, qz, qw


def pose_from_point_and_axes(frame_id, stamp, point, normal, long_axis):
    z_axis = normalize(normal)
    y_axis = normalize(long_axis)
    if z_axis is None:
        z_axis = np.array([0.0, 0.0, -1.0])
    if y_axis is None:
        y_axis = np.array([1.0, 0.0, 0.0])
    x_axis = normalize(np.cross(y_axis, z_axis))
    if x_axis is None:
        x_axis = np.array([0.0, 1.0, 0.0])
    y_axis = normalize(np.cross(z_axis, x_axis))
    qx, qy, qz, qw = quaternion_from_axes(x_axis, y_axis, z_axis)
    pose = PoseStamped()
    pose.header.frame_id = frame_id
    pose.header.stamp = stamp
    pose.pose.position.x = float(point[0])
    pose.pose.position.y = float(point[1])
    pose.pose.position.z = float(point[2])
    pose.pose.orientation.x = qx
    pose.pose.orientation.y = qy
    pose.pose.orientation.z = qz
    pose.pose.orientation.w = qw
    return pose
