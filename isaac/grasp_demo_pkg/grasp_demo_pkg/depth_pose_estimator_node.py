#!/usr/bin/env python3
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, Vector3Stamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float32MultiArray


class DepthPoseEstimatorNode(Node):
    def __init__(self):
        super().__init__('depth_pose_estimator_node')
        self.declare_parameter('depth_topic', '/arm_camera/depth')
        self.declare_parameter('camera_info_topic', '/arm_camera/camera_info')
        self.declare_parameter('camera_frame', '')
        self.declare_parameter('bbox_topic', '/demo_grasp/bbox')
        self.declare_parameter('mask_topic', '/demo_grasp/object_mask')
        self.declare_parameter('point_topic', '/demo_grasp/object_point_camera')
        self.declare_parameter('long_axis_topic', '/demo_grasp/long_axis_camera')
        self.declare_parameter('depth_window_px', 5)
        self.declare_parameter('axis_step_px', 10)
        self.bridge = CvBridge()
        self.depth = None
        self.depth_header = None
        self.info = None
        self.bbox = None
        self.mask = None
        self.point_pub = self.create_publisher(PointStamped, self.get_parameter('point_topic').value, 10)
        self.axis_pub = self.create_publisher(Vector3Stamped, self.get_parameter('long_axis_topic').value, 10)
        self.create_subscription(Image, self.get_parameter('depth_topic').value, self.on_depth, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, self.get_parameter('camera_info_topic').value, self.on_info, qos_profile_sensor_data)
        self.create_subscription(Float32MultiArray, self.get_parameter('bbox_topic').value, self.on_bbox, 10)
        self.create_subscription(Image, self.get_parameter('mask_topic').value, self.on_mask, 10)
        self.create_timer(0.2, self.try_publish)
        self.get_logger().info('depth_pose_estimator_node started')

    def on_depth(self, msg):
        if msg.encoding.upper() == '32FC1' and len(msg.data) == msg.height * msg.width * 4:
            self.depth = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width).copy()
        else:
            arr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            arr = np.asarray(arr)
            if arr.ndim == 3 and arr.shape[-1] == 4 and msg.encoding.upper() == '32FC1':
                self.depth = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width).copy()
            else:
                self.depth = arr.astype(np.float32, copy=False)
        self.depth_header = msg.header

    def on_info(self, msg):
        self.info = msg

    def on_bbox(self, msg):
        if len(msg.data) >= 4:
            self.bbox = list(msg.data)

    def on_mask(self, msg):
        self.mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8') > 0

    def _backproject(self, u, v, z):
        fx = float(self.info.k[0])
        fy = float(self.info.k[4])
        cx = float(self.info.k[2])
        cy = float(self.info.k[5])
        return np.array([(u - cx) * z / fx, (v - cy) * z / fy, z], dtype=np.float64)

    def _depth_at(self, u, v):
        r = int(self.get_parameter('depth_window_px').value)
        h, w = self.depth.shape[:2]
        u0, u1 = max(0, int(round(u)) - r), min(w, int(round(u)) + r + 1)
        v0, v1 = max(0, int(round(v)) - r), min(h, int(round(v)) + r + 1)
        patch = self.depth[v0:v1, u0:u1]
        vals = patch[np.isfinite(patch)]
        vals = vals[vals > 0.0]
        if vals.size == 0:
            return None
        return float(np.median(vals))

    def _axis_from_mask(self):
        if self.mask is None:
            return np.array([1.0, 0.0], dtype=np.float64)
        ys, xs = np.where(self.mask)
        if xs.size < 20:
            return np.array([1.0, 0.0], dtype=np.float64)
        pts = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
        pts = pts - pts.mean(axis=0)
        cov = (pts.T @ pts) / float(pts.shape[0])
        vals, vecs = np.linalg.eigh(cov)
        axis = vecs[:, int(np.argmax(vals))]
        norm = float(np.linalg.norm(axis))
        return axis / norm if norm >= 1e-9 else np.array([1.0, 0.0], dtype=np.float64)

    def try_publish(self):
        if self.depth is None or self.info is None or self.bbox is None:
            return
        u = float(self.bbox[0])
        v = float(self.bbox[1])
        z = self._depth_at(u, v)
        if z is None:
            return
        point_xyz = self._backproject(u, v, z)
        point = PointStamped()
        point.header = self.depth_header
        camera_frame = str(self.get_parameter('camera_frame').value)
        if camera_frame:
            point.header.frame_id = camera_frame
        elif self.info.header.frame_id:
            point.header.frame_id = self.info.header.frame_id
        point.point.x = float(point_xyz[0])
        point.point.y = float(point_xyz[1])
        point.point.z = float(point_xyz[2])
        self.point_pub.publish(point)

        axis2 = self._axis_from_mask()
        step = max(2, int(self.get_parameter('axis_step_px').value))
        u2 = float(np.clip(u + axis2[0] * step, 0, self.depth.shape[1] - 1))
        v2 = float(np.clip(v + axis2[1] * step, 0, self.depth.shape[0] - 1))
        z2 = self._depth_at(u2, v2) or z
        axis3 = self._backproject(u2, v2, z2) - point_xyz
        norm = float(np.linalg.norm(axis3))
        axis3 = axis3 / norm if norm >= 1e-9 else np.array([1.0, 0.0, 0.0], dtype=np.float64)
        axis_msg = Vector3Stamped()
        axis_msg.header = point.header
        axis_msg.vector.x = float(axis3[0])
        axis_msg.vector.y = float(axis3[1])
        axis_msg.vector.z = float(axis3[2])
        self.axis_pub.publish(axis_msg)


def main(args=None):
    rclpy.init(args=args)
    node = DepthPoseEstimatorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
