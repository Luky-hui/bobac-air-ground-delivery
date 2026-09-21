#!/usr/bin/env python3
import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped, Vector3Stamped
from rclpy.node import Node
from tf2_geometry_msgs import do_transform_point, do_transform_vector3
from tf2_ros import Buffer, TransformException, TransformListener


class TfTransformDemoNode(Node):
    def __init__(self):
        super().__init__('tf_transform_demo_node')
        self.declare_parameter('base_frame', 'base_link_arm')
        self.declare_parameter('point_camera_topic', '/demo_grasp/object_point_camera')
        self.declare_parameter('long_axis_camera_topic', '/demo_grasp/long_axis_camera')
        self.declare_parameter('point_base_topic', '/demo_grasp/object_point_base')
        self.declare_parameter('normal_base_topic', '/demo_grasp/normal_base')
        self.declare_parameter('long_axis_base_topic', '/demo_grasp/long_axis_base')
        self.declare_parameter('normal_mode', 'fixed_base')
        self.declare_parameter('fixed_normal_base', [0.0, 0.0, -1.0])
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.point_pub = self.create_publisher(PointStamped, self.get_parameter('point_base_topic').value, 10)
        self.normal_pub = self.create_publisher(Vector3Stamped, self.get_parameter('normal_base_topic').value, 10)
        self.axis_pub = self.create_publisher(Vector3Stamped, self.get_parameter('long_axis_base_topic').value, 10)
        self.create_subscription(PointStamped, self.get_parameter('point_camera_topic').value, self.on_point, 10)
        self.create_subscription(Vector3Stamped, self.get_parameter('long_axis_camera_topic').value, self.on_axis, 10)
        self.create_timer(0.2, self.publish_fixed_normal)
        self.last_stamp = None
        self.get_logger().info('tf_transform_demo_node started')

    def _lookup(self, source_frame):
        target = self.get_parameter('base_frame').value
        return self.tf_buffer.lookup_transform(target, source_frame, rclpy.time.Time())

    def publish_fixed_normal(self):
        if self.get_parameter('normal_mode').value != 'fixed_base':
            return
        vec = np.array(self.get_parameter('fixed_normal_base').value, dtype=np.float64)
        norm = float(np.linalg.norm(vec))
        if norm < 1e-9:
            vec = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        else:
            vec = vec / norm
        msg = Vector3Stamped()
        msg.header.frame_id = self.get_parameter('base_frame').value
        msg.header.stamp = self.get_clock().now().to_msg() if self.last_stamp is None else self.last_stamp
        msg.vector.x = float(vec[0])
        msg.vector.y = float(vec[1])
        msg.vector.z = float(vec[2])
        self.normal_pub.publish(msg)

    def on_point(self, msg):
        try:
            tf = self._lookup(msg.header.frame_id)
            out = do_transform_point(msg, tf)
            self.last_stamp = out.header.stamp
            self.point_pub.publish(out)
        except TransformException as exc:
            self.get_logger().warn(f'point TF failed: {exc}')

    def on_axis(self, msg):
        try:
            tf = self._lookup(msg.header.frame_id)
            out = do_transform_vector3(msg, tf)
            self.axis_pub.publish(out)
        except TransformException as exc:
            self.get_logger().warn(f'axis TF failed: {exc}')


def main(args=None):
    rclpy.init(args=args)
    node = TfTransformDemoNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
