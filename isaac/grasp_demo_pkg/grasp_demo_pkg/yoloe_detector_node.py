#!/usr/bin/env python3
import shutil
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_prefix, get_package_share_directory
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, Vector3Stamped
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Float32MultiArray, String

from grasp_demo_interfaces.action import DetectObject


class YoloEDetectorNode(Node):
    def __init__(self):
        super().__init__('yoloe_detector_node')
        self.declare_parameter('rgb_topic', '/arm_camera/rgb')
        self.declare_parameter('model_path', 'yoloe-26l-seg.pt')
        self.declare_parameter('classes', ['pencil', 'pen'])
        self.declare_parameter('conf', 0.01)
        self.declare_parameter('device', 'cuda:0')
        self.declare_parameter('auto_detect', True)
        self.declare_parameter('run_period_sec', 0.5)
        self.declare_parameter('mask_topic', '/demo_grasp/object_mask')
        self.declare_parameter('bbox_topic', '/demo_grasp/bbox')
        self.declare_parameter('label_topic', '/demo_grasp/label')
        self.declare_parameter('confidence_topic', '/demo_grasp/confidence')
        self.declare_parameter('debug_image_topic', '/demo_grasp/debug_image')
        self.declare_parameter('save_debug_images', True)
        self.declare_parameter('debug_image_dir', '/home/u-zhuang/ros2_ws/src/isaac/isaac-sim/demo_ws/grasp_demo_debug')
        self.declare_parameter('prefer_white_target', False)
        self.declare_parameter('target_roi_norm', [0.0, 0.0, 1.0, 1.0])
        self.declare_parameter('min_white_fraction', 0.0)
        self.declare_parameter('max_bbox_area_fraction', 1.0)

        self.bridge = CvBridge()
        self.last_rgb: Optional[Image] = None
        self.last_detection = None
        self._throttle_last = {}

        self.mask_pub = self.create_publisher(Image, self.get_parameter('mask_topic').value, 10)
        self.bbox_pub = self.create_publisher(Float32MultiArray, self.get_parameter('bbox_topic').value, 10)
        self.label_pub = self.create_publisher(String, self.get_parameter('label_topic').value, 10)
        self.conf_pub = self.create_publisher(Float32, self.get_parameter('confidence_topic').value, 10)
        self.debug_pub = self.create_publisher(Image, self.get_parameter('debug_image_topic').value, 10)
        self.create_subscription(Image, self.get_parameter('rgb_topic').value, self.on_rgb, qos_profile_sensor_data)

        try:
            from ultralytics import YOLOE
        except Exception as exc:
            raise RuntimeError('ultralytics is required to run yoloe_detector_node') from exc

        self.pkg_share = Path(get_package_share_directory('grasp_demo_pkg'))
        self._prepare_mobileclip()
        classes = [str(v) for v in self.get_parameter('classes').value]
        model_path = self._resolve_weight_path(self.get_parameter('model_path').value)
        if not model_path.is_file():
            raise RuntimeError(f'YOLOE model file not found: {model_path}')
        self.model = YOLOE(str(model_path))
        if classes:
            self.model.set_classes(classes)

        self.server = ActionServer(
            self,
            DetectObject,
            '/demo_detect_object',
            execute_callback=self.execute_callback,
            goal_callback=lambda goal: GoalResponse.ACCEPT,
            cancel_callback=lambda goal: CancelResponse.ACCEPT,
        )

        if bool(self.get_parameter('auto_detect').value):
            period = max(0.1, float(self.get_parameter('run_period_sec').value))
            self.create_timer(period, self.detect_once)

        self.get_logger().info(f'yoloe_detector_node started: rgb={self.get_parameter("rgb_topic").value}')
        self.get_logger().info(f'model={model_path}, classes={classes}')

    def _resolve_weight_path(self, value):
        path = Path(str(value)).expanduser()
        if path.is_absolute():
            return path
        return self.pkg_share / 'weights' / path

    def _prepare_mobileclip(self):
        source = self.pkg_share / 'weights' / 'mobileclip2_b.ts'
        if not source.is_file():
            return
        prefix = Path(get_package_prefix('grasp_demo_pkg'))
        workspace = prefix.parent.parent if prefix.parent.name == 'install' else prefix.parent
        target = workspace / 'mobileclip2_b.ts'
        try:
            if not target.exists() or target.stat().st_size != source.stat().st_size:
                shutil.copy2(source, target)
        except OSError as exc:
            self.get_logger().warn(f'could not prepare mobileclip2_b.ts in workspace root: {exc}')

    def on_rgb(self, msg):
        self.last_rgb = msg

    def _warn_throttle(self, key, period_sec, message):
        now = time.monotonic()
        last = self._throttle_last.get(key, 0.0)
        if now - last >= period_sec:
            self._throttle_last[key] = now
            self.get_logger().warn(message)

    def _save_debug_files(self, image, debug, mask, label, confidence, bbox_xyxy, bbox_data):
        if not bool(self.get_parameter('save_debug_images').value):
            return
        debug_dir = Path(str(self.get_parameter('debug_image_dir').value)).expanduser()
        try:
            debug_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(debug_dir / 'last_detection_raw.png'), image)
            cv2.imwrite(str(debug_dir / 'last_detection_annotated.png'), debug)
            cv2.imwrite(str(debug_dir / 'last_detection_mask.png'), mask)
            x1, y1, x2, y2 = bbox_xyxy
            center_u, center_v, width, height, area, _ = bbox_data
            lines = [
                f'label={label}',
                f'conf={confidence}',
                f'uv=({int(round(center_u))},{int(round(center_v))})',
                f'bbox_xyxy=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})',
                'bbox_center_wh_area_conf='
                f'({center_u:.1f},{center_v:.1f},{width:.1f},{height:.1f},{area:.1f},{confidence:.6f})',
                f'image_shape=({image.shape[1]},{image.shape[0]})',
                f'camera_frame={self.last_rgb.header.frame_id}',
                f'mask_area={area:.1f}',
                'note=2D YOLOE output only; depth and TF are handled by later demo nodes.',
            ]
            (debug_dir / 'last_detection.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
        except OSError as exc:
            self.get_logger().warn(f'failed to save YOLOE debug files: {exc}')

    def _candidate_scores(self, image, result, confs):
        h, w = image.shape[:2]
        roi = [float(v) for v in self.get_parameter('target_roi_norm').value]
        if len(roi) != 4:
            roi = [0.0, 0.0, 1.0, 1.0]
        rx1, ry1, rx2, ry2 = roi
        min_white = float(self.get_parameter('min_white_fraction').value)
        max_area_frac = float(self.get_parameter('max_bbox_area_fraction').value)
        prefer_white = bool(self.get_parameter('prefer_white_target').value)
        xyxy = result.boxes.xyxy.detach().cpu().numpy()
        scores = []
        for i, (x1, y1, x2, y2) in enumerate(xyxy):
            cx = float((x1 + x2) * 0.5) / max(1.0, float(w))
            cy = float((y1 + y2) * 0.5) / max(1.0, float(h))
            bw = max(1.0, float(x2 - x1))
            bh = max(1.0, float(y2 - y1))
            area_frac = (bw * bh) / max(1.0, float(w * h))
            inside_roi = rx1 <= cx <= rx2 and ry1 <= cy <= ry2
            x1i, y1i = max(0, int(x1)), max(0, int(y1))
            x2i, y2i = min(w, int(x2)), min(h, int(y2))
            crop = image[y1i:y2i, x1i:x2i]
            white_fraction = 0.0
            if crop.size:
                hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                white_fraction = float(np.mean((hsv[:, :, 1] < 45) & (hsv[:, :, 2] > 150)))
            passes_filter = True
            if prefer_white:
                passes_filter = (
                    inside_roi
                    and white_fraction >= min_white
                    and area_frac <= max_area_frac
                )
            area_penalty = 1.0 / (1.0 + 30.0 * area_frac)
            roi_bonus = 1.0 if inside_roi else 0.15
            white_bonus = 0.25 + white_fraction
            score = float(confs[i]) * area_penalty * roi_bonus * white_bonus
            if prefer_white and not passes_filter:
                score *= 0.01
            scores.append(
                {
                    'index': i,
                    'score': score,
                    'white_fraction': white_fraction,
                    'area_frac': area_frac,
                    'inside_roi': inside_roi,
                }
            )
        return scores

    def detect_once(self, conf_override=None):
        if self.last_rgb is None:
            self._warn_throttle('no_rgb', 2.0, f'waiting for RGB image on {self.get_parameter("rgb_topic").value}')
            return None
        try:
            image = self.bridge.imgmsg_to_cv2(self.last_rgb, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().warn(f'failed to convert RGB image: {exc}')
            return None

        conf = float(conf_override) if conf_override is not None else float(self.get_parameter('conf').value)
        device = self.get_parameter('device').value
        try:
            results = self.model.predict(image, conf=conf, device=device, verbose=False)
        except Exception as exc:
            self.get_logger().error(f'YOLOE predict failed: {exc}')
            return None
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            self._warn_throttle(
                'no_detection',
                2.0,
                f'YOLOE found no target: classes={self.get_parameter("classes").value}, conf={conf}',
            )
            return None

        result = results[0]
        confs = result.boxes.conf.detach().cpu().numpy()
        scores = self._candidate_scores(image, result, confs)
        best_i = int(max(scores, key=lambda item: item['score'])['index']) if scores else int(np.argmax(confs))
        best_conf = float(confs[best_i])
        x1, y1, x2, y2 = result.boxes.xyxy[best_i].detach().cpu().numpy().tolist()
        cls_id = int(result.boxes.cls[best_i].detach().cpu().item()) if result.boxes.cls is not None else -1
        label = result.names.get(cls_id, str(cls_id)) if hasattr(result, 'names') else str(cls_id)

        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        if getattr(result, 'masks', None) is not None and result.masks is not None and result.masks.data is not None:
            raw_mask = result.masks.data[best_i].detach().cpu().numpy()
            mask = (raw_mask > 0.5).astype(np.uint8) * 255
            if mask.shape[:2] != image.shape[:2]:
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            x1i, y1i = max(0, int(x1)), max(0, int(y1))
            x2i, y2i = min(image.shape[1], int(x2)), min(image.shape[0], int(y2))
            mask[y1i:y2i, x1i:x2i] = 255

        area = float(np.count_nonzero(mask))
        bbox = Float32MultiArray()
        bbox.data = [
            float((x1 + x2) * 0.5),
            float((y1 + y2) * 0.5),
            float(x2 - x1),
            float(y2 - y1),
            area,
            best_conf,
        ]
        self.bbox_pub.publish(bbox)

        mask_msg = self.bridge.cv2_to_imgmsg(mask, encoding='mono8')
        mask_msg.header = self.last_rgb.header
        self.mask_pub.publish(mask_msg)

        label_msg = String()
        label_msg.data = label
        self.label_pub.publish(label_msg)

        conf_msg = Float32()
        conf_msg.data = best_conf
        self.conf_pub.publish(conf_msg)

        debug = image.copy()
        if np.count_nonzero(mask) > 0:
            color = np.zeros_like(debug, dtype=np.uint8)
            color[:, :] = (255, 0, 0)
            debug = np.where(mask[..., None] > 0, (debug * 0.65 + color * 0.35).astype(np.uint8), debug)
        cv2.rectangle(debug, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.drawMarker(
            debug,
            (int(round(bbox.data[0])), int(round(bbox.data[1]))),
            (0, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=12,
            thickness=2,
        )
        cv2.putText(
            debug,
            f'{label} {best_conf:.2f}',
            (int(x1), max(20, int(y1) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        debug_msg = self.bridge.cv2_to_imgmsg(debug, encoding='bgr8')
        debug_msg.header = self.last_rgb.header
        self.debug_pub.publish(debug_msg)

        self._save_debug_files(image, debug, mask, label, best_conf, (x1, y1, x2, y2), bbox.data)

        self.last_detection = {
            'label': label,
            'confidence': best_conf,
            'bbox': bbox.data,
            'bbox_xyxy': [float(x1), float(y1), float(x2), float(y2)],
            'frame_id': self.last_rgb.header.frame_id,
        }
        selected_score = next((item for item in scores if item['index'] == best_i), None)
        if selected_score is not None:
            self.get_logger().info(
                f'YOLOE selected candidate: score={selected_score["score"]:.4f}, '
                f'white={selected_score["white_fraction"]:.2f}, '
                f'area_frac={selected_score["area_frac"]:.4f}, roi={selected_score["inside_roi"]}'
            )
        self.get_logger().info(
            f'YOLOE detected {label}: conf={best_conf:.3f}, '
            f'uv=({bbox.data[0]:.1f},{bbox.data[1]:.1f}), '
            f'bbox=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})'
        )
        return self.last_detection

    async def execute_callback(self, goal_handle):
        goal = goal_handle.request
        result = DetectObject.Result()
        feedback = DetectObject.Feedback()
        target_classes = [str(v) for v in goal.target_classes] if goal.target_classes else [str(v) for v in self.get_parameter('classes').value]
        if target_classes:
            self.model.set_classes(target_classes)
        conf_override = float(goal.confidence_threshold) if goal.confidence_threshold > 0.0 else None
        timeout = goal.timeout if goal.timeout > 0.0 else 10.0
        start = self.get_clock().now()
        attempts = 0
        while rclpy.ok():
            elapsed = (self.get_clock().now() - start).nanoseconds / 1e9
            if elapsed > timeout:
                result.success = False
                result.message = 'YOLOE detection timed out'
                goal_handle.abort()
                return result
            if goal_handle.is_cancel_requested:
                result.success = False
                result.message = 'YOLOE detection canceled'
                goal_handle.canceled()
                return result
            attempts += 1
            feedback.state = 'DETECTING'
            feedback.detection_attempts = attempts
            goal_handle.publish_feedback(feedback)
            det = self.detect_once(conf_override=conf_override)
            if det is not None:
                result.success = True
                result.message = 'YOLOE detection succeeded; 2D mask and bbox have been published'
                result.detected_class = det['label']
                result.confidence = float(det['confidence'])
                result.object_position = PointStamped()
                result.object_position.header.frame_id = det['frame_id']
                result.normal = Vector3Stamped()
                result.long_axis = Vector3Stamped()
                goal_handle.succeed()
                return result
            time.sleep(0.05)


def main(args=None):
    rclpy.init(args=args)
    node = YoloEDetectorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
