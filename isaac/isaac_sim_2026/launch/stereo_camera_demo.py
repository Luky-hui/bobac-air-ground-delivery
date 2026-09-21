#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stereo_camera_demo.py

双目相机数据发布与接收演示脚本

功能：
1. 订阅 Isaac Sim 发布的双目相机图像数据
2. 在终端打印接收到的图像信息（分辨率、时间戳、数据大小等）
3. 可选：将图像数据转换并显示统计信息

使用方式：
方式1 - 使用 Isaac Sim Python 环境运行：
cd ~/ros2_ws/src/isaacsim2026/isaac_sim_2026
~/isaac-sim-4.5.0/python.sh launch/stereo_camera_demo.py --ros-args -p use_sim_time:=true

方式2 - 使用系统 ROS2 环境运行（如果已安装 cv_bridge）：
ros2 run isaac_sim stereo_camera_demo.py --ros-args -p use_sim_time:=true

话题订阅：
- /camera/left/image_raw  (左目相机图像)
- /camera/right/image_raw (右目相机图像)
- /camera/left/camera_info  (左目相机参数)
- /camera/right/camera_info (右目相机参数)

验证命令：
ros2 topic list | grep camera
ros2 topic hz /camera/left/image_raw
ros2 topic echo /camera/left/camera_info --once
"""

import sys
import time
from typing import Dict, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from sensor_msgs.msg import Image, CameraInfo

# 尝试导入 cv_bridge（可选，用于图像转换）
try:
    from cv_bridge import CvBridge
    import numpy as np
    CV_BRIDGE_AVAILABLE = True
except ImportError:
    CV_BRIDGE_AVAILABLE = False
    print("[警告] cv_bridge 未安装，将只显示基本图像信息")


class StereoCameraReceiver(Node):
    """双目相机数据接收节点"""

    def __init__(self):
        super().__init__("stereo_camera_receiver")

        # ---- 参数声明 ----
        self.declare_parameter("left_image_topic", "/camera/left/image_raw")
        self.declare_parameter("right_image_topic", "/camera/right/image_raw")
        self.declare_parameter("left_info_topic", "/camera/left/camera_info")
        self.declare_parameter("right_info_topic", "/camera/right/camera_info")
        self.declare_parameter("print_interval", 1.0)  # 打印间隔（秒）
        self.declare_parameter("show_statistics", True)  # 是否显示统计信息

        # 获取参数
        left_image_topic = self.get_parameter("left_image_topic").value
        right_image_topic = self.get_parameter("right_image_topic").value
        left_info_topic = self.get_parameter("left_info_topic").value
        right_info_topic = self.get_parameter("right_info_topic").value

        # QoS 配置（与 Isaac Sim 兼容）
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        # ---- 订阅话题 ----
        self.sub_left_image = self.create_subscription(
            Image,
            left_image_topic,
            self._on_left_image,
            qos
        )
        self.sub_right_image = self.create_subscription(
            Image,
            right_image_topic,
            self._on_right_image,
            qos
        )
        self.sub_left_info = self.create_subscription(
            CameraInfo,
            left_info_topic,
            self._on_left_info,
            qos
        )
        self.sub_right_info = self.create_subscription(
            CameraInfo,
            right_info_topic,
            self._on_right_info,
            qos
        )

        # ---- 状态变量 ----
        self.left_image_count = 0
        self.right_image_count = 0
        self.left_info_count = 0
        self.right_info_count = 0

        self.last_left_image: Optional[Image] = None
        self.last_right_image: Optional[Image] = None
        self.last_left_info: Optional[CameraInfo] = None
        self.last_right_info: Optional[CameraInfo] = None

        self.last_print_time = time.time()
        self.start_time = time.time()

        # cv_bridge（如果可用）
        self.bridge = CvBridge() if CV_BRIDGE_AVAILABLE else None

        # 统计信息
        self.stats: Dict[str, Dict] = {
            "left": {"count": 0, "total_bytes": 0, "last_time": 0.0},
            "right": {"count": 0, "total_bytes": 0, "last_time": 0.0},
        }

        self.get_logger().info("=" * 70)
        self.get_logger().info("双目相机数据接收节点已启动")
        self.get_logger().info("=" * 70)
        self.get_logger().info(f"订阅话题:")
        self.get_logger().info(f"  - 左目图像: {left_image_topic}")
        self.get_logger().info(f"  - 右目图像: {right_image_topic}")
        self.get_logger().info(f"  - 左目参数: {left_info_topic}")
        self.get_logger().info(f"  - 右目参数: {right_info_topic}")
        self.get_logger().info(f"cv_bridge 可用: {CV_BRIDGE_AVAILABLE}")
        self.get_logger().info("=" * 70)

        # 创建定时器，定期打印统计信息
        print_interval = float(self.get_parameter("print_interval").value)
        self.timer = self.create_timer(print_interval, self._print_statistics)

    # ---------------------- 回调函数 ----------------------
    def _on_left_image(self, msg: Image):
        """左目图像回调"""
        self.left_image_count += 1
        self.last_left_image = msg
        self._update_stats("left", msg)

    def _on_right_image(self, msg: Image):
        """右目图像回调"""
        self.right_image_count += 1
        self.last_right_image = msg
        self._update_stats("right", msg)

    def _on_left_info(self, msg: CameraInfo):
        """左目相机参数回调"""
        self.left_info_count += 1
        self.last_left_info = msg

    def _on_right_info(self, msg: CameraInfo):
        """右目相机参数回调"""
        self.right_info_count += 1
        self.last_right_info = msg

    # ---------------------- 统计与打印 ----------------------
    def _update_stats(self, camera: str, msg: Image):
        """更新统计信息"""
        data_size = len(msg.data)
        self.stats[camera]["count"] += 1
        self.stats[camera]["total_bytes"] += data_size
        self.stats[camera]["last_time"] = time.time()

    def _print_statistics(self):
        """定期打印统计信息"""
        current_time = time.time()
        elapsed = current_time - self.start_time

        self.get_logger().info("")
        self.get_logger().info("=" * 70)
        self.get_logger().info(f"运行时间: {elapsed:.1f}s")
        self.get_logger().info("-" * 70)

        # 左目统计
        self._print_camera_stats("左目", "left", self.last_left_image, self.last_left_info)

        self.get_logger().info("-" * 70)

        # 右目统计
        self._print_camera_stats("右目", "right", self.last_right_image, self.last_right_info)

        self.get_logger().info("=" * 70)

    def _print_camera_stats(
        self,
        name: str,
        key: str,
        last_image: Optional[Image],
        last_info: Optional[CameraInfo]
    ):
        """打印单个相机的统计信息"""
        stats = self.stats[key]
        count = stats["count"]
        total_mb = stats["total_bytes"] / (1024 * 1024)
        last_time = stats["last_time"]

        self.get_logger().info(f"{name}相机:")

        if count == 0:
            self.get_logger().warn(f"  [警告] 未接收到图像数据")
            return

        # 计算频率
        elapsed = time.time() - self.start_time
        freq = count / elapsed if elapsed > 0 else 0.0

        self.get_logger().info(f"  接收帧数: {count}")
        self.get_logger().info(f"  接收频率: {freq:.2f} Hz")
        self.get_logger().info(f"  总数据量: {total_mb:.2f} MB")

        # 最新图像信息
        if last_image is not None:
            age = time.time() - last_time
            self.get_logger().info(f"  最新图像:")
            self.get_logger().info(f"    - 分辨率: {last_image.width} x {last_image.height}")
            self.get_logger().info(f"    - 编码格式: {last_image.encoding}")
            self.get_logger().info(f"    - 数据大小: {len(last_image.data)} bytes")
            self.get_logger().info(f"    - 时间戳: {last_image.header.stamp.sec}.{last_image.header.stamp.nanosec:09d}")
            self.get_logger().info(f"    - 数据年龄: {age:.3f}s")

            # 如果 cv_bridge 可用，显示图像统计
            if self.bridge is not None and self.get_parameter("show_statistics").value:
                try:
                    cv_image = self.bridge.imgmsg_to_cv2(last_image, desired_encoding="passthrough")
                    self._print_image_statistics(cv_image)
                except Exception as e:
                    self.get_logger().warn(f"    [图像转换失败] {e}")

        # 相机参数信息
        if last_info is not None:
            self.get_logger().info(f"  相机参数:")
            self.get_logger().info(f"    - 分辨率: {last_info.width} x {last_info.height}")
            self.get_logger().info(f"    - 畸变模型: {last_info.distortion_model}")
            if len(last_info.k) >= 5:
                fx, fy = last_info.k[0], last_info.k[4]
                cx, cy = last_info.k[2], last_info.k[5]
                self.get_logger().info(f"    - 焦距 (fx, fy): ({fx:.2f}, {fy:.2f})")
                self.get_logger().info(f"    - 主点 (cx, cy): ({cx:.2f}, {cy:.2f})")

    def _print_image_statistics(self, cv_image):
        """打印图像统计信息（需要 cv_bridge）"""
        try:
            import numpy as np

            if len(cv_image.shape) == 3:
                # 彩色图像
                mean_val = np.mean(cv_image, axis=(0, 1))
                std_val = np.std(cv_image, axis=(0, 1))
                self.get_logger().info(f"    - 像素均值 (BGR): [{mean_val[0]:.1f}, {mean_val[1]:.1f}, {mean_val[2]:.1f}]")
                self.get_logger().info(f"    - 像素标准差 (BGR): [{std_val[0]:.1f}, {std_val[1]:.1f}, {std_val[2]:.1f}]")
            else:
                # 灰度图像
                mean_val = np.mean(cv_image)
                std_val = np.std(cv_image)
                min_val = np.min(cv_image)
                max_val = np.max(cv_image)
                self.get_logger().info(f"    - 像素范围: [{min_val}, {max_val}]")
                self.get_logger().info(f"    - 像素均值: {mean_val:.1f}")
                self.get_logger().info(f"    - 像素标准差: {std_val:.1f}")
        except Exception as e:
            self.get_logger().warn(f"    [统计计算失败] {e}")


class StereoCameraPublisher(Node):
    """
    双目相机数据发布节点（可选）

    注意：通常 Isaac Sim 会自动发布相机数据，此节点仅用于测试或独立运行
    """

    def __init__(self):
        super().__init__("stereo_camera_publisher")

        self.declare_parameter("publish_rate", 30.0)  # Hz
        self.declare_parameter("image_width", 640)
        self.declare_parameter("image_height", 480)

        # QoS 配置
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        # 发布器
        self.pub_left_image = self.create_publisher(
            Image,
            "/camera/left/image_raw",
            qos
        )
        self.pub_right_image = self.create_publisher(
            Image,
            "/camera/right/image_raw",
            qos
        )
        self.pub_left_info = self.create_publisher(
            CameraInfo,
            "/camera/left/camera_info",
            qos
        )
        self.pub_right_info = self.create_publisher(
            CameraInfo,
            "/camera/right/camera_info",
            qos
        )

        # 定时器
        rate = float(self.get_parameter("publish_rate").value)
        self.timer = self.create_timer(1.0 / rate, self._publish_data)

        self.frame_count = 0

        self.get_logger().info("=" * 70)
        self.get_logger().info("双目相机数据发布节点已启动（测试模式）")
        self.get_logger().info(f"发布频率: {rate} Hz")
        self.get_logger().info("=" * 70)

    def _publish_data(self):
        """发布测试数据"""
        self.frame_count += 1

        width = int(self.get_parameter("image_width").value)
        height = int(self.get_parameter("image_height").value)

        # 创建测试图像（渐变图案）
        if CV_BRIDGE_AVAILABLE:
            import numpy as np
            # 创建渐变图案
            test_image = np.zeros((height, width, 3), dtype=np.uint8)
            test_image[:, :, 0] = (self.frame_count % 256)  # B
            test_image[:, :, 1] = ((self.frame_count * 2) % 256)  # G
            test_image[:, :, 2] = ((self.frame_count * 3) % 256)  # R

            bridge = CvBridge()
            left_msg = bridge.cv2_to_imgmsg(test_image, encoding="bgr8")
            right_msg = bridge.cv2_to_imgmsg(test_image, encoding="bgr8")
        else:
            # 创建空白图像消息
            left_msg = Image()
            right_msg = Image()
            left_msg.width = width
            left_msg.height = height
            left_msg.encoding = "rgb8"
            left_msg.step = width * 3
            left_msg.data = [0] * (width * height * 3)
            right_msg = left_msg

        # 设置时间戳
        stamp = self.get_clock().now().to_msg()
        left_msg.header.stamp = stamp
        left_msg.header.frame_id = "Left_Camera"
        right_msg.header.stamp = stamp
        right_msg.header.frame_id = "Right_Camera"

        # 发布图像
        self.pub_left_image.publish(left_msg)
        self.pub_right_image.publish(right_msg)

        # 发布相机参数
        left_info = self._create_camera_info(stamp, "Left_Camera", width, height)
        right_info = self._create_camera_info(stamp, "Right_Camera", width, height)
        self.pub_left_info.publish(left_info)
        self.pub_right_info.publish(right_info)

        if self.frame_count % 30 == 0:
            self.get_logger().info(f"已发布 {self.frame_count} 帧")

    def _create_camera_info(self, stamp, frame_id: str, width: int, height: int) -> CameraInfo:
        """创建相机参数消息"""
        info = CameraInfo()
        info.header.stamp = stamp
        info.header.frame_id = frame_id
        info.width = width
        info.height = height
        info.distortion_model = "plumb_bob"

        # 简单的相机内参（示例）
        fx = fy = 500.0
        cx = width / 2.0
        cy = height / 2.0

        info.k = [
            fx, 0.0, cx,
            0.0, fy, cy,
            0.0, 0.0, 1.0
        ]

        info.p = [
            fx, 0.0, cx, 0.0,
            0.0, fy, cy, 0.0,
            0.0, 0.0, 1.0, 0.0
        ]

        info.d = [0.0, 0.0, 0.0, 0.0, 0.0]

        return info


def main(args=None):
    """主函数"""
    rclpy.init(args=args)

    # 检查命令行参数
    mode = "receiver"  # 默认为接收模式
    if len(sys.argv) > 1:
        if sys.argv[1] == "--publisher":
            mode = "publisher"
        elif sys.argv[1] == "--help":
            print(__doc__)
            return

    try:
        if mode == "publisher":
            node = StereoCameraPublisher()
            print("\n[模式] 发布测试数据（通常不需要，Isaac Sim 会自动发布）")
        else:
            node = StereoCameraReceiver()
            print("\n[模式] 接收并显示双目相机数据")

        print("\n按 Ctrl+C 退出\n")
        rclpy.spin(node)

    except KeyboardInterrupt:
        print("\n[退出] 用户中断")
    except Exception as e:
        print(f"\n[错误] {e}")
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
