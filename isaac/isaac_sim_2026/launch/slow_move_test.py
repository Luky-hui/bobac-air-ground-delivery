#!/usr/bin/env python3
import argparse
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import TransformStamped
import sys
import time
from collections import deque
import math

class PoseEcho(Node):
    def __init__(self, ee_frame: str, gripper_left: str, gripper_right: str, tag_frame: str, tag_source: str, max_tf_age: float):
        super().__init__('pose_echo_tool')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.ee_frame = ee_frame
        self.gripper_left = gripper_left
        self.gripper_right = gripper_right
        self.tag_frame = tag_frame
        self.tag_source = tag_source
        self.max_tf_age = float(max_tf_age)
        
        # === 新增：滑动平均滤波器配置 ===
        # 如果左右目同时发布，数据量会翻倍，适当增加缓冲区长度以保持稳定
        self.history_len = 40  
        self.tag_history_x = deque(maxlen=self.history_len)
        self.tag_history_y = deque(maxlen=self.history_len)
        self.tag_history_z = deque(maxlen=self.history_len)
        self.tag_history_bx = deque(maxlen=self.history_len)
        self.tag_history_by = deque(maxlen=self.history_len)
        self.tag_history_bz = deque(maxlen=self.history_len)
        # ============================
        
        self.get_logger().info("=== 坐标融合提取工具已启动 ===")
        self.get_logger().info("支持多Tag命名: tag36h11:79, left_tag..., right_tag...")
        self.get_logger().info(f"tag_source={self.tag_source} | max_tf_age={self.max_tf_age:.2f}s")
        self.get_logger().info("正在对双目/抖动的TF数据进行平滑融合处理...")
        
        # 采样频率 10Hz
        self.timer = self.create_timer(0.1, self.print_poses)
        
        # 控制打印频率（不刷屏）
        self.print_count = 0

    def get_transform(self, target_frame, source_frame):
        try:
            # 查找从 source_frame 到 target_frame 的变换
            t = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time())
            return t
        except Exception as e:
            return None


    def _tf_age(self, tf_msg: TransformStamped) -> float:
        try:
            stamp = Time.from_msg(tf_msg.header.stamp, clock_type=self.get_clock().clock_type)
        except TypeError:
            stamp = Time.from_msg(tf_msg.header.stamp)
        return (self.get_clock().now() - stamp).nanoseconds * 1e-9

    def lookup_stereo_tf(self, parent_frame: str):
        left = f"left_{self.tag_frame}"
        right = f"right_{self.tag_frame}"
        tag_source = str(self.tag_source).lower()

        left_tf = self.get_transform(parent_frame, left)
        right_tf = self.get_transform(parent_frame, right)

        def _age(tf_msg: TransformStamped) -> float:
            return self._tf_age(tf_msg)

        def _age_ok(tf_msg: TransformStamped) -> bool:
            return tf_msg is not None and _age(tf_msg) < self.max_tf_age

        if tag_source == "left":
            if _age_ok(left_tf):
                t = left_tf.transform.translation
                return (t.x, t.y, t.z, _age(left_tf))
            if _age_ok(right_tf):
                t = right_tf.transform.translation
                return (t.x, t.y, t.z, _age(right_tf))
            direct_tf = self.get_transform(parent_frame, self.tag_frame)
            if direct_tf is None:
                return None
            t = direct_tf.transform.translation
            return (t.x, t.y, t.z, _age(direct_tf))

        if tag_source == "right":
            if _age_ok(right_tf):
                t = right_tf.transform.translation
                return (t.x, t.y, t.z, _age(right_tf))
            if _age_ok(left_tf):
                t = left_tf.transform.translation
                return (t.x, t.y, t.z, _age(left_tf))
            direct_tf = self.get_transform(parent_frame, self.tag_frame)
            if direct_tf is None:
                return None
            t = direct_tf.transform.translation
            return (t.x, t.y, t.z, _age(direct_tf))

        if tag_source == "direct":
            direct_tf = self.get_transform(parent_frame, self.tag_frame)
            if direct_tf is None:
                return None
            t = direct_tf.transform.translation
            return (t.x, t.y, t.z, _age(direct_tf))

        # fused (default)
        if not _age_ok(left_tf):
            left_tf = None
        if not _age_ok(right_tf):
            right_tf = None

        if left_tf is None and right_tf is None:
            direct_tf = self.get_transform(parent_frame, self.tag_frame)
            if direct_tf is None:
                return None
            t = direct_tf.transform.translation
            return (t.x, t.y, t.z, _age(direct_tf))
        if left_tf is None:
            t = right_tf.transform.translation
            return (t.x, t.y, t.z, _age(right_tf))
        if right_tf is None:
            t = left_tf.transform.translation
            return (t.x, t.y, t.z, _age(left_tf))

        lt = left_tf.transform.translation
        rt = right_tf.transform.translation
        fx = (lt.x + rt.x) * 0.5
        fy = (lt.y + rt.y) * 0.5
        fz = (lt.z + rt.z) * 0.5
        return (fx, fy, fz, min(_age(left_tf), _age(right_tf)))

    def print_poses(self):
        # 1. 获取机器人坐标
        robot_tf = self.get_transform('map', 'base_link')
        
        # 2. 获取 Tag 坐标 (兼容 left/right 前缀)
        # 遍历所有可能的 Tag 名字，谁存在就用谁，都存在就一起用（融合）
        # fused tag pose (map/base_link)
        tag_map = self.lookup_stereo_tf("map")
        tag_base = self.lookup_stereo_tf("base_link")
        left_tag_map = self.get_transform("map", f"left_{self.tag_frame}")
        right_tag_map = self.get_transform("map", f"right_{self.tag_frame}")
        left_tag_base = self.get_transform("base_link", f"left_{self.tag_frame}")
        right_tag_base = self.get_transform("base_link", f"right_{self.tag_frame}")

        if tag_map is not None:
            avg_x, avg_y, avg_z, _ = tag_map
            self.tag_history_x.append(avg_x)
            self.tag_history_y.append(avg_y)
            self.tag_history_z.append(avg_z)

        if tag_base is not None:
            bavg_x, bavg_y, bavg_z, _ = tag_base
            self.tag_history_bx.append(bavg_x)
            self.tag_history_by.append(bavg_y)
            self.tag_history_bz.append(bavg_z)
        self.print_count += 1
        if self.print_count < 10:
            return
        self.print_count = 0

        print("\n" + "="*50)
        
        # --- 打印机器人位置 ---
        if robot_tf:
            trans = robot_tf.transform.translation
            rot = robot_tf.transform.rotation
            print(f"[1] 机器人本体 (map系) -> 用于导航代码")
            print(f"    x: {trans.x:.3f} | y: {trans.y:.3f} | z: {trans.z:.3f}")
            print(f"    w: {rot.w:.2f} (朝向)")
        else:
            print("[1] 等待机器人定位 (map -> base_link)...")

        print("-" * 50)

        # --- EE / gripper center (map) ---
        ee_tf = self.get_transform("map", self.ee_frame)
        left_tf = self.get_transform("map", self.gripper_left)
        right_tf = self.get_transform("map", self.gripper_right)
        center_xyz = None
        left_age = self._tf_age(left_tf) if left_tf else None
        right_age = self._tf_age(right_tf) if right_tf else None
        if ee_tf:
            ee_t = ee_tf.transform.translation
            print(f"[3] EE (map) -> {self.ee_frame} | x: {ee_t.x:.3f} | y: {ee_t.y:.3f} | z: {ee_t.z:.3f}")
        else:
            print(f"[3] EE (map) -> {self.ee_frame} | unavailable")
        if left_tf and right_tf:
            lt = left_tf.transform.translation
            rt = right_tf.transform.translation
            cx = (lt.x + rt.x) * 0.5
            cy = (lt.y + rt.y) * 0.5
            cz = (lt.z + rt.z) * 0.5
            center_xyz = (cx, cy, cz)
            print(f"[4] Gripper center (map) -> ({cx:.3f}, {cy:.3f}, {cz:.3f})")
        if left_tf:
            lt = left_tf.transform.translation
            print(f"    [4L] gripper_left(map) x={lt.x:.3f} y={lt.y:.3f} z={lt.z:.3f} age={left_age:.2f}s")
        if right_tf:
            rt = right_tf.transform.translation
            print(f"    [4R] gripper_right(map) x={rt.x:.3f} y={rt.y:.3f} z={rt.z:.3f} age={right_age:.2f}s")
        else:
            print("[4] Gripper center (map) -> unavailable")

        # --- 打印融合后的 Tag 位置 ---
        if len(self.tag_history_x) > 0:
            # 计算平均值 (融合)
            avg_x = sum(self.tag_history_x) / len(self.tag_history_x)
            avg_y = sum(self.tag_history_y) / len(self.tag_history_y)
            avg_z = sum(self.tag_history_z) / len(self.tag_history_z)
            
            # 计算波动范围 (标准差)，让你知道误差有多大
            std_x = 0.0
            if len(self.tag_history_x) > 1:
                std_x = math.sqrt(sum((x - avg_x) ** 2 for x in self.tag_history_x) / len(self.tag_history_x))

            print(f"[2] AprilTag 融合位置 (map系) -> 用于文档填写")
            if left_tag_map:
                lt = left_tag_map.transform.translation
                print(f"    [2L] left_tag(map) x={lt.x:.4f} y={lt.y:.4f} z={lt.z:.4f} age={self._tf_age(left_tag_map):.2f}s")
            if right_tag_map:
                rt = right_tag_map.transform.translation
                print(f"    [2R] right_tag(map) x={rt.x:.4f} y={rt.y:.4f} z={rt.z:.4f} age={self._tf_age(right_tag_map):.2f}s")
            print(f"    状态: 已融合最近 {len(self.tag_history_x)} 帧数据 (左右目自动平均)")
            print(f"    X: {avg_x:.4f} (波动: ±{std_x:.4f})")
            print(f"    Y: {avg_y:.4f}")
            print(f"    Z: {avg_z:.4f}")
            if len(self.tag_history_bx) > 0:
                bavg_x = sum(self.tag_history_bx) / len(self.tag_history_bx)
                bavg_y = sum(self.tag_history_by) / len(self.tag_history_by)
                bavg_z = sum(self.tag_history_bz) / len(self.tag_history_bz)
                print(f"[5] Tag (base_link) -> X: {bavg_x:.4f} | Y: {bavg_y:.4f} | Z: {bavg_z:.4f}")
                if left_tag_base:
                    lt_b = left_tag_base.transform.translation
                    print(f"    [5L] left_tag(base_link) x={lt_b.x:.4f} y={lt_b.y:.4f} z={lt_b.z:.4f} age={self._tf_age(left_tag_base):.2f}s")
                if right_tag_base:
                    rt_b = right_tag_base.transform.translation
                    print(f"    [5R] right_tag(base_link) x={rt_b.x:.4f} y={rt_b.y:.4f} z={rt_b.z:.4f} age={self._tf_age(right_tag_base):.2f}s")
            else:
                bavg_x = bavg_y = bavg_z = None
            left_tf_b = self.get_transform("base_link", self.gripper_left)
            right_tf_b = self.get_transform("base_link", self.gripper_right)
            if left_tf_b and right_tf_b:
                lt_b = left_tf_b.transform.translation
                rt_b = right_tf_b.transform.translation
                cbx = (lt_b.x + rt_b.x) * 0.5
                cby = (lt_b.y + rt_b.y) * 0.5
                cbz = (lt_b.z + rt_b.z) * 0.5
                print(f"[6] Gripper center (base_link) -> ({cbx:.3f}, {cby:.3f}, {cbz:.3f})")
                print(f"    [6L] gripper_left(base_link) x={lt_b.x:.3f} y={lt_b.y:.3f} z={lt_b.z:.3f}")
                print(f"    [6R] gripper_right(base_link) x={rt_b.x:.3f} y={rt_b.y:.3f} z={rt_b.z:.3f}")
                if bavg_x is not None:
                    dx = cbx - bavg_x
                    dy = cby - bavg_y
                    dz = cbz - bavg_z
                    err = math.sqrt(dx * dx + dy * dy + dz * dz)
                    print(f"    Tag->Gripper delta (base_link): ({dx:.3f}, {dy:.3f}, {dz:.3f}) | err={err:.3f}m")
            else:
                print("[6] Gripper center (base_link) -> unavailable")
            if center_xyz is not None:
                dx = center_xyz[0] - avg_x
                dy = center_xyz[1] - avg_y
                dz = center_xyz[2] - avg_z
                err = math.sqrt(dx * dx + dy * dy + dz * dz)
                print(f"    Tag->Gripper delta: ({dx:.3f}, {dy:.3f}, {dz:.3f}) | err={err:.3f}m")
            
            if std_x > 0.05:
                print("    [提示] 波动较大，建议暂停移动等待数值稳定")
        else:
            print(f"[2] ???? Tag (???: {[self.tag_frame, f'left_{self.tag_frame}', f'right_{self.tag_frame}']})")
            
        print("="*50)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ee-frame", default="link7_R")
    parser.add_argument("--gripper-left-frame", default="gripper01_left1")
    parser.add_argument("--gripper-right-frame", default="gripper01_right1")
    parser.add_argument("--tag-frame", default="tag36h11:79")
    parser.add_argument("--tag-source", default="fused", choices=["fused", "left", "right", "direct"])
    parser.add_argument("--max-tf-age", type=float, default=0.5)
    args = parser.parse_args()
    rclpy.init()
    node = PoseEcho(args.ee_frame, args.gripper_left_frame, args.gripper_right_frame, args.tag_frame, args.tag_source, args.max_tf_age)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
