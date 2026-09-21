#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机械臂 ROS2 链路诊断脚本。

用途：
  1. 检查 Isaac Sim 是否发布 /joint_states 和 /joint_left_states
  2. 检查关节名是否与现有脚本匹配
  3. 可选发布一个很小的右臂关节测试动作，验证 /joint_command 是否被 Isaac 接收

运行：
  python3 /home/u-zhuang/ros2_ws/src/isaac/isaac_sim_2026/launch/arm_diagnostics.py --ros-args -p use_sim_time:=true

带动作测试：
  python3 /home/u-zhuang/ros2_ws/src/isaac/isaac_sim_2026/launch/arm_diagnostics.py --ros-args \
    -p use_sim_time:=true -p send_test_motion:=true
"""

import time
from typing import Dict, Iterable, List, Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


RIGHT_ARM = ["joint1_R", "joint2_R", "joint3_R", "joint4_R", "joint5_R", "joint6_R"]
LEFT_ARM = ["joint1_L", "joint2_L", "joint3_L", "joint4_L", "joint5_L", "joint6_L"]
RIGHT_GRIPPER = "gripper_controller"
LEFT_GRIPPER = "gripper_left_controller"


class ArmDiagnostics(Node):
    def __init__(self) -> None:
        super().__init__("arm_diagnostics")

        self.declare_parameter("wait_sec", 5.0)
        self.declare_parameter("send_test_motion", False)
        self.declare_parameter("test_joint", "joint1_R")
        self.declare_parameter("test_delta", 0.08)
        self.declare_parameter("test_duration", 2.0)
        self.declare_parameter("settle_sec", 1.0)
        self.declare_parameter("change_tolerance", 0.01)

        self.main_state: Dict[str, float] = {}
        self.left_state: Dict[str, float] = {}
        self.main_msg_count = 0
        self.left_msg_count = 0
        self.last_main_names: List[str] = []
        self.last_left_names: List[str] = []

        self.create_subscription(JointState, "/joint_states", self._main_cb, 10)
        self.create_subscription(JointState, "/joint_left_states", self._left_cb, 10)
        self.pub_main = self.create_publisher(JointState, "/joint_command", 10)
        self.pub_left = self.create_publisher(JointState, "/joint_left_command", 10)

    def _main_cb(self, msg: JointState) -> None:
        self.main_msg_count += 1
        self.last_main_names = list(msg.name)
        for name, pos in zip(msg.name, msg.position):
            self.main_state[name] = float(pos)

    def _left_cb(self, msg: JointState) -> None:
        self.left_msg_count += 1
        self.last_left_names = list(msg.name)
        for name, pos in zip(msg.name, msg.position):
            self.left_state[name] = float(pos)

    def _wait(self, duration: float) -> None:
        end = time.time() + duration
        while rclpy.ok() and time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

    def _missing(self, names: Iterable[str], state: Dict[str, float]) -> List[str]:
        return [name for name in names if name not in state]

    def _print_report(self) -> bool:
        wait_sec = float(self.get_parameter("wait_sec").value)
        self.get_logger().info(f"等待关节状态 {wait_sec:.1f}s ...")
        self._wait(wait_sec)

        ok = True
        if self.main_msg_count == 0:
            self.get_logger().error("没有收到 /joint_states。请确认 Isaac Sim 场景已 Play，ROS2 Bridge/ActionGraph 已启用。")
            ok = False
        else:
            self.get_logger().info(f"/joint_states 已收到 {self.main_msg_count} 条，最后一条包含 {len(self.last_main_names)} 个关节")

        if self.left_msg_count == 0:
            self.get_logger().warn("没有收到 /joint_left_states。左夹爪可能无法单独控制，但双臂主关节仍可继续检查。")
        else:
            self.get_logger().info(f"/joint_left_states 已收到 {self.left_msg_count} 条，最后一条包含 {len(self.last_left_names)} 个关节")

        checks = [
            ("右臂", RIGHT_ARM, self.main_state),
            ("左臂", LEFT_ARM, self.main_state),
            ("右夹爪", [RIGHT_GRIPPER], self.main_state),
            ("左夹爪", [LEFT_GRIPPER], self.left_state),
        ]
        for label, names, state in checks:
            missing = self._missing(names, state)
            if missing:
                if label in ("右臂", "左臂"):
                    self.get_logger().error(f"{label}缺少关节: {missing}")
                    ok = False
                else:
                    self.get_logger().warn(
                        f"{label}缺少关节: {missing}。"
                        "2026 官方场景可能只发布夹爪 mimic 关节；主关节测试不受影响。"
                    )
            else:
                values = ", ".join(f"{name}={state[name]:+.3f}" for name in names)
                self.get_logger().info(f"{label}关节名匹配: {values}")

        return ok

    def _publish_hold(self, override: Optional[Dict[str, float]] = None) -> None:
        override = override or {}

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = RIGHT_ARM + LEFT_ARM
        msg.position = [float(override.get(j, self.main_state.get(j, 0.0))) for j in msg.name]
        if RIGHT_GRIPPER in self.main_state:
            msg.name.append(RIGHT_GRIPPER)
            msg.position.append(float(override.get(RIGHT_GRIPPER, self.main_state[RIGHT_GRIPPER])))
        self.pub_main.publish(msg)

        if LEFT_GRIPPER in self.left_state:
            msg_left = JointState()
            msg_left.header.stamp = msg.header.stamp
            msg_left.name = [LEFT_GRIPPER]
            msg_left.position = [float(override.get(LEFT_GRIPPER, self.left_state[LEFT_GRIPPER]))]
            self.pub_left.publish(msg_left)

    def _send_test_motion(self) -> None:
        joint = str(self.get_parameter("test_joint").value)
        if joint not in self.main_state:
            self.get_logger().error(f"无法测试 {joint}: /joint_states 中没有这个关节")
            return

        start = float(self.main_state[joint])
        delta = float(self.get_parameter("test_delta").value)
        duration = max(0.1, float(self.get_parameter("test_duration").value))
        target = start + delta
        steps = max(10, int(duration * 50.0))

        self.get_logger().info(f"发布小幅测试动作: {joint} {start:+.3f} -> {target:+.3f}")
        for i in range(steps + 1):
            alpha = i / steps
            q = start + alpha * (target - start)
            self._publish_hold({joint: q})
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(duration / steps)

        settle_sec = float(self.get_parameter("settle_sec").value)
        self._wait(settle_sec)
        observed = float(self.main_state.get(joint, start))
        moved = abs(observed - start)
        tol = float(self.get_parameter("change_tolerance").value)
        if moved >= tol:
            self.get_logger().info(f"动作链路正常: {joint} 状态变化 {moved:.3f} rad，当前 {observed:+.3f}")
        else:
            self.get_logger().error(
                f"已发布 /joint_command，但 {joint} 状态几乎没变 ({moved:.3f} rad)。"
                "重点检查 Isaac Sim 中订阅 /joint_command 的 ActionGraph/Articulation Controller。"
            )

    def run(self) -> None:
        ok = self._print_report()
        if bool(self.get_parameter("send_test_motion").value):
            if ok:
                self._send_test_motion()
            else:
                self.get_logger().error("基础关节状态不完整，跳过动作测试。")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ArmDiagnostics()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
