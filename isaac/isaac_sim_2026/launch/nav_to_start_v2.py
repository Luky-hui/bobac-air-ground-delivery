#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nav_to_start_v2.py - 机器人导航与动作准备脚本 (完整版)

功能流程：
1. 导航阶段：使用 Nav2 导航到指定起始点
2. 动作准备：控制机械臂执行"打开双臂"动作
3. 接近目标：底盘直线前进接近抓取台面

运行方式：
python3 /home/u-zhuang/ros2_ws/src/isaac/isaac_sim_2026/launch/nav_to_start_v2.py --ros-args -p use_sim_time:=true

前置条件：
1. Isaac Sim 已启动并加载场景
2. RTAB-Map 建图已启动: ros2 launch isaac_sim rtab-map-scan.launch.py
3. Nav2 导航已启动: ros2 launch isaac_sim navigation2.launch.py

参考文档：
- navigation2.launch.py: Nav2 启动配置
- task1_final.py: 机械臂控制方法
- task2_direct.py: 底盘速度控制方法

注意：
- 本脚本使用 RTAB-Map 进行定位，不使用 AMCL
- 因此跳过 waitUntilNav2Active() 中的 AMCL 检查
"""

import time
import math

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.action import ActionClient

from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus
from tf2_ros import Buffer, TransformListener, TransformException


class NavToStartNode(Node):
    """导航与动作准备节点"""

    def __init__(self):
        super().__init__('nav_to_start_node')

        # ================= 0. 仿真时间配置 =================
        # use_sim_time 是 ROS2 内置参数，无需声明，直接读取
        use_sim_time = self.get_parameter('use_sim_time').value
        self.use_sim_time = bool(use_sim_time)
        self.get_logger().info(f'use_sim_time = {use_sim_time}')

        # ================= 1. 参数配置 =================
        # 官方赛段1办公区停车点，来自 /odom 实测。
        self.declare_parameter('goal_frame', 'odom')
        self.declare_parameter('goal_x', 5.107463422879308)
        self.declare_parameter('goal_y', -1.9848852519564644)
        self.declare_parameter('goal_yaw', -0.12103799852925523)

        # localization check params
        self.declare_parameter('localization_timeout', 30.0)
        self.declare_parameter('tf_warmup_sec', 5.0)
        self.declare_parameter('tf_query_timeout', 0.2)
        self.declare_parameter('allow_odom_fallback', True)

        # 底盘前进参数
        self.declare_parameter('forward_distance', 1.9)  # 前进距离 (米)
        self.declare_parameter('forward_speed', 0.15)    # 前进速度 (m/s)

        # 机械臂展开姿态
        self.declare_parameter('arm_open_duration', 3.0)  # 展开动作时间 (秒)

        # ================= 2. 关节定义 =================
        # 右臂关节
        self.joints_R = ['joint1_R', 'joint2_R', 'joint3_R',
                         'joint4_R', 'joint5_R', 'joint6_R']
        # 左臂关节
        self.joints_L = ['joint1_L', 'joint2_L', 'joint3_L',
                         'joint4_L', 'joint5_L', 'joint6_L']

        # 夹爪
        self.gripper_R_name = 'gripper_controller'
        self.gripper_L_name = 'gripper_left_controller'

        # 双臂展开姿态 (参考 open_arms.py)
        self.arm_open_pose = {
            # 右臂
            'joint1_R': 0.9, 'joint2_R': 0.0, 'joint3_R': 0.0,
            'joint4_R': -0.5, 'joint5_R': 0.0, 'joint6_R': 0.0,
            # 左臂
            'joint1_L': 0.9, 'joint2_L': -0.3, 'joint3_L': 0.0,
            'joint4_L': -0.5, 'joint5_L': 0.0, 'joint6_L': 0.0,
        }

        # ================= 3. 通信配置 =================
        # 机械臂控制 - 发布者
        self.pub_joint_cmd = self.create_publisher(JointState, '/joint_command', 10)
        self.pub_joint_left_cmd = self.create_publisher(JointState, '/joint_left_command', 10)

        # 底盘速度控制 - 发布者
        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 10)

        # 关节状态 - 订阅者
        self.sub_joint_states = self.create_subscription(
            JointState, '/joint_states', self.joint_states_callback, 10)
        self.sub_joint_left_states = self.create_subscription(
            JointState, '/joint_left_states', self.joint_left_states_callback, 10)

        # 里程计 - 订阅者 (用于前进距离估算)
        self.sub_odom = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)

        # TF 监听器 (用于定位检查)
        # 用于检查 map -> base_link 的 TF 变换，确认定位系统正常工作
        self.tf_buffer = Buffer(cache_time=Duration(seconds=30.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Nav2 Action Client
        self.nav_action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # ================= 4. 状态变量 =================
        # 关节状态
        self.current_joints = {}
        self.joint_data_ready = False

        # 里程计状态
        self.current_odom = None
        self.odom_start_pos = None

        self.get_logger().info('=' * 60)
        self.get_logger().info('NavToStart 节点初始化完成')
        self.get_logger().info('=' * 60)

    # ================= 回调函数 =================
    def joint_states_callback(self, msg):
        """主关节状态回调"""
        for i, name in enumerate(msg.name):
            self.current_joints[name] = msg.position[i]
        self.joint_data_ready = True

    def joint_left_states_callback(self, msg):
        """左夹爪状态回调"""
        for i, name in enumerate(msg.name):
            self.current_joints[name] = msg.position[i]

    def odom_callback(self, msg):
        """里程计回调"""
        self.current_odom = msg

    # ================= 辅助函数 =================
    def wait_and_spin(self, duration):
        """等待并保持ROS通信"""
        start = time.time()
        while time.time() - start < duration and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.01)
            time.sleep(0.01)

    def quaternion_to_yaw(self, q):
        """四元数转换为 yaw 角"""
        # yaw = atan2(2*(w*z + x*y), 1 - 2*(y^2 + z^2))
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def yaw_to_quaternion(self, yaw):
        """yaw 角转换为四元数"""
        from geometry_msgs.msg import Quaternion
        q = Quaternion()
        q.x = 0.0
        q.y = 0.0
        q.z = math.sin(yaw / 2.0)
        q.w = math.cos(yaw / 2.0)
        return q

    # ================= 定位检查 =================
    def wait_for_localization(self):
        """
        等待定位系统就绪
        检查 map -> base_link 的 TF 变换是否可用
        """
        self.get_logger().info('>>> 等待定位系统就绪...')

        timeout = self.get_parameter('localization_timeout').value
        tf_warmup = self.get_parameter('tf_warmup_sec').value
        allow_odom = self.get_parameter('allow_odom_fallback').value

        start_time = time.time()

        # TF 热启动
        self.get_logger().info(f'TF 热启动中 ({tf_warmup} 秒)...')
        while time.time() - start_time < tf_warmup and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)

        # 检查 map -> base_link 变换
        while time.time() - start_time < timeout and rclpy.ok():
            try:
                # 尝试获取 map -> base_link 变换
                transform = self.tf_buffer.lookup_transform(
                    'map', 'base_link', rclpy.time.Time())

                self.get_logger().info('✓ 定位系统就绪 (map -> base_link 可用)')
                return True

            except TransformException as ex:
                # 如果允许 odom 回退，检查 odom 是否可用
                if allow_odom:
                    try:
                        odom_transform = self.tf_buffer.lookup_transform(
                            'odom', 'base_link', rclpy.time.Time())
                        self.get_logger().warn('使用 odom 定位 (map 不可用)')
                        return True
                    except:
                        pass

                elapsed = time.time() - start_time
                self.get_logger().info(
                    f'等待定位... ({elapsed:.1f}/{timeout:.1f}s)',
                    throttle_duration_sec=2.0)
                rclpy.spin_once(self, timeout_sec=0.1)

        self.get_logger().error('定位系统超时！')
        return False

    def wait_for_nav2_active(self):
        """等待 Nav2 Action Server 就绪"""
        self.get_logger().info('>>> 等待 Nav2 Action Server...')

        timeout = 30.0
        start_time = time.time()

        while time.time() - start_time < timeout and rclpy.ok():
            if self.nav_action_client.wait_for_server(timeout_sec=1.0):
                self.get_logger().info('✓ Nav2 Action Server 就绪')
                return True

            elapsed = time.time() - start_time
            self.get_logger().info(
                f'等待 Nav2... ({elapsed:.1f}/{timeout:.1f}s)',
                throttle_duration_sec=2.0)

        self.get_logger().error('Nav2 Action Server 超时！')
        return False

    # ================= 导航功能 =================
    def navigate_to_goal(self, x, y, yaw):
        """
        使用 Nav2 导航到目标点

        Args:
            x: 目标 x 坐标
            y: 目标 y 坐标
            yaw: 目标朝向 (弧度)

        Returns:
            bool: 导航是否成功
        """
        goal_frame = str(self.get_parameter('goal_frame').value)
        self.get_logger().info(
            f'>>> 开始导航到目标点: frame={goal_frame}, x={x:.3f}, y={y:.3f}, yaw={yaw:.3f}'
        )

        # 构建目标姿态
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = goal_frame
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        goal_msg.pose.pose.position.z = 0.0
        goal_msg.pose.pose.orientation = self.yaw_to_quaternion(yaw)

        # 发送目标
        send_goal_future = self.nav_action_client.send_goal_async(goal_msg)

        # 等待目标被接受
        rclpy.spin_until_future_complete(self, send_goal_future, timeout_sec=5.0)

        if not send_goal_future.done():
            self.get_logger().error('发送导航目标超时！')
            return False

        goal_handle = send_goal_future.result()

        if not goal_handle.accepted:
            self.get_logger().error('导航目标被拒绝！')
            return False

        self.get_logger().info('导航目标已接受，等待完成...')

        # 等待导航完成
        result_future = goal_handle.get_result_async()

        # 监控导航进度
        while not result_future.done() and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)

            # 可以在这里添加进度显示
            try:
                transform = self.tf_buffer.lookup_transform(
                    'map', 'base_link', rclpy.time.Time())
                current_x = transform.transform.translation.x
                current_y = transform.transform.translation.y
                distance = math.sqrt((x - current_x)**2 + (y - current_y)**2)

                self.get_logger().info(
                    f'导航中... 距离目标: {distance:.2f}m',
                    throttle_duration_sec=2.0)
            except:
                pass

        if not result_future.done():
            self.get_logger().error('导航超时！')
            return False

        result = result_future.result()

        if result.status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('✓ 导航成功！')
            return True
        else:
            self.get_logger().error(f'导航失败！状态码: {result.status}')
            return False

    # ================= 机械臂控制 =================
    def smooth_move_arms(self, target_pose, duration=3.0):
        """
        平滑移动机械臂到目标姿态

        Args:
            target_pose: 目标关节角度字典
            duration: 运动时间 (秒)
        """
        self.get_logger().info('>>> 机械臂平滑运动中...')

        # 等待关节数据就绪
        timeout = 0
        while not self.joint_data_ready and timeout < 50:
            rclpy.spin_once(self, timeout_sec=0.1)
            timeout += 1

        if not self.joint_data_ready:
            self.get_logger().error('关节数据未就绪！')
            return False

        # 获取起始位置
        start_positions = {}
        for joint_name in target_pose.keys():
            if joint_name in self.current_joints:
                start_positions[joint_name] = self.current_joints[joint_name]
            else:
                self.get_logger().warn(f'关节 {joint_name} 数据未找到')
                start_positions[joint_name] = 0.0

        # 插值运动
        steps = int(duration * 50)  # 50Hz
        dt = duration / max(steps, 1)

        for i in range(steps + 1):
            alpha = i / steps

            # 准备消息
            msg_main = JointState()
            msg_main.header.stamp = self.get_clock().now().to_msg()
            msg_left = JointState()
            msg_left.header.stamp = self.get_clock().now().to_msg()

            # 计算插值位置
            for name, target in target_pose.items():
                start = start_positions[name]
                current = start + alpha * (target - start)

                if name == self.gripper_L_name:
                    msg_left.name.append(name)
                    msg_left.position.append(float(current))
                else:
                    msg_main.name.append(name)
                    msg_main.position.append(float(current))

            # 发布
            if msg_main.name:
                self.pub_joint_cmd.publish(msg_main)
            if msg_left.name:
                self.pub_joint_left_cmd.publish(msg_left)

            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(dt)

        self.get_logger().info('✓ 机械臂运动完成')
        return True

    def open_arms(self):
        """执行双臂展开动作"""
        self.get_logger().info('>>> 执行双臂展开动作...')
        duration = self.get_parameter('arm_open_duration').value
        return self.smooth_move_arms(self.arm_open_pose, duration)

    # ================= 底盘控制 =================
    def move_forward(self, distance, speed):
        """
        底盘直线前进指定距离（基于时间控制）

        Args:
            distance: 前进距离 (米)
            speed: 前进速度 (m/s)

        Returns:
            bool: 是否成功
        """
        self.get_logger().info(f'>>> 底盘前进: {distance}m @ {speed}m/s')

        # 计算需要的时间
        duration = distance / speed
        self.get_logger().info(f'预计时间: {duration:.2f}s')

        # 记录起始位置（用于显示进度）
        start_x = 0.0
        start_y = 0.0
        if self.current_odom is not None:
            start_x = self.current_odom.pose.pose.position.x
            start_y = self.current_odom.pose.pose.position.y

        # 发布速度指令
        twist = Twist()
        twist.linear.x = speed
        twist.angular.z = 0.0

        start_time = time.time()
        last_log_time = start_time

        while time.time() - start_time < duration and rclpy.ok():
            # 发布速度
            self.pub_cmd_vel.publish(twist)

            # 显示进度（每0.5秒一次）
            current_time = time.time()
            if current_time - last_log_time >= 0.5:
                elapsed = current_time - start_time
                progress = min(elapsed / duration, 1.0) * distance

                # 如果有里程计数据，显示实际距离
                if self.current_odom is not None:
                    current_x = self.current_odom.pose.pose.position.x
                    current_y = self.current_odom.pose.pose.position.y
                    actual_traveled = math.sqrt((current_x - start_x)**2 + (current_y - start_y)**2)
                    self.get_logger().info(
                        f'前进中... 预计: {progress:.2f}m, 实际: {actual_traveled:.2f}m / {distance:.2f}m')
                else:
                    self.get_logger().info(
                        f'前进中... {progress:.2f}/{distance:.2f}m')

                last_log_time = current_time

            rclpy.spin_once(self, timeout_sec=0.01)
            time.sleep(0.02)

        # 停止
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        for _ in range(5):
            self.pub_cmd_vel.publish(twist)
            time.sleep(0.02)

        # 计算实际行驶距离
        actual_traveled = 0.0
        if self.current_odom is not None:
            current_x = self.current_odom.pose.pose.position.x
            current_y = self.current_odom.pose.pose.position.y
            actual_traveled = math.sqrt((current_x - start_x)**2 + (current_y - start_y)**2)
            self.get_logger().info(f'✓ 前进完成，实际距离: {actual_traveled:.2f}m')
        else:
            self.get_logger().info(f'✓ 前进完成（基于时间: {duration:.2f}s）')

        return True

    # ================= 主流程 =================
    def run(self):
        """执行完整流程"""
        self.get_logger().info('=' * 60)
        self.get_logger().info('开始执行导航与动作准备流程')
        self.get_logger().info('=' * 60)

        # 1. 等待定位系统就绪
        if not self.wait_for_localization():
            self.get_logger().error('定位系统未就绪，退出')
            return False

        # 2. 等待 Nav2 就绪
        if not self.wait_for_nav2_active():
            self.get_logger().error('Nav2 未就绪，退出')
            return False

        # 3. 导航到目标点
        goal_x = self.get_parameter('goal_x').value
        goal_y = self.get_parameter('goal_y').value
        goal_yaw = self.get_parameter('goal_yaw').value

        if not self.navigate_to_goal(goal_x, goal_y, goal_yaw):
            self.get_logger().error('导航失败，退出')
            return False

        # 等待稳定
        self.get_logger().info('等待机器人稳定...')
        self.wait_and_spin(2.0)

        # 4. 执行双臂展开动作
        if not self.open_arms():
            self.get_logger().error('机械臂动作失败')
            return False

        # 等待动作完成
        self.wait_and_spin(1.0)

        # 5. 底盘前进接近目标
        forward_distance = self.get_parameter('forward_distance').value
        forward_speed = self.get_parameter('forward_speed').value

        if not self.move_forward(forward_distance, forward_speed):
            self.get_logger().error('底盘前进失败')
            return False

        # 完成
        self.get_logger().info('=' * 60)
        self.get_logger().info('✓ 所有任务完成！')
        self.get_logger().info('=' * 60)

        return True


def main(args=None):
    rclpy.init(args=args)
    node = NavToStartNode()

    try:
        success = node.run()
        if success:
            node.get_logger().info('程序正常结束')
        else:
            node.get_logger().error('程序异常结束')
    except KeyboardInterrupt:
        node.get_logger().info('用户中断')
    except Exception as e:
        node.get_logger().error(f'发生异常: {e}')
        import traceback
        traceback.print_exc()
    finally:
        # 停止机器人（在 destroy 之前）
        try:
            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            for _ in range(5):
                node.pub_cmd_vel.publish(twist)
                time.sleep(0.02)
        except:
            pass

        try:
            node.destroy_node()
        except:
            pass

        try:
            rclpy.shutdown()
        except:
            pass


if __name__ == '__main__':
    main()
