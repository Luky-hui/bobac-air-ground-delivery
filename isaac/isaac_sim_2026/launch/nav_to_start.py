#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nav_to_start.py - robot navigation and arm prep.

Usage:
  python3 /home/u-zhuang/ros2_ws/src/isaac/isaac_sim_2026/launch/nav_to_start.py --ros-args -p use_sim_time:=true
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
from tf2_ros import Buffer, TransformListener


class NavToStartNode(Node):
    """?????????"""

    def __init__(self):
        super().__init__('nav_to_start_node')

        # ================= 0. ?????? =================
        use_sim_time = self.get_parameter('use_sim_time').value
        self.use_sim_time = bool(use_sim_time)
        self.get_logger().info(f'use_sim_time = {use_sim_time}')

        # ================= 1. ???? =================
        # Official stage-1 office parking point measured from /odom.
        self.declare_parameter('goal_frame', 'odom')
        self.declare_parameter('goal_x', 5.107463422879308)
        self.declare_parameter('goal_y', -1.9848852519564644)
        self.declare_parameter('goal_yaw', -0.12103799852925523)

        # localization check params
        self.declare_parameter('localization_timeout', 30.0)
        self.declare_parameter('tf_warmup_sec', 5.0)
        self.declare_parameter('tf_query_timeout', 0.2)
        self.declare_parameter('allow_odom_fallback', True)

        # ??????
        self.declare_parameter('forward_distance', 1.9)  # ???? (?)
        self.declare_parameter('forward_speed', 0.15)    # ???? (m/s)

        # ???????
        self.declare_parameter('arm_open_duration', 3.0)  # ?????? (?)

        # ================= 2. ???? =================
        self.joints_R = ['joint1_R', 'joint2_R', 'joint3_R',
                         'joint4_R', 'joint5_R', 'joint6_R']
        self.joints_L = ['joint1_L', 'joint2_L', 'joint3_L',
                         'joint4_L', 'joint5_L', 'joint6_L']

        self.gripper_R_name = 'gripper_controller'
        self.gripper_L_name = 'gripper_left_controller'

        self.arm_open_pose = {
            'joint1_R': 0.9, 'joint2_R': 0.0, 'joint3_R': 0.0,
            'joint4_R': -0.5, 'joint5_R': 0.0, 'joint6_R': 0.0,
            'joint1_L': 0.9, 'joint2_L': -0.3, 'joint3_L': 0.0,
            'joint4_L': -0.5, 'joint5_L': 0.0, 'joint6_L': 0.0,
        }

        # ================= 3. ???? =================
        self.pub_joint_cmd = self.create_publisher(JointState, '/joint_command', 10)
        self.pub_joint_left_cmd = self.create_publisher(JointState, '/joint_left_command', 10)
        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 10)

        self.sub_joint_states = self.create_subscription(
            JointState, '/joint_states', self.joint_states_callback, 10)
        self.sub_joint_left_states = self.create_subscription(
            JointState, '/joint_left_states', self.joint_left_states_callback, 10)
        self.sub_odom = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)

        # TF ??? (????????)
        self.tf_buffer = Buffer(cache_time=Duration(seconds=30.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ================= 4. ???? =================
        self.current_joints = {}
        self.joint_data_ready = False
        self.current_odom = None
        self._warned_map_missing = False

        self.get_logger().info('=' * 60)
        self.get_logger().info('????????????')
        self.get_logger().info('=' * 60)

    # ---------------------- ???? ----------------------
    def joint_states_callback(self, msg):
        """???????"""
        for i, name in enumerate(msg.name):
            self.current_joints[name] = msg.position[i]
        self.joint_data_ready = True

    def joint_left_states_callback(self, msg):
        """???????"""
        for i, name in enumerate(msg.name):
            self.current_joints[name] = msg.position[i]

    def odom_callback(self, msg):
        """?????"""
        self.current_odom = msg

    # ---------------------- ???? ----------------------
    def wait_and_spin(self, duration):
        """????? ROS ??"""
        start = time.time()
        while time.time() - start < duration:
            rclpy.spin_once(self, timeout_sec=0.01)
            time.sleep(0.01)

    def _wait_for_sim_time(self, timeout=5.0):
        """Wait until /clock starts when use_sim_time is enabled."""
        if not self.use_sim_time:
            return True
        start = time.time()
        while time.time() - start < timeout:
            if self.get_clock().now().nanoseconds > 0:
                return True
            rclpy.spin_once(self, timeout_sec=0.05)
            time.sleep(0.05)
        self.get_logger().warn('sim time not active yet; continuing anyway')
        return False

    def stop_robot(self):
        """????"""
        msg = Twist()
        msg.linear.x = 0.0
        msg.angular.z = 0.0
        for _ in range(5):
            self.pub_cmd_vel.publish(msg)
            time.sleep(0.02)

    def check_localization(self, timeout=15.0):
        """???????????"""
        self.get_logger().info('?????????...')

        tf_warmup = float(self.get_parameter('tf_warmup_sec').value)
        tf_query_timeout = float(self.get_parameter('tf_query_timeout').value)
        allow_odom_fallback = bool(self.get_parameter('allow_odom_fallback').value)

        self._wait_for_sim_time(timeout=5.0)

        self.get_logger().info(f'TF warmup ({tf_warmup:.1f}s)...')
        warmup_start = time.time()
        while time.time() - warmup_start < tf_warmup:
            rclpy.spin_once(self, timeout_sec=0.05)
            time.sleep(0.05)

        start_time = time.time()
        while time.time() - start_time < timeout:
            for _ in range(20):
                rclpy.spin_once(self, timeout_sec=0.05)

            try:
                if self.tf_buffer.can_transform(
                    'map', 'base_link', rclpy.time.Time(),
                    timeout=Duration(seconds=tf_query_timeout)
                ):
                    transform = self.tf_buffer.lookup_transform(
                        'map', 'base_link',
                        rclpy.time.Time(),
                        timeout=Duration(seconds=tf_query_timeout))

                    x = transform.transform.translation.x
                    y = transform.transform.translation.y

                    if abs(x) < 1000 and abs(y) < 1000:
                        self.get_logger().info(f'????: x={x:.3f}, y={y:.3f}')
                        return True
                    self.get_logger().warn(f'????: x={x:.3f}, y={y:.3f}')
                else:
                    if (not self._warned_map_missing and
                            self.tf_buffer.can_transform(
                                'odom', 'base_link', rclpy.time.Time(),
                                timeout=Duration(seconds=tf_query_timeout))):
                        self.get_logger().warn('map->base_link missing but odom->base_link is available')
                        self._warned_map_missing = True

                    if allow_odom_fallback:
                        if self.tf_buffer.can_transform(
                            'odom', 'base_link', rclpy.time.Time(),
                            timeout=Duration(seconds=tf_query_timeout)):
                            transform = self.tf_buffer.lookup_transform(
                                'odom', 'base_link',
                                rclpy.time.Time(),
                                timeout=Duration(seconds=tf_query_timeout))
                            x = transform.transform.translation.x
                            y = transform.transform.translation.y
                            if abs(x) < 1000 and abs(y) < 1000:
                                self.get_logger().warn('using odom->base_link as localization fallback')
                                return True

                    elapsed = time.time() - start_time
                    self.get_logger().info(
                        f'?? TF ??... ({elapsed:.1f}s) - ??????',
                        throttle_duration_sec=2.0)

            except Exception as e:
                elapsed = time.time() - start_time
                self.get_logger().info(
                    f'?? TF ??... ({elapsed:.1f}s) - {type(e).__name__}: {str(e)}',
                    throttle_duration_sec=2.0)

            time.sleep(0.5)

        self.get_logger().error('???????? RTAB-Map ??????')
        self.get_logger().error('??????? "ros2 run tf2_ros tf2_echo map base_link" ?? TF')
        return False

    # ---------------------- ????? ----------------------
    def smooth_move_joints(self, joint_targets, duration=2.0):
        """?????????"""
        timeout = 5.0
        start_wait = time.time()
        while not self.joint_data_ready and time.time() - start_wait < timeout:
            self.get_logger().info('????????...', throttle_duration_sec=1.0)
            rclpy.spin_once(self, timeout_sec=0.1)

        if not self.joint_data_ready:
            self.get_logger().error('???????????????')
            return False

        start_positions = {}
        for joint_name in joint_targets.keys():
            if joint_name in self.current_joints:
                start_positions[joint_name] = self.current_joints[joint_name]
            else:
                self.get_logger().warn(f'?? {joint_name} ???????????')
                start_positions[joint_name] = joint_targets[joint_name]

        steps = int(duration * 60)
        dt = duration / max(steps, 1)

        self.get_logger().info(f'???? {len(joint_targets)} ??? (?? {duration:.1f}s)')

        for i in range(steps + 1):
            alpha = i / steps
            msg_main = JointState()
            msg_main.header.stamp = self.get_clock().now().to_msg()
            msg_left = JointState()
            msg_left.header.stamp = self.get_clock().now().to_msg()

            for name, target in joint_targets.items():
                start = start_positions.get(name, target)
                curr = start + alpha * (target - start)
                if name == self.gripper_L_name:
                    msg_left.name.append(name)
                    msg_left.position.append(float(curr))
                else:
                    msg_main.name.append(name)
                    msg_main.position.append(float(curr))

            if msg_main.name:
                self.pub_joint_cmd.publish(msg_main)
            if msg_left.name:
                self.pub_joint_left_cmd.publish(msg_left)

            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(dt)

        self.get_logger().info('??????')
        return True

    def open_arms(self):
        """????????"""
        self.get_logger().info('')
        self.get_logger().info('=' * 60)
        self.get_logger().info('[????] ?????????')
        self.get_logger().info('=' * 60)

        duration = float(self.get_parameter('arm_open_duration').value)
        success = self.smooth_move_joints(self.arm_open_pose, duration=duration)

        if success:
            self.get_logger().info('?????')
        else:
            self.get_logger().warn('?????????')

        self.wait_and_spin(0.5)
        return success

    # ---------------------- ???? ----------------------
    def move_forward(self, distance, speed):
        """??????"""
        self.get_logger().info('')
        self.get_logger().info('=' * 60)
        self.get_logger().info(f'[????] ?????? {distance:.2f}m @ {speed:.2f}m/s')
        self.get_logger().info('=' * 60)

        duration = distance / speed
        self.get_logger().info(f'????: {duration:.2f}s')

        msg = Twist()
        msg.linear.x = speed
        msg.angular.z = 0.0

        start_time = time.time()
        while time.time() - start_time < duration:
            self.pub_cmd_vel.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.01)
            time.sleep(0.05)

        self.stop_robot()
        self.get_logger().info('??????')
        self.wait_and_spin(0.5)

    # ---------------------- ???? ----------------------
    def navigate_to_goal(self):
        """?? Nav2 ??????"""
        self.get_logger().info('')
        self.get_logger().info('=' * 60)
        self.get_logger().info('[????] ?? Nav2 ??????')
        self.get_logger().info('=' * 60)

        goal_x = float(self.get_parameter('goal_x').value)
        goal_y = float(self.get_parameter('goal_y').value)
        goal_yaw = float(self.get_parameter('goal_yaw').value)
        goal_frame = str(self.get_parameter('goal_frame').value)

        qz = math.sin(goal_yaw / 2.0)
        qw = math.cos(goal_yaw / 2.0)

        self.get_logger().info(f'???? ({goal_frame} frame):')
        self.get_logger().info(f'  x = {goal_x:.3f}')
        self.get_logger().info(f'  y = {goal_y:.3f}')
        self.get_logger().info(f'  yaw = {goal_yaw:.3f} rad ({math.degrees(goal_yaw):.1f} deg)')
        self.get_logger().info(f'  ??? z={qz:.3f}, w={qw:.3f}')

        nav_action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        self.get_logger().info('?? Nav2 navigate_to_pose Action Server...')
        if not nav_action_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error('Nav2 Action Server ??????? Nav2 ???')
            return False
        self.get_logger().info('Nav2 Action Server ???')

        goal_pose = PoseStamped()
        goal_pose.header.frame_id = goal_frame
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_pose.pose.position.x = goal_x
        goal_pose.pose.position.y = goal_y
        goal_pose.pose.position.z = 0.0
        goal_pose.pose.orientation.x = 0.0
        goal_pose.pose.orientation.y = 0.0
        goal_pose.pose.orientation.z = qz
        goal_pose.pose.orientation.w = qw

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = goal_pose

        self.get_logger().info('??????...')
        send_goal_future = nav_action_client.send_goal_async(
            goal_msg,
            feedback_callback=self._nav_feedback_callback
        )

        rclpy.spin_until_future_complete(self, send_goal_future, timeout_sec=5.0)
        goal_handle = send_goal_future.result()

        if not goal_handle or not goal_handle.accepted:
            self.get_logger().error('???????')
            return False

        self.get_logger().info('????????????...')
        result_future = goal_handle.get_result_async()

        while not result_future.done():
            rclpy.spin_once(self, timeout_sec=0.1)

        result = result_future.result()
        status = result.status

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('?????????')
            return True
        if status == GoalStatus.STATUS_CANCELED:
            self.get_logger().warn('?????')
            return False
        if status == GoalStatus.STATUS_ABORTED:
            self.get_logger().error('?????????')
            return False

        self.get_logger().warn(f'?????? {status}')
        return False

    def _nav_feedback_callback(self, feedback_msg):
        """??????"""
        feedback = feedback_msg.feedback
        if hasattr(feedback, 'distance_remaining'):
            distance = feedback.distance_remaining
            self.get_logger().info(
                f'???... ????: {distance:.2f}m',
                throttle_duration_sec=1.0)

    # ---------------------- ??? ----------------------
    def run(self):
        """??????"""
        self.get_logger().info('')
        self.get_logger().info('#' * 60)
        self.get_logger().info('#  ????????????')
        self.get_logger().info('#' * 60)
        self.get_logger().info('')

        self.get_logger().info('=' * 60)
        self.get_logger().info('[????] ???????')
        self.get_logger().info('=' * 60)

        localization_timeout = float(self.get_parameter('localization_timeout').value)
        if not self.check_localization(timeout=localization_timeout):
            self.get_logger().error('???????????')
            self.get_logger().error('???:')
            self.get_logger().error('  1. Isaac Sim ????????')
            self.get_logger().error('  2. RTAB-Map ???: ros2 launch isaac_sim rtab-map-scan.launch.py')
            self.get_logger().error('  3. ?????????')
            return False

        nav_success = self.navigate_to_goal()
        if not nav_success:
            self.get_logger().error('?????????')
            return False

        self.get_logger().info('?????????...')
        self.wait_and_spin(1.0)

        self.open_arms()

        forward_distance = float(self.get_parameter('forward_distance').value)
        forward_speed = float(self.get_parameter('forward_speed').value)
        self.move_forward(forward_distance, forward_speed)

        self.get_logger().info('')
        self.get_logger().info('#' * 60)
        self.get_logger().info('#  ??????')
        self.get_logger().info('#' * 60)
        self.get_logger().info('')
        self.get_logger().info('????:')
        self.get_logger().info('  1. Nav2 ?????? - ??')
        self.get_logger().info('  2. ???????? - ??')
        self.get_logger().info('  3. ???????? - ??')
        self.get_logger().info('')
        self.get_logger().info('???????????????')
        return True


def main():
    """???"""
    rclpy.init()
    node = NavToStartNode()

    try:
        node.run()
    except KeyboardInterrupt:
        node.get_logger().info('????')
    except Exception as e:
        node.get_logger().error(f'????: {e}')
        import traceback
        traceback.print_exc()
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
