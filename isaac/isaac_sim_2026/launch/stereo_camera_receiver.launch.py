#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stereo_camera_receiver.launch.py

双目相机数据接收 Launch 文件

功能：
启动双目相机数据接收节点，订阅 Isaac Sim 发布的图像数据并在终端打印统计信息

使用方式：
ros2 launch isaac_sim stereo_camera_receiver.launch.py

或使用 Isaac Sim Python 环境：
cd ~/ros2_ws/src/isaacsim2026/isaac_sim_2026
~/isaac-sim-4.5.0/python.sh launch/stereo_camera_demo.py --ros-args -p use_sim_time:=true

验证命令：
ros2 topic list | grep camera
ros2 topic hz /camera/left/image_raw
ros2 node list | grep stereo
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """生成 Launch 描述"""

    # ---- Launch 参数 ----
    arg_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time from Isaac Sim'
    )

    arg_print_interval = DeclareLaunchArgument(
        'print_interval',
        default_value='2.0',
        description='Statistics print interval in seconds'
    )

    arg_show_statistics = DeclareLaunchArgument(
        'show_statistics',
        default_value='true',
        description='Show detailed image statistics (requires cv_bridge)'
    )

    # ---- 双目相机接收节点 ----
    stereo_receiver_node = Node(
        package='isaac_sim',
        executable='stereo_camera_demo.py',
        name='stereo_camera_receiver',
        output='screen',
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
            {'print_interval': LaunchConfiguration('print_interval')},
            {'show_statistics': LaunchConfiguration('show_statistics')},
            {'left_image_topic': '/camera/left/image_raw'},
            {'right_image_topic': '/camera/right/image_raw'},
            {'left_info_topic': '/camera/left/camera_info'},
            {'right_info_topic': '/camera/right/camera_info'},
        ],
        emulate_tty=True,
    )

    return LaunchDescription([
        arg_use_sim_time,
        arg_print_interval,
        arg_show_statistics,
        stereo_receiver_node,
    ])
