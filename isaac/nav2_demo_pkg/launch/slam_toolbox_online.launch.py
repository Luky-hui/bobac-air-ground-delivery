#!/usr/bin/env python3
"""
SLAM Toolbox 在线建图 Launch 文件
功能：使用 slam_toolbox 进行实时 SLAM 建图
用法：ros2 launch nav2_demo_pkg slam_toolbox_online.launch.py
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # 获取包路径
    pkg_dir = get_package_share_directory('nav2_demo_pkg')

    # 配置文件路径
    slam_params_file = os.path.join(pkg_dir, 'config', 'slam_toolbox_params.yaml')

    # 声明启动参数
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='使用仿真时间'
    )

    # SLAM Toolbox 在线建图节点
    slam_toolbox_node = Node(
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        output='screen',
        parameters=[
            slam_params_file,
            {'use_sim_time': use_sim_time}
        ]
    )

    # RViz2 可视化节点
    rviz_config_file = os.path.join(pkg_dir, 'rviz', 'slam_config.rviz')
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_file],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    return LaunchDescription([
        declare_use_sim_time_cmd,
        slam_toolbox_node,
        rviz_node
    ])
