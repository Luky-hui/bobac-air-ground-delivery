#!/usr/bin/env python3
import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_dir = get_package_share_directory('grasp_demo_pkg')
    params = os.path.join(pkg_dir, 'config', 'demo_params.yaml')
    return LaunchDescription([
        Node(package='grasp_demo_pkg', executable='yoloe_detector_node', name='yoloe_detector_node', parameters=[params], output='screen'),
        Node(package='grasp_demo_pkg', executable='depth_pose_estimator_node', name='depth_pose_estimator_node', parameters=[params], output='screen'),
    ])
