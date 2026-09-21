#!/usr/bin/env python3
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="isaac_sim_2026",
                executable="pointcloud_to_laserscan_2026.py",
                name="pointcloud_to_laserscan_2026",
                output="screen",
                parameters=[{"use_sim_time": True}],
            )
        ]
    )
