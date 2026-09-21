#!/usr/bin/env python3
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory("isaac_sim_2026")
    nav2_pkg = get_package_share_directory("nav2_demo_pkg")
    use_sim_time = LaunchConfiguration("use_sim_time")
    cloud_topic = LaunchConfiguration("cloud_topic")
    scan_topic = LaunchConfiguration("scan_topic")
    secondary_scan_topic = LaunchConfiguration("secondary_scan_topic")
    target_frame = LaunchConfiguration("target_frame")
    override_params = os.path.join(pkg_dir, "config", "nav2_2026_overrides.yaml")

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("cloud_topic", default_value="/avoidance/lidar/pointcloud"),
            DeclareLaunchArgument("scan_topic", default_value="/laser_scan_fuse"),
            DeclareLaunchArgument("secondary_scan_topic", default_value="/scan"),
            DeclareLaunchArgument("target_frame", default_value="fuse_lidar_link"),
            Node(
                package="isaac_sim_2026",
                executable="pointcloud_to_laserscan_2026.py",
                name="pointcloud_to_laserscan_2026",
                output="screen",
                parameters=[
                    {"use_sim_time": use_sim_time},
                    {"cloud_topic": cloud_topic},
                    {"scan_topic": scan_topic},
                    {"secondary_scan_topic": secondary_scan_topic},
                    {"target_frame": target_frame},
                ],
            ),
            Node(
                package="slam_toolbox",
                executable="async_slam_toolbox_node",
                name="slam_toolbox",
                output="screen",
                parameters=[
                    os.path.join(nav2_pkg, "config", "slam_toolbox_params.yaml"),
                    override_params,
                    {"use_sim_time": use_sim_time},
                ],
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                arguments=["-d", os.path.join(nav2_pkg, "rviz", "slam_config.rviz")],
                parameters=[{"use_sim_time": use_sim_time}],
                output="screen",
            ),
        ]
    )
