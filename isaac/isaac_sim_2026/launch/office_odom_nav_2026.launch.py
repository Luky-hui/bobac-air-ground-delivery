#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("max_linear", default_value="1.20"),
            DeclareLaunchArgument("max_angular", default_value="1.60"),
            DeclareLaunchArgument("odom_topic", default_value="/odom"),
            DeclareLaunchArgument("cmd_vel_topic", default_value="/bobac_kinematic_cmd_vel"),
            DeclareLaunchArgument(
                "waypoints",
                default_value=(
                    "[1.760000, -1.100000, 1.661177, "
                    "3.436775, -0.727569, 1.661177, "
                    "3.970959, -0.368233, 1.577041]"
                ),
            ),
            Node(
                package="isaac_sim_2026",
                executable="bobac_sort_odom_nav_2026.py",
                name="office_odom_nav_2026",
                output="screen",
                parameters=[
                    {"use_sim_time": LaunchConfiguration("use_sim_time")},
                    {"odom_topic": LaunchConfiguration("odom_topic")},
                    {"cmd_vel_topic": LaunchConfiguration("cmd_vel_topic")},
                    {"waypoints": LaunchConfiguration("waypoints")},
                    {"max_linear": LaunchConfiguration("max_linear")},
                    {"max_angular": LaunchConfiguration("max_angular")},
                    {"world_frame_cmd": True},
                    {"forward_only_mode": False},
                    {"min_linear": 0.30},
                    {"fine_min_linear": 0.10},
                    {"min_angular": 0.36},
                    {"prestop_xy_tolerance": 0.14},
                    {"final_xy_tolerance": 0.055},
                    {"yaw_tolerance": 0.08},
                    {"front_stop_range": 0.34},
                    {"front_slow_range": 0.55},
                    {"goal_timeout_sec": 180.0},
                ],
            ),
        ]
    )
