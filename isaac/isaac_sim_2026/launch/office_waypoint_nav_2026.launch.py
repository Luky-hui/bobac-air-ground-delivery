#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("frame_id", default_value="odom"),
            DeclareLaunchArgument("goal_timeout_sec", default_value="180.0"),
            DeclareLaunchArgument("nav_handoff_xy_tolerance", default_value="0.45"),
            DeclareLaunchArgument("final_nav_handoff_xy_tolerance", default_value="1.20"),
            DeclareLaunchArgument("final_failure_handoff_xy_tolerance", default_value="1.30"),
            DeclareLaunchArgument("fine_dock_enabled", default_value="true"),
            DeclareLaunchArgument("fine_dock_timeout_sec", default_value="180.0"),
            DeclareLaunchArgument("fine_dock_xy_tolerance", default_value="0.10"),
            DeclareLaunchArgument("fine_dock_yaw_tolerance", default_value="0.15"),
            DeclareLaunchArgument("fine_dock_max_linear", default_value="0.055"),
            DeclareLaunchArgument("fine_dock_max_angular", default_value="0.18"),
            Node(
                package="isaac_sim_2026",
                executable="office_waypoint_nav_2026.py",
                name="office_waypoint_nav_2026",
                output="screen",
                parameters=[
                    {"use_sim_time": LaunchConfiguration("use_sim_time")},
                    {"frame_id": LaunchConfiguration("frame_id")},
                    {"goal_timeout_sec": LaunchConfiguration("goal_timeout_sec")},
                    {"nav_handoff_xy_tolerance": LaunchConfiguration("nav_handoff_xy_tolerance")},
                    {
                        "final_nav_handoff_xy_tolerance": LaunchConfiguration(
                            "final_nav_handoff_xy_tolerance"
                        )
                    },
                    {
                        "final_failure_handoff_xy_tolerance": LaunchConfiguration(
                            "final_failure_handoff_xy_tolerance"
                        )
                    },
                    {"fine_dock_enabled": LaunchConfiguration("fine_dock_enabled")},
                    {"fine_dock_timeout_sec": LaunchConfiguration("fine_dock_timeout_sec")},
                    {"fine_dock_xy_tolerance": LaunchConfiguration("fine_dock_xy_tolerance")},
                    {"fine_dock_yaw_tolerance": LaunchConfiguration("fine_dock_yaw_tolerance")},
                    {"fine_dock_max_linear": LaunchConfiguration("fine_dock_max_linear")},
                    {"fine_dock_max_angular": LaunchConfiguration("fine_dock_max_angular")},
                ],
            ),
        ]
    )
