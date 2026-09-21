#!/usr/bin/env python3
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    pkg_dir = get_package_share_directory("isaac_sim_2026")

    rtabmap_launch = os.path.join(pkg_dir, "launch", "rtab-map-scan.launch.py")
    nav2_launch = os.path.join(pkg_dir, "launch", "navigation2.launch.py")
    office_nav_launch = os.path.join(pkg_dir, "launch", "office_waypoint_nav_2026.launch.py")

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("launch_viz", default_value="false"),
            DeclareLaunchArgument("delete_db_on_start", default_value="true"),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(rtabmap_launch),
                launch_arguments={
                    "launch_viz": LaunchConfiguration("launch_viz"),
                    "delete_db_on_start": LaunchConfiguration("delete_db_on_start"),
                }.items(),
            ),
            TimerAction(
                period=8.0,
                actions=[
                    IncludeLaunchDescription(
                        PythonLaunchDescriptionSource(nav2_launch),
                        launch_arguments={
                            "use_sim_time": LaunchConfiguration("use_sim_time"),
                        }.items(),
                    )
                ],
            ),
            TimerAction(
                period=45.0,
                actions=[
                    IncludeLaunchDescription(
                        PythonLaunchDescriptionSource(office_nav_launch),
                        launch_arguments={
                            "use_sim_time": LaunchConfiguration("use_sim_time"),
                            "goal_timeout_sec": "240.0",
                        }.items(),
                    )
                ],
            ),
        ]
    )
