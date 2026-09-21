#!/usr/bin/env python3
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory("isaac_sim_2026")
    nav2_pkg = get_package_share_directory("nav2_demo_pkg")
    params = os.path.join(pkg_dir, "config", "nav2_official_x1_2026.yaml")

    use_sim_time = LaunchConfiguration("use_sim_time")
    autostart = LaunchConfiguration("autostart")
    ros_domain_id = LaunchConfiguration("ros_domain_id")
    stow_arms = LaunchConfiguration("stow_arms")

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("autostart", default_value="true"),
            DeclareLaunchArgument(
                "ros_domain_id",
                default_value="55",
                description="Official Mercury X1 ActionGraph ROS domain",
            ),
            DeclareLaunchArgument(
                "stow_arms",
                default_value="true",
                description="Move both arms into a compact navigation posture before Nav2 starts",
            ),
            SetEnvironmentVariable("ROS_DOMAIN_ID", ros_domain_id),
            Node(
                condition=IfCondition(stow_arms),
                package="isaac_sim_2026",
                executable="stow_arms_for_navigation.py",
                name="stow_arms_for_navigation",
                output="screen",
                parameters=[{"use_sim_time": use_sim_time}],
            ),
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="official_x1_lidar_tf",
                output="screen",
                arguments=[
                    "0.10",
                    "0.0",
                    "0.15",
                    "0.0",
                    "0.0",
                    "0.0",
                    "1.0",
                    "base_link",
                    "lidar",
                ],
            ),
            Node(
                package="isaac_sim_2026",
                executable="laser_scan_republisher_2026.py",
                name="laser_scan_republisher_2026",
                output="screen",
                parameters=[
                    {"use_sim_time": use_sim_time},
                    {"input_topic": "/laser_scan"},
                    {"output_topic": "/laser_scan_nav"},
                    {"cloud_topic": "/laser_obstacles_cloud"},
                    {"publish_cloud": True},
                    {"target_frame": "lidar"},
                    {"restamp_with_now": True},
                    {"replace_negative_with_inf": True},
                ],
            ),
            TimerAction(
                period=8.0,
                actions=[
                    Node(
                        package="nav2_planner",
                        executable="planner_server",
                        name="planner_server",
                        output="screen",
                        parameters=[params, {"use_sim_time": use_sim_time}],
                    ),
                    Node(
                        package="nav2_controller",
                        executable="controller_server",
                        name="controller_server",
                        output="screen",
                        parameters=[params, {"use_sim_time": use_sim_time}],
                    ),
                    Node(
                        package="nav2_behaviors",
                        executable="behavior_server",
                        name="behavior_server",
                        output="screen",
                        parameters=[params, {"use_sim_time": use_sim_time}],
                    ),
                    Node(
                        package="nav2_bt_navigator",
                        executable="bt_navigator",
                        name="bt_navigator",
                        output="screen",
                        parameters=[
                            os.path.join(nav2_pkg, "config", "bt_navigator_params.yaml"),
                            params,
                            {"use_sim_time": use_sim_time},
                        ],
                    ),
                    Node(
                        package="nav2_lifecycle_manager",
                        executable="lifecycle_manager",
                        name="lifecycle_manager_navigation_official_x1",
                        output="screen",
                        parameters=[
                            {"use_sim_time": use_sim_time},
                            {"autostart": autostart},
                            {
                                "node_names": [
                                    "planner_server",
                                    "controller_server",
                                    "behavior_server",
                                    "bt_navigator",
                                ]
                            },
                        ],
                    ),
                ],
            ),
        ]
    )
