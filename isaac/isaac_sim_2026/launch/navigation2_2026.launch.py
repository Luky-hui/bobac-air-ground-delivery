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
    default_map = os.path.join(nav2_pkg, "maps", "map.yaml")
    override_params = os.path.join(pkg_dir, "config", "nav2_2026_overrides.yaml")
    use_sim_time = LaunchConfiguration("use_sim_time")
    map_file = LaunchConfiguration("map")
    autostart = LaunchConfiguration("autostart")
    cloud_topic = LaunchConfiguration("cloud_topic")
    scan_topic = LaunchConfiguration("scan_topic")
    secondary_scan_topic = LaunchConfiguration("secondary_scan_topic")
    target_frame = LaunchConfiguration("target_frame")

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("map", default_value=default_map),
            DeclareLaunchArgument("autostart", default_value="true"),
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
                package="nav2_map_server",
                executable="map_server",
                name="map_server",
                output="screen",
                parameters=[{"yaml_filename": map_file}, {"use_sim_time": use_sim_time}],
            ),
            Node(
                package="nav2_amcl",
                executable="amcl",
                name="amcl",
                output="screen",
                parameters=[
                    os.path.join(nav2_pkg, "config", "amcl_params.yaml"),
                    override_params,
                    {"use_sim_time": use_sim_time},
                ],
            ),
            Node(
                package="nav2_planner",
                executable="planner_server",
                name="planner_server",
                output="screen",
                parameters=[
                    os.path.join(nav2_pkg, "config", "planner_params.yaml"),
                    os.path.join(nav2_pkg, "config", "costmap_params.yaml"),
                    override_params,
                    {"use_sim_time": use_sim_time},
                ],
            ),
            Node(
                package="nav2_controller",
                executable="controller_server",
                name="controller_server",
                output="screen",
                parameters=[
                    os.path.join(nav2_pkg, "config", "controller_params.yaml"),
                    os.path.join(nav2_pkg, "config", "costmap_params.yaml"),
                    override_params,
                    {"use_sim_time": use_sim_time},
                ],
            ),
            Node(
                package="nav2_behaviors",
                executable="behavior_server",
                name="behavior_server",
                output="screen",
                parameters=[
                    os.path.join(nav2_pkg, "config", "behavior_params.yaml"),
                    override_params,
                    {"use_sim_time": use_sim_time},
                ],
            ),
            Node(
                package="nav2_bt_navigator",
                executable="bt_navigator",
                name="bt_navigator",
                output="screen",
                parameters=[
                    os.path.join(nav2_pkg, "config", "bt_navigator_params.yaml"),
                    override_params,
                    {"use_sim_time": use_sim_time},
                ],
            ),
            Node(
                package="nav2_lifecycle_manager",
                executable="lifecycle_manager",
                name="lifecycle_manager_navigation_2026",
                output="screen",
                parameters=[
                    {"use_sim_time": use_sim_time},
                    {"autostart": autostart},
                    {
                        "node_names": [
                            "map_server",
                            "amcl",
                            "planner_server",
                            "controller_server",
                            "behavior_server",
                            "bt_navigator",
                        ]
                    },
                ],
            ),
        ]
    )
