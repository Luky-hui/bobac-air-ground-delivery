#!/usr/bin/env python3
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory("isaac_sim_2026")
    params = os.path.join(pkg_dir, "config", "stage1_2026.yaml")

    start_nav = LaunchConfiguration("start_nav")
    start_sensors = LaunchConfiguration("start_sensors")
    start_perception = LaunchConfiguration("start_perception")
    start_manager = LaunchConfiguration("start_manager")
    skip_navigation = LaunchConfiguration("skip_navigation")

    nav2_pkg = get_package_share_directory("nav2_demo_pkg")
    nav2_map = os.path.join(nav2_pkg, "maps", "map.yaml")
    nav2_overrides = os.path.join(pkg_dir, "config", "nav2_2026_overrides.yaml")

    grasp_pkg = get_package_share_directory("grasp_demo_pkg")
    grasp_params = os.path.join(grasp_pkg, "config", "demo_params.yaml")

    return LaunchDescription(
        [
            DeclareLaunchArgument("start_nav", default_value="false"),
            DeclareLaunchArgument("start_sensors", default_value="true"),
            DeclareLaunchArgument("start_perception", default_value="true"),
            DeclareLaunchArgument("start_manager", default_value="true"),
            DeclareLaunchArgument("skip_navigation", default_value="false"),
            Node(
                condition=IfCondition(start_sensors),
                package="isaac_sim_2026",
                executable="pointcloud_to_laserscan_2026.py",
                name="pointcloud_to_laserscan_2026",
                output="screen",
                parameters=[{"use_sim_time": True}],
            ),
            Node(
                condition=IfCondition(start_nav),
                package="nav2_map_server",
                executable="map_server",
                name="map_server",
                output="screen",
                parameters=[{"yaml_filename": nav2_map}, {"use_sim_time": True}],
            ),
            Node(
                condition=IfCondition(start_nav),
                package="nav2_amcl",
                executable="amcl",
                name="amcl",
                output="screen",
                parameters=[
                    os.path.join(nav2_pkg, "config", "amcl_params.yaml"),
                    nav2_overrides,
                    {"use_sim_time": True},
                ],
            ),
            Node(
                condition=IfCondition(start_nav),
                package="nav2_planner",
                executable="planner_server",
                name="planner_server",
                output="screen",
                parameters=[
                    os.path.join(nav2_pkg, "config", "planner_params.yaml"),
                    os.path.join(nav2_pkg, "config", "costmap_params.yaml"),
                    nav2_overrides,
                    {"use_sim_time": True},
                ],
            ),
            Node(
                condition=IfCondition(start_nav),
                package="nav2_controller",
                executable="controller_server",
                name="controller_server",
                output="screen",
                parameters=[
                    os.path.join(nav2_pkg, "config", "controller_params.yaml"),
                    os.path.join(nav2_pkg, "config", "costmap_params.yaml"),
                    nav2_overrides,
                    {"use_sim_time": True},
                ],
            ),
            Node(
                condition=IfCondition(start_nav),
                package="nav2_behaviors",
                executable="behavior_server",
                name="behavior_server",
                output="screen",
                parameters=[
                    os.path.join(nav2_pkg, "config", "behavior_params.yaml"),
                    nav2_overrides,
                    {"use_sim_time": True},
                ],
            ),
            Node(
                condition=IfCondition(start_nav),
                package="nav2_bt_navigator",
                executable="bt_navigator",
                name="bt_navigator",
                output="screen",
                parameters=[
                    os.path.join(nav2_pkg, "config", "bt_navigator_params.yaml"),
                    nav2_overrides,
                    {"use_sim_time": True},
                ],
            ),
            Node(
                condition=IfCondition(start_nav),
                package="nav2_lifecycle_manager",
                executable="lifecycle_manager",
                name="lifecycle_manager_navigation_stage1",
                output="screen",
                parameters=[
                    {"use_sim_time": True},
                    {"autostart": True},
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
            Node(
                condition=IfCondition(start_perception),
                package="grasp_demo_pkg",
                executable="yoloe_detector_node",
                name="yoloe_detector_node",
                parameters=[grasp_params, {"auto_detect": False}],
                output="screen",
            ),
            Node(
                condition=IfCondition(start_perception),
                package="grasp_demo_pkg",
                executable="depth_pose_estimator_node",
                name="depth_pose_estimator_node",
                parameters=[grasp_params],
                output="screen",
            ),
            Node(
                condition=IfCondition(start_perception),
                package="grasp_demo_pkg",
                executable="tf_transform_demo_node",
                name="tf_transform_demo_node",
                parameters=[grasp_params],
                output="screen",
            ),
            Node(
                condition=IfCondition(start_perception),
                package="grasp_demo_pkg",
                executable="plan_to_pose_node",
                name="plan_to_pose_node",
                parameters=[grasp_params],
                output="screen",
            ),
            Node(
                condition=IfCondition(start_perception),
                package="grasp_demo_pkg",
                executable="gripper_demo_node",
                name="gripper_demo_node",
                parameters=[grasp_params],
                output="screen",
            ),
            TimerAction(
                period=45.0,
                actions=[
                    Node(
                        condition=IfCondition(start_manager),
                        package="isaac_sim_2026",
                        executable="stage1_task_manager.py",
                        name="stage1_task_manager",
                        parameters=[params, {"skip_navigation": skip_navigation}],
                        output="screen",
                    )
                ],
            ),
        ]
    )
