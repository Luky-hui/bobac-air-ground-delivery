#!/usr/bin/env python3
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory('nav2_demo_pkg')
    amcl_params_file = os.path.join(pkg_dir, 'config', 'amcl_params.yaml')
    costmap_params_file = os.path.join(pkg_dir, 'config', 'costmap_params.yaml')
    planner_params_file = os.path.join(pkg_dir, 'config', 'planner_params.yaml')
    controller_params_file = os.path.join(pkg_dir, 'config', 'controller_params.yaml')
    behavior_params_file = os.path.join(pkg_dir, 'config', 'behavior_params.yaml')
    bt_params_file = os.path.join(pkg_dir, 'config', 'bt_navigator_params.yaml')
    default_map_file = os.path.join(pkg_dir, 'maps', 'map.yaml')
    rviz_config_file = os.path.join(pkg_dir, 'rviz', 'nav2_config.rviz')

    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    map_yaml_file = LaunchConfiguration('map', default=default_map_file)
    autostart = LaunchConfiguration('autostart', default='true')

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('map', default_value=default_map_file),
        DeclareLaunchArgument('autostart', default_value='true'),
        Node(
            package='nav2_map_server',
            executable='map_server',
            name='map_server',
            output='screen',
            parameters=[{'yaml_filename': map_yaml_file}, {'use_sim_time': use_sim_time}],
        ),
        Node(
            package='nav2_amcl',
            executable='amcl',
            name='amcl',
            output='screen',
            parameters=[amcl_params_file, {'use_sim_time': use_sim_time}],
        ),
        Node(
            package='nav2_planner',
            executable='planner_server',
            name='planner_server',
            output='screen',
            parameters=[planner_params_file, costmap_params_file, {'use_sim_time': use_sim_time}],
        ),
        Node(
            package='nav2_controller',
            executable='controller_server',
            name='controller_server',
            output='screen',
            parameters=[controller_params_file, costmap_params_file, {'use_sim_time': use_sim_time}],
        ),
        Node(
            package='nav2_behaviors',
            executable='behavior_server',
            name='behavior_server',
            output='screen',
            parameters=[behavior_params_file, {'use_sim_time': use_sim_time}],
        ),
        Node(
            package='nav2_bt_navigator',
            executable='bt_navigator',
            name='bt_navigator',
            output='screen',
            parameters=[bt_params_file, {'use_sim_time': use_sim_time}],
        ),
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_controller_test',
            output='screen',
            parameters=[
                {'use_sim_time': use_sim_time},
                {'autostart': autostart},
                {'node_names': ['map_server', 'amcl', 'planner_server', 'controller_server', 'behavior_server', 'bt_navigator']},
            ],
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config_file],
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen',
        ),
    ])
