#!/usr/bin/env python3
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory('nav2_demo_pkg')
    costmap_params_file = os.path.join(pkg_dir, 'config', 'costmap_params.yaml')
    amcl_params_file = os.path.join(pkg_dir, 'config', 'amcl_params.yaml')
    default_map_file = os.path.join(pkg_dir, 'maps', 'map.yaml')
    rviz_config_file = os.path.join(pkg_dir, 'rviz', 'costmap_config.rviz')

    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    map_yaml_file = LaunchConfiguration('map', default=default_map_file)
    autostart = LaunchConfiguration('autostart', default='true')

    # The planner server owns the global costmap. We provide only the minimal
    # inline planner plugin config needed for the node to configure; all costmap
    # parameters still come from costmap_params.yaml.
    minimal_planner_params = {
        'use_sim_time': use_sim_time,
        'expected_planner_frequency': 20.0,
        'planner_plugins': ['GridBased'],
        'GridBased.plugin': 'nav2_navfn_planner/NavfnPlanner',
        'GridBased.tolerance': 0.5,
        'GridBased.use_astar': True,
        'GridBased.allow_unknown': True,
    }

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('map', default_value=default_map_file),
        DeclareLaunchArgument('autostart', default_value='true'),
        Node(
            package='nav2_map_server',
            executable='map_server',
            name='map_server',
            output='screen',
            parameters=[
                {'yaml_filename': map_yaml_file},
                {'use_sim_time': use_sim_time},
            ],
        ),
        Node(
            package='nav2_amcl',
            executable='amcl',
            name='amcl',
            output='screen',
            parameters=[
                amcl_params_file,
                {'use_sim_time': use_sim_time},
            ],
        ),
        Node(
            package='nav2_planner',
            executable='planner_server',
            name='planner_server',
            output='screen',
            parameters=[
                costmap_params_file,
                minimal_planner_params,
            ],
        ),
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_costmap',
            output='screen',
            parameters=[
                {'use_sim_time': use_sim_time},
                {'autostart': autostart},
                {'node_names': ['map_server', 'amcl', 'planner_server']},
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
