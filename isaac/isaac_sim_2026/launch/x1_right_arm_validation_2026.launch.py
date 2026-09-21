#!/usr/bin/env python3
"""Launch the X1 right-arm URDF/right_tcp validation TF chain."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = Path(get_package_share_directory("isaac_sim_2026"))
    urdf_path = pkg_share / "config" / "TuringStack_X1_2.urdf"
    robot_description = urdf_path.read_text(encoding="utf-8")

    use_sim_time = LaunchConfiguration("use_sim_time")

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            Node(
                package="isaac_sim_2026",
                executable="x1_joint_state_alias_bridge_2026.py",
                name="x1_joint_state_alias_bridge_2026",
                output="screen",
                parameters=[
                    {
                        "use_sim_time": use_sim_time,
                        "input_topic": "/joint_states",
                        "output_topic": "/x1_model_joint_states",
                    }
                ],
            ),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                name="x1_right_arm_robot_state_publisher_2026",
                output="screen",
                parameters=[
                    {
                        "use_sim_time": use_sim_time,
                        "robot_description": robot_description,
                    }
                ],
                remappings=[("joint_states", "/x1_model_joint_states")],
            ),
        ]
    )
