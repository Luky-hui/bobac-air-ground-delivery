#!/usr/bin/env python3
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory('grasp_demo_pkg')
    params = os.path.join(pkg_dir, 'config', 'demo_params.yaml')
    urdf_path = '/home/u-zhuang/ros2_ws/src/isaac/isaac_sim_2026/config/bobac_eco65_right_tcp.urdf'
    with open(urdf_path, 'r', encoding='utf-8') as f:
        robot_description = f.read()

    python_sh = os.environ.get('ISAAC_PYTHON', '/home/u-zhuang/isaac-sim-4.5.0/python.sh')
    plan_cmd = [
        python_sh,
        '-m',
        'grasp_demo_pkg.plan_to_pose_node',
        '--ros-args',
        '--params-file',
        params,
        '-p',
        'rotation_threshold:=0.75',
        '-p',
        'position_threshold:=0.025',
        '-p',
        'num_seeds:=384',
        '-p',
        'local_seed_count:=64',
        '-p',
        'max_solution_jump:=2.80',
    ]

    return LaunchDescription([
        Node(
            package='isaac_sim_2026',
            executable='bobac_hand_command_bridge_2026.py',
            name='bobac_hand_command_bridge_2026',
            output='screen',
        ),
        Node(
            package='grasp_demo_pkg',
            executable='joint_state_filter_node',
            name='joint_state_filter_node',
            parameters=[params],
            output='screen',
        ),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='bobac_eco65_tcp_state_publisher',
            parameters=[{'robot_description': robot_description}],
            remappings=[('joint_states', '/demo_grasp/arm_joint_states')],
            output='screen',
        ),
        Node(
            package='grasp_demo_pkg',
            executable='yoloe_detector_node',
            name='yoloe_detector_node',
            parameters=[params, {'auto_detect': False}],
            output='screen',
        ),
        Node(package='grasp_demo_pkg', executable='depth_pose_estimator_node', name='depth_pose_estimator_node', parameters=[params], output='screen'),
        Node(package='grasp_demo_pkg', executable='tf_transform_demo_node', name='tf_transform_demo_node', parameters=[params], output='screen'),
        ExecuteProcess(
            cmd=plan_cmd,
            name='plan_to_pose_node',
            output='screen',
        ),
        Node(package='grasp_demo_pkg', executable='gripper_demo_node', name='gripper_demo_node', parameters=[params], output='screen'),
        TimerAction(period=4.0, actions=[
            Node(package='grasp_demo_pkg', executable='pick_place_state_machine', name='pick_place_state_machine', parameters=[params], output='screen')
        ]),
    ])
