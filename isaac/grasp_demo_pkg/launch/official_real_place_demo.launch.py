#!/usr/bin/env python3
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory('grasp_demo_pkg')
    isaac_pkg_dir = get_package_share_directory('isaac_sim_2026')
    params = os.path.join(pkg_dir, 'config', 'demo_params.yaml')
    urdf_path = os.path.join(isaac_pkg_dir, 'config', 'bobac_eco65_right_tcp.urdf')
    with open(urdf_path, 'r', encoding='utf-8') as f:
        robot_description = f.read()

    python_sh = os.environ.get('ISAAC_PYTHON', '/home/u-zhuang/isaac-sim-4.5.0/python.sh')
    enable_legacy_place = os.environ.get('ENABLE_LEGACY_PLACE', '').strip().lower() in {
        '1',
        'true',
        'yes',
        'on',
    }
    enable_video_place = os.environ.get('ENABLE_VIDEO_PLACE', '').strip().lower() in {
        '1',
        'true',
        'yes',
        'on',
    }
    place_only = enable_legacy_place and os.environ.get('PLACE_ONLY', '').strip().lower() in {
        '1',
        'true',
        'yes',
        'on',
    }
    stop_after_lift = not enable_video_place
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
        'max_solution_jump:=2.50',
    ]
    place_overrides = {
        'stop_after_lift': stop_after_lift,
        'place_only': place_only,
        'video_place_enabled': enable_video_place,
        'cargo_target_id': 'white_pencil',
        'place_pre_offset': 0.14,
        'place_retreat_offset': 0.20,
        'object_place_z_offset': 0.015,
        'place_scene_point_is_tcp': True,
        'control_cargo_doors': True,
        'retreat_after_place': True,
        'place_stand_before_place': False,
        'place_stand_pose': [3.84, -0.38, 0.0],
        'place_stand_speed': 0.16,
        'place_stand_angular_speed': 0.65,
        'place_stand_timeout': 35.0,
        'reposition_base_before_place': True,
        'place_base_target_xy': [0.27, 0.40],
        'place_base_xy_tolerance': 0.045,
        'place_base_max_step': 0.18,
        'place_base_reposition_iterations': 4,
        'place_base_reposition_speed': 0.06,
        'place_base_reposition_timeout': 8.0,
        'retreat_base_before_lift': False,
        'retreat_base_before_lift_delta': [-0.32, 0.0],
        'retreat_base_before_lift_speed': 0.08,
        'retreat_base_before_lift_timeout': 12.0,
        'escape_lift_before_base_retreat': False,
        'escape_lift_z': 0.32,
        'post_retreat_lift_from_measured_tcp': False,
        'finger_tip_offset': 0.070,
        'min_grasp_z': 0.120,
        'lift_offset': 0.46,
        'min_lift_z': 0.56,
        'lift_steps': 8,
        'min_place_z_from_cargo_body': True,
        'cargo_body_tcp_clearance': 0.10,
        'carry_pose_before_place': False,
        'carry_pose': [0.0, 0.58, -1.67, -0.50, 1.51, 0.0],
        'carry_pose_hold_seconds': 0.8,
        'close_side_door_after_place': True,
        'place_validation_tolerance': 0.12,
        'video_place_face_yaw_offset': 0.0,
        'video_place_yaw_tolerance': 0.05,
        'video_place_yaw_speed': 0.25,
        'video_place_target_base_xy': [0.42, 0.0],
        'video_place_base_xy_tolerance': 0.04,
        'video_place_base_max_step': 0.08,
        'video_place_base_reposition_iterations': 6,
        'video_place_base_reposition_speed': 0.04,
        'video_place_base_reposition_timeout': 7.0,
        'video_place_normal_base': [0.0, 0.0, -1.0],
        'video_place_long_axis_base': [0.0, 1.0, 0.0],
        'video_place_pre_offset_x': 0.16,
        'video_place_retreat_offset_x': 0.18,
        'video_place_post_release_lift': 0.12,
        'video_place_tcp_clearance_from_cargo_body': 0.10,
        'video_place_close_side_door_after_place': False,
    }

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
        Node(
            package='grasp_demo_pkg',
            executable='depth_pose_estimator_node',
            name='depth_pose_estimator_node',
            parameters=[params],
            output='screen',
        ),
        Node(
            package='grasp_demo_pkg',
            executable='tf_transform_demo_node',
            name='tf_transform_demo_node',
            parameters=[params],
            output='screen',
        ),
        ExecuteProcess(
            cmd=plan_cmd,
            name='plan_to_pose_node',
            output='screen',
        ),
        Node(
            package='grasp_demo_pkg',
            executable='gripper_demo_node',
            name='gripper_demo_node',
            parameters=[params],
            output='screen',
        ),
        TimerAction(period=4.0, actions=[
            Node(
                package='grasp_demo_pkg',
                executable='pick_place_state_machine',
                name='pick_place_state_machine',
                parameters=[params, place_overrides],
                output='screen',
            )
        ]),
    ])
