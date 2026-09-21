cmake_minimum_required(VERSION 3.8)
project(isaac_sim_2026)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()


find_package(ament_cmake REQUIRED)


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)

  set(ament_cmake_copyright_FOUND TRUE)

  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

install(
  DIRECTORY launch 
  DESTINATION share/${PROJECT_NAME}
)

install(
  DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS
    scripts/pointcloud_to_laserscan_2026.py
    scripts/laser_scan_republisher_2026.py
    scripts/stow_arms_for_navigation.py
    scripts/safe_base_jog_2026.py
    scripts/slow_base_move_2026.py
    scripts/office_odom_nav_2026.py
    scripts/office_waypoint_nav_2026.py
    scripts/stage1_task_manager.py
    scripts/material_task_client_2026.py
    scripts/x1_joint_state_alias_bridge_2026.py
    scripts/bobac_cmd_vel_to_joint_2026.py
    scripts/bobac_hand_command_bridge_2026.py
    scripts/bobac_sort_odom_nav_2026.py
    scripts/bobac_nav_then_grasp_demo_2026.py
    scripts/prepare_pencil_rect_dataset_2026.py
    scripts/cargo_bay_command_2026.py
    scripts/bobac_keyboard_base_teleop_2026.py
    scripts/bobac_keyboard_arm_joints_2026.py
    scripts/bobac_print_pose_2026.py
    scripts/bobac_print_joints_2026.py
    scripts/bobac_execute_joint_pose_2026.py
    scripts/bobac_manual_place_sequence_2026.py
    scripts/bobac_drone_command_2026.py
    scripts/bobac_drone_drop_mission_2026.py
    scripts/bobac_drone_visual_align_drop_2026.py
    scripts/bobac_keyboard_drone_teleop_2026.py
    scripts/bobac_px4_offboard_mission_2026.py
    scripts/bobac_px4_keyboard_teleop_2026.py
    scripts/bobac_px4_print_pose_2026.py
    scripts/bobac_grasp_perception_viewer_2026.py
    scripts/bobac_drone_bluebox_viewer_2026.py
    scripts/bobac_drop_precision_check_2026.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
