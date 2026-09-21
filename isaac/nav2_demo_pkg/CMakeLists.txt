cmake_minimum_required(VERSION 3.8)
project(nav2_demo_pkg)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)

# 安装 launch 文件
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

# 安装 config 文件
install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}/
)

# 安装 maps 文件
install(DIRECTORY
  maps
  DESTINATION share/${PROJECT_NAME}/
)

# 安装 rviz 配置
install(DIRECTORY
  rviz
  DESTINATION share/${PROJECT_NAME}/
)

# 安装 scripts
install(DIRECTORY
  scripts
  DESTINATION share/${PROJECT_NAME}/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
