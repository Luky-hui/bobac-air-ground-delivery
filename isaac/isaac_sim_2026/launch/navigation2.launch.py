from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='True',
        description='Use simulation time if true'
    )

    # 新增：允许自定义 Nav2 参数文件
    pkg_share = get_package_share_directory('isaac_sim_2026')
    default_params = os.path.join(pkg_share, 'config', 'my_robot_nav2_params.yaml')
    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=default_params,
        description='Full path to the ROS2 parameters file for Nav2'
    )

    nav2_bringup_dir = get_package_share_directory('nav2_bringup')
    navigation_launch_file = os.path.join(nav2_bringup_dir, 'launch', 'navigation_launch.py')

    navigation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(navigation_launch_file),
        launch_arguments={
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'params_file': LaunchConfiguration('params_file'),
        }.items()
    )

    return LaunchDescription([
        use_sim_time_arg,
        params_file_arg,
        navigation_launch
    ])
