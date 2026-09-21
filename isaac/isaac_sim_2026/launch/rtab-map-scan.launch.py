import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.substitutions import LaunchConfiguration, PythonExpression, PathJoinSubstitution
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.conditions import IfCondition 

def generate_launch_description():
    # 获取 rtabmap_slam 包的共享目录
    rtabmap_slam_share_dir = get_package_share_directory('rtabmap_slam')
    rtabmap_viz_share_dir = get_package_share_directory('rtabmap_viz')

    # 定义 Launch 参数
    declare_frame_id_cmd = DeclareLaunchArgument(
        'frame_id', default_value='base_link',
        description='Robot base frame_id'
    )
    declare_odom_topic_cmd = DeclareLaunchArgument(
        'odom_topic', default_value='odom',
        description='Odometry topic name'
    )
    declare_scan_topic_cmd = DeclareLaunchArgument(
        'scan_topic', default_value='scan',
        description='Laser scan topic name'
    )
    declare_database_path_cmd = DeclareLaunchArgument(
        'database_path', default_value=os.path.join(os.path.expanduser('~'), '.ros', 'rtabmap.db'),
        description='Path to RTAB-Map database'
    )
    declare_delete_db_cmd = DeclareLaunchArgument(
        'delete_db_on_start', default_value='true', 
        description='Whether to delete the RTAB-Map database on start'
    )
    declare_viz_cmd = DeclareLaunchArgument(
        'launch_viz', default_value='true', 
        description='Whether to launch rtabmap_viz'
    )

    # 构建 rtabmap 节点的 arguments 列表
    rtabmap_arguments = [
        PythonExpression([
            "'--delete_db_on_start' if '",
            LaunchConfiguration('delete_db_on_start'),
            "' == 'true' else ''"
        ])
    ]

    # rtabmap 节点
    rtabmap_node = Node(
        package='rtabmap_slam',
        executable='rtabmap',
        name='rtabmap',
        output='screen',
        parameters=[{
            'frame_id': LaunchConfiguration('frame_id'),
            'subscribe_depth': False,
            'subscribe_rgb': False,
            'subscribe_stereo': False,
            'subscribe_rgbd': False,
            'subscribe_scan': True,
            'subscribe_scan_cloud': False,
            'approx_sync': True,
            'queue_size': 30,
            'qos_scan': 1,  
            'qos_odom': 1,
            
            # 启用 use_sim_time ***
            'use_sim_time': True, # <--- 这一行是关键！

            'database_path': LaunchConfiguration('database_path'),

            # RTAB-Map 核心参数
            'RGBD/NeighborLinkRefining': 'true',
            'RGBD/ProximityBySpace': 'true',
            'RGBD/AngularUpdate': '0.01',
            'RGBD/LinearUpdate': '0.01',
            'RGBD/OptimizeFromGraphEnd': 'false',
            'Grid/FromDepth': 'false', # 从激光雷达数据创建占用栅格图 (而不是深度相机)
            'Reg/Force3DoF': 'true',   # 强制 3DoF 注册 (平面运动，不估计滚转、俯仰和Z轴)
            'Reg/Strategy': '1',       # 注册策略：1 = ICP (Iterative Closest Point)，通常配合激光
            'Icp/VoxelSize': '0.05',
            'Icp/MaxCorrespondenceDistance': '0.1',
            'Vis/MinInliers': '10',    # 视觉回环检测最小内点数 (有视觉输入了，这个参数会生效)
        }],
        remappings=[
            ('odom', LaunchConfiguration('odom_topic')),
            ('scan', LaunchConfiguration('scan_topic')),
        ],
        arguments=rtabmap_arguments,
        # prefix = ['xterm -e gdb --args'] # 调试时启用，正式运行时注释
    )

    # RTAB-Map Viz 可视化节点 
    rtabmap_viz_node = Node(
        package='rtabmap_viz',
        executable='rtabmap_viz',
        name='rtabmap_viz',
        output='screen',
        parameters=[{
            'frame_id': LaunchConfiguration('frame_id'),
            'subscribe_odom_info': True,
            'subscribe_rgb': True,
            'subscribe_stereo': False,
            'subscribe_rgbd': False,
            'subscribe_scan': True,
            'queue_size': 10,

            # 启用 use_sim_time ***
            'use_sim_time': True, 
        }],
        remappings=[
            ('rgb/image', '/camera/rgb/image_rect_color'),
            ('rgb/camera_info', '/camera/rgb/camera_info'),
            ('odom', LaunchConfiguration('odom_topic')),
            ('scan', LaunchConfiguration('scan_topic')),
        ],
        condition=IfCondition(LaunchConfiguration('launch_viz'))
    )

    return LaunchDescription([
        declare_frame_id_cmd,
        declare_odom_topic_cmd,
        declare_scan_topic_cmd,
        declare_database_path_cmd,
        declare_delete_db_cmd,
        declare_viz_cmd,
        
        rtabmap_node,
        rtabmap_viz_node,
    ])


