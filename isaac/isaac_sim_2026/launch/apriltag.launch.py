from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node


def generate_launch_description():
    """
    双目 AprilTag 检测 launch 文件

    启动两个 apriltag_node：
    - 左目节点：订阅 /camera/left/image_raw，发布 TF: Left_Camera -> left_tag36h11:79
    - 右目节点：订阅 /camera/right/image_raw，发布 TF: Right_Camera -> right_tag36h11:79

    使用方式：
    ros2 launch isaac_sim apriltag.launch.py

    TF 输出：
    - left_tag36h11:79  (左目检测)
    - right_tag36h11:79 (右目检测)

    验证命令：
    ros2 param get /camera/left/apriltag_left size
    ros2 param dump /camera/left/apriltag_left | grep -A5 "tag:"
    ros2 run tf2_ros tf2_echo Left_Camera left_tag36h11:79
    """
    # 独立配置文件（包含 tag.ids/frames/sizes）
    left_config = '/home/u-zhuang/ros2_ws/src/isaac/isaac_sim_2026/config/tags_left.yaml'
    right_config = '/home/u-zhuang/ros2_ws/src/isaac/isaac_sim_2026/config/tags_right.yaml'

    arg_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time'
    )

    # ========== 左目节点 ==========
    apriltag_left = Node(
        package='apriltag_ros',
        executable='apriltag_node',
        name='apriltag_left',
        namespace='camera/left',
        output='screen',
        remappings=[
            ('image_rect', '/camera/left/image_raw'),
            ('camera_info', '/camera/left/camera_info'),
        ],
        parameters=[left_config]
    )

    # ========== 右目节点 ==========
    apriltag_right = Node(
        package='apriltag_ros',
        executable='apriltag_node',
        name='apriltag_right',
        namespace='camera/right',
        output='screen',
        remappings=[
            ('image_rect', '/camera/right/image_raw'),
            ('camera_info', '/camera/right/camera_info'),
        ],
        parameters=[right_config]
    )

    return LaunchDescription([
        arg_use_sim_time,
        apriltag_left,
        apriltag_right,
        # 注意：移除了 mono 兼容节点，避免 TF 冲突
    ])
