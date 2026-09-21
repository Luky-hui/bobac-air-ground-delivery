# 双目相机数据发布与接收演示

## 功能说明

本脚本实现了双目相机数据的接收和显示功能，用于验证 Isaac Sim 与 ROS2 之间的图像数据通信。

### 主要功能

1. **订阅双目相机话题**
   - `/camera/left/image_raw` - 左目相机图像
   - `/camera/right/image_raw` - 右目相机图像
   - `/camera/left/camera_info` - 左目相机参数
   - `/camera/right/camera_info` - 右目相机参数

2. **实时统计显示**
   - 接收帧数和频率
   - 图像分辨率和编码格式
   - 数据大小和传输速率
   - 相机内参（焦距、主点等）
   - 像素统计信息（均值、标准差等）

3. **终端打印输出**
   - 每隔指定时间（默认 2 秒）打印统计信息
   - 显示数据接收状态和质量

## 使用方法

### 方法 1：使用 Isaac Sim Python 环境（推荐）

```bash
# 进入项目目录
cd ~/ros2_ws/src/isaacsim2026/isaac_sim_2026

# 使用 Isaac Sim 的 Python 运行脚本
~/isaac-sim-4.5.0/python.sh launch/stereo_camera_demo.py --ros-args -p use_sim_time:=true
```

### 方法 2：使用 ROS2 Launch 文件

```bash
# 确保已编译工作空间
cd ~/ros2_ws
colcon build --packages-select isaac_sim

# 使用 launch 文件启动
ros2 launch isaac_sim stereo_camera_receiver.launch.py
```

### 方法 3：直接运行 Python 脚本

```bash
# 如果系统已安装 cv_bridge
cd ~/ros2_ws/src/isaacsim2026/isaac_sim_2026/launch
python3 stereo_camera_demo.py --ros-args -p use_sim_time:=true
```

## 参数配置

可以通过 ROS2 参数自定义行为：

```bash
# 修改打印间隔为 1 秒
ros2 run isaac_sim stereo_camera_demo.py --ros-args -p print_interval:=1.0

# 禁用详细统计信息
ros2 run isaac_sim stereo_camera_demo.py --ros-args -p show_statistics:=false

# 自定义话题名称
ros2 run isaac_sim stereo_camera_demo.py --ros-args \
  -p left_image_topic:=/my_camera/left/image \
  -p right_image_topic:=/my_camera/right/image
```

## 输出示例

运行脚本后，终端会显示类似以下内容：

```
======================================================================
运行时间: 10.5s
----------------------------------------------------------------------
左目相机:
  接收帧数: 315
  接收频率: 30.00 Hz
  总数据量: 289.45 MB
  最新图像:
    - 分辨率: 1280 x 720
    - 编码格式: rgb8
    - 数据大小: 2764800 bytes
    - 时间戳: 10.500000000
    - 数据年龄: 0.003s
    - 像素均值 (BGR): [128.5, 130.2, 125.8]
    - 像素标准差 (BGR): [45.3, 42.1, 48.6]
  相机参数:
    - 分辨率: 1280 x 720
    - 畸变模型: plumb_bob
    - 焦距 (fx, fy): (800.00, 800.00)
    - 主点 (cx, cy): (640.00, 360.00)
----------------------------------------------------------------------
右目相机:
  接收帧数: 315
  接收频率: 30.00 Hz
  总数据量: 289.45 MB
  最新图像:
    - 分辨率: 1280 x 720
    - 编码格式: rgb8
    - 数据大小: 2764800 bytes
    - 时间戳: 10.500000000
    - 数据年龄: 0.003s
    - 像素均值 (BGR): [127.8, 129.5, 126.3]
    - 像素标准差 (BGR): [44.9, 41.8, 47.2]
  相机参数:
    - 分辨率: 1280 x 720
    - 畸变模型: plumb_bob
    - 焦距 (fx, fy): (800.00, 800.00)
    - 主点 (cx, cy): (640.00, 360.00)
======================================================================
```

## 验证命令

### 1. 检查话题是否存在

```bash
# 列出所有相机相关话题
ros2 topic list | grep camera

# 预期输出：
# /camera/left/image_raw
# /camera/right/image_raw
# /camera/left/camera_info
# /camera/right/camera_info
```

### 2. 检查话题频率

```bash
# 检查左目图像发布频率
ros2 topic hz /camera/left/image_raw

# 预期输出：
# average rate: 30.000
#   min: 0.033s max: 0.033s std dev: 0.00001s window: 30
```

### 3. 查看话题内容

```bash
# 查看一帧图像信息
ros2 topic echo /camera/left/image_raw --once

# 查看相机参数
ros2 topic echo /camera/left/camera_info --once
```

### 4. 检查节点状态

```bash
# 列出运行的节点
ros2 node list | grep stereo

# 预期输出：
# /stereo_camera_receiver
```

### 5. 检查数据带宽

```bash
# 查看话题带宽
ros2 topic bw /camera/left/image_raw

# 预期输出（1280x720 RGB @ 30Hz）：
# average: 79.10MB/s
#   mean: 2.64MB min: 2.64MB max: 2.64MB window: 30
```

## 录屏展示要点

在录屏时，建议展示以下内容：

1. **启动脚本**
   - 显示完整的启动命令
   - 展示节点启动成功的日志

2. **数据接收状态**
   - 终端打印的统计信息（每 2 秒更新）
   - 显示左右相机都在正常接收数据

3. **验证命令**
   - 运行 `ros2 topic list` 显示话题列表
   - 运行 `ros2 topic hz` 显示接收频率
   - 运行 `ros2 topic echo --once` 显示数据内容

4. **关键信息**
   - 接收帧数持续增加
   - 接收频率稳定（约 30 Hz）
   - 图像分辨率和编码格式正确
   - 相机参数正确显示

## 故障排查

### 问题 1：未接收到图像数据

**症状**：终端显示 `[警告] 未接收到图像数据`

**解决方法**：
1. 确认 Isaac Sim 已启动并加载了包含双目相机的场景
2. 检查相机是否已启用 ROS2 发布功能
3. 验证话题名称是否正确：`ros2 topic list | grep camera`
4. 检查 QoS 设置是否匹配（脚本使用 BEST_EFFORT）

### 问题 2：cv_bridge 导入失败

**症状**：`[警告] cv_bridge 未安装，将只显示基本图像信息`

**解决方法**：
```bash
# 使用 Isaac Sim Python 环境运行（推荐）
~/isaac-sim-4.5.0/python.sh launch/stereo_camera_demo.py

# 或安装 cv_bridge（系统 Python）
sudo apt install ros-humble-cv-bridge python3-opencv
```

### 问题 3：接收频率过低

**症状**：接收频率远低于 30 Hz

**解决方法**：
1. 检查系统资源占用（CPU、GPU）
2. 降低图像分辨率或发布频率
3. 使用 BEST_EFFORT QoS（脚本默认）
4. 检查网络延迟（如果使用远程连接）

### 问题 4：时间戳不同步

**症状**：数据年龄过大或时间戳异常

**解决方法**：
1. 确保使用仿真时间：`-p use_sim_time:=true`
2. 检查 Isaac Sim 的时钟发布：`ros2 topic echo /clock`
3. 验证节点的 use_sim_time 参数：`ros2 param get /stereo_camera_receiver use_sim_time`

## 依赖项

### 必需依赖
- ROS2 Humble
- sensor_msgs
- rclpy

### 可选依赖
- cv_bridge（用于图像转换和统计）
- numpy（用于数值计算）
- opencv-python（用于图像处理）

## 文件说明

- `stereo_camera_demo.py` - 主脚本，包含接收和发布节点
- `stereo_camera_receiver.launch.py` - Launch 文件，方便启动
- `STEREO_CAMERA_README.md` - 本说明文档

## 扩展功能

如需扩展功能，可以修改脚本实现：

1. **图像保存**：将接收到的图像保存为文件
2. **图像显示**：使用 OpenCV 显示实时图像
3. **深度计算**：基于双目图像计算深度图
4. **特征检测**：在图像上运行特征检测算法
5. **数据录制**：使用 rosbag 录制图像数据

## 参考资料

- [ROS2 Image Transport](https://github.com/ros-perception/image_common)
- [Isaac Sim ROS2 Bridge](https://docs.omniverse.nvidia.com/isaacsim/latest/ros2_tutorials/index.html)
- [sensor_msgs/Image](https://docs.ros2.org/latest/api/sensor_msgs/msg/Image.html)
- [sensor_msgs/CameraInfo](https://docs.ros2.org/latest/api/sensor_msgs/msg/CameraInfo.html)
