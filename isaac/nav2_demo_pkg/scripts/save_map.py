#!/usr/bin/env python3
"""
保存地图脚本
功能：使用 nav2_map_server 保存当前 SLAM 建图结果
用法：python3 save_map.py --map-name my_map
"""

import os
import sys
import subprocess
import argparse


def save_map(map_name, output_dir):
    """保存地图到指定目录"""

    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 完整路径
    map_path = os.path.join(output_dir, map_name)

    print(f"正在保存地图到: {map_path}")

    # 调用 map_saver_cli
    cmd = ['ros2', 'run', 'nav2_map_server', 'map_saver_cli', '-f', map_path]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("地图保存成功！")
        print(f"生成文件：\n  - {map_path}.yaml\n  - {map_path}.pgm")
        return True
    except subprocess.CalledProcessError as e:
        print(f"地图保存失败: {e}")
        print(f"错误输出: {e.stderr}")
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='保存 SLAM 建图结果')
    parser.add_argument('--map-name', type=str, default='my_map',
                        help='地图文件名（不含扩展名）')
    parser.add_argument('--output-dir', type=str,
                        default=os.path.dirname(os.path.abspath(__file__)) + '/../maps',
                        help='输出目录路径')

    args = parser.parse_args()

    save_map(args.map_name, args.output_dir)
