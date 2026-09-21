#!/usr/bin/env python3
import argparse
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class CargoBayCommand(Node):
    def __init__(self):
        super().__init__('cargo_bay_command_2026')
        self.pub = self.create_publisher(String, '/cargo_bay/command', 10)
        self.status = []
        self.create_subscription(String, '/cargo_bay/status', self._on_status, 10)

    def _on_status(self, msg):
        self.status.append(msg.data)
        print(msg.data)

    def send(self, command, wait_sec):
        deadline = time.time() + 1.0
        while rclpy.ok() and time.time() < deadline and self.pub.get_subscription_count() == 0:
            rclpy.spin_once(self, timeout_sec=0.05)
        msg = String()
        msg.data = command
        self.pub.publish(msg)
        end = time.time() + wait_sec
        while rclpy.ok() and time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'command',
        choices=[
            'side_open',
            'side_close',
            'left_open',
            'left_close',
            'bottom_open',
            'bottom_close',
            'payload_lock',
            'payload_release',
            'status',
        ],
    )
    parser.add_argument('--wait-sec', type=float, default=1.0)
    args = parser.parse_args()

    rclpy.init()
    node = CargoBayCommand()
    try:
        node.send(args.command, args.wait_sec)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
