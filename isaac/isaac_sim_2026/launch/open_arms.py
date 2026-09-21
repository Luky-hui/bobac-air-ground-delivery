import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import time

class OpenArmsSafe(Node):
    def __init__(self):
        super().__init__('open_arms_safe_node')

        # ================= 1. 关节定义 =================
        self.joints_R = ['joint1_R', 'joint2_R', 'joint3_R', 'joint4_R', 'joint5_R', 'joint6_R']
        self.joints_L = ['joint1_L', 'joint2_L', 'joint3_L', 'joint4_L', 'joint5_L', 'joint6_L']
        
        # 你的右夹爪名字是 gripper_controller
        self.gripper_R_name = ['gripper_controller']
        self.gripper_L_name = ['gripper_left_controller']

        # 缓存
        self.current_pose = {}
        self.data_ready = False

        # ================= 2. 通信配置 =================
        self.pub_main = self.create_publisher(JointState, '/joint_command', 10)
        self.pub_left = self.create_publisher(JointState, '/joint_left_command', 10)

        self.create_subscription(JointState, '/joint_states', self.main_callback, 10)
        self.create_subscription(JointState, '/joint_left_states', self.left_callback, 10)

        self.get_logger().info('>>> [安全版] 正在初始化... 等待读取当前状态 <<<')

    def main_callback(self, msg):
        try:
            # 记录右臂6个关节角度
            if 'joint1_R' in msg.name:
                self.current_pose['R_arm'] = [msg.position[msg.name.index(j)] for j in self.joints_R if j in msg.name]
            
            # 记录左臂 (防止遗漏)
            if 'joint1_L' in msg.name:
                self.current_pose['L_arm'] = [msg.position[msg.name.index(j)] for j in self.joints_L if j in msg.name]

            # 记录右夹爪当前开度 (关键！只读不写，后面用到)
            if self.gripper_R_name[0] in msg.name:
                val = msg.position[msg.name.index(self.gripper_R_name[0])]
                self.current_pose['R_grip'] = [val]
                self.data_ready = True
        except:
            pass

    def left_callback(self, msg):
        if self.gripper_L_name[0] in msg.name:
            val = msg.position[msg.name.index(self.gripper_L_name[0])]
            self.current_pose['L_grip'] = [val]

    def smooth_move(self, target_R, target_L, duration=3.0):
        steps = int(duration * 50)
        
        start_R = list(self.current_pose.get('R_arm', [0.0]*6)) # 起始位置（当前）
        start_L = list(self.current_pose.get('L_arm', [0.0]*6))
        
        # === 核心修改：锁定夹爪 ===
        # 锁定夹爪为当前读到的值，不强行发0.0
        # 这样物理引擎就不会打架了
        fixed_grip_R = self.current_pose.get('R_grip', [0.0])
        fixed_grip_L = self.current_pose.get('L_grip', [0.0])

        for i in range(steps + 1):
            alpha = i / steps
            # 线性插值计算当前步的目标角度
            now_R = [s + (e - s) * alpha for s, e in zip(start_R, target_R)]
            now_L = [s + (e - s) * alpha for s, e in zip(start_L, target_L)]

            # 发布给主控制器 (右臂+左臂+右爪)
            msg_main = JointState()
            msg_main.header.stamp = self.get_clock().now().to_msg()
            msg_main.name = self.joints_R + self.joints_L + self.gripper_R_name
            msg_main.position = now_R + now_L + fixed_grip_R # <--- 使用当前读到的值
            self.pub_main.publish(msg_main)

            # 发布给左控制器 (左爪)
            msg_left = JointState()
            msg_left.header.stamp = self.get_clock().now().to_msg()
            msg_left.name = self.gripper_L_name
            msg_left.position = fixed_grip_L # <--- 使用当前读到的值
            self.pub_left.publish(msg_left)

            rclpy.spin_once(self, timeout_sec=0.01)
            time.sleep(0.02)

    def execute(self):
        # 必须等待数据完全同步
        timeout = 0
        while not self.data_ready:
            rclpy.spin_once(self, timeout_sec=0.1)
            timeout += 1
            if timeout > 50:
                self.get_logger().warn("等待数据超时...请检查仿真是否在播放")
                return
        
        self.get_logger().info('>>> 状态同步完成，开始安全张开... <<<')

        # 安全角度 (手肘下垂，不撞轮子)
        target_R_Safe = [0.9, 0.0, 0.0, -0.5, 0.0, 0.0]
        target_L_Safe = [0.9, -0.3, 0.0, -0.5, 0.0, 0.0]

        self.smooth_move(target_R_Safe, target_L_Safe, duration=4.0) # 动作放慢到4秒
        
        self.get_logger().info('>>> 动作完成，夹爪保持当前状态。 <<<')

def main():
    rclpy.init()
    node = OpenArmsSafe()
    try:
        node.execute()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
