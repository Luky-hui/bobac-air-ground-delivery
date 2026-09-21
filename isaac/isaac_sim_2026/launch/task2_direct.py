#!/usr/bin/env python3
import time as pytime

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration

from geometry_msgs.msg import Twist
from tf2_ros import TransformListener, Buffer, TransformException


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def stamp_to_ns(stamp_msg) -> int:
    # builtin_interfaces/msg/Time
    return int(stamp_msg.sec) * 1_000_000_000 + int(stamp_msg.nanosec)


class TagLockCmdVel(Node):
    """
    自主搜寻 AR 码并对准:
      - 找不到 tag 时: 自动转向搜索
      - 找到 tag 时: 根据 x 偏差控制 angular.z 进行对准
    输出到 /cmd_vel，驱动 Isaac 的 DifferentialController -> wheel_left_link / wheel_right_link
    """

    def __init__(self):
        super().__init__("tag_lock_cmdvel")

        # -------- 基本参数 --------
        self.declare_parameter("target_frame", "tag36h11:79")
        self.declare_parameter("base_frame", "turtle")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("rate_hz", 10.0)

        # -------- TF 新鲜度 --------
        self.declare_parameter("max_tf_age_sec", 0.5)

        # -------- 转向控制(对准) --------
        self.declare_parameter("k_yaw", 2.0)              # rad/s per meter(x)
        self.declare_parameter("max_yaw_rate", 1.0)       # rad/s
        self.declare_parameter("deadband_x", 0.02)        # m，x 小于该值不转
        self.declare_parameter("yaw_sign", -1.0)          # 方向不对就改成 +1.0

        # -------- 渐进式搜索模式 --------
        # 搜索策略: 右小幅 -> 左小幅 -> 右大幅 -> 左大幅 -> 持续大幅旋转
        self.declare_parameter("search_yaw_rate", 0.5)       # rad/s，搜索时的转向速度
        self.declare_parameter("small_sweep_angle", 30.0)    # 度，小幅度摆动角度
        self.declare_parameter("large_sweep_angle", 90.0)    # 度，大幅度摆动角度

        # -------- 平滑(可选) --------
        self.declare_parameter("smoothing_alpha", 0.35)    # 0~1，越大越跟手；0=不动
        # 说明：w = alpha*w_new + (1-alpha)*w_last

        # -------- 读取参数 --------
        self.target_frame = self.get_parameter("target_frame").value
        self.base_frame = self.get_parameter("base_frame").value
        self.cmd_vel_topic = self.get_parameter("cmd_vel_topic").value

        self.rate_hz = float(self.get_parameter("rate_hz").value)
        self.period_wall = 1.0 / max(self.rate_hz, 1e-6)

        self.max_tf_age = float(self.get_parameter("max_tf_age_sec").value)

        self.k_yaw = float(self.get_parameter("k_yaw").value)
        self.max_w = float(self.get_parameter("max_yaw_rate").value)
        self.deadband_x = float(self.get_parameter("deadband_x").value)
        self.yaw_sign = float(self.get_parameter("yaw_sign").value)

        self.search_yaw_rate = float(self.get_parameter("search_yaw_rate").value)
        self.small_sweep_angle = float(self.get_parameter("small_sweep_angle").value)
        self.large_sweep_angle = float(self.get_parameter("large_sweep_angle").value)
        # 转换为弧度
        self.small_sweep_rad = self.small_sweep_angle * 3.14159 / 180.0
        self.large_sweep_rad = self.large_sweep_angle * 3.14159 / 180.0

        self.alpha = float(self.get_parameter("smoothing_alpha").value)
        self.alpha = clamp(self.alpha, 0.0, 1.0)

        # -------- TF --------
        self.tf_buffer = Buffer(cache_time=Duration(seconds=30.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # -------- Publisher (/cmd_vel) --------
        self.pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)

        # -------- 状态 --------
        self._seq = 0
        self._last_warn_wall = 0.0
        self._warn_period = 2.0
        self._searching = False  # 是否处于搜索模式

        # 渐进式搜索状态
        # 搜索阶段: 0=右小幅, 1=左小幅, 2=右大幅, 3=左大幅, 4=持续旋转
        self._search_phase = 0
        self._search_start_time = 0.0
        self._search_direction = -1.0  # -1=右转, 1=左转
        self._current_sweep_target = 0.0  # 当前阶段目标角度(弧度)
        self._accumulated_angle = 0.0  # 已累积转过的角度

        self._last_w = 0.0

        self.get_logger().info(
            f'>>> 启动 TagLockCmdVel: base="{self.base_frame}" -> target="{self.target_frame}" | '
            f'pub="{self.cmd_vel_topic}" @ {self.rate_hz:.1f}Hz <<<'
        )
        self.get_logger().info('建议加： --ros-args -p use_sim_time:=true')

    def warn_throttled(self, text: str) -> None:
        now_wall = pytime.time()
        if now_wall - self._last_warn_wall >= self._warn_period:
            self.get_logger().warn(text)
            self._last_warn_wall = now_wall

    def publish_twist(self, w: float) -> None:
        """发布转向指令 (无前进)"""
        msg = Twist()
        msg.linear.x = 0.0
        msg.angular.z = float(w)
        self.pub.publish(msg)

    def stop(self) -> None:
        self.publish_twist(0.0)
        self._last_w = 0.0

    def reset_search(self) -> None:
        """重置搜索状态，从头开始搜索"""
        self._search_phase = 0
        self._search_direction = -1.0  # 从右转开始
        self._current_sweep_target = self.small_sweep_rad
        self._accumulated_angle = 0.0
        self._search_start_time = pytime.time()

    def get_search_phase_name(self) -> str:
        """获取当前搜索阶段名称"""
        names = ["右小幅", "左小幅", "右大幅", "左大幅", "持续旋转"]
        return names[min(self._search_phase, 4)]

    def search(self) -> float:
        """
        渐进式搜索模式：
        阶段0: 右转小幅度 (small_sweep_angle)
        阶段1: 左转小幅度 (small_sweep_angle * 2，回到中心再过去)
        阶段2: 右转大幅度 (large_sweep_angle)
        阶段3: 左转大幅度 (large_sweep_angle * 2)
        阶段4: 持续左转搜索

        返回当前角速度
        """
        # 计算本周期转过的角度
        dt = self.period_wall
        angle_this_step = abs(self.search_yaw_rate * dt)
        self._accumulated_angle += angle_this_step

        # 检查是否完成当前阶段
        if self._accumulated_angle >= self._current_sweep_target:
            self._search_phase += 1
            self._accumulated_angle = 0.0

            if self._search_phase == 1:
                # 阶段1: 左转小幅度 (需要转回中心再过去，所以是2倍)
                self._search_direction = 1.0
                self._current_sweep_target = self.small_sweep_rad * 2
                self.get_logger().info("搜索: 切换到左小幅")
            elif self._search_phase == 2:
                # 阶段2: 右转大幅度 (从左边回到中心再到右边)
                self._search_direction = -1.0
                self._current_sweep_target = self.small_sweep_rad + self.large_sweep_rad
                self.get_logger().info("搜索: 切换到右大幅")
            elif self._search_phase == 3:
                # 阶段3: 左转大幅度
                self._search_direction = 1.0
                self._current_sweep_target = self.large_sweep_rad * 2
                self.get_logger().info("搜索: 切换到左大幅")
            elif self._search_phase >= 4:
                # 阶段4: 持续旋转
                self._search_phase = 4
                self._search_direction = 1.0  # 持续左转
                self._current_sweep_target = float('inf')
                self.get_logger().info("搜索: 进入持续旋转模式")

        w = self._search_direction * self.search_yaw_rate
        self.publish_twist(w)
        self._last_w = w
        return w

    def run_loop(self) -> None:
        # TF 热启动
        t0 = pytime.time()
        while rclpy.ok() and pytime.time() - t0 < 0.5:
            rclpy.spin_once(self, timeout_sec=0.05)

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.01)
            self._seq += 1

            try:
                if not self.tf_buffer.can_transform(self.base_frame, self.target_frame, Time()):
                    # 找不到 tag，进入搜索模式
                    if not self._searching:
                        self.get_logger().info(f"未找到 {self.target_frame}，开始渐进式搜索...")
                        self._searching = True
                        self.reset_search()
                    w = self.search()
                    phase_name = self.get_search_phase_name()
                    print(
                        f">>> seq={self._seq:06d} [搜索-{phase_name}] w={w:+.3f} rad/s    \r",
                        end="",
                        flush=True,
                    )
                    pytime.sleep(self.period_wall)
                    continue

                t = self.tf_buffer.lookup_transform(self.base_frame, self.target_frame, Time())

                tf_ns = stamp_to_ns(t.header.stamp)
                now_ns = int(self.get_clock().now().nanoseconds)
                age = max(0.0, (now_ns - tf_ns) / 1e9)

                if age > self.max_tf_age:
                    # TF 过期，进入搜索模式
                    if not self._searching:
                        self.get_logger().info(f"TF 过期 (tag 丢失?)，开始渐进式搜索...")
                        self._searching = True
                        self.reset_search()
                    w = self.search()
                    phase_name = self.get_search_phase_name()
                    print(
                        f">>> seq={self._seq:06d} [搜索-{phase_name}-TF过期] age={age:.2f}s w={w:+.3f} rad/s    \r",
                        end="",
                        flush=True,
                    )
                    pytime.sleep(self.period_wall)
                    continue

                # 找到 tag，退出搜索模式
                if self._searching:
                    self.get_logger().info(f"找到 {self.target_frame}，开始对准...")
                    self._searching = False

                x = t.transform.translation.x
                z = t.transform.translation.z

                # ---------- yaw(转向对准) ----------
                if abs(x) < self.deadband_x:
                    w_cmd = 0.0
                else:
                    w_cmd = clamp(self.yaw_sign * self.k_yaw * x, -self.max_w, self.max_w)

                # ---------- 平滑 ----------
                w = self.alpha * w_cmd + (1.0 - self.alpha) * self._last_w
                self._last_w = w

                # ---------- 发布 ----------
                self.publish_twist(w)

                # ---------- 终端心跳 ----------
                status = "[已对准]" if abs(x) < self.deadband_x else "[对准中]"
                print(
                    f">>> seq={self._seq:06d} {status} age={age:.2f}s x={x:+.3f} z={z:.3f} | w={w:+.3f} rad/s    \r",
                    end="",
                    flush=True,
                )

            except TransformException as ex:
                # TF 异常，进入搜索模式
                if not self._searching:
                    self._searching = True
                    self.reset_search()
                self.search()
                self.warn_throttled(f"TF 异常: {type(ex).__name__}: {ex}")
            except Exception as ex:
                self.stop()
                self.warn_throttled(f"运行异常: {type(ex).__name__}: {ex}")

            pytime.sleep(self.period_wall)


def main(args=None):
    rclpy.init(args=args)
    node = TagLockCmdVel()
    try:
        node.run_loop()
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

