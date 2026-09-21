#!/usr/bin/env python3
import math
import time as pytime
from typing import List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
from tf2_ros import TransformListener, Buffer, TransformException


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def stamp_to_ns(stamp_msg) -> int:
    return int(stamp_msg.sec) * 1_000_000_000 + int(stamp_msg.nanosec)


class TagLockCmdVel(Node):
    """
    目标：
      1) 搜索转速别太快（给相机足够识别时间）
      2) 不允许持续旋转；导航后物块应在正前方，最大转向不超过一定角度（默认 60 度）
      3) 识别逻辑要“搜 TF 树”：同时检查 mono 以及双目 (left_/right_) 的 tag frame
      4) 一旦 TF 出现就立即锁定并对准，停止搜寻

    输出 /cmd_vel (仅 angular.z；linear.x=0)
    """

    def __init__(self):
        super().__init__("tag_lock_cmdvel")

        # -------- 基本参数 --------
        self.declare_parameter("target_frame", "tag36h11:79")
        # 可显式指定多个候选 tag frame（会在 TF 树里逐个查）
        self.declare_parameter("target_frames", [])
        self.declare_parameter("base_frame", "turtle")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("rate_hz", 10.0)

        # -------- TF 新鲜度/丢失处理 --------
        self.declare_parameter("max_tf_age_sec", 0.5)
        self.declare_parameter("lost_grace_sec", 0.25)  # 刚丢失时先刹住别立刻开转（防抖）
        self.declare_parameter("stop_when_lost", True)

        # -------- 转向控制(对准) --------
        self.declare_parameter("k_yaw", 2.0)            # rad/s per meter(x)
        self.declare_parameter("max_yaw_rate", 0.45)    # rad/s（手动参数：对准时最大转速）
        self.declare_parameter("deadband_x", 0.02)      # m，x 小于该值不转
        self.declare_parameter("yaw_sign", -1.0)        # 方向不对就改成 +1.0

        # -------- 搜索(限幅摆动，不允许持续旋转) --------
        self.declare_parameter("search_yaw_rate", 0.25)       # rad/s（手动参数：搜索转速）
        self.declare_parameter("max_search_angle_deg", 60)    # 整数！最大搜索偏航角（左右各 60 度）
        self.declare_parameter("search_edge_pause_sec", 0.15) # 到边界停一下再反向

        # -------- 平滑 --------
        self.declare_parameter("smoothing_alpha", 0.35)  # 0~1

        # -------- 对准后停止 --------
        self.declare_parameter("stop_after_aligned", True)      # 对准后是否停止运行
        self.declare_parameter("aligned_hold_time", 0.0)         # 对准后保持时间（秒）
        self.declare_parameter("aligned_stable_count", 1)       # 需要连续对准的次数
        self.declare_parameter("publish_alignment_done", True)
        self.declare_parameter("alignment_done_topic", "/tag_alignment_done")
        self.declare_parameter("alignment_done_linger_sec", 1.0)
        self.declare_parameter("alignment_done_repeat_hz", 10.0)

        # -------- 读取参数 --------
        self.target_frame = str(self.get_parameter("target_frame").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)

        self.rate_hz = float(self.get_parameter("rate_hz").value)
        self.period_wall = 1.0 / max(self.rate_hz, 1e-6)

        self.max_tf_age = float(self.get_parameter("max_tf_age_sec").value)
        self.lost_grace_sec = float(self.get_parameter("lost_grace_sec").value)
        self.stop_when_lost = bool(self.get_parameter("stop_when_lost").value)

        self.k_yaw = float(self.get_parameter("k_yaw").value)
        self.max_w = float(self.get_parameter("max_yaw_rate").value)
        self.deadband_x = float(self.get_parameter("deadband_x").value)
        self.yaw_sign = float(self.get_parameter("yaw_sign").value)

        self.search_yaw_rate = float(self.get_parameter("search_yaw_rate").value)
        self.max_search_angle_deg = int(self.get_parameter("max_search_angle_deg").value)
        self.max_search_rad = math.radians(float(self.max_search_angle_deg))
        self.search_edge_pause_sec = float(self.get_parameter("search_edge_pause_sec").value)

        self.alpha = clamp(float(self.get_parameter("smoothing_alpha").value), 0.0, 1.0)

        # 对准后停止参数
        self.stop_after_aligned = bool(self.get_parameter("stop_after_aligned").value)
        self.aligned_hold_time = float(self.get_parameter("aligned_hold_time").value)
        self.aligned_stable_count = int(self.get_parameter("aligned_stable_count").value)
        self.publish_alignment_done = bool(self.get_parameter("publish_alignment_done").value)
        self.alignment_done_topic = str(self.get_parameter("alignment_done_topic").value)
        self.alignment_done_linger_sec = float(self.get_parameter("alignment_done_linger_sec").value)
        self.alignment_done_repeat_hz = float(self.get_parameter("alignment_done_repeat_hz").value)

        # target_frames: 若未显式给，则自动兼容 stereo 的 left_/right_ 前缀
        raw_list = self.get_parameter("target_frames").value
        target_frames: List[str] = []
        if isinstance(raw_list, list) and len(raw_list) > 0:
            target_frames = [str(x) for x in raw_list]
        else:
            # 兼容你的 apriltag.launch.py: left_tag36h11:79 / right_tag36h11:79
            target_frames = [
                self.target_frame,
                f"left_{self.target_frame}",
                f"right_{self.target_frame}",
            ]
        # 去重但保序
        seen = set()
        self.target_frames = []
        for f in target_frames:
            if f and f not in seen:
                self.target_frames.append(f)
                seen.add(f)

        # -------- TF --------
        self.tf_buffer = Buffer(cache_time=Duration(seconds=30.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # -------- Publisher (/cmd_vel) --------
        self.pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        alignment_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.alignment_pub = self.create_publisher(Bool, self.alignment_done_topic, alignment_qos)

        # -------- 状态 --------
        self._seq = 0
        self._last_warn_wall = 0.0
        self._warn_period = 2.0

        self._searching = False
        self._search_dir = -1.0        # -1 右转, +1 左转（与你 yaw_sign 无关）
        self._search_angle = 0.0       # 积分出来的“相对搜索角”（近似）
        self._edge_pause_until = 0.0

        self._last_w = 0.0
        self._last_seen_wall = 0.0

        # 对准稳定计数器
        self._aligned_count = 0
        self._aligned_start_time = None
        self._alignment_complete = False
        self._alignment_published = False

        self.get_logger().info(
            f'>>> TagLockCmdVel FIXED: base="{self.base_frame}" | pub="{self.cmd_vel_topic}" @ {self.rate_hz:.1f}Hz'
        )
        self.get_logger().info(
            f"候选 tag frames: {self.target_frames} | max_search_angle={self.max_search_angle_deg}deg"
        )
        self.get_logger().info(
            f"搜索参数: search_yaw_rate={self.search_yaw_rate:.2f} rad/s | 对准参数: max_yaw_rate={self.max_w:.2f} rad/s, k_yaw={self.k_yaw:.2f}"
        )
        self.get_logger().info('建议加： --ros-args -p use_sim_time:=true')

    def warn_throttled(self, text: str) -> None:
        now_wall = pytime.time()
        if now_wall - self._last_warn_wall >= self._warn_period:
            self.get_logger().warn(text)
            self._last_warn_wall = now_wall

    def publish_twist(self, w: float) -> None:
        msg = Twist()
        msg.linear.x = 0.0
        msg.angular.z = float(w)
        self.pub.publish(msg)

    def _publish_alignment_done(self) -> None:
        if not self.publish_alignment_done or self._alignment_published:
            return
        msg = Bool()
        msg.data = True
        self.alignment_pub.publish(msg)
        self._alignment_published = True

    def _linger_after_alignment(self) -> None:
        linger = max(0.0, float(self.alignment_done_linger_sec))
        repeat_hz = max(1.0, float(self.alignment_done_repeat_hz))
        if linger <= 0.0:
            return
        interval = 1.0 / repeat_hz
        end_time = pytime.time() + linger
        msg = Bool()
        msg.data = True
        while rclpy.ok() and pytime.time() < end_time:
            self.alignment_pub.publish(msg)
            self.publish_twist(0.0)
            rclpy.spin_once(self, timeout_sec=0.01)
            pytime.sleep(interval)

    def stop(self) -> None:
        self.publish_twist(0.0)
        self._last_w = 0.0

    def _pick_best_transform(self) -> Optional[Tuple[str, object, float]]:
        """在 TF 树里搜索：从多个候选 frame 中挑选最新鲜的那个。"""
        best = None  # (age, frame, transform)
        now_ns = int(self.get_clock().now().nanoseconds)

        for frame in self.target_frames:
            if not self.tf_buffer.can_transform(self.base_frame, frame, Time()):
                continue
            try:
                t = self.tf_buffer.lookup_transform(self.base_frame, frame, Time())
            except TransformException:
                continue

            tf_ns = stamp_to_ns(t.header.stamp)
            age = max(0.0, (now_ns - tf_ns) / 1e9)
            if best is None or age < best[0]:
                best = (age, frame, t)

        if best is None:
            return None
        age, frame, t = best
        return frame, t, age

    def _enter_search(self) -> None:
        if not self._searching:
            self._searching = True
            self._search_dir = -1.0
            self._search_angle = 0.0
            self._edge_pause_until = 0.0
            self.get_logger().info(
                f"未找到有效 TF（或过期），进入限幅摆动搜索: +/-{self.max_search_angle_deg:.0f}deg"
            )

    def _search_step(self) -> float:
        """限幅摆动：角度到边界就反向，不会持续旋转。"""
        now = pytime.time()
        if now < self._edge_pause_until:
            self.publish_twist(0.0)
            self._last_w = 0.0
            return 0.0

        dt = self.period_wall
        w = self._search_dir * self.search_yaw_rate
        next_angle = self._search_angle + w * dt

        # 触边：把本步角度截断到边界，并反向
        if next_angle > self.max_search_rad:
            next_angle = self.max_search_rad
            w = (next_angle - self._search_angle) / dt if dt > 0 else 0.0
            self._search_angle = next_angle
            self._search_dir = -1.0
            self._edge_pause_until = now + self.search_edge_pause_sec
        elif next_angle < -self.max_search_rad:
            next_angle = -self.max_search_rad
            w = (next_angle - self._search_angle) / dt if dt > 0 else 0.0
            self._search_angle = next_angle
            self._search_dir = 1.0
            self._edge_pause_until = now + self.search_edge_pause_sec
        else:
            self._search_angle = next_angle

        # 轻微平滑，避免“抖”
        w = self.alpha * w + (1.0 - self.alpha) * self._last_w
        self._last_w = w

        self.publish_twist(w)
        return w

    def run_loop(self) -> bool:
        """
        主循环。
        返回值：
          - True: 对准完成（stop_after_aligned=True 时）
          - False: 被中断或异常退出
        """
        # TF 热启动
        t0 = pytime.time()
        while rclpy.ok() and pytime.time() - t0 < 0.5:
            rclpy.spin_once(self, timeout_sec=0.05)

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.01)
            self._seq += 1

            try:
                picked = self._pick_best_transform()

                if picked is None:
                    # TF 树里压根没有：若刚丢失，先刹住
                    if self.stop_when_lost:
                        self.stop()
                    if (pytime.time() - self._last_seen_wall) >= self.lost_grace_sec:
                        self._enter_search()
                        w = self._search_step()
                        print(
                            f">>> seq={self._seq:06d} [搜索-无TF] w={w:+.3f} rad/s\r",
                            end="",
                            flush=True,
                        )
                    else:
                        print(
                            f">>> seq={self._seq:06d} [丢失防抖] hold\r",
                            end="",
                            flush=True,
                        )
                    pytime.sleep(self.period_wall)
                    continue

                frame, t, age = picked
                if age > self.max_tf_age:
                    # 有 frame，但过期：同样防抖再进入搜索
                    if self.stop_when_lost:
                        self.stop()
                    if (pytime.time() - self._last_seen_wall) >= self.lost_grace_sec:
                        self._enter_search()
                        w = self._search_step()
                        print(
                            f">>> seq={self._seq:06d} [搜索-TF过期] age={age:.2f}s w={w:+.3f} rad/s\r",
                            end="",
                            flush=True,
                        )
                    else:
                        print(
                            f">>> seq={self._seq:06d} [过期防抖] age={age:.2f}s hold\r",
                            end="",
                            flush=True,
                        )
                    pytime.sleep(self.period_wall)
                    continue

                # ====== 找到新鲜 TF：立即退出搜索并对准 ======
                self._last_seen_wall = pytime.time()
                if self._searching:
                    self.get_logger().info(f"发现 tag TF: {frame}（age={age:.2f}s），停止搜寻并对准")
                    self._searching = False
                    # 注意：不强制把 _search_angle 清零；避免来回切换时突然加速

                x = t.transform.translation.x
                z = t.transform.translation.z

                # 对准：只用 x 控制 yaw
                if abs(x) < self.deadband_x:
                    w_cmd = 0.0
                    # 对准稳定计数
                    self._aligned_count += 1
                    if self._aligned_start_time is None:
                        self._aligned_start_time = pytime.time()
                else:
                    w_cmd = clamp(self.yaw_sign * self.k_yaw * x, -self.max_w, self.max_w)
                    # 重置对准计数
                    self._aligned_count = 0
                    self._aligned_start_time = None

                # 平滑
                w = self.alpha * w_cmd + (1.0 - self.alpha) * self._last_w
                self._last_w = w

                self.publish_twist(w)

                # 检查是否满足对准停止条件
                if self.stop_after_aligned and abs(x) < self.deadband_x:
                    if self._aligned_count >= self.aligned_stable_count:
                        if self._aligned_start_time is not None:
                            hold_elapsed = pytime.time() - self._aligned_start_time
                            if hold_elapsed >= self.aligned_hold_time:
                                self.get_logger().info(
                                    f"[对准完成] 稳定对准 {self._aligned_count} 次，保持 {hold_elapsed:.1f}s，停止运行"
                                )
                                self.stop()
                                self._publish_alignment_done()
                                self._linger_after_alignment()
                                self._alignment_complete = True
                                return True

                status = "[已对准]" if abs(x) < self.deadband_x else "[对准中]"
                aligned_info = ""
                if self.stop_after_aligned and abs(x) < self.deadband_x:
                    aligned_info = f" | 稳定={self._aligned_count}/{self.aligned_stable_count}"
                print(
                    f">>> seq={self._seq:06d} {status} frame={frame} age={age:.2f}s x={x:+.3f} z={z:.3f} | w={w:+.3f} rad/s{aligned_info}\r",
                    end="",
                    flush=True,
                )

            except Exception as ex:
                if self.stop_when_lost:
                    self.stop()
                self.warn_throttled(f"运行异常: {type(ex).__name__}: {ex}")

            pytime.sleep(self.period_wall)

        return False  # 被中断退出

    def is_aligned(self) -> bool:
        """返回是否已完成对准"""
        return self._alignment_complete


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
