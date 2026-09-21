#!/usr/bin/env python3
import math
import time
from typing import Optional, Tuple

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PointStamped, Vector3Stamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from grasp_demo_interfaces.action import DetectObject, GripperCommand, PlanToPose


class Stage1TaskManager(Node):
    def __init__(self):
        super().__init__("stage1_task_manager")

        self.declare_parameter("map_frame", "odom")
        self.declare_parameter(
            "office_waypoints",
            [
                4.279537,
                -0.458060,
                -1.350366,
                4.555933,
                -1.358829,
                -0.823705,
                5.141703,
                -2.005444,
                -0.208234,
            ],
        )
        self.declare_parameter(
            "office_goal",
            [5.141703, -2.005444, -0.208234],
        )
        self.declare_parameter("nav_timeout_sec", 180.0)
        self.declare_parameter("require_navigation_success", False)
        self.declare_parameter("skip_navigation", False)
        self.declare_parameter("action_server_wait_sec", 90.0)

        self.declare_parameter("target_classes", ["pencil", "pen"])
        self.declare_parameter("confidence_threshold", 0.01)
        self.declare_parameter("detect_timeout_sec", 20.0)
        self.declare_parameter("perception_wait_sec", 8.0)

        self.declare_parameter("joint_command_topic", "/joint_command")
        self.declare_parameter("arm_joint_names", ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"])
        self.declare_parameter("observation_pose", [0.0, 0.58, -1.67, -0.5, 1.51, 0.0])
        self.declare_parameter("observation_hold_sec", 3.0)

        self.declare_parameter("object_point_topic", "/demo_grasp/object_point_base")
        self.declare_parameter("normal_topic", "/demo_grasp/normal_base")
        self.declare_parameter("long_axis_topic", "/demo_grasp/long_axis_base")

        self.declare_parameter("pre_grasp_offset", 0.08)
        self.declare_parameter("finger_tip_offset", 0.03)
        self.declare_parameter("lift_offset", 0.12)
        self.declare_parameter("cargo_place_position", [0.65, -0.22, 0.18])
        self.declare_parameter("execute_demo_motion", True)

        self.declare_parameter("open_gripper_before_grasp", True)
        self.declare_parameter("gripper_open_position", 100.0)
        self.declare_parameter("gripper_close_position", 0.0)

        self.declare_parameter("cargo_command_topic", "/cargo_bay/command")
        self.declare_parameter("cargo_status_topic", "/cargo_bay/status")
        self.declare_parameter("cargo_status_wait_sec", 3.0)
        self.declare_parameter("cargo_prepare_command", "left_open")
        self.declare_parameter("cargo_finish_command", "left_close")
        self.declare_parameter("publish_cargo_commands", True)

        self.object_point: Optional[PointStamped] = None
        self.normal: Optional[Vector3Stamped] = None
        self.long_axis: Optional[Vector3Stamped] = None
        self.cargo_status: Optional[str] = None
        self.arm_pub = self.create_publisher(
            JointState, self.get_parameter("joint_command_topic").value, 10
        )

        self.create_subscription(
            PointStamped,
            self.get_parameter("object_point_topic").value,
            lambda msg: setattr(self, "object_point", msg),
            10,
        )
        self.create_subscription(
            Vector3Stamped,
            self.get_parameter("normal_topic").value,
            lambda msg: setattr(self, "normal", msg),
            10,
        )
        self.create_subscription(
            Vector3Stamped,
            self.get_parameter("long_axis_topic").value,
            lambda msg: setattr(self, "long_axis", msg),
            10,
        )

        self.nav_client = ActionClient(self, NavigateToPose, "navigate_to_pose")
        self.detect_client = ActionClient(self, DetectObject, "/demo_detect_object")
        self.plan_client = ActionClient(self, PlanToPose, "/demo_plan_to_pose")
        self.gripper_client = ActionClient(self, GripperCommand, "/demo_gripper_command")
        self.cargo_pub = self.create_publisher(
            String, self.get_parameter("cargo_command_topic").value, 10
        )
        self.create_subscription(
            String,
            self.get_parameter("cargo_status_topic").value,
            self.on_cargo_status,
            10,
        )

    def move_to_observation_pose(self) -> None:
        names = [str(v) for v in self.get_parameter("arm_joint_names").value]
        pose = [float(v) for v in self.get_parameter("observation_pose").value]
        if len(names) != len(pose):
            raise RuntimeError("arm_joint_names length must match observation_pose length")
        hold = max(0.1, float(self.get_parameter("observation_hold_sec").value))
        msg = JointState()
        msg.name = names
        end = time.time() + hold
        while rclpy.ok() and time.time() < end:
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.position = pose
            self.arm_pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(0.1)
        self.get_logger().info("observation pose command sent")

    def on_cargo_status(self, msg: String) -> None:
        self.cargo_status = msg.data
        self.get_logger().info(f"cargo status: {msg.data}")

    def _spin_until(self, future, timeout_sec: float):
        deadline = time.time() + timeout_sec
        while rclpy.ok() and not future.done():
            if time.time() > deadline:
                return None
            rclpy.spin_once(self, timeout_sec=0.1)
        return future.result() if future.done() else None

    @staticmethod
    def _yaw_to_quat(yaw: float) -> Tuple[float, float, float, float]:
        return 0.0, 0.0, math.sin(yaw * 0.5), math.cos(yaw * 0.5)

    def navigate_to_office(self) -> bool:
        waypoint_values = [float(v) for v in self.get_parameter("office_waypoints").value]
        if waypoint_values:
            if len(waypoint_values) % 3 != 0:
                raise RuntimeError("office_waypoints must be [x1, y1, yaw1, ...]")
            waypoints = [
                waypoint_values[i : i + 3] for i in range(0, len(waypoint_values), 3)
            ]
        else:
            goal_values = [float(v) for v in self.get_parameter("office_goal").value]
            if len(goal_values) != 3:
                raise RuntimeError("office_goal must be [x, y, yaw]")
            waypoints = [goal_values]

        timeout = float(self.get_parameter("nav_timeout_sec").value)
        self.get_logger().info(
            f"waiting for Nav2 action, office waypoints={waypoints}"
        )
        if not self.nav_client.wait_for_server(timeout_sec=10.0):
            msg = "navigate_to_pose action is unavailable"
            if self.get_parameter("require_navigation_success").value:
                raise RuntimeError(msg)
            self.get_logger().warn(msg + "; continuing because require_navigation_success=false")
            return False

        for index, goal_values in enumerate(waypoints, start=1):
            goal = NavigateToPose.Goal()
            goal.pose.header.frame_id = self.get_parameter("map_frame").value
            # Use latest TF for odom-frame waypoints; old stamps can expire while
            # Nav2 replans through the RTAB-Map map->odom transform.
            goal.pose.header.stamp.sec = 0
            goal.pose.header.stamp.nanosec = 0
            goal.pose.pose.position.x = goal_values[0]
            goal.pose.pose.position.y = goal_values[1]
            qx, qy, qz, qw = self._yaw_to_quat(goal_values[2])
            goal.pose.pose.orientation.x = qx
            goal.pose.pose.orientation.y = qy
            goal.pose.pose.orientation.z = qz
            goal.pose.pose.orientation.w = qw

            self.get_logger().info(
                f"office waypoint {index}/{len(waypoints)}: {goal_values}"
            )
            goal_handle = self._spin_until(
                self.nav_client.send_goal_async(goal),
                timeout_sec=10.0,
            )
            if goal_handle is None or not goal_handle.accepted:
                raise RuntimeError("office navigation goal was rejected or timed out")

            result = self._spin_until(goal_handle.get_result_async(), timeout_sec=timeout)
            if result is None:
                raise RuntimeError("office navigation timed out")
            self.get_logger().info(
                f"office waypoint {index}/{len(waypoints)} finished with status={result.status}"
            )
            if (
                result.status != GoalStatus.STATUS_SUCCEEDED
                and self.get_parameter("require_navigation_success").value
            ):
                raise RuntimeError(
                    f"office waypoint {index}/{len(waypoints)} failed with status={result.status}"
                )
        return True

    def detect_material(self):
        wait_sec = float(self.get_parameter("action_server_wait_sec").value)
        if not self.detect_client.wait_for_server(timeout_sec=wait_sec):
            raise RuntimeError("/demo_detect_object action is unavailable")
        self.object_point = None
        self.normal = None
        self.long_axis = None
        goal = DetectObject.Goal()
        goal.target_classes = [str(v) for v in self.get_parameter("target_classes").value]
        goal.confidence_threshold = float(self.get_parameter("confidence_threshold").value)
        goal.timeout = float(self.get_parameter("detect_timeout_sec").value)

        goal_handle = self._spin_until(
            self.detect_client.send_goal_async(goal),
            timeout_sec=10.0,
        )
        if goal_handle is None or not goal_handle.accepted:
            raise RuntimeError("detect goal was rejected or timed out")
        result_msg = self._spin_until(
            goal_handle.get_result_async(),
            timeout_sec=goal.timeout + 5.0,
        )
        if result_msg is None:
            raise RuntimeError("detect result timed out")
        result = result_msg.result
        if not result.success:
            raise RuntimeError(f"detection failed: {result.message}")
        self.get_logger().info(
            f"detected {result.detected_class}, confidence={result.confidence:.3f}; waiting for depth/TF pose"
        )

    def wait_for_perception_topics(self) -> None:
        deadline = time.time() + float(self.get_parameter("perception_wait_sec").value)
        while rclpy.ok() and time.time() < deadline:
            if self.object_point and self.normal and self.long_axis:
                return
            rclpy.spin_once(self, timeout_sec=0.1)
        raise RuntimeError("perception topics did not publish object pose/axis")

    def _normalized_normal(self) -> Tuple[float, float, float]:
        msg = self.normal
        if msg is None:
            return 0.0, 0.0, -1.0
        nx, ny, nz = msg.vector.x, msg.vector.y, msg.vector.z
        norm = math.sqrt(nx * nx + ny * ny + nz * nz)
        if norm < 1e-9:
            return 0.0, 0.0, -1.0
        return nx / norm, ny / norm, nz / norm

    def _offset_point(self, source: PointStamped, distance: float) -> PointStamped:
        nx, ny, nz = self._normalized_normal()
        msg = PointStamped()
        msg.header = source.header
        msg.point.x = source.point.x - nx * distance
        msg.point.y = source.point.y - ny * distance
        msg.point.z = source.point.z - nz * distance
        return msg

    def _point_from_xyz(self, xyz) -> PointStamped:
        if self.object_point is None:
            raise RuntimeError("object_point is unavailable")
        msg = PointStamped()
        msg.header = self.object_point.header
        msg.point.x = float(xyz[0])
        msg.point.y = float(xyz[1])
        msg.point.z = float(xyz[2])
        return msg

    def send_plan(self, point: PointStamped, label: str, execute: bool) -> None:
        if self.normal is None or self.long_axis is None:
            raise RuntimeError("normal/long_axis are unavailable")
        wait_sec = float(self.get_parameter("action_server_wait_sec").value)
        if not self.plan_client.wait_for_server(timeout_sec=wait_sec):
            raise RuntimeError("/demo_plan_to_pose action is unavailable")
        goal = PlanToPose.Goal()
        goal.target = point
        goal.normal = self.normal
        goal.long_axis = self.long_axis
        goal.execute = bool(execute)
        goal.label = label
        goal_handle = self._spin_until(
            self.plan_client.send_goal_async(goal),
            timeout_sec=10.0,
        )
        if goal_handle is None or not goal_handle.accepted:
            raise RuntimeError(f"plan goal rejected or timed out: {label}")
        result_msg = self._spin_until(goal_handle.get_result_async(), timeout_sec=60.0)
        if result_msg is None or not result_msg.result.success:
            reason = "timed out" if result_msg is None else result_msg.result.message
            raise RuntimeError(f"plan failed: {label}: {reason}")
        self.get_logger().info(f"{label}: {result_msg.result.message}")

    def send_gripper(self, command: str, position: float) -> None:
        wait_sec = float(self.get_parameter("action_server_wait_sec").value)
        if not self.gripper_client.wait_for_server(timeout_sec=wait_sec):
            raise RuntimeError("/demo_gripper_command action is unavailable")
        goal = GripperCommand.Goal()
        goal.command = command
        goal.position = float(position)
        goal.speed = 50.0
        goal.wait_for_completion = True
        goal_handle = self._spin_until(
            self.gripper_client.send_goal_async(goal),
            timeout_sec=10.0,
        )
        if goal_handle is None or not goal_handle.accepted:
            raise RuntimeError(f"gripper goal rejected or timed out: {command}")
        result_msg = self._spin_until(goal_handle.get_result_async(), timeout_sec=20.0)
        if result_msg is None or not result_msg.result.success:
            reason = "timed out" if result_msg is None else result_msg.result.message
            raise RuntimeError(f"gripper failed: {reason}")

    def publish_cargo_command(self, command: str) -> None:
        if not self.get_parameter("publish_cargo_commands").value or not command:
            return
        self.cargo_status = None
        msg = String()
        msg.data = command
        for _ in range(3):
            self.cargo_pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.05)
        self.get_logger().info(f"cargo command published: {command}")
        self.wait_for_cargo_status(command)

    def wait_for_cargo_status(self, command: str) -> None:
        expected = {
            "left_open": "left_opened",
            "side_open": "left_opened",
            "left_close": "left_closed",
            "side_close": "left_closed",
            "bottom_open": "bottom_opened",
            "bottom_close": "bottom_closed",
        }.get(command)
        if not expected:
            return
        deadline = time.time() + float(self.get_parameter("cargo_status_wait_sec").value)
        while rclpy.ok() and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.cargo_status and expected in self.cargo_status:
                return
        self.get_logger().warn(
            f"cargo status did not confirm '{expected}' within timeout"
        )

    def run(self) -> bool:
        execute = bool(self.get_parameter("execute_demo_motion").value)
        if self.get_parameter("skip_navigation").value:
            self.get_logger().info("skip_navigation=true; starting perception/grasp in current pose")
        else:
            self.navigate_to_office()
        self.publish_cargo_command(self.get_parameter("cargo_prepare_command").value)
        if self.get_parameter("open_gripper_before_grasp").value:
            self.send_gripper("open", self.get_parameter("gripper_open_position").value)
        self.move_to_observation_pose()
        self.detect_material()
        self.wait_for_perception_topics()

        tip = float(self.get_parameter("finger_tip_offset").value)
        pre = float(self.get_parameter("pre_grasp_offset").value)
        lift = float(self.get_parameter("lift_offset").value)
        place = self._point_from_xyz(self.get_parameter("cargo_place_position").value)
        pre_place = self._offset_point(place, pre)

        self.send_plan(self._offset_point(self.object_point, tip + pre), "pre_grasp", execute)
        self.send_plan(self._offset_point(self.object_point, tip), "grasp", execute)
        self.send_gripper("close", self.get_parameter("gripper_close_position").value)
        self.send_plan(self._offset_point(self.object_point, tip + lift), "lift", execute)
        self.send_plan(pre_place, "pre_place_cargo", execute)
        self.send_plan(place, "place_cargo", execute)
        self.send_gripper("open", self.get_parameter("gripper_open_position").value)
        self.publish_cargo_command(self.get_parameter("cargo_finish_command").value)
        self.get_logger().info("stage 1 task sequence completed")
        return True


def main(args=None):
    rclpy.init(args=args)
    node = Stage1TaskManager()
    ok = False
    try:
        ok = node.run()
    except KeyboardInterrupt:
        node.get_logger().info("stage 1 task manager interrupted")
    except RuntimeError as exc:
        node.get_logger().error(str(exc))
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
