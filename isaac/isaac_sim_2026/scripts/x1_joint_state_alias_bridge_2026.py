#!/usr/bin/env python3
"""Publish URDF-compatible joint states for the X1 right-arm validation chain.

The Isaac 2026 integrated runtime publishes the arm joints with the expected
names, but the gripper joints use newer USD names such as
``right_gripper_left_joint2``.  The archived right_tcp/cuRobo validation tools
expect the older URDF names, especially ``gripper_controller`` and
``gripper01_*`` mimic joints.  This bridge republishes the original joint state
message with those aliases added on a separate topic so robot_state_publisher
can publish the full right_tcp/finger TF tree without feeding back into the
runtime's /joint_states topic.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


RIGHT_ARM_JOINTS = (
    "joint1_R",
    "joint2_R",
    "joint3_R",
    "joint4_R",
    "joint5_R",
    "joint6_R",
)

# New Isaac runtime gripper joint -> old URDF validation joint.
# The multipliers follow the current direct-articulation sign convention used by
# scene_app.py and the archived URDF mimic signs.
RIGHT_GRIPPER_ALIASES: Tuple[Tuple[str, str, float], ...] = (
    ("right_gripper_left_joint2", "gripper_controller", 1.0),
    ("right_gripper_left_joint2", "gripper01_base_to_gripper_left2", -1.0),
    ("right_gripper_right_join2", "gripper01_base_to_gripper_right2", 1.0),
    ("right_gripper_right_finger_joint", "gripper01_base_to_gripper_right3", 1.0),
    ("right_gripper_right_finger_joint", "gripper01_right3_to_gripper_right1", -1.0),
    ("right_gripper_left_finger_joint", "gripper01_left3_to_gripper_left1", -1.0),
)

LEFT_GRIPPER_ALIASES: Tuple[Tuple[str, str, float], ...] = (
    ("left_gripper_left_joint2", "gripper_left_controller", 1.0),
    ("left_gripper_left_joint2", "gripper01_base_to_gripper_left2_L", -1.0),
    ("left_gripper_right_join2", "gripper01_base_to_gripper_right2_L", 1.0),
    ("left_gripper_right_finger_joint", "gripper01_base_to_gripper_right3_L", 1.0),
    ("left_gripper_right_finger_joint", "gripper01_right3_to_gripper_right1_L", -1.0),
    ("left_gripper_left_finger_joint", "gripper01_left3_to_gripper_left1_L", -1.0),
)


def _value(source: str, scale: float, positions: Dict[str, float]) -> float | None:
    if source not in positions:
        return None
    return float(positions[source]) * float(scale)


def _append_unique(
    names: List[str],
    positions: List[float],
    source_positions: Dict[str, float],
    aliases: Iterable[Tuple[str, str, float]],
) -> None:
    present = set(names)
    for source, target, scale in aliases:
        if target in present:
            continue
        value = _value(source, scale, source_positions)
        if value is None:
            continue
        names.append(target)
        positions.append(value)
        present.add(target)


class X1JointStateAliasBridge(Node):
    def __init__(self) -> None:
        super().__init__("x1_joint_state_alias_bridge_2026")
        self.declare_parameter("input_topic", "/joint_states")
        self.declare_parameter("output_topic", "/x1_model_joint_states")
        self.declare_parameter("publish_left_aliases", True)
        input_topic = self.get_parameter("input_topic").get_parameter_value().string_value
        output_topic = self.get_parameter("output_topic").get_parameter_value().string_value
        self.publish_left_aliases = (
            self.get_parameter("publish_left_aliases").get_parameter_value().bool_value
        )
        self.pub = self.create_publisher(JointState, output_topic, 10)
        self.create_subscription(JointState, input_topic, self._on_joint_state, 10)
        self.get_logger().info(
            f"bridging {input_topic} -> {output_topic} with URDF gripper aliases"
        )

    def _on_joint_state(self, msg: JointState) -> None:
        source_positions = {
            name: float(pos) for name, pos in zip(msg.name, msg.position)
        }
        out = JointState()
        out.header = msg.header
        out.name = list(msg.name)
        out.position = [float(v) for v in msg.position]
        out.velocity = []
        out.effort = []

        _append_unique(out.name, out.position, source_positions, RIGHT_GRIPPER_ALIASES)
        if self.publish_left_aliases:
            _append_unique(out.name, out.position, source_positions, LEFT_GRIPPER_ALIASES)

        # Ensure robot_state_publisher sees all primary arm joints even if the
        # runtime temporarily omits one while initializing.
        present = set(out.name)
        for joint in RIGHT_ARM_JOINTS:
            if joint not in present:
                out.name.append(joint)
                out.position.append(0.0)
                present.add(joint)

        self.pub.publish(out)


def main() -> None:
    rclpy.init()
    node = X1JointStateAliasBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
