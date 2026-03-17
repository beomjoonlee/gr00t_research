#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Any

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float32MultiArray

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy


DEFAULT_MODEL_PATH = "gr00t/model/gr00t_rgb_run/checkpoint-200000"


def _parse_name_list(value: str | None) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _image_msg_to_rgb_array(msg: Image) -> np.ndarray:
    dtype = np.uint8
    channels = max(1, msg.step // max(1, msg.width))
    image = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width, channels)

    if msg.encoding == "rgb8":
        rgb = image[:, :, :3]
    elif msg.encoding == "bgr8":
        rgb = image[:, :, :3][:, :, ::-1]
    elif msg.encoding == "bgra8":
        rgb = image[:, :, :3][:, :, ::-1]
    elif msg.encoding == "rgba8":
        rgb = image[:, :, :3]
    elif msg.encoding in {"mono8", "8UC1"}:
        rgb = np.repeat(image[:, :, :1], 3, axis=2)
    else:
        raise ValueError(f"Unsupported image encoding: {msg.encoding}")

    return np.ascontiguousarray(rgb)


class Gr00tRos2InferenceNode(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("gr00t_ros2_inference")
        self.args = args
        self.last_image: np.ndarray | None = None
        self.last_joint_state: JointState | None = None
        self.last_action_time = 0.0

        model_path = Path(args.model_path)
        self.policy = Gr00tPolicy(
            embodiment_tag=EmbodimentTag.GR1,
            model_path=str(model_path),
            device=args.device,
        )
        self.modality = self.policy.get_modality_config()
        self.video_key = self.modality["video"].modality_keys[0]
        self.state_keys = self.modality["state"].modality_keys
        self.action_keys = self.modality["action"].modality_keys
        self.language_key = self.modality["language"].modality_keys[0]

        self.group_dims = self._infer_group_dims()
        self.input_name_groups = self._build_input_name_groups(args)
        self.output_name_groups = self._build_output_name_groups(args)

        self.image_sub = self.create_subscription(
            Image, args.image_topic, self._on_image, qos_profile=10
        )
        self.joint_sub = self.create_subscription(
            JointState, args.joint_state_topic, self._on_joint_state, qos_profile=50
        )
        self.command_pub = self.create_publisher(JointState, args.command_topic, 10)
        self.chunk_pub = self.create_publisher(Float32MultiArray, args.chunk_topic, 10)
        self.timer = self.create_timer(1.0 / args.inference_hz, self._run_inference)

        self.get_logger().info(f"Loaded model from {model_path}")
        self.get_logger().info(f"Using image topic: {args.image_topic}")
        self.get_logger().info(f"Using joint topic: {args.joint_state_topic}")
        self.get_logger().info(f"Publishing commands to: {args.command_topic}")
        self.get_logger().info(
            "State groups: "
            + ", ".join(f"{key}={self.group_dims[key]}" for key in self.state_keys)
        )

    def _infer_group_dims(self) -> dict[str, int]:
        stats = self.policy.processor.statistics["gr1"]["state"]
        return {key: len(stats[key]["mean"]) for key in self.state_keys}

    def _build_input_name_groups(self, args: argparse.Namespace) -> dict[str, list[str]]:
        configured = {
            "left_arm": args.left_arm_names,
            "right_arm": args.right_arm_names,
            "left_hand": args.left_hand_names,
            "right_hand": args.right_hand_names,
            "waist": args.waist_names,
        }
        if all(configured[key] for key in self.state_keys):
            return configured

        self.get_logger().warn(
            "Joint name mapping parameters were not fully provided. "
            "Falling back to JointState.position ordering."
        )
        return {key: [] for key in self.state_keys}

    def _build_output_name_groups(self, args: argparse.Namespace) -> dict[str, list[str]]:
        configured = {
            "left_arm": args.output_left_arm_names or args.left_arm_names,
            "right_arm": args.output_right_arm_names or args.right_arm_names,
            "left_hand": args.output_left_hand_names or args.left_hand_names,
            "right_hand": args.output_right_hand_names or args.right_hand_names,
            "waist": args.output_waist_names or args.waist_names,
        }
        if all(configured[key] for key in self.action_keys):
            return configured

        return {key: [] for key in self.action_keys}

    def _on_image(self, msg: Image) -> None:
        try:
            self.last_image = _image_msg_to_rgb_array(msg)
        except Exception as exc:
            self.get_logger().error(f"Failed to parse image: {exc}")

    def _on_joint_state(self, msg: JointState) -> None:
        self.last_joint_state = msg

    def _extract_group(self, msg: JointState, key: str, start_idx: int) -> tuple[np.ndarray, int]:
        dim = self.group_dims[key]
        names = self.input_name_groups[key]

        if names:
            position_map = {name: pos for name, pos in zip(msg.name, msg.position)}
            values = [position_map[name] for name in names]
            return np.asarray(values, dtype=np.float32), start_idx

        end_idx = start_idx + dim
        if len(msg.position) < end_idx:
            raise ValueError(
                f"JointState.position has {len(msg.position)} entries, need at least {end_idx}"
            )
        values = np.asarray(msg.position[start_idx:end_idx], dtype=np.float32)
        return values, end_idx

    def _build_observation(self) -> dict[str, Any]:
        if self.last_image is None or self.last_joint_state is None:
            raise RuntimeError("Waiting for both image and joint state")

        joint_msg = self.last_joint_state
        state: dict[str, np.ndarray] = {}
        cursor = 0
        for key in self.state_keys:
            values, cursor = self._extract_group(joint_msg, key, cursor)
            state[key] = values[None, None, :]

        return {
            "video": {self.video_key: self.last_image[None, None, :, :, :]},
            "state": state,
            "language": {self.language_key: [[self.args.instruction]]},
        }

    def _flatten_action_step(
        self, action_chunk: dict[str, np.ndarray], step_idx: int
    ) -> tuple[list[str], list[float]]:
        names: list[str] = []
        positions: list[float] = []

        for key in self.action_keys:
            values = action_chunk[key][0, step_idx].astype(np.float32).tolist()
            group_names = self.output_name_groups[key]
            if group_names and len(group_names) == len(values):
                names.extend(group_names)
            else:
                names.extend([f"{key}_{i}" for i in range(len(values))])
            positions.extend(values)

        return names, positions

    def _flatten_full_chunk(self, action_chunk: dict[str, np.ndarray]) -> list[float]:
        flat_chunk: list[float] = []
        horizon = next(iter(action_chunk.values())).shape[1]
        for step_idx in range(horizon):
            for key in self.action_keys:
                flat_chunk.extend(action_chunk[key][0, step_idx].astype(np.float32).tolist())
        return flat_chunk

    def _publish_action(self, action_chunk: dict[str, np.ndarray]) -> None:
        names, positions = self._flatten_action_step(action_chunk, 0)

        joint_msg = JointState()
        joint_msg.header.stamp = self.get_clock().now().to_msg()
        joint_msg.name = names
        joint_msg.position = positions
        self.command_pub.publish(joint_msg)

        chunk_msg = Float32MultiArray()
        chunk_msg.data = self._flatten_full_chunk(action_chunk)
        self.chunk_pub.publish(chunk_msg)

    def _run_inference(self) -> None:
        now = time.monotonic()
        if now - self.last_action_time < 1.0 / self.args.inference_hz:
            return

        try:
            obs = self._build_observation()
        except RuntimeError:
            return
        except Exception as exc:
            self.get_logger().error(f"Failed to build observation: {exc}")
            return

        try:
            action_chunk, _ = self.policy.get_action(obs)
            self._publish_action(action_chunk)
            self.last_action_time = now
        except Exception as exc:
            self.get_logger().error(f"Inference failed: {exc}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ROS2 GR00T inference bridge for GR1 models.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--instruction", default="pick and place the target object")
    parser.add_argument("--image-topic", default="/ec_sensor_1/camera/color/image_raw")
    parser.add_argument("--joint-state-topic", default="/ec_robot_1/joint_states")
    parser.add_argument("--command-topic", default="/ec_robot_1/pos_cmd")
    parser.add_argument("--chunk-topic", default="/gr00t/action_chunk")
    parser.add_argument("--inference-hz", type=float, default=5.0)
    parser.add_argument(
        "--left-arm-names",
        type=_parse_name_list,
        default=None,
        help="Comma-separated joint names for GR1 left_arm group.",
    )
    parser.add_argument(
        "--right-arm-names",
        type=_parse_name_list,
        default=None,
        help="Comma-separated joint names for GR1 right_arm group.",
    )
    parser.add_argument(
        "--left-hand-names",
        type=_parse_name_list,
        default=None,
        help="Comma-separated joint names for GR1 left_hand group.",
    )
    parser.add_argument(
        "--right-hand-names",
        type=_parse_name_list,
        default=None,
        help="Comma-separated joint names for GR1 right_hand group.",
    )
    parser.add_argument(
        "--waist-names",
        type=_parse_name_list,
        default=None,
        help="Comma-separated joint names for GR1 waist group.",
    )
    parser.add_argument(
        "--output-left-arm-names",
        type=_parse_name_list,
        default=None,
        help="Optional output joint names for left_arm commands.",
    )
    parser.add_argument(
        "--output-right-arm-names",
        type=_parse_name_list,
        default=None,
        help="Optional output joint names for right_arm commands.",
    )
    parser.add_argument(
        "--output-left-hand-names",
        type=_parse_name_list,
        default=None,
        help="Optional output joint names for left_hand commands.",
    )
    parser.add_argument(
        "--output-right-hand-names",
        type=_parse_name_list,
        default=None,
        help="Optional output joint names for right_hand commands.",
    )
    parser.add_argument(
        "--output-waist-names",
        type=_parse_name_list,
        default=None,
        help="Optional output joint names for waist commands.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    rclpy.init(args=None)
    node = Gr00tRos2InferenceNode(args)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
