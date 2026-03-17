#!/usr/bin/env python3

from __future__ import annotations

import socket
import struct
import threading
import time
from collections import deque
from typing import Any, Optional

import msgpack
import msgpack_numpy as m
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float32MultiArray

m.patch()

# ====== EDIT HERE (no CLI params) ======
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 5555

IMAGE_H = 256
IMAGE_W = 256
PROMPT_TEXT = "pick and place the target object"

TOPIC_JS_IN = "/ec_robot_1/joint_states_single"
TOPIC_IMG_1 = "/ec_sensor_1/camera/color/image_raw"
TOPIC_IMG_5 = "/ec_sensor_5/camera/color/image_raw"
TOPIC_IMG_6 = "/ec_sensor_6/camera/color/image_raw"
TOPIC_JS_OUT = "/ec_robot_1/joint_states"
TOPIC_CHUNK_OUT = "/gr00t/action_chunk"

INFER_HZ = 5.0
EXEC_HZ = 20.0
ACTION_CHUNK_LEN = 8

PIPER_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]

GR1_STATE_LAYOUT = {
    "left_arm": 7,
    "right_arm": 7,
    "left_hand": 6,
    "right_hand": 6,
    "waist": 3,
}
PIPER_CMD_DIM = 7

CONNECT_TIMEOUT_SEC = 2.0
RPC_TIMEOUT_SEC = 180.0
# ======================================


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("socket closed")
        buf += chunk
    return buf


def recv_obj(sock: socket.socket) -> Any:
    header = _recv_exact(sock, 8)
    (size,) = struct.unpack("!Q", header)
    payload = _recv_exact(sock, size)
    return msgpack.unpackb(payload, raw=False)


def send_obj(sock: socket.socket, obj: Any) -> None:
    payload = msgpack.packb(obj, use_bin_type=True)
    header = struct.pack("!Q", len(payload))
    sock.sendall(header + payload)


def image_msg_to_hwc_uint8(msg: Image, target_h: int, target_w: int) -> np.ndarray:
    enc = (msg.encoding or "").lower()
    h, w = int(msg.height), int(msg.width)

    if enc in ("rgb8", "bgr8"):
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 3)
        if enc == "bgr8":
            img = img[..., ::-1]
    elif enc in ("rgba8", "bgra8"):
        img4 = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 4)
        if enc == "bgra8":
            img = img4[..., [2, 1, 0]]
        else:
            img = img4[..., :3]
    elif enc in ("mono8",):
        mono = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 1)
        img = np.repeat(mono, 3, axis=2)
    else:
        raise ValueError(f"Unsupported image encoding: '{msg.encoding}'")

    if (h, w) != (target_h, target_w):
        try:
            import cv2

            img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        except Exception:
            out = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            hh = min(target_h, h)
            ww = min(target_w, w)
            sy0 = (h - hh) // 2
            sx0 = (w - ww) // 2
            dy0 = (target_h - hh) // 2
            dx0 = (target_w - ww) // 2
            out[dy0 : dy0 + hh, dx0 : dx0 + ww] = img[sy0 : sy0 + hh, sx0 : sx0 + ww]
            img = out

    return img.astype(np.uint8, copy=False)


def blank_image(target_h: int, target_w: int) -> np.ndarray:
    return np.zeros((target_h, target_w, 3), dtype=np.uint8)


def make_three_camera_mosaic(
    img1: Optional[Image], img5: Optional[Image], img6: Optional[Image]
) -> np.ndarray:
    views = []
    for msg in (img1, img5, img6):
        if msg is None:
            views.append(blank_image(IMAGE_H, IMAGE_W))
        else:
            views.append(image_msg_to_hwc_uint8(msg, IMAGE_H, IMAGE_W))
    return np.concatenate(views, axis=1)


class RosBridge(Node):
    def __init__(self):
        super().__init__("gr00t_ros_bridge")

        sensor_qos = QoSProfile(depth=1)
        sensor_qos.reliability = ReliabilityPolicy.BEST_EFFORT
        sensor_qos.durability = DurabilityPolicy.VOLATILE

        self.joint_state: Optional[JointState] = None
        self.image_1: Optional[Image] = None
        self.image_5: Optional[Image] = None
        self.image_6: Optional[Image] = None
        self._lock = threading.Lock()

        self.create_subscription(JointState, TOPIC_JS_IN, self._cb_js, sensor_qos)
        self.create_subscription(Image, TOPIC_IMG_1, self._cb_img_1, sensor_qos)
        self.create_subscription(Image, TOPIC_IMG_5, self._cb_img_5, sensor_qos)
        self.create_subscription(Image, TOPIC_IMG_6, self._cb_img_6, sensor_qos)

        self.pub_js = self.create_publisher(JointState, TOPIC_JS_OUT, 1)
        self.pub_chunk = self.create_publisher(Float32MultiArray, TOPIC_CHUNK_OUT, 1)

        self.sock: Optional[socket.socket] = None
        self._connect()

        self._action_queue: deque[dict[str, np.ndarray]] = deque()
        self._last_action: Optional[dict[str, np.ndarray]] = None
        self._stop_event = threading.Event()
        self._infer_thread = threading.Thread(target=self._infer_loop, daemon=True)
        self._infer_thread.start()
        self.exec_timer = self.create_timer(1.0 / EXEC_HZ, self._exec_tick)

    def _connect(self):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(CONNECT_TIMEOUT_SEC)
            s.connect((SERVER_HOST, SERVER_PORT))
            s.settimeout(RPC_TIMEOUT_SEC)
            self.sock = s
            self.get_logger().info(f"Connected to GR00T server {SERVER_HOST}:{SERVER_PORT}")
        except Exception as e:
            self.sock = None
            self.get_logger().error(f"Failed to connect GR00T server: {e}")

    def _ensure_conn(self) -> bool:
        if self.sock is not None:
            return True
        self._connect()
        return self.sock is not None

    def _cb_js(self, msg: JointState):
        with self._lock:
            self.joint_state = msg

    def _cb_img_1(self, msg: Image):
        with self._lock:
            self.image_1 = msg

    def _cb_img_5(self, msg: Image):
        with self._lock:
            self.image_5 = msg

    def _cb_img_6(self, msg: Image):
        with self._lock:
            self.image_6 = msg

    def _extract_state(self, msg: JointState) -> dict[str, np.ndarray]:
        q = np.asarray(msg.position, dtype=np.float32)
        if q.shape[0] < PIPER_CMD_DIM:
            q = np.pad(q, (0, PIPER_CMD_DIM - q.shape[0]))
        else:
            q = q[:PIPER_CMD_DIM]

        # Minimal Piper -> GR1 mapping:
        # joint1..joint6 map to the first 6 right-arm channels, joint7(gripper) maps to right-hand[0].
        ordered = {
            "left_arm": np.zeros(GR1_STATE_LAYOUT["left_arm"], dtype=np.float32),
            "right_arm": np.concatenate([q[:6], np.zeros(1, dtype=np.float32)], axis=0),
            "left_hand": np.zeros(GR1_STATE_LAYOUT["left_hand"], dtype=np.float32),
            "right_hand": np.concatenate(
                [q[6:7], np.zeros(GR1_STATE_LAYOUT["right_hand"] - 1, dtype=np.float32)], axis=0
            ),
            "waist": np.zeros(GR1_STATE_LAYOUT["waist"], dtype=np.float32),
        }
        return ordered

    def _gr1_action_to_piper_cmd(self, action: dict[str, np.ndarray]) -> np.ndarray:
        cmd = np.zeros(PIPER_CMD_DIM, dtype=np.float32)
        cmd[:6] = np.asarray(action["right_arm"][:6], dtype=np.float32)
        cmd[6] = float(np.asarray(action["right_hand"], dtype=np.float32)[0])
        return cmd

    def _infer_loop(self) -> None:
        period = 1.0 / INFER_HZ
        next_t = time.perf_counter()

        while not self._stop_event.is_set():
            now = time.perf_counter()
            if now < next_t:
                time.sleep(next_t - now)
                continue
            next_t += period

            with self._lock:
                if len(self._action_queue) > ACTION_CHUNK_LEN:
                    continue
                joint_state = self.joint_state
                image_1 = self.image_1
                image_5 = self.image_5
                image_6 = self.image_6

            if joint_state is None:
                continue
            if not self._ensure_conn():
                continue

            try:
                req = {
                    "image": make_three_camera_mosaic(image_1, image_5, image_6),
                    "state": self._extract_state(joint_state),
                    "prompt": PROMPT_TEXT,
                }
                send_obj(self.sock, req)
                resp = recv_obj(self.sock)
            except Exception as e:
                self.get_logger().error(f"Socket error: {e} (reconnect)")
                try:
                    if self.sock:
                        self.sock.close()
                except Exception:
                    pass
                self.sock = None
                continue

            if not resp.get("ok", False):
                self.get_logger().error(f"GR00T server error: {resp}")
                continue

            actions = {
                key: np.asarray(value, dtype=np.float32)
                for key, value in resp["actions"].items()
            }
            horizon = next(iter(actions.values())).shape[0]
            chunk_len = min(ACTION_CHUNK_LEN, horizon)

            with self._lock:
                if len(self._action_queue) > ACTION_CHUNK_LEN:
                    self._action_queue.clear()
                for i in range(chunk_len):
                    self._action_queue.append({key: value[i] for key, value in actions.items()})
                self._last_action = {key: value[0] for key, value in actions.items()}

            flat = []
            for i in range(chunk_len):
                for key in GR1_STATE_LAYOUT:
                    flat.extend(actions[key][i].tolist())
            msg = Float32MultiArray()
            msg.data = flat
            self.pub_chunk.publish(msg)

    def _exec_tick(self) -> None:
        with self._lock:
            joint_state = self.joint_state
            if self._action_queue:
                action = self._action_queue.popleft()
                self._last_action = action
            else:
                action = self._last_action

        if action is None or joint_state is None:
            return

        cmd = self._gr1_action_to_piper_cmd(action)

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = PIPER_JOINT_NAMES
        msg.position = cmd.tolist()
        msg.velocity = [0.0] * 6 + [10.0]
        msg.effort = [0.0] * 6 + [0.5]
        self.pub_js.publish(msg)

    def shutdown(self) -> None:
        self._stop_event.set()
        try:
            if self._infer_thread.is_alive():
                self._infer_thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            if self.sock:
                self.sock.close()
        except Exception:
            pass
        self.sock = None


def main() -> None:
    rclpy.init()
    node = RosBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
