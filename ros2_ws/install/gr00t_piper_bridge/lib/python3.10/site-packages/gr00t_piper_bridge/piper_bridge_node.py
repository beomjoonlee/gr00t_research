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


def image_msg_to_hwc_uint8(msg: Image) -> np.ndarray:
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
    elif enc == "mono8":
        mono = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 1)
        img = np.repeat(mono, 3, axis=2)
    else:
        raise ValueError(f"Unsupported image encoding: '{msg.encoding}'")

    return img.astype(np.uint8, copy=False)


def blank_image(target_h: int, target_w: int) -> np.ndarray:
    return np.zeros((target_h, target_w, 3), dtype=np.uint8)


def maybe_rotate_180(img: np.ndarray, enabled: bool) -> np.ndarray:
    if not enabled:
        return img
    return np.ascontiguousarray(img[::-1, ::-1])


class Gr00tPiperBridge(Node):
    def __init__(self) -> None:
        super().__init__("gr00t_piper_bridge")

        self.declare_parameter("server_host", "127.0.0.1")
        self.declare_parameter("server_port", 5555)
        self.declare_parameter("image_h", 256)
        self.declare_parameter("image_w", 256)
        self.declare_parameter("prompt_text", "pick up the red object from the table and put it on the white plate")
        self.declare_parameter("topic_js_in", "/ec_robot_1/joint_states_single")
        self.declare_parameter("topic_image_1", "/ec_sensor_3/camera/color/image_raw")
        self.declare_parameter("topic_image_2", "/ec_sensor_2/camera/color/image_raw")
        self.declare_parameter("topic_image_3", "/ec_sensor_1/camera/color/image_raw")
        self.declare_parameter("rotate_image_1_180", False)
        self.declare_parameter("rotate_image_2_180", False)
        self.declare_parameter("rotate_image_3_180", False)
        self.declare_parameter("topic_js_out", "/ec_robot_1/joint_states")
        self.declare_parameter("topic_chunk_out", "/gr00t/action_chunk")
        self.declare_parameter("infer_hz", 2.0)
        self.declare_parameter("exec_hz", 5.0)
        self.declare_parameter("action_chunk_len", 16)
        self.declare_parameter(
            "piper_joint_names",
            ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"],
        )
        self.declare_parameter("piper_cmd_dim", 7)
        self.declare_parameter("connect_timeout_sec", 2.0)
        self.declare_parameter("rpc_timeout_sec", 180.0)
        self.declare_parameter("gripper_velocity", 10.0)
        self.declare_parameter("gripper_effort", 0.5)

        self.server_host = self.get_parameter("server_host").value
        self.server_port = int(self.get_parameter("server_port").value)
        self.image_h = int(self.get_parameter("image_h").value)
        self.image_w = int(self.get_parameter("image_w").value)
        self.prompt_text = str(self.get_parameter("prompt_text").value)
        self.topic_js_in = str(self.get_parameter("topic_js_in").value)
        self.topic_image_1 = str(self.get_parameter("topic_image_1").value)
        self.topic_image_2 = str(self.get_parameter("topic_image_2").value)
        self.topic_image_3 = str(self.get_parameter("topic_image_3").value)
        self.rotate_image_1_180 = bool(self.get_parameter("rotate_image_1_180").value)
        self.rotate_image_2_180 = bool(self.get_parameter("rotate_image_2_180").value)
        self.rotate_image_3_180 = bool(self.get_parameter("rotate_image_3_180").value)
        self.topic_js_out = str(self.get_parameter("topic_js_out").value)
        self.topic_chunk_out = str(self.get_parameter("topic_chunk_out").value)
        self.infer_hz = float(self.get_parameter("infer_hz").value)
        self.exec_hz = float(self.get_parameter("exec_hz").value)
        self.action_chunk_len = int(self.get_parameter("action_chunk_len").value)
        self.piper_joint_names = list(self.get_parameter("piper_joint_names").value)
        self.piper_cmd_dim = int(self.get_parameter("piper_cmd_dim").value)
        self.connect_timeout_sec = float(self.get_parameter("connect_timeout_sec").value)
        self.rpc_timeout_sec = float(self.get_parameter("rpc_timeout_sec").value)
        self.gripper_velocity = float(self.get_parameter("gripper_velocity").value)
        self.gripper_effort = float(self.get_parameter("gripper_effort").value)

        sensor_qos = QoSProfile(depth=1)
        sensor_qos.reliability = ReliabilityPolicy.BEST_EFFORT
        sensor_qos.durability = DurabilityPolicy.VOLATILE

        self.joint_state: Optional[JointState] = None
        self.image_1: Optional[Image] = None
        self.image_2: Optional[Image] = None
        self.image_3: Optional[Image] = None
        self._lock = threading.Lock()

        self.create_subscription(JointState, self.topic_js_in, self._cb_js, sensor_qos)
        self.create_subscription(Image, self.topic_image_1, self._cb_img_1, sensor_qos)
        self.create_subscription(Image, self.topic_image_2, self._cb_img_2, sensor_qos)
        self.create_subscription(Image, self.topic_image_3, self._cb_img_3, sensor_qos)

        self.pub_js = self.create_publisher(JointState, self.topic_js_out, 1)
        self.pub_chunk = self.create_publisher(Float32MultiArray, self.topic_chunk_out, 1)

        self.sock: Optional[socket.socket] = None
        self._connect()

        self._action_queue: deque[np.ndarray] = deque()
        self._last_action: Optional[np.ndarray] = None
        self._stop_event = threading.Event()
        self._infer_thread = threading.Thread(target=self._infer_loop, daemon=True)
        self._infer_thread.start()
        self.exec_timer = self.create_timer(1.0 / self.exec_hz, self._exec_tick)

    def _connect(self) -> None:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.connect_timeout_sec)
            sock.connect((self.server_host, self.server_port))
            sock.settimeout(self.rpc_timeout_sec)
            self.sock = sock
            self.get_logger().info(
                f"Connected to GR00T server {self.server_host}:{self.server_port}"
            )
        except Exception as exc:
            self.sock = None
            self.get_logger().error(f"Failed to connect GR00T server: {exc}")

    def _ensure_conn(self) -> bool:
        if self.sock is not None:
            return True
        self._connect()
        return self.sock is not None

    def _cb_js(self, msg: JointState) -> None:
        with self._lock:
            self.joint_state = msg

    def _cb_img_1(self, msg: Image) -> None:
        with self._lock:
            self.image_1 = msg

    def _cb_img_2(self, msg: Image) -> None:
        with self._lock:
            self.image_2 = msg

    def _cb_img_3(self, msg: Image) -> None:
        with self._lock:
            self.image_3 = msg

    def _infer_loop(self) -> None:
        period = 1.0 / self.infer_hz
        next_t = time.perf_counter()

        while not self._stop_event.is_set():
            now = time.perf_counter()
            if now < next_t:
                time.sleep(next_t - now)
                continue
            next_t += period

            with self._lock:
                if len(self._action_queue) > 12:
                    continue
                joint_state = self.joint_state
                image_1 = self.image_1
                image_2 = self.image_2
                image_3 = self.image_3

            if joint_state is None:
                continue
            if not self._ensure_conn():
                continue

            try:
                state = np.asarray(joint_state.position, dtype=np.float32)
                if state.shape[0] < self.piper_cmd_dim:
                    state = np.pad(state, (0, self.piper_cmd_dim - state.shape[0]))
                else:
                    state = state[: self.piper_cmd_dim]
                req = {
                    "images": {
                        "rgb.head_256_256": maybe_rotate_180(
                            image_msg_to_hwc_uint8(image_1), self.rotate_image_1_180
                        )
                        if image_1 is not None
                        else blank_image(self.image_h, self.image_w),
                        "rgb.left_wrist_256_256": maybe_rotate_180(
                            image_msg_to_hwc_uint8(image_2), self.rotate_image_2_180
                        )
                        if image_2 is not None
                        else blank_image(self.image_h, self.image_w),
                        "rgb.right_wrist_256_256": maybe_rotate_180(
                            image_msg_to_hwc_uint8(image_3), self.rotate_image_3_180
                        )
                        if image_3 is not None
                        else blank_image(self.image_h, self.image_w),
                    },
                    "state": state,
                    "prompt": self.prompt_text,
                }
                send_obj(self.sock, req)
                resp = recv_obj(self.sock)
            except Exception as exc:
                self.get_logger().error(f"Socket error: {exc} (reconnect)")
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
                key: np.asarray(value, dtype=np.float32) for key, value in resp["actions"].items()
            }
            control = actions["control"]
            chunk_len = min(self.action_chunk_len, control.shape[0])

            with self._lock:
                if len(self._action_queue) > self.action_chunk_len:
                    self._action_queue.clear()
                for i in range(chunk_len):
                    self._action_queue.append(control[i].copy())
                self._last_action = control[0].copy()

            msg = Float32MultiArray()
            msg.data = control[:chunk_len].reshape(-1).tolist()
            self.pub_chunk.publish(msg)

    def _exec_tick(self) -> None:
        with self._lock:
            joint_state = self.joint_state
            if self._action_queue:
                action = self._action_queue.popleft()
                self._last_action = action
            else:
                action = None

        if action is None or joint_state is None:
            return

        cmd = np.asarray(action, dtype=np.float32)[: self.piper_cmd_dim]

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.piper_joint_names
        msg.position = cmd.tolist()
        msg.velocity = [0.0] * max(0, self.piper_cmd_dim - 1) + [self.gripper_velocity]
        msg.effort = [0.0] * max(0, self.piper_cmd_dim - 1) + [self.gripper_effort]
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
    rclpy.init(args=None)
    node = Gr00tPiperBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
