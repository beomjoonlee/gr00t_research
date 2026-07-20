from __future__ import annotations

import socket
import struct
import threading
import time
from typing import Any, Optional

import cv2
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


def stamp_to_sec(msg: Any) -> float | None:
    header = getattr(msg, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return None
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class Gr00tPiperBridge(Node):
    def __init__(self) -> None:
        super().__init__("gr00t_piper_bridge")

        self.declare_parameter("server_host", "127.0.0.1")
        self.declare_parameter("server_port", 5555)
        # 학습 데이터는 EasyTrainer가 cv2.resize로 320x240 squish 저장 → 추론도 동일 전처리
        self.declare_parameter("image_h", 240)
        self.declare_parameter("image_w", 320)
        # 학습 language annotation과 정확히 일치해야 함 (datasets/{1,2,3}_augment/meta/tasks.jsonl):
        #   1(blue):  "Pick up the blue object from the table and place it on the white plate"
        #   2(red):   "Pick up the red object and place it on the white plate"
        #   3(green): "Pick up the green object and place it on the white plate"
        self.declare_parameter("prompt_text", "Pick up the blue object from the table and place it on the white plate")
        self.declare_parameter("topic_js_in", "/ec_robot_1/joint_states_single")
        # 2026-07 카메라 재배치: sensor_7=wrist(D405), sensor_8=right(D435), sensor_11=left(D435)
        # sensor id는 EasyTrainer DB에 따라 달라질 수 있음 → 실행 전 `ros2 topic list`로 확인
        self.declare_parameter("topic_image_wrist", "/ec_sensor_7/camera/color/image_rect_raw")
        self.declare_parameter("topic_image_right", "/ec_sensor_8/camera/color/image_raw")
        self.declare_parameter("topic_image_left", "/ec_sensor_11/camera/color/image_raw")
        self.declare_parameter("rotate_wrist_180", False)
        self.declare_parameter("rotate_right_180", False)
        self.declare_parameter("rotate_left_180", False)
        self.declare_parameter("topic_js_out", "/ec_robot_1/joint_states")
        self.declare_parameter("topic_chunk_out", "/gr00t/action_chunk")
        self.declare_parameter("exec_hz", 10.0)  # 데이터 수집 10Hz와 동일
        self.declare_parameter("action_chunk_len", 16)
        self.declare_parameter("exec_steps", 8)
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
        self.topic_image_wrist = str(self.get_parameter("topic_image_wrist").value)
        self.topic_image_right = str(self.get_parameter("topic_image_right").value)
        self.topic_image_left = str(self.get_parameter("topic_image_left").value)
        self.rotate_wrist_180 = bool(self.get_parameter("rotate_wrist_180").value)
        self.rotate_right_180 = bool(self.get_parameter("rotate_right_180").value)
        self.rotate_left_180 = bool(self.get_parameter("rotate_left_180").value)
        self.topic_js_out = str(self.get_parameter("topic_js_out").value)
        self.topic_chunk_out = str(self.get_parameter("topic_chunk_out").value)
        self.exec_hz = float(self.get_parameter("exec_hz").value)
        self.action_chunk_len = int(self.get_parameter("action_chunk_len").value)
        self.exec_steps = int(self.get_parameter("exec_steps").value)
        self.piper_joint_names = list(self.get_parameter("piper_joint_names").value)
        self.piper_cmd_dim = int(self.get_parameter("piper_cmd_dim").value)
        self.connect_timeout_sec = float(self.get_parameter("connect_timeout_sec").value)
        self.rpc_timeout_sec = float(self.get_parameter("rpc_timeout_sec").value)
        self.gripper_velocity = float(self.get_parameter("gripper_velocity").value)
        self.gripper_effort = float(self.get_parameter("gripper_effort").value)

        sensor_qos = QoSProfile(depth=1)
        sensor_qos.reliability = ReliabilityPolicy.BEST_EFFORT
        sensor_qos.durability = DurabilityPolicy.VOLATILE

        self.latest_joint_state: Optional[JointState] = None
        self.latest_image_wrist: Optional[Image] = None
        self.latest_image_right: Optional[Image] = None
        self.latest_image_left: Optional[Image] = None
        self._last_wait_warning_time = 0.0
        self._lock = threading.Lock()

        # 각 토픽을 독립적으로 구독 - 싱크 없이 각자의 최신 메시지만 저장.
        self.js_sub = self.create_subscription(
            JointState, self.topic_js_in, self._cb_joint_state, qos_profile=sensor_qos
        )
        self.img_wrist_sub = self.create_subscription(
            Image, self.topic_image_wrist, self._cb_image_wrist, qos_profile=sensor_qos
        )
        self.img_right_sub = self.create_subscription(
            Image, self.topic_image_right, self._cb_image_right, qos_profile=sensor_qos
        )
        self.img_left_sub = self.create_subscription(
            Image, self.topic_image_left, self._cb_image_left, qos_profile=sensor_qos
        )
        self.get_logger().info(
            "Camera topics: wrist=%s, right=%s, left=%s"
            % (self.topic_image_wrist, self.topic_image_right, self.topic_image_left)
        )

        self.pub_js = self.create_publisher(JointState, self.topic_js_out, 1)
        self.pub_chunk = self.create_publisher(Float32MultiArray, self.topic_chunk_out, 1)

        self.sock: Optional[socket.socket] = None
        self._connect()

        self._stop_event = threading.Event()
        self._sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
        self._sync_thread.start()

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

    def _cb_joint_state(self, msg: JointState) -> None:
        with self._lock:
            self.latest_joint_state = msg

    def _cb_image_wrist(self, msg: Image) -> None:
        with self._lock:
            self.latest_image_wrist = msg

    def _cb_image_right(self, msg: Image) -> None:
        with self._lock:
            self.latest_image_right = msg

    def _cb_image_left(self, msg: Image) -> None:
        with self._lock:
            self.latest_image_left = msg

    def _prepare_image(self, msg: Image, rotate_180: bool) -> np.ndarray:
        img = maybe_rotate_180(image_msg_to_hwc_uint8(msg), rotate_180)
        # EasyTrainer 기록과 동일한 squish resize (aspect ratio 무시)
        if img.shape[0] != self.image_h or img.shape[1] != self.image_w:
            img = cv2.resize(img, (self.image_w, self.image_h))
        return np.ascontiguousarray(img)

    def _sync_loop(self) -> None:
        exec_period = 1.0 / self.exec_hz

        while not self._stop_event.is_set():
            # --- 관측 수집: 각 토픽의 최신 메시지를 싱크 없이 그대로 사용 ---
            with self._lock:
                joint_state = self.latest_joint_state
                image_wrist = self.latest_image_wrist
                image_right = self.latest_image_right
                image_left = self.latest_image_left

            if joint_state is None or image_wrist is None or image_right is None or image_left is None:
                now = time.monotonic()
                if now - self._last_wait_warning_time > 2.0:
                    stamps = {
                        "joint": stamp_to_sec(joint_state),
                        "wrist": stamp_to_sec(image_wrist),
                        "right": stamp_to_sec(image_right),
                        "left": stamp_to_sec(image_left),
                    }
                    valid_stamps = {k: v for k, v in stamps.items() if v is not None}
                    stamp_summary = ", ".join(f"{k}={v:.3f}" for k, v in valid_stamps.items())
                    self.get_logger().warning(
                        "Latest observation incomplete, waiting"
                        + (f" ({stamp_summary})" if stamp_summary else "")
                    )
                    self._last_wait_warning_time = now
                time.sleep(0.1)
                continue
            if not self._ensure_conn():
                time.sleep(1.0)
                continue

            # --- 추론 요청 ---
            try:
                state = np.asarray(joint_state.position, dtype=np.float32)
                if state.shape[0] < self.piper_cmd_dim:
                    state = np.pad(state, (0, self.piper_cmd_dim - state.shape[0]))
                else:
                    state = state[: self.piper_cmd_dim]
                req = {
                    "images": {
                        "wrist": self._prepare_image(image_wrist, self.rotate_wrist_180),
                        "right": self._prepare_image(image_right, self.rotate_right_180),
                        "left": self._prepare_image(image_left, self.rotate_left_180),
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
            run_steps = min(self.exec_steps, chunk_len)

            # --- 디버그용 chunk 발행 ---
            chunk_msg = Float32MultiArray()
            chunk_msg.data = control[:chunk_len].reshape(-1).tolist()
            self.pub_chunk.publish(chunk_msg)

            # --- 동기 실행: action을 exec_hz로 순차 발행 ---
            for i in range(run_steps):
                if self._stop_event.is_set():
                    break
                cmd = control[i][: self.piper_cmd_dim]
                msg = JointState()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.name = self.piper_joint_names
                msg.position = cmd.tolist()
                msg.velocity = [0.0] * max(0, self.piper_cmd_dim - 1) + [self.gripper_velocity]
                msg.effort = [0.0] * max(0, self.piper_cmd_dim - 1) + [self.gripper_effort]
                self.pub_js.publish(msg)
                time.sleep(exec_period)

    def shutdown(self) -> None:
        self._stop_event.set()
        try:
            if self._sync_thread.is_alive():
                self._sync_thread.join(timeout=1.0)
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
