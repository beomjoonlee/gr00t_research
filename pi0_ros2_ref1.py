#!/usr/bin/env python3
import socket
import struct
import msgpack
import msgpack_numpy as m
import threading
import time
import sys
from collections import deque
from typing import Optional, Any, Dict, Deque

import numpy as np

m.patch()

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from sensor_msgs.msg import JointState, Image

# ====== EDIT HERE (no CLI params) ======
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 5555

IMAGE_H = 240
IMAGE_W = 320
PROMPT_TEXT = "push the button"

# Left wrist image fallback. If enabled, cam_left_wrist always uses this image.
USE_LEFT_BACKGROUND_IMAGE = True
LEFT_BACKGROUND_IMAGE_PATH = "/opt/easytrainer/project/src/backend/scripts/left_background.jpg"

# Configuration for active components (True = Real, False = Dummy/Zero-padded)
ENABLE_ROBOT_LEFT = False   # ec_robot_4
ENABLE_ROBOT_RIGHT = True   # ec_robot_7

ENABLE_CAM_HIGH = True      # ec_sensor_1
ENABLE_CAM_LW = False       # ec_sensor_3
ENABLE_CAM_RW = True        # ec_sensor_2

# Updated topics based on previous context (Left=4, Right=7)
TOPIC_JS1_IN = "/ec_robot_4/joint_states_single"
TOPIC_JS2_IN = "/ec_robot_7/joint_states_single"

# Camera topics (Using the ones from the provided snippet, assuming they are correct for the new setup)
TOPIC_CAM_HIGH = "/ec_sensor_1/camera/color/image_raw"
TOPIC_CAM_LW = "/ec_sensor_3/camera/color/image_raw"
TOPIC_CAM_RW = "/ec_sensor_2/camera/color/image_raw"

TOPIC_JS1_OUT = "/ec_robot_4/joint_states"
TOPIC_JS2_OUT = "/ec_robot_7/joint_states"

# Desired rates
INFER_HZ = 0.5       # run model inference at 10 Hz (every 100 ms)
EXEC_HZ = 50       # apply actions at 100 Hz (every 10 ms)
ACTION_CHUNK_LEN = 50 # take first N actions from the horizon for the next 100 ms

EXEC_PERIOD_SEC = 1.0 / EXEC_HZ

ROBOT_DOF = 7
EXPECTED_ACTION_DIM = 2 * ROBOT_DOF

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
    return msgpack.unpackb(payload)


def send_obj(sock: socket.socket, obj: Any) -> None:
    payload = msgpack.packb(obj)
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

    # resize (cv2 있으면 resize, 없으면 center crop/pad)
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
            out[dy0:dy0 + hh, dx0:dx0 + ww] = img[sy0:sy0 + hh, sx0:sx0 + ww]
            img = out

    return img.astype(np.uint8, copy=False)


def get_padded_arm_data(msg: Optional[JointState], dof: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns (pos, effort) for an arm. Zero-pads if msg is None."""
    if msg is None:
        return np.zeros(dof, dtype=np.float32), np.zeros(dof, dtype=np.float32)
    
    pos = np.asarray(msg.position, dtype=np.float32)
    if len(pos) >= dof:
        pos = pos[:dof]
    else:
        pos = np.pad(pos, (0, dof - len(pos)))
        
    if msg.effort:
        eff = np.asarray(msg.effort, dtype=np.float32)
        if len(eff) >= dof:
            eff = eff[:dof]
        else:
            eff = np.pad(eff, (0, dof - len(eff)))
    else:
        eff = np.zeros(dof, dtype=np.float32)
        
    return pos, eff

def get_padded_image(msg: Optional[Image], h: int, w: int) -> np.ndarray:
    """Returns HWC uint8 image. Returns black image if msg is None."""
    if msg is None:
        return np.zeros((h, w, 3), dtype=np.uint8)
    return image_msg_to_hwc_uint8(msg, h, w)


def load_background_image(path: str, h: int, w: int) -> np.ndarray:
    try:
        import cv2

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to read background image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        return img.astype(np.uint8, copy=False)
    except Exception:
        return np.zeros((h, w, 3), dtype=np.uint8)


class RosBridge(Node):
    def __init__(self):
        super().__init__("openpi_ros_bridge")

        sensor_qos = QoSProfile(depth=1)
        sensor_qos.reliability = ReliabilityPolicy.BEST_EFFORT
        sensor_qos.durability = DurabilityPolicy.VOLATILE

        self.js1: Optional[JointState] = None
        self.js2: Optional[JointState] = None
        self.im_high: Optional[Image] = None
        self.im_lw: Optional[Image] = None
        self.im_rw: Optional[Image] = None

        self._lock = threading.Lock()
        self._left_background = load_background_image(LEFT_BACKGROUND_IMAGE_PATH, IMAGE_H, IMAGE_W)

        if ENABLE_ROBOT_LEFT:
            self.create_subscription(JointState, TOPIC_JS1_IN, self._cb_js1, sensor_qos)
        if ENABLE_ROBOT_RIGHT:
            self.create_subscription(JointState, TOPIC_JS2_IN, self._cb_js2, sensor_qos)
        if ENABLE_CAM_HIGH:
            self.create_subscription(Image, TOPIC_CAM_HIGH, self._cb_high, sensor_qos)
        if ENABLE_CAM_LW:
            self.create_subscription(Image, TOPIC_CAM_LW, self._cb_lw, sensor_qos)
        if ENABLE_CAM_RW:
            self.create_subscription(Image, TOPIC_CAM_RW, self._cb_rw, sensor_qos)

        # 100 Hz publish: keep pub queue small to avoid command backlog.
        self.pub_js1 = self.create_publisher(JointState, TOPIC_JS1_OUT, 1) if ENABLE_ROBOT_LEFT else None
        self.pub_js2 = self.create_publisher(JointState, TOPIC_JS2_OUT, 1) if ENABLE_ROBOT_RIGHT else None

        self.sock: Optional[socket.socket] = None
        self._connect()

        # Action queue: each element is (14,) float32 for one 10ms step at 100Hz.
        self._action_queue: Deque[np.ndarray] = deque()
        self._queue_max = int(2 * ACTION_CHUNK_LEN)  # buffer up to two chunks
        self._effort_history: Deque[np.ndarray] = deque(maxlen=10)
        self._last_action: Optional[np.ndarray] = None

        self._stop_event = threading.Event()
        self._infer_thread = threading.Thread(target=self._infer_loop, daemon=True)
        self._infer_thread.start()

        # 100 Hz execution timer
        self.exec_timer = self.create_timer(EXEC_PERIOD_SEC, self._exec_tick)

        self.get_logger().info(
            f"ROS bridge started (infer={INFER_HZ:.1f}Hz, exec={EXEC_HZ:.1f}Hz, chunk={ACTION_CHUNK_LEN})."
        )

    def _connect(self):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(CONNECT_TIMEOUT_SEC)
            s.connect((SERVER_HOST, SERVER_PORT))
            s.settimeout(RPC_TIMEOUT_SEC)
            self.sock = s
            self.get_logger().info(f"Connected to OpenPI server {SERVER_HOST}:{SERVER_PORT}")
        except Exception as e:
            self.sock = None
            self.get_logger().error(f"Failed to connect OpenPI server: {e}")

    def _ensure_conn(self) -> bool:
        if self.sock is not None:
            return True
        self._connect()
        return self.sock is not None

    def _cb_js1(self, msg: JointState):
        with self._lock:
            self.js1 = msg

    def _cb_js2(self, msg: JointState):
        with self._lock:
            self.js2 = msg

    def _cb_high(self, msg: Image):
        with self._lock:
            self.im_high = msg

    def _cb_lw(self, msg: Image):
        with self._lock:
            self.im_lw = msg

    def _cb_rw(self, msg: Image):
        with self._lock:
            self.im_rw = msg

    def _infer_loop(self) -> None:
        """Run inference at INFER_HZ and enqueue ACTION_CHUNK_LEN actions for 100 Hz execution."""
        period = 1.0 / float(INFER_HZ)
        next_t = time.perf_counter()

        while not self._stop_event.is_set():
            now = time.perf_counter()
            if now < next_t:
                time.sleep(next_t - now)
                continue
            next_t += period

            # Snapshot obs + queue length
            with self._lock:
                qlen = len(self._action_queue)
                js1 = self.js1
                js2 = self.js2
                im_high = self.im_high
                im_lw = self.im_lw
                im_rw = self.im_rw

            # If we already have enough buffered actions, skip this cycle.
            # (This allows pipelining: while executing current chunk, compute next chunk.)
            if qlen > ACTION_CHUNK_LEN:
                continue

            # Check only enabled components
            if ENABLE_ROBOT_LEFT and js1 is None: continue
            if ENABLE_ROBOT_RIGHT and js2 is None: continue
            if ENABLE_CAM_HIGH and im_high is None: continue
            if ENABLE_CAM_LW and im_lw is None: continue
            if ENABLE_CAM_RW and im_rw is None: continue

            if not self._ensure_conn():
                continue

            # state: robot1 -> robot2 (slice to 7-DOF each)
            q1, eff1 = get_padded_arm_data(js1, ROBOT_DOF)
            q2, eff2 = get_padded_arm_data(js2, ROBOT_DOF)
            
            state = np.concatenate([q1, q2], axis=0).astype(np.float32, copy=False)
            effort_now = np.concatenate([eff1, eff2], axis=0).astype(np.float32, copy=False)

            self._effort_history.append(effort_now)
            # Fill history if not enough data
            while len(self._effort_history) < 10:
                self._effort_history.appendleft(effort_now)
            effort = np.stack(self._effort_history, axis=0)

            try:
                cam_high = get_padded_image(im_high, IMAGE_H, IMAGE_W)
                cam_lw = self._left_background if USE_LEFT_BACKGROUND_IMAGE else get_padded_image(im_lw, IMAGE_H, IMAGE_W)
                cam_rw = get_padded_image(im_rw, IMAGE_H, IMAGE_W)
            except Exception as e:
                self.get_logger().error(f"Image conversion failed: {e}")
                continue

            req: Dict[str, Any] = {
                "images": {
                    "cam_high": cam_high,
                    "cam_left_wrist": cam_lw,
                    "cam_right_wrist": cam_rw,
                },
                "image_masks": {
                    "cam_high": np.array(True, dtype=bool),
                    "cam_left_wrist": np.array(True, dtype=bool),
                    "cam_right_wrist": np.array(True, dtype=bool),
                },
                "state": state,
                "effort": effort,
                "prompt": PROMPT_TEXT,
            }

            t0 = time.perf_counter()
            try:
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
                self.get_logger().error(f"OpenPI server error: {resp.get('err')}")
                continue

            # Preferred response: {"actions": (N,14)}. Backward compat: {"action": (14,)}.
            if "actions" in resp:
                actions = np.asarray(resp["actions"], dtype=np.float32)
                if actions.ndim == 1:
                    actions = actions.reshape(1, -1)
            else:
                actions = np.asarray(resp["action"], dtype=np.float32).reshape(1, -1)

            if actions.ndim != 2 or actions.shape[1] < EXPECTED_ACTION_DIM:
                self.get_logger().error(f"Unexpected actions shape: {actions.shape}")
                continue

            chunk_len = min(int(ACTION_CHUNK_LEN), int(actions.shape[0]))
            chunk = actions[:chunk_len, :EXPECTED_ACTION_DIM]

            with self._lock:
                # Prevent unbounded growth. If backlog exists, drop old commands and keep the most recent.
                if len(self._action_queue) > self._queue_max - chunk_len:
                    self._action_queue.clear()

                for i in range(chunk_len):
                    a = np.asarray(chunk[i], dtype=np.float32).reshape(-1)
                    self._action_queue.append(a)
                    self._last_action = a

            dt_ms = (time.perf_counter() - t0) * 1000.0
            if dt_ms > 150.0:
                self.get_logger().warn(f"Inference RPC took {dt_ms:.1f} ms")

    def _exec_tick(self) -> None:
        """Publish one action at 100 Hz. If queue is empty, hold the last action."""
        with self._lock:
            js1 = self.js1
            js2 = self.js2
            if self._action_queue:
                action = self._action_queue.popleft()
                self._last_action = action
            else:
                action = self._last_action

        if action is None:
            return
        if (ENABLE_ROBOT_LEFT and js1 is None) or (ENABLE_ROBOT_RIGHT and js2 is None):
            return

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] < EXPECTED_ACTION_DIM:
            return

        a1 = action[:ROBOT_DOF]
        a2 = action[ROBOT_DOF:2 * ROBOT_DOF]

        now = self.get_clock().now().to_msg()

        if ENABLE_ROBOT_LEFT and self.pub_js1:
            m1 = JointState()
            m1.header.stamp = now
            m1.name = list(js1.name)[:ROBOT_DOF]
            m1.position = a1.tolist()
            self.pub_js1.publish(m1)

        if ENABLE_ROBOT_RIGHT and self.pub_js2:
            m2 = JointState()
            m2.header.stamp = now
            m2.name = list(js2.name)[:ROBOT_DOF]
            m2.position = a2.tolist()
            self.pub_js2.publish(m2)

        self.get_logger().info(f"Action: {action}")

    def shutdown(self) -> None:
        self._stop_event.set()
        try:
            if hasattr(self, "_infer_thread") and self._infer_thread.is_alive():
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
