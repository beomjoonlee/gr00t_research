#!/usr/bin/env python3
"""Socket inference server compatible with EasyCollector one-arm bridge."""

import logging
import socket
import struct
from pathlib import Path
from typing import Any

import msgpack
import msgpack_numpy as m
import numpy as np
import tyro

from openpi.policies import policy_config
from openpi.training import config as config_loader

m.patch()

DEFAULT_CONFIG = "seohan_13_push"
DEFAULT_CHECKPOINT = Path("/workspace/TA-VLA/checkpoints/seohan_13_push/push_button_v1/29999")
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 5555
RIGHT_DOF = 7
FULL_DOF = 14
ACTION_CHUNK_LEN = 50


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


def adapt_request(req: dict) -> dict:
    state = np.asarray(req["state"], dtype=np.float32).reshape(-1)
    effort = np.asarray(req["effort"], dtype=np.float32)
    images = {
        "cam_high": np.asarray(req["images"]["cam_high"], dtype=np.uint8),
        "cam_left_wrist": np.asarray(req["images"]["cam_left_wrist"], dtype=np.uint8),
        "cam_right_wrist": np.asarray(req["images"]["cam_right_wrist"], dtype=np.uint8),
    }
    obs = {
        "images": images,
        "prompt": req.get("prompt"),
        "state": state[-RIGHT_DOF:],
        "effort": effort[..., -RIGHT_DOF:],
    }
    if "actions" in req:
        actions = np.asarray(req["actions"], dtype=np.float32)
        obs["actions"] = actions[..., -RIGHT_DOF:]
    return obs


def adapt_actions(actions: np.ndarray) -> np.ndarray:
    right = np.asarray(actions, dtype=np.float32)
    if right.ndim == 1:
        right = right.reshape(1, -1)
    left = np.zeros((right.shape[0], FULL_DOF - RIGHT_DOF), dtype=np.float32)
    return np.concatenate([left, right[:, :RIGHT_DOF]], axis=1)


def main(
    config_name: str = DEFAULT_CONFIG,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    default_prompt: str = "push the button",
) -> None:
    logging.basicConfig(level=logging.INFO, force=True)

    config = config_loader.get_config(config_name)
    logging.info("Loading checkpoint from %s", checkpoint_path)
    policy = policy_config.create_trained_policy(config, str(checkpoint_path), default_prompt=default_prompt)
    logging.info("Policy loaded. Listening on %s:%s", host, port)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)

    while True:
        conn, addr = srv.accept()
        logging.info("Client connected: %s", addr)
        try:
            while True:
                req = recv_obj(conn)
                obs = adapt_request(req)
                out = policy.infer(obs)
                actions = np.asarray(out["actions"], dtype=np.float32)
                wrapped = adapt_actions(actions)
                chunk = wrapped[: min(ACTION_CHUNK_LEN, wrapped.shape[0])]
                send_obj(conn, {"ok": True, "actions": chunk, "chunk_len": int(chunk.shape[0])})
        except Exception as exc:
            logging.info("Client disconnected / error: %s", exc)
        finally:
            try:
                conn.close()
            except Exception:
                pass


if __name__ == "__main__":
    tyro.cli(main)