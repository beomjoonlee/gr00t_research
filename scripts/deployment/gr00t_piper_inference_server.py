#!/usr/bin/env python3

from __future__ import annotations

import logging
import socket
import struct
from pathlib import Path
from typing import Any

import msgpack
import msgpack_numpy as m
import numpy as np
import tyro

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy

m.patch()

DEFAULT_MODEL_PATH = Path("gr00t/model/gr00t_rgb_run/checkpoint-200000")
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 5555


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


class Gr00tSocketInferenceServer:
    def __init__(self, model_path: Path, device: str):
        self.policy = Gr00tPolicy(
            embodiment_tag=EmbodimentTag.GR1,
            model_path=str(model_path),
            device=device,
        )
        self.modality = self.policy.get_modality_config()
        self.video_key = self.modality["video"].modality_keys[0]
        self.state_keys = self.modality["state"].modality_keys
        self.language_key = self.modality["language"].modality_keys[0]

    def adapt_request(self, req: dict[str, Any]) -> dict[str, Any]:
        image = np.asarray(req["image"], dtype=np.uint8)
        state = req["state"]
        instruction = str(req.get("prompt", "pick and place the target object"))

        obs_state: dict[str, np.ndarray] = {}
        for key in self.state_keys:
            value = np.asarray(state[key], dtype=np.float32).reshape(1, 1, -1)
            obs_state[key] = value

        return {
            "video": {self.video_key: image.reshape(1, 1, *image.shape)},
            "state": obs_state,
            "language": {self.language_key: [[instruction]]},
        }

    def infer(self, req: dict[str, Any]) -> dict[str, Any]:
        obs = self.adapt_request(req)
        action_chunk, _ = self.policy.get_action(obs)
        serializable = {key: value[0].astype(np.float32) for key, value in action_chunk.items()}
        return {"ok": True, "actions": serializable}


def main(
    model_path: Path = DEFAULT_MODEL_PATH,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    device: str = "cuda",
) -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    logging.info("Loading GR00T checkpoint from %s", model_path)
    server = Gr00tSocketInferenceServer(model_path=model_path, device=device)
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
                resp = server.infer(req)
                send_obj(conn, resp)
        except Exception as exc:
            logging.info("Client disconnected / error: %s", exc)
        finally:
            try:
                conn.close()
            except Exception:
                pass


if __name__ == "__main__":
    tyro.cli(main)
