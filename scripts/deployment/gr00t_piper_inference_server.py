#!/usr/bin/env python3

from __future__ import annotations

import logging
import socket
import struct
from pathlib import Path
from typing import Any

import torch.nn.functional as F

import msgpack
import msgpack_numpy as m
import numpy as np
import torch
import tyro
from PIL import Image

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import MessageType
from gr00t.policy.gr00t_policy import Gr00tPolicy, _rec_to_dtype

# Register Piper modality config before policy creation
import examples.Piper.piper_config  # noqa: F401

m.patch()

DEFAULT_MODEL_PATH = Path("gr00t/model/gr00t_rgb_run/checkpoint-199800")
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 5555
PATCH_PIXELS = 28


class CaptureAttnProcessor2_0:
    def __init__(self, block_idx: int, attends_image: bool):
        self.block_idx = block_idx
        self.attends_image = attends_image
        self.last_attention_probs = None

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        temb: torch.Tensor | None = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        scores = torch.matmul(query, key.transpose(-1, -2)) * attn.scale
        if attention_mask is not None:
            scores = scores + attention_mask
        attention_probs = scores.softmax(dim=-1)
        self.last_attention_probs = attention_probs.detach().float().cpu()

        hidden_states = torch.matmul(attention_probs, value)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states



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
    def __init__(
        self,
        model_path: Path,
        device: str,
        show_images: bool = False,
        dump_image_saliency: bool = False,
        dump_action_attention: bool = False,
    ):
        self.policy = Gr00tPolicy(
            embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
            model_path=str(model_path),
            device=device,
        )
        self.show_images = show_images
        self.dump_image_saliency = dump_image_saliency
        self.dump_action_attention = dump_action_attention
        self.modality = self.policy.get_modality_config()
        self.video_keys = self.modality["video"].modality_keys
        self.state_keys = self.modality["state"].modality_keys   # ["arm", "gripper"]
        self.action_keys = self.modality["action"].modality_keys  # ["arm", "gripper"]
        self.language_key = self.modality["language"].modality_keys[0]
        self._action_attention_modules = []
        if self.dump_action_attention:
            self._install_action_attention_hooks()

    def _install_action_attention_hooks(self) -> None:
        dit_model = self.policy.model.action_head.model
        use_alternate = getattr(self.policy.model.action_head.config, "use_alternate_vl_dit", False)
        attend_text_every_n_blocks = getattr(dit_model, "attend_text_every_n_blocks", 1)

        for block_idx, block in enumerate(dit_model.transformer_blocks):
            attn = block.attn1
            if getattr(attn, "cross_attention_dim", None) is None:
                continue
            if getattr(attn, "_action_attention_hooked", False):
                self._action_attention_modules.append(attn)
                continue

            attends_image = True
            if use_alternate:
                attends_image = block_idx % (2 * attend_text_every_n_blocks) != 0

            processor = CaptureAttnProcessor2_0(block_idx=block_idx, attends_image=attends_image)
            attn.set_processor(processor)
            attn._action_attention_hooked = True
            attn._last_attention_processor = processor
            self._action_attention_modules.append(attn)

    def _reset_action_attention_cache(self) -> None:
        for module in self._action_attention_modules:
            processor = getattr(module, "_last_attention_processor", None)
            if processor is not None:
                processor.last_attention_probs = None

    def _select_action_attention_capture(self, encoder_seq_len: int):
        captures = []
        for module in self._action_attention_modules:
            processor = getattr(module, "_last_attention_processor", None)
            if processor is None or processor.last_attention_probs is None:
                continue
            probs = processor.last_attention_probs
            is_cross_attention = probs.shape[-1] == encoder_seq_len
            captures.append((processor.block_idx, bool(processor.attends_image), bool(is_cross_attention), module))

        if not captures:
            return None

        preferred = [item for item in captures if item[1] and item[2]]
        fallback = [item for item in captures if item[2]]
        target = max(preferred or fallback or captures, key=lambda item: item[0])
        module = target[3]
        processor = getattr(module, "_last_attention_processor", None)
        return processor.last_attention_probs, {"block_idx": processor.block_idx, "attends_image": processor.attends_image, "heads": module.heads}

    def _prepare_collated_inputs(self, obs: dict[str, Any]):
        single_obs = {
            "video": {k: v[0] for k, v in obs["video"].items()},
            "state": {k: v[0] for k, v in obs["state"].items()},
            "language": {k: v[0] for k, v in obs["language"].items()},
        }
        vla_step_data = self.policy._to_vla_step_data(single_obs)
        messages = [{"type": MessageType.EPISODE_STEP.value, "content": vla_step_data}]
        processed_inputs = self.policy.processor(messages)
        collated_inputs = self.policy.collate_fn([processed_inputs])
        return _rec_to_dtype(collated_inputs, dtype=torch.bfloat16)

    def _prepare_model_inputs(self, obs: dict[str, Any]):
        collated_inputs = self._prepare_collated_inputs(obs)
        model_inputs = collated_inputs["inputs"]
        return self.policy.model.prepare_input(model_inputs)

    def _predict_action_with_grad(self, backbone_inputs, action_inputs):
        model = self.policy.model
        backbone_outputs = model.backbone(backbone_inputs)
        features = model.action_head._encode_features(backbone_outputs, action_inputs)
        return model.action_head.get_action_with_features.__wrapped__(
            model.action_head,
            backbone_features=features.backbone_features,
            state_features=features.state_features,
            embodiment_id=action_inputs.embodiment_id,
            backbone_output=backbone_outputs,
        )

    def _normalize_map(self, heatmap: np.ndarray) -> np.ndarray:
        heatmap = heatmap.astype(np.float32, copy=False)
        heatmap = heatmap - heatmap.min()
        max_value = float(heatmap.max())
        if max_value > 0:
            heatmap = heatmap / max_value
        return heatmap

    def _save_overlay(self, base_path: Path, out_path: Path, heatmap: np.ndarray) -> None:
        if not base_path.exists():
            return
        base = np.asarray(Image.open(base_path).convert("RGB"), dtype=np.float32)
        heat = np.asarray(
            Image.fromarray((heatmap * 255).astype(np.uint8)).resize((base.shape[1], base.shape[0]), Image.BILINEAR),
            dtype=np.float32,
        ) / 255.0
        overlay = base.copy()
        overlay[..., 0] = np.clip(base[..., 0] * 0.5 + heat * 255.0 * 0.5, 0, 255)
        overlay[..., 1] = np.clip(base[..., 1] * 0.7, 0, 255)
        overlay[..., 2] = np.clip(base[..., 2] * 0.7, 0, 255)
        Image.fromarray(overlay.astype(np.uint8)).save(out_path)

    def _pixel_value_specs(self, pixel_values) -> list[dict[str, int]]:
        tensors = pixel_values if isinstance(pixel_values, list) else [pixel_values]
        specs = []
        for tensor in tensors:
            if tensor.ndim == 4:
                _, _, height, width = tensor.shape
            elif tensor.ndim == 5:
                _, _, _, height, width = tensor.shape
            else:
                raise ValueError(f"Unexpected pixel_values tensor shape: {tuple(tensor.shape)}")
            grid_h = int(height // PATCH_PIXELS)
            grid_w = int(width // PATCH_PIXELS)
            specs.append(
                {
                    "height": int(height),
                    "width": int(width),
                    "grid_h": grid_h,
                    "grid_w": grid_w,
                    "token_len": grid_h * grid_w,
                }
            )
        return specs

    def _dump_action_attention(self, backbone_inputs, action_inputs) -> None:
        dump_dir = Path("/tmp/gr00t_infer_debug")
        dump_dir.mkdir(parents=True, exist_ok=True)

        self._reset_action_attention_cache()
        with torch.inference_mode():
            backbone_outputs = self.policy.model.backbone(backbone_inputs)
            _ = self.policy.model.action_head.get_action(backbone_outputs, action_inputs)

        capture = self._select_action_attention_capture(int(backbone_outputs.image_mask.shape[-1]))
        if capture is None:
            logging.warning("Action expert returned no cross-attention tensors; skipping action-attention dump")
            return

        attention_probs, meta = capture
        batch_size = int(backbone_outputs.backbone_features.shape[0])
        heads = int(meta["heads"])
        try:
            attention_probs = attention_probs.reshape(batch_size, heads, attention_probs.shape[-2], attention_probs.shape[-1])
        except Exception as exc:
            logging.warning("Failed to reshape action attention tensor %s: %s", tuple(attention_probs.shape), exc)
            return

        key_scores = attention_probs.mean(dim=1).mean(dim=1)[0]
        image_mask = backbone_outputs.image_mask[0].detach().cpu().bool()
        if key_scores.shape[0] != image_mask.shape[0]:
            logging.warning(
                "Action attention key length %s does not match image mask length %s; skipping dump",
                key_scores.shape[0],
                image_mask.shape[0],
            )
            return

        image_token_scores = key_scores[image_mask].numpy()
        specs = self._pixel_value_specs(backbone_inputs["pixel_values"])
        total_expected_tokens = sum(spec["token_len"] for spec in specs)
        if image_token_scores.shape[0] != total_expected_tokens:
            logging.warning(
                "Image token count mismatch: attention has %s image tokens but pixel grids expect %s; skipping dump",
                image_token_scores.shape[0],
                total_expected_tokens,
            )
            return

        offset = 0
        for key, spec in zip(self.video_keys, specs):
            next_offset = offset + spec["token_len"]
            patch_map = image_token_scores[offset:next_offset].reshape(spec["grid_h"], spec["grid_w"])
            patch_map = self._normalize_map(patch_map)
            stem = key.replace('/', '_')
            np.save(dump_dir / f"{stem}_action_attn.npy", patch_map)
            self._save_overlay(
                dump_dir / f"{stem}.png",
                dump_dir / f"{stem}_action_attn_overlay.png",
                patch_map,
            )
            offset = next_offset

    def _dump_image_saliency(self, backbone_inputs, action_inputs) -> None:
        dump_dir = Path('/tmp/gr00t_infer_debug')
        dump_dir.mkdir(parents=True, exist_ok=True)

        pixel_values = backbone_inputs["pixel_values"]
        backbone_inputs = dict(backbone_inputs)
        if isinstance(pixel_values, list):
            grad_inputs = [pv.detach().clone().requires_grad_(True) for pv in pixel_values]
            backbone_inputs["pixel_values"] = grad_inputs
        else:
            grad_inputs = pixel_values.detach().clone().requires_grad_(True)
            backbone_inputs["pixel_values"] = grad_inputs

        model = self.policy.model
        model.zero_grad(set_to_none=True)

        with torch.enable_grad():
            action_outputs = self._predict_action_with_grad(backbone_inputs, action_inputs)
            target = action_outputs["action_pred"][0, 0].pow(2).sum()
            target.backward()

        if isinstance(grad_inputs, list):
            grads = torch.cat([g.grad.detach().float().cpu() for g in grad_inputs], dim=0)
        else:
            grads = grad_inputs.grad.detach().float().cpu()
        saliency = grads.abs().mean(dim=1)

        for idx, key in enumerate(self.video_keys):
            sal_np = self._normalize_map(saliency[idx].numpy())
            stem = key.replace('/', '_')
            np.save(dump_dir / f"{stem}_saliency.npy", sal_np)
            self._save_overlay(
                dump_dir / f"{stem}.png",
                dump_dir / f"{stem}_saliency_overlay.png",
                sal_np,
            )

    def _dump_model_input_images(self, backbone_inputs) -> None:
        dump_dir = Path('/tmp/gr00t_infer_debug')
        dump_dir.mkdir(parents=True, exist_ok=True)
        pixel_values = backbone_inputs["pixel_values"]
        tensors = pixel_values if isinstance(pixel_values, list) else [pixel_values]
        for key, tensor in zip(self.video_keys, tensors):
            image_tensor = tensor[0].detach().float().cpu()
            image_tensor = ((image_tensor + 1.0) * 127.5).clamp(0, 255)
            frame = image_tensor.permute(1, 2, 0).numpy().astype(np.uint8)
            Image.fromarray(frame).save(dump_dir / f"{key.replace('/', '_')}.png")

    def adapt_request(self, req: dict[str, Any]) -> dict[str, Any]:
        images = req["images"]
        full_state = np.asarray(req["state"], dtype=np.float32)  # (7,)
        instruction = str(req.get("prompt", "pick and place the target object"))
        obs_video = {}
        for key in self.video_keys:
            image = np.asarray(images[key], dtype=np.uint8)
            obs_video[key] = image.reshape(1, 1, *image.shape)

        # Split state into arm (6D) and gripper (1D) matching modality.json
        obs_state = {
            "arm": full_state[:6].reshape(1, 1, -1),
            "gripper": full_state[6:7].reshape(1, 1, -1),
        }

        obs = {
            "video": obs_video,
            "state": obs_state,
            "language": {self.language_key: [[instruction]]},
        }

        backbone_inputs = action_inputs = None
        if self.show_images or self.dump_image_saliency or self.dump_action_attention:
            backbone_inputs, action_inputs = self._prepare_model_inputs(obs)
        if self.show_images or self.dump_image_saliency or self.dump_action_attention:
            self._dump_model_input_images(backbone_inputs)
        if self.dump_image_saliency:
            self._dump_image_saliency(backbone_inputs, action_inputs)
        if self.dump_action_attention:
            self._dump_action_attention(backbone_inputs, action_inputs)

        return obs

    def infer(self, req: dict[str, Any]) -> dict[str, Any]:
        obs = self.adapt_request(req)
        action_chunk, _ = self.policy.get_action(obs)
        # Concatenate arm (6D) + gripper (1D) into "control" (7D) for bridge compatibility
        arm = action_chunk["arm"][0].astype(np.float32)      # (T, 6)
        gripper = action_chunk["gripper"][0].astype(np.float32)  # (T, 1)
        control = np.concatenate([arm, gripper], axis=-1)     # (T, 7)
        return {"ok": True, "actions": {"control": control}}


def main(
    model_path: Path = DEFAULT_MODEL_PATH,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    device: str = "cuda",
    show_images: bool = False,
    dump_image_saliency: bool = False,
    dump_action_attention: bool = False,
) -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    logging.info("Loading GR00T checkpoint from %s", model_path)
    server = Gr00tSocketInferenceServer(
        model_path=model_path,
        device=device,
        show_images=show_images,
        dump_image_saliency=dump_image_saliency,
        dump_action_attention=dump_action_attention,
    )
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
