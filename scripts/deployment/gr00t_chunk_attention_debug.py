#!/usr/bin/env python3

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import tyro
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import MessageType
from gr00t.policy.gr00t_policy import Gr00tPolicy, _rec_to_dtype
from gr00t.utils.video_utils import get_frames_by_indices


DEFAULT_MODEL_PATH = Path("gr00t/model/gr00t_rgb_run/checkpoint-199800")
DEFAULT_CHUNK_DIR = Path("gr00t/data/chunk")
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


class Gr00tChunkAttentionDebugger:
    def __init__(
        self,
        model_path: Path,
        device: str,
        output_dir: Path,
        show_images: bool = True,
        dump_image_saliency: bool = False,
        dump_action_attention: bool = True,
    ):
        self.policy = Gr00tPolicy(
            embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
            model_path=str(model_path),
            device=device,
        )
        self.output_dir = output_dir
        self.show_images = show_images
        self.dump_image_saliency = dump_image_saliency
        self.dump_action_attention = dump_action_attention
        self.modality = self.policy.get_modality_config()
        self.video_keys = self.modality["video"].modality_keys
        self.state_key = self.modality["state"].modality_keys[0]
        self.action_key = self.modality["action"].modality_keys[0]
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

            attends_image = True
            if use_alternate:
                attends_image = block_idx % (2 * attend_text_every_n_blocks) != 0

            processor = CaptureAttnProcessor2_0(block_idx=block_idx, attends_image=attends_image)
            attn.set_processor(processor)
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
        return processor.last_attention_probs, {
            "block_idx": processor.block_idx,
            "attends_image": processor.attends_image,
            "heads": module.heads,
        }

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
            stem = key.replace("/", "_")
            np.save(self.output_dir / f"{stem}_action_attn.npy", patch_map)
            self._save_overlay(
                self.output_dir / f"{stem}.png",
                self.output_dir / f"{stem}_action_attn_overlay.png",
                patch_map,
            )
            offset = next_offset

    def _dump_image_saliency(self, backbone_inputs, action_inputs) -> None:
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
            stem = key.replace("/", "_")
            np.save(self.output_dir / f"{stem}_saliency.npy", sal_np)
            self._save_overlay(
                self.output_dir / f"{stem}.png",
                self.output_dir / f"{stem}_saliency_overlay.png",
                sal_np,
            )

    def _dump_model_input_images(self, backbone_inputs) -> None:
        pixel_values = backbone_inputs["pixel_values"]
        tensors = pixel_values if isinstance(pixel_values, list) else [pixel_values]

        for key, tensor in zip(self.video_keys, tensors):
            image_tensor = tensor[0].detach().float().cpu()
            image_tensor = ((image_tensor + 1.0) * 127.5).clamp(0, 255)
            image = image_tensor.permute(1, 2, 0).numpy().astype(np.uint8)
            Image.fromarray(image).save(self.output_dir / f"{key.replace('/', '_')}.png")

    def build_observation(
        self,
        images: dict[str, np.ndarray],
        state: np.ndarray,
        instruction: str,
    ) -> dict[str, Any]:
        obs_video = {}
        for key in self.video_keys:
            image = np.asarray(images[key], dtype=np.uint8)
            obs_video[key] = image.reshape(1, 1, *image.shape)

        obs = {
            "video": obs_video,
            "state": {self.state_key: np.asarray(state, dtype=np.float32).reshape(1, 1, -1)},
            "language": {self.language_key: [[instruction]]},
        }

        backbone_inputs, action_inputs = self._prepare_model_inputs(obs)
        if self.show_images or self.dump_image_saliency or self.dump_action_attention:
            self._dump_model_input_images(backbone_inputs)
        if self.dump_image_saliency:
            self._dump_image_saliency(backbone_inputs, action_inputs)
        if self.dump_action_attention:
            self._dump_action_attention(backbone_inputs, action_inputs)

        return obs


def _ensure_numpy_frame(frame: np.ndarray) -> np.ndarray:
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError(f"Expected RGB frame with shape (H, W, 3), got {frame.shape}")
    return frame


def _load_parquet_row(chunk_dir: Path, episode_idx: int, frame_idx: int) -> tuple[pd.DataFrame, pd.Series]:
    parquet_path = chunk_dir / f"episode_{episode_idx:06d}.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing parquet file: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    if frame_idx < 0 or frame_idx >= len(df):
        raise IndexError(
            f"frame_idx={frame_idx} is out of range for {parquet_path} with {len(df)} rows"
        )
    return df, df.iloc[frame_idx]


def _load_views(
    chunk_dir: Path,
    episode_idx: int,
    frame_idx: int,
    view_dir_names: dict[str, str],
) -> dict[str, np.ndarray]:
    frames = {}
    for logical_name, directory_name in view_dir_names.items():
        video_path = chunk_dir / directory_name / f"episode_{episode_idx:06d}.mp4"
        if not video_path.exists():
            raise FileNotFoundError(f"Missing video file for {logical_name}: {video_path}")
        frame = get_frames_by_indices(str(video_path), [frame_idx])[0]
        frames[logical_name] = _ensure_numpy_frame(np.asarray(frame))
    return frames


def _build_request_images(model_video_keys: list[str], frames: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    key_map = {
        "rgb.head_256_256": "head",
        "rgb.left_wrist_256_256": "left",
        "rgb.right_wrist_256_256": "right",
    }
    images = {}
    for model_key in model_video_keys:
        logical_name = key_map.get(model_key)
        if logical_name is None:
            raise KeyError(f"Unsupported model video key: {model_key}")
        images[model_key] = frames[logical_name]
    return images


def _save_action_plot(
    pred_action: np.ndarray,
    gt_action: np.ndarray,
    out_path: Path,
) -> None:
    indices = np.arange(len(gt_action))
    width = 0.38

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    axes[0].bar(indices - width / 2, gt_action, width=width, label="GT")
    axes[0].bar(indices + width / 2, pred_action, width=width, label="Pred")
    axes[0].set_ylabel("Joint Value")
    axes[0].set_title("Prediction vs Ground Truth")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.2)

    abs_error = np.abs(pred_action - gt_action)
    axes[1].bar(indices, abs_error, width=0.6, color="tab:red")
    axes[1].set_ylabel("Absolute Error")
    axes[1].set_xlabel("Joint Index")
    axes[1].set_title("Absolute Error")
    axes[1].grid(axis="y", alpha=0.2)

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _save_summary(
    out_dir: Path,
    episode_idx: int,
    frame_idx: int,
    prompt: str,
    pred_action: np.ndarray,
    gt_action: np.ndarray,
    pred_action_chunk: np.ndarray,
    gt_action_chunk: np.ndarray,
) -> None:
    abs_error = np.abs(pred_action - gt_action)
    summary = {
        "episode_idx": int(episode_idx),
        "frame_idx": int(frame_idx),
        "prompt": prompt,
        "pred_action_chunk_shape": list(pred_action_chunk.shape),
        "gt_action_chunk_shape": list(gt_action_chunk.shape),
        "pred_action": pred_action.tolist(),
        "gt_action": gt_action.tolist(),
        "abs_error": abs_error.tolist(),
        "l1_mean": float(abs_error.mean()),
        "l2": float(np.linalg.norm(pred_action - gt_action)),
        "max_abs_error": float(abs_error.max()),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    np.save(out_dir / "pred_action.npy", pred_action)
    np.save(out_dir / "gt_action.npy", gt_action)
    np.save(out_dir / "pred_action_chunk.npy", pred_action_chunk)
    np.save(out_dir / "gt_action_chunk.npy", gt_action_chunk)
    np.save(out_dir / "abs_error.npy", abs_error)
    _save_action_plot(pred_action, gt_action, out_dir / "action_compare.png")


def run(
    chunk_dir: Path = DEFAULT_CHUNK_DIR,
    model_path: Path = DEFAULT_MODEL_PATH,
    episode_idx: int = 0,
    frame_idx: int = 0,
    prompt: str = "pick and place the target object",
    output_dir: Path = Path("/tmp/gr00t_chunk_debug"),
    device: str = "cuda",
    show_images: bool = True,
    dump_image_saliency: bool = False,
    dump_action_attention: bool = True,
    head_dir_name: str = "head",
    left_dir_name: str = "left",
    right_dir_name: str = "wrist",
) -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    torch.set_grad_enabled(False)

    df, row = _load_parquet_row(chunk_dir, episode_idx, frame_idx)
    state = np.asarray(row["observation.state"], dtype=np.float32)
    gt_action = np.asarray(row["action"], dtype=np.float32)

    frames = _load_views(
        chunk_dir=chunk_dir,
        episode_idx=episode_idx,
        frame_idx=frame_idx,
        view_dir_names={
            "head": head_dir_name,
            "left": left_dir_name,
            "right": right_dir_name,
        },
    )

    step_dir = output_dir / f"episode_{episode_idx:06d}" / f"frame_{frame_idx:06d}"
    step_dir.mkdir(parents=True, exist_ok=True)

    debugger = Gr00tChunkAttentionDebugger(
        model_path=model_path,
        device=device,
        output_dir=step_dir,
        show_images=show_images,
        dump_image_saliency=dump_image_saliency,
        dump_action_attention=dump_action_attention,
    )

    images = _build_request_images(debugger.video_keys, frames)
    obs = debugger.build_observation(images=images, state=state, instruction=prompt)
    action_chunk, _ = debugger.policy.get_action(obs)

    pred_action_chunk = np.asarray(action_chunk[debugger.action_key][0], dtype=np.float32)
    if pred_action_chunk.ndim == 1:
        pred_action_chunk = pred_action_chunk[None, :]

    gt_end_idx = min(frame_idx + pred_action_chunk.shape[0], len(df))
    gt_action_chunk = np.stack(df.iloc[frame_idx:gt_end_idx]["action"].to_list(), axis=0).astype(np.float32)
    if gt_action_chunk.shape[0] < pred_action_chunk.shape[0]:
        pad_count = pred_action_chunk.shape[0] - gt_action_chunk.shape[0]
        pad = np.repeat(gt_action_chunk[-1][None, :], pad_count, axis=0)
        gt_action_chunk = np.concatenate([gt_action_chunk, pad], axis=0)

    pred_action = pred_action_chunk[0]
    _save_summary(
        out_dir=step_dir,
        episode_idx=episode_idx,
        frame_idx=frame_idx,
        prompt=prompt,
        pred_action=pred_action,
        gt_action=gt_action,
        pred_action_chunk=pred_action_chunk,
        gt_action_chunk=gt_action_chunk,
    )

    logging.info("Saved debug outputs to %s", step_dir)
    logging.info("Pred action: %s", np.array2string(pred_action, precision=5))
    logging.info("GT action:   %s", np.array2string(gt_action, precision=5))
    logging.info("Abs error:   %s", np.array2string(np.abs(pred_action - gt_action), precision=5))
    logging.info("Episode length: %d", len(df))


if __name__ == "__main__":
    tyro.cli(run)
