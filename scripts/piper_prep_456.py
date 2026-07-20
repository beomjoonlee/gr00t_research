#!/usr/bin/env python3
"""Preprocess new EasyTrainer datasets (dirs 4/5/6) -> GR00T-flavored LeRobot format.

One atomic pass per dataset, writing to `datasets/{4,5,6}_groot` (originals preserved):
  - fps: unify to 20Hz. dir4/dir5 are already 20Hz (keep all frames); dir6 is 30Hz
    -> downsample to 20Hz by dropping every 3rd frame (keep n where n%3 != 2).
  - reindex episodes 0..N-1 (already contiguous here) and rebuild global `index`,
    per-episode `frame_index`, and `timestamp` (= arange(n)/20).
  - language: single unified string per dir, all task_index -> 0.
  - parquet columns -> GR00T: observation.qpos->observation.state, action.joint->action,
    add annotation.human.action.task_description (=0); drop qvel/qeffort/eepos/succeed.
  - videos: re-encode H.264/yuv420p @20fps, rename folders sensor_{7,8,11}->wrist/right/left.
  - meta: modality.json (state arm[0:6]+gripper[6:7]), info.json (v2 path templates,
    video shape [480,640,3]); drop episodes_stats.jsonl (GR00T regenerates stats at train time).

Color labels (collector-confirmed; metadata/frames were unreliable):
    dir4 = blue, dir5 = green, dir6 = red
Camera mapping (unchanged): sensor_7 -> wrist, sensor_8 -> right, sensor_11 -> left
"""
import glob
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = "/home/wook/Airlab/gr00t_research/datasets"
OUT_SUFFIX = "_groot"
OUT_FPS = 20

CAM_MAP = {"sensor_7": "wrist", "sensor_8": "right", "sensor_11": "left"}
VIEW_ORDER = ["wrist", "right", "left"]
JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]
VIDEO_SHAPE = [480, 640, 3]  # H, W, C (actual, ffprobe-verified)

DATASETS = {
    "4": {"src_fps": 20, "lang": "Pick up the blue object from the table and place it on the white plate"},
    "5": {"src_fps": 20, "lang": "Pick up the green object from the table and place it on the white plate"},
    "6": {"src_fps": 30, "lang": "Pick up the red object from the table and place it on the white plate"},
}

# Collector-specified bad episodes to drop, by ORIGINAL episode_index.
# Result: 22-2 / 20-0 / 21-1 -> 20 episodes each (balanced).
DROP = {
    "4": {0, 8},
    "5": set(),
    "6": {3},
}


def keep_mask(n_frames, src_fps):
    """Row indices to keep when normalizing src_fps -> OUT_FPS."""
    if src_fps == OUT_FPS:
        return np.arange(n_frames)
    if src_fps == 30 and OUT_FPS == 20:  # 3:2 -> drop every 3rd frame (n%3==2)
        return np.array([n for n in range(n_frames) if n % 3 != 2])
    raise ValueError(f"unsupported fps conversion {src_fps}->{OUT_FPS}")


def ffprobe_nframes(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", path],
        capture_output=True, text=True,
    ).stdout.strip()
    return int(out) if out.isdigit() else None


def reencode_video(src, dst, src_fps):
    """Re-encode to H.264/yuv420p @OUT_FPS; drop every 3rd frame for 30->20."""
    if src_fps == OUT_FPS:
        vf = f"setpts=N/{OUT_FPS}/TB"
    else:  # 30 -> 20
        vf = f"select='not(eq(mod(n\\,3)\\,2))',setpts=N/{OUT_FPS}/TB"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", src,
         "-vf", vf, "-r", str(OUT_FPS),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", dst],
        check=True,
    )


def build_modality_json():
    return {
        "state": {"arm": {"start": 0, "end": 6}, "gripper": {"start": 6, "end": 7}},
        "action": {"arm": {"start": 0, "end": 6}, "gripper": {"start": 6, "end": 7}},
        "video": {v: {"original_key": f"observation.images.{v}"} for v in VIEW_ORDER},
        "annotation": {"human.action.task_description": {"original_key": "task_index"}},
    }


def build_info_features():
    vec7 = lambda: {"dtype": "float32", "shape": [7], "names": JOINT_NAMES}
    scalar_i = lambda: {"dtype": "int64", "shape": [1], "names": None}
    video_feat = lambda: {"dtype": "video", "shape": VIDEO_SHAPE,
                          "names": ["height", "width", "channels"]}
    feats = {"observation.state": vec7(), "action": vec7()}
    for v in VIEW_ORDER:
        feats[f"observation.images.{v}"] = video_feat()
    feats["timestamp"] = {"dtype": "float32", "shape": [1], "names": None}
    feats["annotation.human.action.task_description"] = scalar_i()
    feats["task_index"] = scalar_i()
    feats["episode_index"] = scalar_i()
    feats["index"] = scalar_i()
    feats["frame_index"] = scalar_i()
    return feats


def process(name, cfg):
    src_dir = os.path.join(ROOT, name)
    out_dir = os.path.join(ROOT, f"{name}{OUT_SUFFIX}")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(os.path.join(out_dir, "data/chunk-000"))
    os.makedirs(os.path.join(out_dir, "meta"))
    for v in VIEW_ORDER:
        os.makedirs(os.path.join(out_dir, f"videos/chunk-000/observation.images.{v}"))

    src_fps, lang = cfg["src_fps"], cfg["lang"]

    # order episodes by their existing episode_index
    src_parquets = sorted(
        glob.glob(os.path.join(src_dir, "data/chunk-000/*.parquet")),
        key=lambda p: int(pd.read_parquet(p, columns=["episode_index"])["episode_index"].iloc[0]),
    )

    # drop the collector-specified bad episodes (by original episode_index)
    drop = DROP.get(name, set())
    usable = []
    for src_pq in src_parquets:
        old_ep = int(pd.read_parquet(src_pq, columns=["episode_index"])["episode_index"].iloc[0])
        if old_ep in drop:
            print(f"[ds{name}] DROP old_ep {old_ep}")
            continue
        usable.append((old_ep, src_pq))

    episodes_meta = []
    running_index = 0
    total_frames = 0

    for new_ep, (old_ep, src_pq) in enumerate(usable):
        df = pd.read_parquet(src_pq).reset_index(drop=True)
        keep = keep_mask(len(df), src_fps)
        df = df.iloc[keep].reset_index(drop=True)
        n = len(df)

        out = pd.DataFrame()
        out["observation.state"] = df["observation.qpos"]
        out["action"] = df["action.joint"]
        out["timestamp"] = (np.arange(n, dtype=np.float32) / OUT_FPS).astype(np.float32)
        out["annotation.human.action.task_description"] = np.int64(0)
        out["task_index"] = np.int64(0)
        out["episode_index"] = np.int64(new_ep)
        out["index"] = np.arange(running_index, running_index + n, dtype=np.int64)
        out["frame_index"] = np.arange(n, dtype=np.int64)
        out.to_parquet(os.path.join(out_dir, f"data/chunk-000/episode_{new_ep:06d}.parquet"))

        for sensor, view in CAM_MAP.items():
            src_v = os.path.join(src_dir, f"videos/chunk-000/observation.images.{sensor}/episode_{old_ep:06d}.mp4")
            dst_v = os.path.join(out_dir, f"videos/chunk-000/observation.images.{view}/episode_{new_ep:06d}.mp4")
            reencode_video(src_v, dst_v, src_fps)
            got = ffprobe_nframes(dst_v)
            assert got == n, f"{name} ep{new_ep} {view}: video frames {got} != parquet {n}"

        episodes_meta.append({"episode_index": new_ep, "tasks": [lang], "length": n})
        running_index += n
        total_frames += n

    meta = os.path.join(out_dir, "meta")
    with open(os.path.join(meta, "tasks.jsonl"), "w") as f:
        f.write(json.dumps({"task_index": 0, "task": lang}) + "\n")
    with open(os.path.join(meta, "episodes.jsonl"), "w") as f:
        for e in episodes_meta:
            f.write(json.dumps(e) + "\n")
    with open(os.path.join(meta, "modality.json"), "w") as f:
        json.dump(build_modality_json(), f, indent=4)

    info = json.load(open(os.path.join(src_dir, "meta/info.json")))
    n_ep = len(episodes_meta)
    info["fps"] = OUT_FPS
    info["total_episodes"] = n_ep
    info["total_frames"] = total_frames
    info["total_tasks"] = 1
    info["total_videos"] = n_ep * len(VIEW_ORDER)
    info["total_chunks"] = 1
    info["splits"] = {"train": f"0:{n_ep}"}
    info["features"] = build_info_features()
    for stale in ("action_key", "tool_qpos_indices"):
        info.pop(stale, None)
    info["data_path"] = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    info["video_path"] = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    with open(os.path.join(meta, "info.json"), "w") as f:
        json.dump(info, f, indent=4)

    print(f"[ds{name}] -> {os.path.basename(out_dir)}: {n_ep} eps, {total_frames} frames @ {OUT_FPS}Hz, "
          f"lang='{lang}'")


if __name__ == "__main__":
    for t in sys.argv[1:] or list(DATASETS):
        process(t, DATASETS[t])
