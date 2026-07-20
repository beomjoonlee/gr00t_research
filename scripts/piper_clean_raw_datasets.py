#!/usr/bin/env python3
"""Clean EasyTrainer raw LeRobot datasets (dirs 1/2/3) for GR00T prep.

Per-dataset operations (keeps the 3 directories separate, LeRobot layout intact):
  - Reindex episodes to a contiguous 0..N-1 range (closes gaps like missing ep15).
  - Rebuild global `index` (0..total-1), per-episode `frame_index`, and `timestamp`.
  - Optionally downsample 20Hz -> 10Hz (every 2nd frame; re-encodes videos w/ ffmpeg).
  - Unify language: single task string, all `task_index` -> 0.
  - Rewrite meta/{tasks,episodes,episodes_stats}.jsonl and info.json accordingly.

Original data is preserved in /home/wook/Downloads/datasets.tar.gz.
Each dataset is built into a temp dir then atomically swapped in.
"""
import glob
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = "/home/wook/Downloads/datasets"
CAMS = ["sensor_7", "sensor_8", "sensor_11"]
STAT_KEYS = [
    "observation.qpos",
    "action.joint",
    "observation.qvel",
    "observation.qeffort",
    "observation.eepos",
]

DATASETS = {
    "1": {
        "downsample": True,  # 20Hz -> 10Hz
        "lang": "Pick up the blue object from the table and place it on the white plate",
    },
    "2": {
        "downsample": False,
        "lang": "Pick up the red object and place it on the white plate",
    },
    "3": {
        "downsample": False,
        "lang": "Pick up the green object and place it on the white plate",
    },
}
OUT_FPS = 10


def ffprobe_nframes(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-count_frames", "-show_entries", "stream=nb_read_frames",
         "-of", "csv=p=0", path],
        capture_output=True, text=True,
    ).stdout.strip()
    return int(out) if out.isdigit() else None


def downsample_video(src, dst):
    """Keep frames 0,2,4,... and re-time to 10 fps (H.264)."""
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", src,
         "-vf", "select='not(mod(n,2))',setpts=N/10/TB",
         "-r", str(OUT_FPS), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", dst],
        check=True,
    )


def compute_stats(df):
    stats = {}
    for k in STAT_KEYS:
        if k not in df.columns:
            continue
        arr = np.stack(df[k].values).astype(np.float64)
        stats[k] = {
            "min": arr.min(0).tolist(),
            "max": arr.max(0).tolist(),
            "mean": arr.mean(0).tolist(),
            "std": arr.std(0).tolist(),
            "count": [int(arr.shape[0])],
        }
    return stats


def process(name, cfg):
    src_dir = os.path.join(ROOT, name)
    tmp_dir = os.path.join(ROOT, f"{name}__tmp")
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    for sub in ["data/chunk-000", "meta"] + [f"videos/chunk-000/observation.images.{c}" for c in CAMS]:
        os.makedirs(os.path.join(tmp_dir, sub), exist_ok=True)

    downsample = cfg["downsample"]
    lang = cfg["lang"]

    # order episodes by their existing episode_index (files may have gaps)
    src_parquets = sorted(
        glob.glob(os.path.join(src_dir, "data/chunk-000/*.parquet")),
        key=lambda p: int(pd.read_parquet(p, columns=["episode_index"])["episode_index"].iloc[0]),
    )

    episodes_meta = []
    episodes_stats = []
    running_index = 0
    total_frames = 0

    for new_ep, src_pq in enumerate(src_parquets):
        df = pd.read_parquet(src_pq).reset_index(drop=True)
        old_ep = int(df["episode_index"].iloc[0])

        if downsample:
            df = df.iloc[0::2].reset_index(drop=True)

        n = len(df)
        df["episode_index"] = new_ep
        df["frame_index"] = np.arange(n, dtype=np.int64)
        df["index"] = np.arange(running_index, running_index + n, dtype=np.int64)
        df["timestamp"] = (np.arange(n, dtype=np.float32) / OUT_FPS).astype(np.float32)
        df["task_index"] = 0
        running_index += n
        total_frames += n

        df.to_parquet(os.path.join(tmp_dir, f"data/chunk-000/episode_{new_ep:06d}.parquet"))

        # videos: rename to new_ep, downsample if needed
        for cam in CAMS:
            src_v = os.path.join(src_dir, f"videos/chunk-000/observation.images.{cam}/episode_{old_ep:06d}.mp4")
            dst_v = os.path.join(tmp_dir, f"videos/chunk-000/observation.images.{cam}/episode_{new_ep:06d}.mp4")
            if downsample:
                downsample_video(src_v, dst_v)
                got = ffprobe_nframes(dst_v)
                assert got == n, f"{name} ep{new_ep} {cam}: video frames {got} != parquet {n}"
            else:
                shutil.copy2(src_v, dst_v)

        episodes_meta.append({"episode_index": new_ep, "length": n, "tasks": [lang]})
        episodes_stats.append({"episode_index": new_ep, "stats": compute_stats(df)})

    # meta files
    meta_out = os.path.join(tmp_dir, "meta")
    with open(os.path.join(meta_out, "tasks.jsonl"), "w") as f:
        f.write(json.dumps({"task_index": 0, "task": lang}) + "\n")
    with open(os.path.join(meta_out, "episodes.jsonl"), "w") as f:
        for e in episodes_meta:
            f.write(json.dumps(e) + "\n")
    with open(os.path.join(meta_out, "episodes_stats.jsonl"), "w") as f:
        for e in episodes_stats:
            f.write(json.dumps(e) + "\n")

    # info.json
    info = json.load(open(os.path.join(src_dir, "meta/info.json")))
    n_ep = len(episodes_meta)
    info["fps"] = OUT_FPS
    info["total_episodes"] = n_ep
    info["total_frames"] = total_frames
    info["total_tasks"] = 1
    info["total_videos"] = n_ep * len(CAMS)
    info["total_chunks"] = 1
    info["splits"] = {"train": f"0:{n_ep}"}
    # fix ds1's wrong declared video shape (actual is 320x240 -> [H,W,C])
    for cam in CAMS:
        key = f"observation.images.{cam}"
        if key in info["features"]:
            info["features"][key]["shape"] = [240, 320, 3]
    json.dump(info, open(os.path.join(meta_out, "info.json"), "w"), indent=2)

    # swap in
    backup = os.path.join(ROOT, f"{name}__orig")
    if os.path.exists(backup):
        shutil.rmtree(backup)
    os.rename(src_dir, backup)
    os.rename(tmp_dir, src_dir)
    print(f"[ds{name}] done: {n_ep} eps, {total_frames} frames, lang='{lang}' "
          f"(orig kept at {os.path.basename(backup)})")


if __name__ == "__main__":
    targets = sys.argv[1:] or list(DATASETS)
    for t in targets:
        process(t, DATASETS[t])
