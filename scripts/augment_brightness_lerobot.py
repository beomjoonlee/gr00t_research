"""Brightness augmentation for a piper LeRobot dataset.

Per task (language annotation), takes the first N episodes (where N == len(brightness_levels))
and assigns brightness levels in order. Episodes that share a within-task index across
different tasks therefore receive the same brightness.

Brightness adjust matches augment_dataset.py reference:
    factor = 1 + lightness / 100   (PIL ImageEnhance.Brightness)

Examples
--------
v2 (drops the 21st red episode automatically):
    python augment_brightness_lerobot.py \\
        --src data/piper_pick_place_v2 \\
        --dst data/piper_pick_place_v2_augment \\
        --bright-min -50 --bright-max 50 --bright-step 5 --skip-zero

v3:
    python augment_brightness_lerobot.py \\
        --src data/piper_pick_place_v3 \\
        --dst data/piper_pick_place_v3_augment \\
        --bright-min -30 --bright-max 70 --bright-step 5 --skip-zero
"""

import argparse
import json
import shutil
import subprocess
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image, ImageEnhance


CHUNK = "chunk-000"


def load_video_keys(src: Path) -> list[str]:
    """Derive video folder keys from the dataset's modality.json (original_key values),
    so the script adapts to whatever camera layout the dataset uses."""
    with open(src / "meta" / "modality.json") as f:
        modality = json.load(f)
    return [v["original_key"] for v in modality["video"].values()]


def build_brightness_levels(bmin: int, bmax: int, step: int, skip_zero: bool) -> list[int]:
    levels = list(range(bmin, bmax + 1, step))
    if skip_zero:
        levels = [v for v in levels if v != 0]
    return levels


def adjust_brightness(frame_rgb: np.ndarray, lightness: int) -> np.ndarray:
    if lightness == 0:
        return frame_rgb
    img = Image.fromarray(frame_rgb)
    img = ImageEnhance.Brightness(img).enhance(1 + lightness / 100.0)
    return np.asarray(img)


def process_video(src_path: Path, dst_path: Path, lightness: int, fps: int) -> int:
    """Decode mp4 with cv2, apply brightness, re-encode as h264/yuv420p via ffmpeg pipe."""
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {src_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ffmpeg_cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{width}x{height}",
        "-pix_fmt", "rgb24",
        "-r", str(fps),
        "-i", "-",
        "-an",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(dst_path),
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    n = 0
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            out = adjust_brightness(frame_rgb, lightness)
            proc.stdin.write(np.ascontiguousarray(out).tobytes())
            n += 1
    finally:
        cap.release()
        proc.stdin.close()
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"ffmpeg failed with exit code {rc} for {dst_path}")
    return n


def rewrite_parquet(src_path: Path, dst_path: Path, new_ep_idx: int, global_offset: int) -> int:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    table = pq.read_table(src_path)
    df = table.to_pandas()
    n = len(df)
    df["episode_index"] = np.int64(new_ep_idx)
    df["index"] = np.arange(global_offset, global_offset + n, dtype=np.int64)
    new_table = pa.Table.from_pandas(df, preserve_index=False).cast(table.schema)
    pq.write_table(new_table, dst_path)
    return n


def build_plan(src: Path, levels: list[int]):
    """Group episodes by task (in first-seen order), take first N per task, pair with levels."""
    grouped: "OrderedDict[str, list[dict]]" = OrderedDict()
    with open(src / "meta" / "episodes.jsonl") as f:
        for line in f:
            e = json.loads(line)
            task = e["tasks"][0]
            grouped.setdefault(task, []).append(e)

    n = len(levels)
    plan = []  # (new_ep, old_ep, lightness, task_str)
    new_idx = 0
    for task, eps in grouped.items():
        if len(eps) < n:
            raise SystemExit(
                f"Task {task!r} has {len(eps)} episodes but {n} brightness levels are needed."
            )
        if len(eps) > n:
            print(f"NOTE: task {task!r} has {len(eps)} eps, dropping last {len(eps) - n}")
        for i in range(n):
            plan.append((new_idx, eps[i]["episode_index"], levels[i], task))
            new_idx += 1
    return plan, grouped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst", type=Path, required=True)
    parser.add_argument("--bright-min", type=int, required=True)
    parser.add_argument("--bright-max", type=int, required=True)
    parser.add_argument("--bright-step", type=int, default=5)
    parser.add_argument("--skip-zero", action="store_true",
                        help="exclude 0 from the brightness level list")
    args = parser.parse_args()

    src: Path = args.src
    dst: Path = args.dst

    if dst.exists():
        raise SystemExit(f"Destination already exists, refusing to overwrite: {dst}")

    levels = build_brightness_levels(args.bright_min, args.bright_max, args.bright_step,
                                     args.skip_zero)
    print(f"Brightness levels ({len(levels)}): {levels}")

    plan, _ = build_plan(src, levels)
    total = len(plan)
    print(f"Total new episodes: {total}")

    with open(src / "meta" / "info.json") as f:
        info = json.load(f)
    fps = info["fps"]

    video_keys = load_video_keys(src)
    print(f"Video keys: {video_keys}")

    (dst / "meta").mkdir(parents=True)
    (dst / "data" / CHUNK).mkdir(parents=True)
    for key in video_keys:
        (dst / "videos" / CHUNK / key).mkdir(parents=True)

    shutil.copy(src / "meta" / "tasks.jsonl", dst / "meta" / "tasks.jsonl")
    shutil.copy(src / "meta" / "modality.json", dst / "meta" / "modality.json")

    new_episodes = []
    global_offset = 0
    for new_ep, old_ep, lightness, task in plan:
        print(f"[{new_ep:03d}/{total}] old={old_ep:03d}  brightness={lightness:+d}")

        video_frame_counts = []
        for key in video_keys:
            src_v = src / "videos" / CHUNK / key / f"episode_{old_ep:06d}.mp4"
            dst_v = dst / "videos" / CHUNK / key / f"episode_{new_ep:06d}.mp4"
            video_frame_counts.append(process_video(src_v, dst_v, lightness, fps))

        src_p = src / "data" / CHUNK / f"episode_{old_ep:06d}.parquet"
        dst_p = dst / "data" / CHUNK / f"episode_{new_ep:06d}.parquet"
        n_rows = rewrite_parquet(src_p, dst_p, new_ep, global_offset)

        if not all(c == n_rows for c in video_frame_counts):
            print(f"  WARN: parquet rows={n_rows} videos={video_frame_counts}")

        new_episodes.append({
            "episode_index": new_ep,
            "tasks": [task],
            "length": n_rows,
        })
        global_offset += n_rows

    with open(dst / "meta" / "episodes.jsonl", "w") as f:
        for e in new_episodes:
            f.write(json.dumps(e) + "\n")

    info["total_episodes"] = len(new_episodes)
    info["total_frames"] = global_offset
    info["total_videos"] = len(new_episodes) * len(video_keys)
    info["splits"] = {"train": f"0:{len(new_episodes)}"}
    with open(dst / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=4)

    print(f"\nDone. {len(new_episodes)} episodes, {global_offset} frames -> {dst}")


if __name__ == "__main__":
    main()
