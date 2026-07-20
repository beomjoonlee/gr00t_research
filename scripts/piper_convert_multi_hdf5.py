"""
Convert multiple Piper HDF5 dataset directories into a single GR00T LeRobot v2 dataset.

Usage:
    python scripts/piper_convert_multi_hdf5.py \
        --input-dirs /path/to/ds3 /path/to/ds4 /path/to/ds5 \
        --tasks "pick up red" "pick up blue" "pick up green" \
        --output-dir /path/to/output \
        --skip-episodes 3:11
"""

import argparse
import json
import os

import cv2
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

CAMERA_MAP = {
    "sensor_2": "observation.images.side",
    "sensor_3": "observation.images.top",
    "sensor_4": "observation.images.wrist",
}

FPS = 10


def images_to_mp4(images: np.ndarray, output_path: str, fps: int):
    """Write (T, H, W, 3) uint8 BGR array to MP4 via OpenCV."""
    T, H, W, C = images.shape
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
    for t in range(T):
        writer.write(images[t])  # Already BGR, write directly
    writer.release()


def build_parquet(qpos, qaction, episode_idx, global_offset, task_idx, fps):
    T = qpos.shape[0]
    rows = {
        "observation.state": [qpos[t].astype(np.float32).tolist() for t in range(T)],
        "action": [qaction[t].astype(np.float32).tolist() for t in range(T)],
        "timestamp": [t / fps for t in range(T)],
        "annotation.human.action.task_description": [task_idx] * T,
        "task_index": [task_idx] * T,
        "episode_index": [episode_idx] * T,
        "index": list(range(global_offset, global_offset + T)),
        "frame_index": list(range(T)),
        "next.done": [False] * (T - 1) + [True],
        "next.reward": [0.0] * T,
    }
    schema = pa.schema([
        ("observation.state", pa.list_(pa.float32())),
        ("action", pa.list_(pa.float32())),
        ("timestamp", pa.float32()),
        ("annotation.human.action.task_description", pa.int64()),
        ("task_index", pa.int64()),
        ("episode_index", pa.int64()),
        ("index", pa.int64()),
        ("frame_index", pa.int64()),
        ("next.done", pa.bool_()),
        ("next.reward", pa.float32()),
    ])
    return pa.table(rows, schema=schema)


def parse_skip_episodes(skip_str_list):
    """Parse skip episodes like '3:11' meaning skip episode 11 from input dir index 3."""
    skips = {}
    if not skip_str_list:
        return skips
    for s in skip_str_list:
        dir_idx, ep_idx = s.split(":")
        dir_idx, ep_idx = int(dir_idx), int(ep_idx)
        if dir_idx not in skips:
            skips[dir_idx] = set()
        skips[dir_idx].add(ep_idx)
    return skips


def main():
    parser = argparse.ArgumentParser(description="Convert multiple Piper HDF5 dirs to GR00T LeRobot v2")
    parser.add_argument("--input-dirs", nargs="+", required=True, help="HDF5 directories in order")
    parser.add_argument("--tasks", nargs="+", required=True, help="Task description per input dir")
    parser.add_argument("--output-dir", required=True, help="Output LeRobot dataset directory")
    parser.add_argument("--skip-episodes", nargs="*", default=[], help="Skip episodes as dir_idx:ep_idx (e.g. 0:11)")
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    assert len(args.input_dirs) == len(args.tasks), "Number of input dirs must match number of tasks"

    skips = parse_skip_episodes(args.skip_episodes)
    fps = args.fps
    output_dir = args.output_dir

    # Deduplicate tasks and build task list
    unique_tasks = list(dict.fromkeys(args.tasks))  # preserve order, remove duplicates
    task_to_idx = {t: i for i, t in enumerate(unique_tasks)}

    # Create directories
    data_dir = os.path.join(output_dir, "data", "chunk-000")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "meta"), exist_ok=True)
    for cam_key in CAMERA_MAP.values():
        os.makedirs(os.path.join(output_dir, "videos", "chunk-000", cam_key), exist_ok=True)

    episodes_meta = []
    global_index = 0
    total_frames = 0
    new_idx = 0

    for dir_i, (input_dir, task) in enumerate(zip(args.input_dirs, args.tasks)):
        task_idx = task_to_idx[task]
        skip_set = skips.get(dir_i, set())

        all_hdf5 = sorted(
            [f for f in os.listdir(input_dir) if f.endswith(".hdf5")],
            key=lambda x: int(x.replace("episode_", "").replace(".hdf5", "")),
        )

        for fname in all_hdf5:
            orig_idx = int(fname.replace("episode_", "").replace(".hdf5", ""))
            if orig_idx in skip_set:
                print(f"  Skipping dir {dir_i} episode {orig_idx}")
                continue

            fpath = os.path.join(input_dir, fname)
            print(f"  [{new_idx:03d}] dir{dir_i}/episode_{orig_idx} → episode_{new_idx:06d} (task: {task[:50]}...)")

            with h5py.File(fpath, "r") as f:
                qpos = np.array(f["observations/qpos/robot_1"])
                qaction = np.array(f["qaction/robot_1"])
                T = qpos.shape[0]

                table = build_parquet(qpos, qaction, new_idx, global_index, task_idx, fps)
                pq.write_table(table, os.path.join(data_dir, f"episode_{new_idx:06d}.parquet"))

                for sensor_key, cam_name in CAMERA_MAP.items():
                    images = np.array(f[f"observations/images/{sensor_key}"])
                    video_path = os.path.join(
                        output_dir, "videos", "chunk-000", cam_name, f"episode_{new_idx:06d}.mp4"
                    )
                    images_to_mp4(images, video_path, fps)

            episodes_meta.append({
                "episode_index": new_idx,
                "tasks": [task],
                "length": T,
            })
            global_index += T
            total_frames += T
            new_idx += 1

    # Write meta files
    with open(os.path.join(output_dir, "meta", "tasks.jsonl"), "w") as f:
        for i, task in enumerate(unique_tasks):
            f.write(json.dumps({"task_index": i, "task": task}) + "\n")

    with open(os.path.join(output_dir, "meta", "episodes.jsonl"), "w") as f:
        for ep in episodes_meta:
            f.write(json.dumps(ep) + "\n")

    modality = {
        "state": {"arm": {"start": 0, "end": 6}, "gripper": {"start": 6, "end": 7}},
        "action": {"arm": {"start": 0, "end": 6}, "gripper": {"start": 6, "end": 7}},
        "video": {
            "side": {"original_key": "observation.images.side"},
            "top": {"original_key": "observation.images.top"},
            "wrist": {"original_key": "observation.images.wrist"},
        },
        "annotation": {"human.action.task_description": {"original_key": "task_index"}},
    }
    with open(os.path.join(output_dir, "meta", "modality.json"), "w") as f:
        json.dump(modality, f, indent=4)

    info = {
        "codebase_version": "v2.1",
        "robot_type": "piper",
        "total_episodes": new_idx,
        "total_frames": total_frames,
        "total_tasks": len(unique_tasks),
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": f"0:{new_idx}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "action": {"dtype": "float32", "names": ["joint1.pos","joint2.pos","joint3.pos","joint4.pos","joint5.pos","joint6.pos","gripper.pos"], "shape": [7]},
            "observation.state": {"dtype": "float32", "names": ["joint1.pos","joint2.pos","joint3.pos","joint4.pos","joint5.pos","joint6.pos","gripper.pos"], "shape": [7]},
            **{cam: {"dtype": "video", "shape": [480,640,3], "names": ["height","width","channels"], "info": {"video.height":480,"video.width":640,"video.codec":"h264","video.pix_fmt":"yuv420p","video.is_depth_map":False,"video.fps":fps,"video.channels":3,"has_audio":False}} for cam in CAMERA_MAP.values()},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
        "total_chunks": 0,
        "total_videos": new_idx * len(CAMERA_MAP),
    }
    with open(os.path.join(output_dir, "meta", "info.json"), "w") as f:
        json.dump(info, f, indent=4)

    print(f"\nDone! {new_idx} episodes, {total_frames} frames → {output_dir}")


if __name__ == "__main__":
    main()
