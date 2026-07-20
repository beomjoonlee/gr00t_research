"""
Convert Piper HDF5 dataset to GR00T LeRobot v2 format.

HDF5 structure:
  observations/qpos/robot_1: (T, 7) - joint positions (6 arm + 1 gripper)
  qaction/robot_1: (T, 7) - joint actions
  observations/images/sensor_2: (T, 480, 640, 3) - side camera (BGR)
  observations/images/sensor_3: (T, 480, 640, 3) - top camera (BGR)
  observations/images/sensor_4: (T, 480, 640, 3) - wrist camera (BGR)
  language_instruction: str

Output: GR00T LeRobot v2 format with:
  - Parquet files (state, action, annotations)
  - MP4 videos per camera
  - Meta files (info.json, episodes.jsonl, tasks.jsonl, modality.json)
"""

import argparse
import json
import os

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


# ── Configuration ──────────────────────────────────────────────────────────────

CAMERA_MAP = {
    "sensor_2": "observation.images.side",
    "sensor_3": "observation.images.top",
    "sensor_4": "observation.images.wrist",
}

TASKS = [
    "pick up the red object from the table and place it on the white plate",
    "pick up the green object from the table and place it on the white plate",
    "pick up the blue object from the table and place it on the white plate",
]

# Episode index (original) → task index
EPISODE_TASK_MAP = {
    **{i: 0 for i in range(0, 5)},      # red: 0-4
    **{i: 0 for i in range(15, 20)},     # red: 15-19
    **{i: 1 for i in range(5, 9)},       # green: 5-8
    10: 1,                                # green: 10
    **{i: 1 for i in range(20, 25)},     # green: 20-24
    **{i: 2 for i in range(11, 15)},     # blue: 11-14
    **{i: 2 for i in range(26, 32)},     # blue: 26-31
}

SKIP_EPISODES = {9, 25}

FPS = 10


# ── Helpers ────────────────────────────────────────────────────────────────────

def images_to_mp4(images: np.ndarray, output_path: str, fps: int):
    """Write (T, H, W, 3) uint8 BGR array to MP4 via OpenCV.
    HDF5 from EasyTrainer stores images in BGR format."""
    import cv2

    T, H, W, C = images.shape
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
    for t in range(T):
        writer.write(images[t])  # Already BGR, write directly
    writer.release()


def build_parquet(qpos, qaction, episode_idx, global_offset, task_idx, fps):
    """Build a pyarrow Table for one episode."""
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
    table = pa.table(rows, schema=schema)
    return table


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Convert Piper HDF5 to GR00T LeRobot v2")
    parser.add_argument("--input-dir", required=True, help="Directory with episode_*.hdf5 files")
    parser.add_argument("--output-dir", required=True, help="Output LeRobot dataset directory")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    # Collect valid episodes in order
    all_hdf5 = sorted(
        [f for f in os.listdir(input_dir) if f.endswith(".hdf5")],
        key=lambda x: int(x.replace("episode_", "").replace(".hdf5", "")),
    )

    valid_episodes = []
    for fname in all_hdf5:
        orig_idx = int(fname.replace("episode_", "").replace(".hdf5", ""))
        if orig_idx in SKIP_EPISODES:
            print(f"Skipping episode {orig_idx}")
            continue
        if orig_idx not in EPISODE_TASK_MAP:
            print(f"Warning: episode {orig_idx} has no task mapping, skipping")
            continue
        valid_episodes.append((orig_idx, fname))

    print(f"Converting {len(valid_episodes)} episodes")

    # Create directories
    data_dir = os.path.join(output_dir, "data", "chunk-000")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "meta"), exist_ok=True)

    for cam_key in CAMERA_MAP.values():
        os.makedirs(os.path.join(output_dir, "videos", "chunk-000", cam_key), exist_ok=True)

    episodes_meta = []
    global_index = 0
    total_frames = 0

    for new_idx, (orig_idx, fname) in enumerate(valid_episodes):
        fpath = os.path.join(input_dir, fname)
        task_idx = EPISODE_TASK_MAP[orig_idx]

        print(f"  [{new_idx:02d}] episode_{orig_idx} → episode_{new_idx:06d} (task: {TASKS[task_idx][:50]}...)")

        with h5py.File(fpath, "r") as f:
            qpos = np.array(f["observations/qpos/robot_1"])
            qaction = np.array(f["qaction/robot_1"])
            T = qpos.shape[0]

            # Write parquet
            table = build_parquet(qpos, qaction, new_idx, global_index, task_idx, FPS)
            pq.write_table(table, os.path.join(data_dir, f"episode_{new_idx:06d}.parquet"))

            # Write videos
            for sensor_key, cam_name in CAMERA_MAP.items():
                images = np.array(f[f"observations/images/{sensor_key}"])
                video_path = os.path.join(
                    output_dir, "videos", "chunk-000", cam_name, f"episode_{new_idx:06d}.mp4"
                )
                images_to_mp4(images, video_path, FPS)

        episodes_meta.append({
            "episode_index": new_idx,
            "tasks": [TASKS[task_idx]],
            "length": T,
        })
        global_index += T
        total_frames += T

    # ── Write meta files ──────────────────────────────────────────────────────

    # tasks.jsonl
    with open(os.path.join(output_dir, "meta", "tasks.jsonl"), "w") as f:
        for i, task in enumerate(TASKS):
            f.write(json.dumps({"task_index": i, "task": task}) + "\n")

    # episodes.jsonl
    with open(os.path.join(output_dir, "meta", "episodes.jsonl"), "w") as f:
        for ep in episodes_meta:
            f.write(json.dumps(ep) + "\n")

    # modality.json
    modality = {
        "state": {
            "arm": {"start": 0, "end": 6},
            "gripper": {"start": 6, "end": 7},
        },
        "action": {
            "arm": {"start": 0, "end": 6},
            "gripper": {"start": 6, "end": 7},
        },
        "video": {
            "side": {"original_key": "observation.images.side"},
            "top": {"original_key": "observation.images.top"},
            "wrist": {"original_key": "observation.images.wrist"},
        },
        "annotation": {
            "human.action.task_description": {"original_key": "task_index"},
        },
    }
    with open(os.path.join(output_dir, "meta", "modality.json"), "w") as f:
        json.dump(modality, f, indent=4)

    # info.json
    info = {
        "codebase_version": "v2.1",
        "robot_type": "piper",
        "total_episodes": len(valid_episodes),
        "total_frames": total_frames,
        "total_tasks": len(TASKS),
        "chunks_size": 1000,
        "fps": FPS,
        "splits": {"train": f"0:{len(valid_episodes)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "action": {
                "dtype": "float32",
                "names": [
                    "joint1.pos", "joint2.pos", "joint3.pos",
                    "joint4.pos", "joint5.pos", "joint6.pos",
                    "gripper.pos",
                ],
                "shape": [7],
            },
            "observation.state": {
                "dtype": "float32",
                "names": [
                    "joint1.pos", "joint2.pos", "joint3.pos",
                    "joint4.pos", "joint5.pos", "joint6.pos",
                    "gripper.pos",
                ],
                "shape": [7],
            },
            **{
                cam_name: {
                    "dtype": "video",
                    "shape": [480, 640, 3],
                    "names": ["height", "width", "channels"],
                    "info": {
                        "video.height": 480,
                        "video.width": 640,
                        "video.codec": "h264",
                        "video.pix_fmt": "yuv420p",
                        "video.is_depth_map": False,
                        "video.fps": FPS,
                        "video.channels": 3,
                        "has_audio": False,
                    },
                }
                for cam_name in CAMERA_MAP.values()
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
        "total_chunks": 0,
        "total_videos": len(valid_episodes) * len(CAMERA_MAP),
    }
    with open(os.path.join(output_dir, "meta", "info.json"), "w") as f:
        json.dump(info, f, indent=4)

    print(f"\nDone! {len(valid_episodes)} episodes, {total_frames} frames → {output_dir}")


if __name__ == "__main__":
    main()
