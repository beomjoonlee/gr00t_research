#!/usr/bin/env python3
"""Convert cleaned EasyTrainer LeRobot datasets (dirs 1/2/3) to GR00T-flavored format.

In-place transform of each dataset dir (originals preserved in *__orig / datasets.tar.gz):
  - parquet columns: observation.qpos -> observation.state, action.joint -> action,
    add annotation.human.action.task_description (= task_index), drop qvel/qeffort/eepos/succeed.
  - videos: rename folders observation.images.sensor_{7,8,11} -> wrist/right/left.
  - meta/modality.json: created (state arm[0:6]+gripper[6:7], video wrist/right/left, annotation).
  - meta/info.json: features rewritten to match new columns; stale EasyTrainer keys dropped.

Camera mapping (fresh-collection layout, cameras were repositioned):
    sensor_7  -> wrist   sensor_8 -> right   sensor_11 -> left
"""
import glob
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = "/home/wook/Downloads/datasets"

# sensor -> semantic view name (new physical layout)
CAM_MAP = {
    "sensor_7": "wrist",
    "sensor_8": "right",
    "sensor_11": "left",
}
VIEW_ORDER = ["wrist", "right", "left"]  # modality_keys order for training

JOINT_NAMES = [
    "joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper",
]


def build_modality_json():
    return {
        "state": {"arm": {"start": 0, "end": 6}, "gripper": {"start": 6, "end": 7}},
        "action": {"arm": {"start": 0, "end": 6}, "gripper": {"start": 6, "end": 7}},
        "video": {
            view: {"original_key": f"observation.images.{view}"} for view in VIEW_ORDER
        },
        "annotation": {"human.action.task_description": {"original_key": "task_index"}},
    }


def build_info_features():
    vec7 = lambda: {"dtype": "float32", "shape": [7], "names": JOINT_NAMES}
    scalar_i = {"dtype": "int64", "shape": [1], "names": None}
    video_feat = {"dtype": "video", "shape": [240, 320, 3],
                  "names": ["height", "width", "channels"]}
    feats = {
        "observation.state": vec7(),
        "action": vec7(),
    }
    for view in VIEW_ORDER:
        feats[f"observation.images.{view}"] = dict(video_feat)
    feats["timestamp"] = {"dtype": "float32", "shape": [1], "names": None}
    feats["annotation.human.action.task_description"] = dict(scalar_i)
    feats["task_index"] = dict(scalar_i)
    feats["episode_index"] = dict(scalar_i)
    feats["index"] = dict(scalar_i)
    feats["frame_index"] = dict(scalar_i)
    return feats


def convert(name):
    d = os.path.join(ROOT, name)

    # 1) parquet columns
    for pq in sorted(glob.glob(os.path.join(d, "data/chunk-000/*.parquet"))):
        df = pd.read_parquet(pq)
        out = pd.DataFrame()
        out["observation.state"] = df["observation.qpos"]
        out["action"] = df["action.joint"]
        out["timestamp"] = df["timestamp"].astype(np.float32)
        out["annotation.human.action.task_description"] = df["task_index"].astype(np.int64)
        out["task_index"] = df["task_index"].astype(np.int64)
        out["episode_index"] = df["episode_index"].astype(np.int64)
        out["index"] = df["index"].astype(np.int64)
        out["frame_index"] = df["frame_index"].astype(np.int64)
        out.to_parquet(pq)

    # 2) rename video folders sensor_* -> view
    vdir = os.path.join(d, "videos/chunk-000")
    for sensor, view in CAM_MAP.items():
        src = os.path.join(vdir, f"observation.images.{sensor}")
        dst = os.path.join(vdir, f"observation.images.{view}")
        if os.path.isdir(src):
            if os.path.isdir(dst):
                shutil.rmtree(dst)
            os.rename(src, dst)

    # 3) modality.json
    with open(os.path.join(d, "meta/modality.json"), "w") as f:
        json.dump(build_modality_json(), f, indent=4)

    # 4) info.json
    info = json.load(open(os.path.join(d, "meta/info.json")))
    info["features"] = build_info_features()
    for stale in ("action_key", "tool_qpos_indices"):
        info.pop(stale, None)
    info["total_videos"] = info["total_episodes"] * len(VIEW_ORDER)
    # GR00T loader expects the v2 path templates ({episode_chunk}/{episode_index}),
    # not EasyTrainer's v3 templates ({chunk_index}/{file_index}).
    info["data_path"] = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    info["video_path"] = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    with open(os.path.join(d, "meta/info.json"), "w") as f:
        json.dump(info, f, indent=4)

    # 5) drop LeRobot-native episodes_stats.jsonl (GR00T regenerates stats.json at train time)
    stale_stats = os.path.join(d, "meta/episodes_stats.jsonl")
    if os.path.exists(stale_stats):
        os.remove(stale_stats)

    print(f"[ds{name}] converted -> GR00T format "
          f"(state/action renamed, videos={VIEW_ORDER}, modality.json written)")


if __name__ == "__main__":
    for t in sys.argv[1:] or ["1", "2", "3"]:
        convert(t)
