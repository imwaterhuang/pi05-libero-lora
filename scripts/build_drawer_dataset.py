"""Build a LeRobot dataset = human demos of one LIBERO task + distilled rollouts.

Runs in the openpi uv env (from /workspace/openpi):
  uv run /path/to/build_drawer_dataset.py --rollout-dir data/distill_drawer
Output: $HF_LEROBOT_HOME/<repo_id> (default ~/.cache/huggingface/lerobot/water/libero_drawer100)
"""

import pathlib
import shutil

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import tqdm
import tyro

TASK = "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate"


def to_hwc_uint8(x):
    x = x.numpy()
    if x.shape[0] == 3:  # CHW -> HWC
        x = x.transpose(1, 2, 0)
    if x.dtype != np.uint8:
        x = (x * 255).round().clip(0, 255).astype(np.uint8)
    return x


def main(rollout_dir: str, repo_id: str = "water/libero_drawer100", source_repo: str = "physical-intelligence/libero"):
    out_path = HF_LEROBOT_HOME / repo_id
    if out_path.exists():
        shutil.rmtree(out_path)
    dst = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="panda",
        fps=10,
        features={
            "image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
            "wrist_image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
            "state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # 1) human demos of this task from the original LIBERO dataset
    src = LeRobotDataset(source_repo)
    eps = [i for i, ep in src.meta.episodes.items() if TASK in ep["tasks"]]
    print(f"Found {len(eps)} human demos for the task")
    for ep in tqdm.tqdm(eps, desc="human demos"):
        start = src.episode_data_index["from"][ep].item()
        end = src.episode_data_index["to"][ep].item()
        for i in range(start, end):
            f = src[i]
            dst.add_frame(
                {
                    "image": to_hwc_uint8(f["image"]),
                    "wrist_image": to_hwc_uint8(f["wrist_image"]),
                    "state": f["state"].numpy().astype(np.float32),
                    "actions": f["actions"].numpy().astype(np.float32),
                    "task": TASK,
                }
            )
        dst.save_episode()

    # 2) distilled rollouts from the official checkpoint
    files = sorted(pathlib.Path(rollout_dir).glob("ep_*.npz"))
    print(f"Found {len(files)} distilled rollouts")
    for p in tqdm.tqdm(files, desc="rollouts"):
        d = np.load(p)
        for t in range(len(d["actions"])):
            dst.add_frame(
                {
                    "image": d["image"][t],
                    "wrist_image": d["wrist_image"][t],
                    "state": d["state"][t],
                    "actions": d["actions"][t],
                    "task": TASK,
                }
            )
        dst.save_episode()
    print(f"Done: {len(eps) + len(files)} episodes at {out_path}")


if __name__ == "__main__":
    tyro.cli(main)
