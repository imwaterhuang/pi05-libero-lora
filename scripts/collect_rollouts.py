"""Record successful rollouts of a policy server on one LIBERO task (for distillation).

Runs in the LIBERO venv (Python 3.8), with a policy server already running (terminal 1).
Initial states come from env.reset() with new seeds, NOT the 50 fixed benchmark init states,
so the evaluation set stays unseen.

Usage (from /workspace/openpi):
  export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
  MUJOCO_GL=egl examples/libero/.venv/bin/python /path/to/collect_rollouts.py \
      --args.num-success 50 --args.out-dir data/distill_drawer
"""

import collections
import dataclasses
import logging
import math
import pathlib

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
RES = 256


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5
    task_suite_name: str = "libero_spatial"
    task_filter: str = "top drawer"  # substring of the task instruction
    num_success: int = 50  # stop after this many successful episodes
    max_attempts: int = 200
    max_steps: int = 220
    num_steps_wait: int = 10
    seed_offset: int = 10000  # seeds far from the eval seed (7)
    out_dir: str = "data/distill_drawer"


def quat2axisangle(quat):
    quat = quat.copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def main(args: Args) -> None:
    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    task = None
    for i in range(suite.n_tasks):
        t = suite.get_task(i)
        if args.task_filter in t.language:
            task = t
            break
    assert task is not None, f"No task matches {args.task_filter!r}"
    logging.info(f"Task: {task.language}")

    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=RES, camera_widths=RES)
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    for attempt in range(args.max_attempts):
        if n_ok >= args.num_success:
            break
        env.seed(args.seed_offset + attempt)
        obs = env.reset()  # random object placement, not a benchmark init state
        plan = collections.deque()
        images, wrists, states, actions = [], [], [], []
        done = False
        for t in range(args.max_steps + args.num_steps_wait):
            if t < args.num_steps_wait:
                obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                continue
            # rotate 180 degrees: same convention as the training dataset
            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
            state = np.concatenate(
                (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
            ).astype(np.float32)
            if not plan:
                element = {
                    "observation/image": image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    ),
                    "observation/wrist_image": image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist, args.resize_size, args.resize_size)
                    ),
                    "observation/state": state,
                    "prompt": str(task.language),
                }
                plan.extend(client.infer(element)["actions"][: args.replan_steps])
            action = np.asarray(plan.popleft(), dtype=np.float32)
            # dataset stores full-res 256x256 images, like the original LIBERO data
            images.append(img.astype(np.uint8))
            wrists.append(wrist.astype(np.uint8))
            states.append(state)
            actions.append(action)
            obs, _, done, _ = env.step(action.tolist())
            if done:
                break

        logging.info(f"attempt {attempt}: success={done}, steps={len(actions)}")
        if done:
            np.savez_compressed(
                out / f"ep_{n_ok:03d}.npz",
                image=np.stack(images),
                wrist_image=np.stack(wrists),
                state=np.stack(states),
                actions=np.stack(actions),
                task=np.array(task.language),
            )
            n_ok += 1
    logging.info(f"Saved {n_ok} successful episodes to {out} ({attempt + 1} attempts)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(main)
