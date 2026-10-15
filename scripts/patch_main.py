"""Patch openpi examples/libero/main.py: add --args.task-filter and per-episode video names.

Run from the openpi repo root: python3 patch_main.py
"""

p = "examples/libero/main.py"
s = open(p).read()
s = s.replace(
    '    num_trials_per_task: int = 50  # Number of rollouts per task\n',
    '    num_trials_per_task: int = 50  # Number of rollouts per task\n'
    '    task_filter: str = ""  # only run tasks whose instruction contains this text\n',
    1,
)
s = s.replace(
    '        task = task_suite.get_task(task_id)\n',
    '        task = task_suite.get_task(task_id)\n'
    '        if args.task_filter and args.task_filter not in task.language:\n'
    '            continue\n',
    1,
)
s = s.replace('f"rollout_{task_segment}_{suffix}.mp4"', 'f"rollout_{task_segment}_ep{episode_idx}_{suffix}.mp4"', 1)
open(p, "w").write(s)
