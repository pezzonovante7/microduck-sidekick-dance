"""Paste this into pollen-robotics/microduck_rl:

    src/mjlab_microduck/tasks/__init__.py

Put the import with the other task imports. Put the register_mjlab_task
call with the other register_mjlab_task calls (after BallKick is a good
spot). Do not run this file by itself.
"""

from .microduck_sidekick_dance_env_cfg import (
    make_microduck_sidekick_dance_env_cfg,
    MicroduckSideKickDanceRlCfg,
)

# SideKickDance — repeating lateral hip_yaw step. Not BallKick (forward tap).
register_mjlab_task(
    task_id="Mjlab-SideKickDance-Flat-MicroDuck",
    env_cfg=make_microduck_sidekick_dance_env_cfg(),
    play_env_cfg=make_microduck_sidekick_dance_env_cfg(play=True),
    rl_cfg=MicroduckSideKickDanceRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
