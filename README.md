# MicroDuck side-kick dance

A new **skill** for [Pollen Robotics MicroDuck](https://github.com/pollen-robotics/microduck): a repeating lateral dance step. Not the soccer kick in the browser (`Q` / `E`). Those swing **forward**. This one opens a hip **to the side**, other foot planted, body up, then recovers, on a beat.

The Hugging Face [sandbox](https://huggingface.co/spaces/pollen-robotics/microduck-simulator) cannot learn this. A policy has to be trained in [microduck_rl](https://github.com/pollen-robotics/microduck_rl), exported to ONNX, then played.

This repo is the drop-in task. It is **not** a trained brain yet. Training needs a CUDA GPU (or [Hugging Face Jobs](https://huggingface.co/docs/hub/spaces-gpus)).

## What we are rewarding

- Support foot stays on the floor.
- Swing-leg `hip_yaw` goes out to the side when the phase says kick, back when it says recover.
- Trunk stays upright at standing height.
- Do **not** walk forward, hop, or fall.

Phase is a 1.2 s metronome. Positive half: right-leg side kick. Negative half: left-leg side kick. That is the dance loop.

## Install (once)

You need Python 3.11+, `uv`, and a machine with an NVIDIA GPU **or** a Hugging Face account for Jobs.

```bash
git clone https://github.com/pollen-robotics/microduck_rl.git
cd microduck_rl
# copy the files from this repo (see below)
```

From this repo:

```bash
cp src/mjlab_microduck/tasks/microduck_sidekick_dance_env_cfg.py \
   /path/to/microduck_rl/src/mjlab_microduck/tasks/
```

Then add the registration block in `src/mjlab_microduck/tasks/__init__.py` (also in `register_snippet.py` here):

```python
from .microduck_sidekick_dance_env_cfg import (
    make_microduck_sidekick_dance_env_cfg,
    MicroduckSideKickDanceRlCfg,
)

register_mjlab_task(
    task_id="Mjlab-SideKickDance-Flat-MicroDuck",
    env_cfg=make_microduck_sidekick_dance_env_cfg(),
    play_env_cfg=make_microduck_sidekick_dance_env_cfg(play=True),
    rl_cfg=MicroduckSideKickDanceRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
```

## Train

Smoke test (CPU is ok, does not learn the dance):

```bash
uv run train Mjlab-SideKickDance-Flat-MicroDuck --env.scene.num-envs 64 --agent.max-iterations 5
```

Real training (GPU):

```bash
uv run train Mjlab-SideKickDance-Flat-MicroDuck --env.scene.num-envs 4096
```

No GPU? Same command with `--hf-jobs` (needs a Hugging Face token; see `scripts/hf/README.md` in microduck_rl).

Walking took Pollen about 1–2 hours at 4096 envs. A short dance trick is usually in that ballpark. Watch whether `hip_yaw` actually opens sideways in the viewer. If it starts walking or falling, the reward weights in the env file are the knobs.

## Export and record

Always use their exporter so the observation normalizer is baked in:

```bash
uv run scripts/export.py Mjlab-SideKickDance-Flat-MicroDuck --wandb-run-path <entity/project/run_id>
uv run scripts/infer_policy.py --walking output.onnx --record
```

That `.mp4` is what people post on X. The sandbox will not load this file until someone wires a new ONNX into the Space.

## Why this is a repo

The code has to live somewhere you own. GitHub is that drawer. Cursor Cloud Agents / Origin need Pro on this account, so the files were written here directly.
