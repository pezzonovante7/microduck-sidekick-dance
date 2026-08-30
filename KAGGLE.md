# Train the side-kick dance on Kaggle (free GPU)

Kaggle: ~30 GPU hours/week, 12 h session cap. Pick **GPU T4 x2**, not P100.
This trainer wants Python **3.12** (`requires-python = ">=3.12, <3.13"`). Use `uv`’s Python, not the notebook’s.

Pollen’s 4096-env recipe is sized for a 24 GB L4. A T4 is 16 GB (T4 x2 is two cards; this trainer uses one). Start at **1024** envs. Bump to 2048 only if `nvidia-smi` still has headroom.

## 0. Once

1. [kaggle.com](https://www.kaggle.com) → sign in (Google is fine).
2. Phone-verify the account. GPU stays locked until you do.
3. Make [pezzonovante7/microduck-sidekick-dance](https://github.com/pezzonovante7/microduck-sidekick-dance) **public**, or upload the two files as a Kaggle dataset. Kaggle cannot clone a private GitHub repo without a token.

## 1. Notebook settings

New Notebook → keep it **private**.

Right sidebar:

- Accelerator: **GPU T4 x2**
- Internet: **On**
- Persistence: **Files** (so `/kaggle/working` survives a Save Version)

Or from this repo, `kaggle-notebook/kernel-metadata.json` already has those flags (`machine_shape`: `NvidiaTeslaT4`, internet on, private). Push and run:

```bash
kaggle kernels push -p kaggle-notebook --accelerator NvidiaTeslaT4 -t 43200
```

## 2. Cells (run in order)

**GPU check**

```python
!nvidia-smi
```

You want a Tesla T4. If you see P100, change the accelerator and restart the session.

**uv + Python 3.12**

```python
import os
os.environ["PATH"] = os.path.expanduser("~/.local/bin") + os.pathsep + os.environ["PATH"]

!curl -LsSf https://astral.sh/uv/install.sh | sh
!uv python install 3.12
```

**clone trainer + drop in the dance task**

```python
%cd /kaggle/working
!git clone --depth 1 https://github.com/pollen-robotics/microduck_rl.git
%cd /kaggle/working/microduck_rl
!git clone --depth 1 https://github.com/pezzonovante7/microduck-sidekick-dance.git /tmp/dance
!cp /tmp/dance/src/mjlab_microduck/tasks/microduck_sidekick_dance_env_cfg.py src/mjlab_microduck/tasks/
```

**register the task** (append once; skip if you re-run)

```python
init = "src/mjlab_microduck/tasks/__init__.py"
block = '''
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
'''
text = open(init).read()
if "Mjlab-SideKickDance-Flat-MicroDuck" not in text:
    open(init, "a").write(block)
    print("registered")
else:
    print("already registered")
```

**install (slow the first time, ~10 min)**

```python
!uv sync
```

The trainer defaults to wandb. Kaggle has no wandb login, so every train cell must (1) keep wandb offline and (2) pass `--agent.logger tensorboard`. `WANDB_MODE=disabled` / `WANDB_DISABLED=true` is not enough: rsl_rl still constructs `WandbLogWriter` and `wandb.init()` raises `UsageError: No API key configured`.

```python
import os
os.environ["PATH"] = os.path.expanduser("~/.local/bin") + os.pathsep + os.environ["PATH"]
os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_SILENT"] = "true"
```

**smoke test (CPU-ok, burns almost no GPU quota, proves the task exists)**

```python
%cd /kaggle/working/microduck_rl
!uv run train Mjlab-SideKickDance-Flat-MicroDuck --env.scene.num-envs 64 --agent.max-iterations 5 --agent.logger tensorboard
```

If this errors on the task id, the register cell did not stick. If it OOMs, the GPU is not attached. If wandb asks for an API key, the logger flag did not stick.

**real train**

```python
%cd /kaggle/working/microduck_rl
!uv run train Mjlab-SideKickDance-Flat-MicroDuck --env.scene.num-envs 1024 --agent.logger tensorboard
```

Leave the tab open. Session dies at 12 h. Checkpoints land under `logs/`. Download them before you stop the session, or **Save Version** so `/kaggle/working` is kept.

If CUDA OOM, drop to 512 envs and rerun. If VRAM is idle, try 2048.

## 3. After it learns

Still on this notebook, with a checkpoint:

```bash
uv run scripts/export.py Mjlab-SideKickDance-Flat-MicroDuck --wandb-run-path <entity/project/run_id>
```

If you did not log in to wandb, point export at the local run folder instead (see `scripts/export.py --help`). Then:

```bash
uv run scripts/infer_policy.py --walking output.onnx --record
```

That mp4 is the clip. The Hugging Face sandbox will not load it until someone wires a new ONNX into the Space.

## Resume after a 12 h kick

```bash
uv run train Mjlab-SideKickDance-Flat-MicroDuck --env.scene.num-envs 1024 \
    --agent.logger tensorboard \
    --agent.run-name resume --agent.load-checkpoint model_XXXX.pt --agent.resume True
```

Put the `.pt` back into the working tree first (Kaggle dataset, or re-upload).
