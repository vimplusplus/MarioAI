"""
Training configuration with YAML serialization support.

All hyperparameters live here. Edit ``configs/default.yaml`` or pass
overrides via the CLI to tune training without touching source code.
"""

from __future__ import annotations

import yaml
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Optional


@dataclass
class TrainingConfig:
    """
    All hyperparameters and directory paths for a training run.

    Attributes can be loaded from a YAML file via :meth:`from_yaml` or
    constructed manually. CLI flags in ``main.py`` override individual fields
    without touching the config file.
    """

    # ── Environment ──────────────────────────────────────────────────────────
    env_name: str = "SuperMarioBros-v0"
    movement: str = "simple"       # "simple" | "right_only" | "complex"
    n_stack: int = 4               # Frames to stack (gives the agent motion cues)

    # ── Algorithm ─────────────────────────────────────────────────────────────
    policy: str = "CnnPolicy"      # CNN reads raw pixel frames
    total_timesteps: int = 1_000_000
    learning_rate: float = 1e-6    # Low LR is critical for PPO stability on visual input
    n_steps: int = 512             # Rollout steps per environment per update
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99            # Discount factor
    gae_lambda: float = 0.95       # GAE smoothing
    clip_range: float = 0.2        # PPO clip (prevents large policy updates)
    ent_coef: float = 0.01         # Entropy bonus — encourages exploration

    # ── Paths ─────────────────────────────────────────────────────────────────
    checkpoint_dir: str = "./train/"
    log_dir: str = "./logs/"
    checkpoint_freq: int = 10_000  # Save a checkpoint every N environment steps

    # ── Resume ────────────────────────────────────────────────────────────────
    resume_path: Optional[str] = None  # Path to checkpoint to continue from

    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainingConfig":
        """Load config from a YAML file, ignoring any unrecognised keys."""
        with open(path, "r") as fh:
            data = yaml.safe_load(fh) or {}
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid})

    def save(self, path: str | Path) -> None:
        """Persist the current config to a YAML file."""
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "w") as fh:
            yaml.safe_dump(
                asdict(self), fh,
                default_flow_style=False,
                sort_keys=False,
            )

    def to_dict(self) -> dict:
        """Return the config as a plain dictionary."""
        return asdict(self)

    def __str__(self) -> str:
        lines = [f"TrainingConfig("]
        for f in fields(self):
            lines.append(f"  {f.name}={getattr(self, f.name)!r},")
        lines.append(")")
        return "\n".join(lines)
