"""
Mario environment builder with the full preprocessing pipeline.

Preprocessing pipeline
----------------------
    Raw RGB frames (240×256×3)
        → GrayScaleObservation  → (240×256×1)
        → DummyVecEnv           → vectorised single env
        → VecFrameStack(n=4)    → (240×256×4)  ← what the CNN policy sees
"""

from __future__ import annotations

from typing import Optional

import gym_super_mario_bros
from gym.wrappers import GrayScaleObservation
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT, RIGHT_ONLY, SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from .config import TrainingConfig

# ── Movement presets ──────────────────────────────────────────────────────────

MOVEMENT_PRESETS: dict[str, list] = {
    "right_only": RIGHT_ONLY,      # 5 actions  — fastest initial learning
    "simple": SIMPLE_MOVEMENT,     # 7 actions  — good balance (default)
    "complex": COMPLEX_MOVEMENT,   # 12 actions — most expressive, hardest to learn
}


class MarioEnvironment:
    """
    Builds and manages the preprocessed Super Mario Bros environment.

    Usage — context manager (recommended)::

        config = TrainingConfig()
        with MarioEnvironment(config) as env:
            obs = env.reset()
            ...

    Usage — manual::

        builder = MarioEnvironment(config)
        env = builder.build()
        ...
        builder.close()

    Args:
        config: A :class:`~mario_ai.config.TrainingConfig` instance that
                controls the environment name, movement preset, and frame
                stack depth.
    """

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self._env: Optional[VecFrameStack] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def build(self) -> VecFrameStack:
        """Construct the full preprocessing pipeline and return the env."""
        if self.config.movement not in MOVEMENT_PRESETS:
            raise ValueError(
                f"Unknown movement preset '{self.config.movement}'. "
                f"Choose from: {list(MOVEMENT_PRESETS)}"
            )

        movement = MOVEMENT_PRESETS[self.config.movement]

        # 1. Base NES environment
        env = gym_super_mario_bros.make(self.config.env_name)

        # 2. Simplify the action space
        env = JoypadSpace(env, movement)

        # 3. Convert RGB → greyscale (keeps channel dim for SB3 compatibility)
        env = GrayScaleObservation(env, keep_dim=True)

        # 4. Wrap in a vectorised container (required by Stable Baselines3)
        env = DummyVecEnv([lambda: env])

        # 5. Stack the last n_stack frames so the agent can perceive velocity
        env = VecFrameStack(env, self.config.n_stack, channels_order="last")

        self._env = env
        return env

    def close(self) -> None:
        """Close the underlying environment and release resources."""
        if self._env is not None:
            self._env.close()
            self._env = None

    @property
    def is_open(self) -> bool:
        """True if the environment has been built and not yet closed."""
        return self._env is not None

    @staticmethod
    def action_count(movement: str) -> int:
        """Return the number of discrete actions for a given movement preset."""
        preset = MOVEMENT_PRESETS.get(movement, SIMPLE_MOVEMENT)
        return len(preset)

    # ── Context Manager ───────────────────────────────────────────────────────

    def __enter__(self) -> VecFrameStack:
        return self.build()

    def __exit__(self, *_) -> None:
        self.close()

    def __repr__(self) -> str:
        status = "open" if self.is_open else "closed"
        return (
            f"MarioEnvironment("
            f"env={self.config.env_name!r}, "
            f"movement={self.config.movement!r}, "
            f"n_stack={self.config.n_stack}, "
            f"status={status})"
        )
