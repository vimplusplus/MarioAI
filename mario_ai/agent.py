"""
The Mario PPO agent — wraps Stable Baselines3's PPO with a clean API.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
from stable_baselines3 import PPO

from .callbacks import TrainAndLoggingCallback
from .config import TrainingConfig

if TYPE_CHECKING:
    from stable_baselines3.common.vec_env import VecEnv


class MarioAgent:
    """
    A PPO-based agent for Super Mario Bros.

    Wraps Stable Baselines3's :class:`~stable_baselines3.PPO` with a simple
    interface for building, loading, training, saving, and running inference.

    Example — train from scratch::

        config = TrainingConfig()
        with MarioEnvironment(config) as env:
            agent = MarioAgent(config, env).build()
            agent.train()

    Example — load and play::

        agent = MarioAgent(config, env).load("./train/best_model_70000")
        state = env.reset()
        while True:
            action, _ = agent.predict(state)
            state, reward, done, info = env.step(action)

    Args:
        config: Hyperparameters and path settings.
        env:    The preprocessed vectorised environment.
    """

    def __init__(self, config: TrainingConfig, env: Optional["VecEnv"] = None) -> None:
        self.config = config
        self.env = env
        self.model: Optional[PPO] = None

    # ── Construction ──────────────────────────────────────────────────────────

    def build(self) -> "MarioAgent":
        """Instantiate a fresh PPO model with the current config."""
        if self.env is None:
            raise RuntimeError("An environment must be provided before calling build().")

        self.model = PPO(
            policy=self.config.policy,
            env=self.env,
            verbose=1,
            tensorboard_log=self.config.log_dir,
            learning_rate=self.config.learning_rate,
            n_steps=self.config.n_steps,
            batch_size=self.config.batch_size,
            n_epochs=self.config.n_epochs,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            clip_range=self.config.clip_range,
            ent_coef=self.config.ent_coef,
        )
        return self

    def load(self, path: str | Path) -> "MarioAgent":
        """
        Load a saved model from *path*.

        Automatically strips the ``.zip`` extension if present, as
        Stable Baselines3 handles that internally.
        """
        path = str(path).removesuffix(".zip")
        self.model = PPO.load(path, env=self.env)
        return self

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self) -> "MarioAgent":
        """
        Run the training loop.

        Saves a checkpoint every ``config.checkpoint_freq`` steps to
        ``config.checkpoint_dir``.  When resuming (``config.resume_path`` is
        set), the timestep counter is **not** reset so TensorBoard curves are
        continuous.
        """
        if self.model is None:
            raise RuntimeError("Call build() or load() before train().")

        callback = TrainAndLoggingCallback(
            check_freq=self.config.checkpoint_freq,
            save_path=self.config.checkpoint_dir,
        )

        self.model.learn(
            total_timesteps=self.config.total_timesteps,
            callback=callback,
            reset_num_timesteps=(self.config.resume_path is None),
        )
        return self

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, state: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Return ``(action, state)`` for the given observation."""
        if self.model is None:
            raise RuntimeError("No model loaded. Call build() or load() first.")
        return self.model.predict(state)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save the current model weights to *path*."""
        if self.model is None:
            raise RuntimeError("No model to save.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))

    # ── Dunder ────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        loaded = self.model is not None
        return (
            f"MarioAgent("
            f"policy={self.config.policy!r}, "
            f"lr={self.config.learning_rate}, "
            f"loaded={loaded})"
        )
