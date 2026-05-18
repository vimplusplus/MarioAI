"""
Custom Stable Baselines3 callbacks for checkpointing and console logging.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback


class TrainAndLoggingCallback(BaseCallback):
    """
    Periodically saves model checkpoints during training and prints
    human-readable progress lines to stdout.

    Checkpoints are written to::

        {save_path}/best_model_{step}.zip

    every ``check_freq`` **environment** steps (not gradient updates).

    Args:
        check_freq: How many environment steps between checkpoints.
        save_path:  Directory where ``.zip`` files are written.
        verbose:    0 = silent, 1 = print on each checkpoint.
    """

    def __init__(self, check_freq: int, save_path: str, verbose: int = 1) -> None:
        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self._start_time: float = 0.0

    # ── SB3 lifecycle hooks ───────────────────────────────────────────────────

    def _init_callback(self) -> None:
        Path(self.save_path).mkdir(parents=True, exist_ok=True)
        self._start_time = time.time()

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            checkpoint = os.path.join(self.save_path, f"best_model_{self.n_calls}")
            self.model.save(checkpoint)

            if self.verbose >= 1:
                elapsed = time.time() - self._start_time
                steps_per_sec = self.n_calls / max(elapsed, 1e-6)
                remaining = (
                    (self.locals.get("total_timesteps", 0) - self.n_calls)
                    / max(steps_per_sec, 1e-6)
                    / 60
                )
                print(
                    f"  [✓] Step {self.n_calls:>10,}"
                    f"  |  Saved → {os.path.basename(checkpoint)}.zip"
                    f"  |  {steps_per_sec:,.0f} steps/s"
                    f"  |  ~{remaining:.0f} min remaining"
                )
        return True
