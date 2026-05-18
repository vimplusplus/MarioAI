"""
Smoke tests for MarioEnvironment.
The actual gym/NES environment is mocked so these run without a display.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from mario_ai.config import TrainingConfig
from mario_ai.environment import MOVEMENT_PRESETS, MarioEnvironment


class TestMovementPresets:
    def test_all_presets_present(self):
        assert "simple" in MOVEMENT_PRESETS
        assert "right_only" in MOVEMENT_PRESETS
        assert "complex" in MOVEMENT_PRESETS

    def test_action_counts(self):
        # simple has 7 actions, right_only has 5, complex has 12
        assert MarioEnvironment.action_count("simple") == 7
        assert MarioEnvironment.action_count("right_only") == 5
        assert MarioEnvironment.action_count("complex") == 12

    def test_unknown_movement_raises(self):
        config = TrainingConfig(movement="teleport")
        builder = MarioEnvironment(config)
        with pytest.raises(ValueError, match="Unknown movement preset"):
            builder.build()


class TestMarioEnvironmentRepr:
    def test_repr_closed(self):
        config = TrainingConfig()
        builder = MarioEnvironment(config)
        assert "closed" in repr(builder)
        assert "simple" in repr(builder)

    def test_is_open_false_before_build(self):
        config = TrainingConfig()
        assert not MarioEnvironment(config).is_open


class TestMarioEnvironmentMocked:
    """Build the full pipeline against a mocked NES env."""

    def _mock_base_env(self):
        """Return a mock that satisfies GrayScaleObservation's interface."""
        import gym
        import numpy as np

        mock_env = MagicMock(spec=gym.Env)
        mock_env.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(240, 256, 3), dtype=np.uint8
        )
        mock_env.action_space = gym.spaces.Discrete(7)
        mock_env.reset.return_value = MagicMock()
        mock_env.step.return_value = (MagicMock(), 0.0, False, {})
        return mock_env

    @patch("mario_ai.environment.gym_super_mario_bros.make")
    @patch("mario_ai.environment.JoypadSpace")
    @patch("mario_ai.environment.GrayScaleObservation")
    @patch("mario_ai.environment.DummyVecEnv")
    @patch("mario_ai.environment.VecFrameStack")
    def test_build_returns_vec_frame_stack(
        self, mock_vfs, mock_dve, mock_gray, mock_joypad, mock_make
    ):
        config = TrainingConfig()
        builder = MarioEnvironment(config)

        env = builder.build()

        # VecFrameStack is the outermost wrapper — build() should return it
        assert env is mock_vfs.return_value
        assert builder.is_open

    @patch("mario_ai.environment.gym_super_mario_bros.make")
    @patch("mario_ai.environment.JoypadSpace")
    @patch("mario_ai.environment.GrayScaleObservation")
    @patch("mario_ai.environment.DummyVecEnv")
    @patch("mario_ai.environment.VecFrameStack")
    def test_close_clears_env(self, mock_vfs, *_):
        config = TrainingConfig()
        builder = MarioEnvironment(config)
        builder.build()
        builder.close()
        assert not builder.is_open

    @patch("mario_ai.environment.gym_super_mario_bros.make")
    @patch("mario_ai.environment.JoypadSpace")
    @patch("mario_ai.environment.GrayScaleObservation")
    @patch("mario_ai.environment.DummyVecEnv")
    @patch("mario_ai.environment.VecFrameStack")
    def test_context_manager(self, mock_vfs, *_):
        config = TrainingConfig()
        builder = MarioEnvironment(config)

        with builder as env:
            assert env is mock_vfs.return_value
            assert builder.is_open

        assert not builder.is_open
