"""
Tests for TrainingConfig — YAML round-trip, defaults, and field validation.
These tests have zero external dependencies (no gym, torch, etc.).
"""

import os
import tempfile

import pytest

from mario_ai.config import TrainingConfig


class TestTrainingConfigDefaults:
    def test_default_env_name(self):
        config = TrainingConfig()
        assert config.env_name == "SuperMarioBros-v0"

    def test_default_movement(self):
        config = TrainingConfig()
        assert config.movement == "simple"

    def test_default_n_stack(self):
        config = TrainingConfig()
        assert config.n_stack == 4

    def test_default_policy(self):
        config = TrainingConfig()
        assert config.policy == "CnnPolicy"

    def test_default_learning_rate(self):
        config = TrainingConfig()
        assert config.learning_rate == pytest.approx(1e-6)

    def test_default_resume_path_is_none(self):
        config = TrainingConfig()
        assert config.resume_path is None

    def test_default_total_timesteps(self):
        config = TrainingConfig()
        assert config.total_timesteps == 1_000_000


class TestTrainingConfigOverride:
    def test_override_lr(self):
        config = TrainingConfig(learning_rate=5e-5)
        assert config.learning_rate == pytest.approx(5e-5)

    def test_override_movement(self):
        config = TrainingConfig(movement="right_only")
        assert config.movement == "right_only"

    def test_to_dict_contains_all_keys(self):
        config = TrainingConfig()
        d = config.to_dict()
        expected_keys = {
            "env_name", "movement", "n_stack", "policy", "total_timesteps",
            "learning_rate", "n_steps", "batch_size", "n_epochs", "gamma",
            "gae_lambda", "clip_range", "ent_coef", "checkpoint_dir",
            "log_dir", "checkpoint_freq", "resume_path",
        }
        assert expected_keys.issubset(d.keys())


class TestTrainingConfigYAML:
    def test_round_trip(self):
        config = TrainingConfig(learning_rate=5e-7, n_steps=256, movement="right_only")

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "config.yaml")
            config.save(path)
            loaded = TrainingConfig.from_yaml(path)

        assert loaded.learning_rate == pytest.approx(5e-7)
        assert loaded.n_steps == 256
        assert loaded.movement == "right_only"

    def test_from_yaml_ignores_unknown_keys(self):
        """Extra keys in the YAML file should not raise an error."""
        import yaml

        data = {"learning_rate": 1e-5, "unknown_future_key": "ignore_me"}

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "config.yaml")
            with open(path, "w") as fh:
                yaml.safe_dump(data, fh)
            # Should not raise
            config = TrainingConfig.from_yaml(path)

        assert config.learning_rate == pytest.approx(1e-5)

    def test_save_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            nested = os.path.join(tmp, "deep", "nested", "config.yaml")
            TrainingConfig().save(nested)
            assert os.path.exists(nested)
