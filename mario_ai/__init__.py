"""
Mario AI — Reinforcement learning agent for Super Mario Bros.

A PPO-based agent that learns to play Super Mario Bros directly from
raw pixel observations using Stable Baselines3 and PyTorch.
"""

__version__ = "1.0.0"
__author__ = "Rhys"
__license__ = "MIT"

from .agent import MarioAgent
from .config import TrainingConfig
from .environment import MarioEnvironment

__all__ = ["MarioAgent", "TrainingConfig", "MarioEnvironment"]
