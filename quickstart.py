"""
quickstart.py — Single-file walkthrough of the Mario AI pipeline.

This script is the native Python equivalent of the original MarioAIPy notebook.
It walks through every stage in order: environment setup → preprocessing →
training → watching the agent play.

Dependencies (install once):
    pip install -r requirements.txt

Usage:
    python quickstart.py demo     # 200 random-action steps to test your setup
    python quickstart.py train    # train from scratch for 1 000 000 steps
    python quickstart.py play     # watch the latest saved model play
"""

from __future__ import annotations

import argparse
import os


# ══════════════════════════════════════════════════════════════════════════════
# 1. ENVIRONMENT SETUP
# ══════════════════════════════════════════════════════════════════════════════

def build_environment(movement: str = "simple", n_stack: int = 4):
    """
    Build the full preprocessing pipeline and return a ready-to-use env.

    Pipeline
    --------
    SuperMarioBros-v0 (240×256 RGB)
        → JoypadSpace          simplify the NES button set
        → GrayScaleObservation convert RGB → greyscale
        → DummyVecEnv          vectorise (required by Stable Baselines3)
        → VecFrameStack        stack the last *n_stack* frames so the agent
                               can perceive motion and velocity
    """
    import gym_super_mario_bros
    from gym.wrappers import GrayScaleObservation
    from gym_super_mario_bros.actions import (
        COMPLEX_MOVEMENT,
        RIGHT_ONLY,
        SIMPLE_MOVEMENT,
    )
    from nes_py.wrappers import JoypadSpace
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

    presets = {
        "simple":     SIMPLE_MOVEMENT,   # 7 actions  — recommended default
        "right_only": RIGHT_ONLY,         # 5 actions  — fastest early learning
        "complex":    COMPLEX_MOVEMENT,   # 12 actions — hardest to master
    }

    if movement not in presets:
        raise ValueError(f"Unknown movement '{movement}'. Choose: {list(presets)}")

    # 1. Base NES environment
    env = gym_super_mario_bros.make("SuperMarioBros-v0")

    # 2. Restrict the action space
    env = JoypadSpace(env, presets[movement])

    # 3. Greyscale (keep channel dim for SB3 compatibility)
    env = GrayScaleObservation(env, keep_dim=True)

    # 4. Vectorise
    env = DummyVecEnv([lambda: env])

    # 5. Stack frames so the agent sees motion, not just a single frozen frame
    env = VecFrameStack(env, n_stack, channels_order="last")

    return env


# ══════════════════════════════════════════════════════════════════════════════
# 2. TRAINING CALLBACK
# ══════════════════════════════════════════════════════════════════════════════

from stable_baselines3.common.callbacks import BaseCallback


class CheckpointCallback(BaseCallback):
    """
    Save a model snapshot every *check_freq* environment steps.

    Checkpoints are written to:
        {save_path}/best_model_{step}.zip
    """

    def __init__(self, check_freq: int, save_path: str, verbose: int = 1) -> None:
        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self) -> None:
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            path = os.path.join(self.save_path, f"best_model_{self.n_calls}")
            self.model.save(path)
            if self.verbose:
                print(f"  [✓] Checkpoint saved at step {self.n_calls:,} → {path}.zip")
        return True


# ══════════════════════════════════════════════════════════════════════════════
# 3. DEMO — random actions (sanity-check your install)
# ══════════════════════════════════════════════════════════════════════════════

def run_demo(steps: int = 200) -> None:
    """
    Run *steps* random actions in the raw (unprocessed) environment.
    A game window should pop up and Mario will flail around randomly.
    Press Ctrl-C to stop early.
    """
    import gym_super_mario_bros
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
    from nes_py.wrappers import JoypadSpace

    print(f"\n── Demo mode: {steps} random steps ──\n")
    env = gym_super_mario_bros.make("SuperMarioBros-v0")
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    done = True
    try:
        for step in range(steps):
            if done:
                state = env.reset()
            state, reward, done, info = env.step(env.action_space.sample())
            env.render()
    except KeyboardInterrupt:
        print("\nStopped early.")
    finally:
        env.close()

    print("Demo complete.")


# ══════════════════════════════════════════════════════════════════════════════
# 4. TRAIN
# ══════════════════════════════════════════════════════════════════════════════

def run_train(
    total_timesteps: int = 1_000_000,
    checkpoint_dir: str = "./train/",
    log_dir: str = "./logs/",
    checkpoint_freq: int = 10_000,
    learning_rate: float = 1e-6,
    n_steps: int = 512,
    resume_path: str | None = None,
) -> None:
    """
    Train (or resume training) a PPO agent on SuperMarioBros-v0.

    Key hyperparameters
    -------------------
    learning_rate   1e-6 is conservative but stable for visual RL.
                    Higher values (> 1e-5) often cause training collapse.
    n_steps         Rollout length per update. 512 is a good default.
    ent_coef        Entropy bonus keeps the agent exploring — important
                    early on so it doesn't get stuck at the first pipe.
    """
    from stable_baselines3 import PPO

    print(f"\n── Training for {total_timesteps:,} steps ──\n")

    env = build_environment()
    callback = CheckpointCallback(check_freq=checkpoint_freq, save_path=checkpoint_dir)

    if resume_path:
        print(f"Resuming from: {resume_path}")
        model = PPO.load(resume_path, env=env)
        reset_timesteps = False
    else:
        model = PPO(
            policy="CnnPolicy",
            env=env,
            verbose=1,
            tensorboard_log=log_dir,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
        )
        reset_timesteps = True

    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        reset_num_timesteps=reset_timesteps,
    )

    env.close()
    print("\nTraining complete.")
    print(f"Checkpoints saved in: {checkpoint_dir}")
    print(f"TensorBoard logs in:  {log_dir}")
    print("  → run: tensorboard --logdir ./logs/")


# ══════════════════════════════════════════════════════════════════════════════
# 5. PLAY — watch the trained agent
# ══════════════════════════════════════════════════════════════════════════════

def run_play(model_path: str | None = None, episodes: int = 5) -> None:
    """
    Load a saved model and watch it play.

    If *model_path* is not given, the latest checkpoint in ./train/ is used.
    """
    from stable_baselines3 import PPO

    # Auto-discover the latest checkpoint
    if model_path is None:
        import re
        from pathlib import Path

        checkpoints = list(Path("./train/").glob("best_model_*.zip"))
        if not checkpoints:
            print("No checkpoints found. Train the agent first:\n  python quickstart.py train")
            return
        model_path = str(max(
            checkpoints,
            key=lambda p: int(m.group(1)) if (m := re.search(r"(\d+)", p.stem)) else 0,
        ))
        print(f"Auto-selected: {model_path}")

    model_path = model_path.removesuffix(".zip")

    print(f"\n── Playing {episodes} episode(s) with: {model_path} ──\n")

    env = build_environment()
    model = PPO.load(model_path)

    for ep in range(1, episodes + 1):
        state = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action, _ = model.predict(state)
            state, reward, done_arr, info = env.step(action)
            total_reward += float(reward[0])
            done = bool(done_arr[0])
            env.render()

        print(f"  Episode {ep}/{episodes}  |  Total reward: {total_reward:.1f}")

    env.close()


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="quickstart",
        description="Mario AI — single-file quickstart script",
    )
    sub = parser.add_subparsers(dest="cmd", metavar="<command>")
    sub.required = True

    # demo
    p_demo = sub.add_parser("demo", help="Run random actions to test your setup")
    p_demo.add_argument("--steps", type=int, default=200, metavar="N")

    # train
    p_train = sub.add_parser("train", help="Train the PPO agent")
    p_train.add_argument("--timesteps",       type=int,   default=1_000_000)
    p_train.add_argument("--resume",          default=None, metavar="PATH")
    p_train.add_argument("--lr",              type=float, default=1e-6)
    p_train.add_argument("--checkpoint-freq", type=int,   default=10_000, dest="checkpoint_freq")

    # play
    p_play = sub.add_parser("play", help="Watch the trained agent play")
    p_play.add_argument("--model",    default=None, metavar="PATH")
    p_play.add_argument("--episodes", type=int,     default=5)

    args = parser.parse_args()

    if args.cmd == "demo":
        run_demo(steps=args.steps)
    elif args.cmd == "train":
        run_train(
            total_timesteps=args.timesteps,
            resume_path=args.resume,
            learning_rate=args.lr,
            checkpoint_freq=args.checkpoint_freq,
        )
    elif args.cmd == "play":
        run_play(model_path=args.model, episodes=args.episodes)


if __name__ == "__main__":
    main()
