"""
Mario AI — Command-line interface

Usage
-----
    python main.py train                               # train from scratch
    python main.py train --resume ./train/best_model_70000
    python main.py train --timesteps 500000 --lr 5e-7
    python main.py play                                # watch latest checkpoint
    python main.py play --model ./train/best_model_70000 --episodes 3
    python main.py checkpoints                         # list saved checkpoints
    python main.py info                                # show config + env info
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ── Optional rich for pretty terminal output ──────────────────────────────────
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    _console = Console()
    HAS_RICH = True
except ImportError:
    _console = None  # type: ignore[assignment]
    HAS_RICH = False

# ── Banner ────────────────────────────────────────────────────────────────────

_BANNER = """\
 ███╗   ███╗ █████╗ ██████╗ ██╗ ██████╗      █████╗ ██╗
 ████╗ ████║██╔══██╗██╔══██╗██║██╔═══██╗    ██╔══██╗██║
 ██╔████╔██║███████║██████╔╝██║██║   ██║    ███████║██║
 ██║╚██╔╝██║██╔══██║██╔══██╗██║██║   ██║    ██╔══██║██║
 ██║ ╚═╝ ██║██║  ██║██║  ██║██║╚██████╔╝    ██║  ██║██║
 ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝ ╚═════╝    ╚═╝  ╚═╝╚═╝
         Reinforcement Learning Agent  ·  PPO + CNN\
"""


def _print_banner() -> None:
    if HAS_RICH:
        _console.print(Panel(_BANNER, style="bold green", expand=False))
    else:
        print(_BANNER)
        print()


def _log(msg: str, style: str = "") -> None:
    if HAS_RICH:
        _console.print(f"[{style}]{msg}[/{style}]" if style else msg)
    else:
        print(msg)


# ── Subcommand: train ─────────────────────────────────────────────────────────

def cmd_train(args: argparse.Namespace) -> None:
    """Train (or resume training) the Mario PPO agent."""
    from mario_ai.agent import MarioAgent
    from mario_ai.config import TrainingConfig
    from mario_ai.environment import MarioEnvironment

    # ── Resolve config ────────────────────────────────────────────────────────
    config_path = Path(args.config)
    if config_path.exists():
        config = TrainingConfig.from_yaml(config_path)
        _log(f"Loaded config from {config_path}", style="dim")
    else:
        config = TrainingConfig()
        _log(f"Config file not found at '{config_path}', using defaults.", style="yellow")

    # ── Apply CLI overrides ───────────────────────────────────────────────────
    if args.timesteps:
        config.total_timesteps = args.timesteps
    if args.lr:
        config.learning_rate = args.lr
    if args.n_steps:
        config.n_steps = args.n_steps
    if args.checkpoint_freq:
        config.checkpoint_freq = args.checkpoint_freq
    if args.movement:
        config.movement = args.movement
    if args.resume:
        config.resume_path = args.resume

    _print_banner()
    _print_config(config)

    # ── Build env ─────────────────────────────────────────────────────────────
    env_builder = MarioEnvironment(config)
    env = env_builder.build()

    try:
        agent = MarioAgent(config, env)

        if config.resume_path:
            _log(f"\nResuming from checkpoint: {config.resume_path}", style="bold yellow")
            agent.load(config.resume_path)
        else:
            agent.build()

        _log(
            f"\nStarting training for {config.total_timesteps:,} timesteps ...\n",
            style="bold cyan",
        )
        agent.train()
        _log("\nTraining complete!", style="bold green")

        # Save the final config alongside checkpoints for reproducibility
        config.save(Path(config.checkpoint_dir) / "run_config.yaml")
        _log(
            f"Run config saved to {config.checkpoint_dir}run_config.yaml",
            style="dim",
        )

    finally:
        env_builder.close()


# ── Subcommand: play ──────────────────────────────────────────────────────────

def cmd_play(args: argparse.Namespace) -> None:
    """Watch a trained agent play Super Mario Bros."""
    from mario_ai.agent import MarioAgent
    from mario_ai.config import TrainingConfig
    from mario_ai.environment import MarioEnvironment
    from mario_ai.utils import find_latest_checkpoint

    config = TrainingConfig()

    # ── Resolve model path ────────────────────────────────────────────────────
    model_path = args.model
    if not model_path:
        model_path = find_latest_checkpoint(config.checkpoint_dir)
        if not model_path:
            _log(
                "No model found. Train one first with:\n"
                "  python main.py train",
                style="bold red",
            )
            sys.exit(1)
        _log(f"Auto-selected latest checkpoint: {model_path}", style="yellow")

    model_path = str(model_path).removesuffix(".zip")

    _print_banner()
    _log(f"Model : {model_path}", style="cyan")
    _log(f"Episodes : {args.episodes}\n", style="cyan")

    env_builder = MarioEnvironment(config)
    env = env_builder.build()

    try:
        agent = MarioAgent(config, env)
        agent.load(model_path)

        total_rewards: list[float] = []

        for ep in range(1, args.episodes + 1):
            state = env.reset()
            done = False
            ep_reward = 0.0
            steps = 0

            while not done:
                action, _ = agent.predict(state)
                state, reward, done_arr, info = env.step(action)

                # Handle both vectorised (array) and scalar reward
                step_reward = (
                    float(reward[0]) if hasattr(reward, "__len__") else float(reward)
                )
                is_done = (
                    bool(done_arr[0]) if hasattr(done_arr, "__len__") else bool(done_arr)
                )
                ep_reward += step_reward
                steps += 1
                done = is_done

                if args.render:
                    env.render()

            total_rewards.append(ep_reward)
            _log(
                f"  Episode {ep:>2}/{args.episodes}"
                f"  |  Reward: {ep_reward:>8.1f}"
                f"  |  Steps: {steps:>5,}",
                style="green",
            )

        if total_rewards:
            avg = sum(total_rewards) / len(total_rewards)
            _log(
                f"\nAverage reward over {args.episodes} episode(s): {avg:.1f}",
                style="bold cyan",
            )

    finally:
        env_builder.close()


# ── Subcommand: checkpoints ───────────────────────────────────────────────────

def cmd_checkpoints(args: argparse.Namespace) -> None:
    """List all saved model checkpoints in the training directory."""
    from mario_ai.utils import find_latest_checkpoint, list_checkpoints

    checkpoints = list_checkpoints(args.dir)

    if not checkpoints:
        _log(f"No checkpoints found in '{args.dir}'.", style="yellow")
        _log("Train the agent first:  python main.py train", style="dim")
        return

    latest_path = find_latest_checkpoint(args.dir)

    if HAS_RICH:
        table = Table(
            title=f"[bold]Checkpoints — {args.dir}[/bold]",
            show_lines=True,
            header_style="bold cyan",
        )
        table.add_column("#", style="dim", width=4, justify="right")
        table.add_column("Step", style="cyan", justify="right")
        table.add_column("Filename", style="white")
        table.add_column("", style="green", width=8)

        for i, (step, path) in enumerate(checkpoints, 1):
            is_latest = str(path) == latest_path
            table.add_row(
                str(i),
                f"{step:,}",
                path.name,
                "← latest" if is_latest else "",
            )

        _console.print(table)
        _log(f"Total: {len(checkpoints)} checkpoint(s)", style="dim")
    else:
        print(f"\nCheckpoints in '{args.dir}':")
        print(f"  {'#':>3}  {'Step':>10}  Filename")
        print("  " + "─" * 50)
        for i, (step, path) in enumerate(checkpoints, 1):
            latest_marker = "  ← latest" if str(path) == latest_path else ""
            print(f"  {i:>3}  {step:>10,}  {path.name}{latest_marker}")
        print(f"\n  Total: {len(checkpoints)} checkpoint(s)")


# ── Subcommand: info ──────────────────────────────────────────────────────────

def cmd_info(_args: argparse.Namespace) -> None:
    """Display default configuration, environment details, and movement presets."""
    from mario_ai.config import TrainingConfig
    from mario_ai.environment import MOVEMENT_PRESETS

    _print_banner()
    config = TrainingConfig()

    if HAS_RICH:
        cfg_table = Table(
            title="Default Configuration",
            header_style="bold cyan",
            show_lines=False,
        )
        cfg_table.add_column("Parameter", style="cyan", width=22)
        cfg_table.add_column("Value", style="white")
        for k, v in config.to_dict().items():
            cfg_table.add_row(k, str(v))
        _console.print(cfg_table)

        mv_table = Table(
            title="Movement Presets",
            header_style="bold cyan",
        )
        mv_table.add_column("Name", style="cyan")
        mv_table.add_column("Actions", justify="right", style="green")
        mv_table.add_column("Notes", style="dim")
        notes = {
            "right_only": "Fastest initial convergence",
            "simple":     "Balanced default (recommended)",
            "complex":    "Full control, hardest to learn",
        }
        for name, actions in MOVEMENT_PRESETS.items():
            mv_table.add_row(name, str(len(actions)), notes.get(name, ""))
        _console.print(mv_table)

    else:
        from mario_ai.utils import format_config_table
        print("\nDefault Configuration:")
        print(format_config_table(config.to_dict()))
        print("\nMovement Presets:")
        for name, actions in MOVEMENT_PRESETS.items():
            print(f"  {name:<14}  {len(actions)} actions")


# ── Internal helpers ──────────────────────────────────────────────────────────

def _print_config(config) -> None:
    if HAS_RICH:
        table = Table(
            title="Training Configuration",
            header_style="bold cyan",
            show_lines=False,
        )
        table.add_column("Parameter", style="cyan", width=22)
        table.add_column("Value", style="white")
        for k, v in config.to_dict().items():
            if v is not None:
                table.add_row(k, str(v))
        _console.print(table)
    else:
        from mario_ai.utils import format_config_table
        print("\nTraining Configuration:")
        print(format_config_table(config.to_dict()))


# ── Argument parser ───────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mario-ai",
        description="Super Mario Bros reinforcement learning agent (PPO + CNN).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python main.py train                                  train from scratch
  python main.py train --timesteps 500000               train for 500k steps
  python main.py train --resume ./train/best_model_70000  resume a checkpoint
  python main.py train --movement right_only            use right-only controls
  python main.py play                                   watch the latest model
  python main.py play --model ./train/best_model_70000 --episodes 3
  python main.py checkpoints                            list all checkpoints
  python main.py info                                   show config + env info
        """,
    )

    sub = parser.add_subparsers(dest="command", metavar="<command>")
    sub.required = True

    # ── train ─────────────────────────────────────────────────────────────────
    p_train = sub.add_parser("train", help="Train a new agent or resume training")
    p_train.add_argument(
        "--config",
        default="configs/default.yaml",
        metavar="PATH",
        help="YAML config file (default: configs/default.yaml)",
    )
    p_train.add_argument(
        "--timesteps", type=int, metavar="N",
        help="Total environment steps to train for",
    )
    p_train.add_argument(
        "--resume", metavar="PATH",
        help="Path to a checkpoint .zip file to continue training from",
    )
    p_train.add_argument(
        "--lr", type=float, metavar="FLOAT",
        help="Learning rate (default: 1e-6)",
    )
    p_train.add_argument(
        "--n-steps", type=int, dest="n_steps", metavar="N",
        help="Rollout steps per update (default: 512)",
    )
    p_train.add_argument(
        "--checkpoint-freq", type=int, dest="checkpoint_freq", metavar="N",
        help="Save a checkpoint every N steps (default: 10000)",
    )
    p_train.add_argument(
        "--movement",
        choices=["simple", "right_only", "complex"],
        help="Action-space preset (default: simple)",
    )

    # ── play ──────────────────────────────────────────────────────────────────
    p_play = sub.add_parser("play", help="Watch a trained agent play Mario")
    p_play.add_argument(
        "--model", metavar="PATH",
        help="Checkpoint path (default: auto-selects latest)",
    )
    p_play.add_argument(
        "--episodes", type=int, default=5, metavar="N",
        help="Number of episodes to run (default: 5)",
    )
    p_play.add_argument(
        "--no-render", dest="render", action="store_false",
        help="Disable the game window (useful for headless evaluation)",
    )
    p_play.set_defaults(render=True)

    # ── checkpoints ───────────────────────────────────────────────────────────
    p_ckpt = sub.add_parser("checkpoints", help="List all saved model checkpoints")
    p_ckpt.add_argument(
        "--dir", default="./train/", metavar="PATH",
        help="Directory to scan (default: ./train/)",
    )

    # ── info ──────────────────────────────────────────────────────────────────
    sub.add_parser("info", help="Show default config and environment info")

    return parser


# ── Entry point ───────────────────────────────────────────────────────────────

_COMMANDS = {
    "train":       cmd_train,
    "play":        cmd_play,
    "checkpoints": cmd_checkpoints,
    "info":        cmd_info,
}


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _COMMANDS[args.command](args)


if __name__ == "__main__":
    main()
