"""
Utility helpers: checkpoint management, display formatting.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional


# ── Checkpoint Helpers ────────────────────────────────────────────────────────

def find_latest_checkpoint(checkpoint_dir: str | Path) -> Optional[str]:
    """
    Scan *checkpoint_dir* and return the path of the checkpoint with the
    highest step number, or ``None`` if none exist.

    Example::

        path = find_latest_checkpoint("./train/")
        # → "./train/best_model_70000.zip"
    """
    path = Path(checkpoint_dir)
    if not path.exists():
        return None
    checkpoints = list(path.glob("best_model_*.zip"))
    if not checkpoints:
        return None
    return str(max(checkpoints, key=_extract_step))


def list_checkpoints(checkpoint_dir: str | Path) -> list[tuple[int, Path]]:
    """
    Return all checkpoints in *checkpoint_dir* sorted by step number
    (ascending).  Each element is a ``(step, path)`` tuple.
    """
    path = Path(checkpoint_dir)
    if not path.exists():
        return []
    pairs = [
        (_extract_step(p), p)
        for p in path.glob("best_model_*.zip")
    ]
    return sorted([(s, p) for s, p in pairs if s >= 0], key=lambda x: x[0])


def _extract_step(path: Path) -> int:
    """Parse the step number embedded in a checkpoint filename."""
    match = re.search(r"best_model_(\d+)", path.stem)
    return int(match.group(1)) if match else -1


# ── Display Helpers ───────────────────────────────────────────────────────────

def format_config_table(config_dict: dict) -> str:
    """Render a config dict as a simple two-column plain-text table."""
    lines = [f"  {'Key':<28}  Value", "  " + "─" * 50]
    for k, v in config_dict.items():
        lines.append(f"  {k:<28}  {v}")
    return "\n".join(lines)


def human_size(num_bytes: int) -> str:
    """Convert bytes to a human-readable string (KB / MB / GB)."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num_bytes) < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024  # type: ignore[assignment]
    return f"{num_bytes:.1f} TB"
