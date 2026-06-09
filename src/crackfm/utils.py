"""Utilities: seeding, device selection, and an optional W&B logger."""
from __future__ import annotations

import csv
import os
import random
from pathlib import Path
from typing import Any

import numpy as np


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def resolve_device(pref: str = "auto") -> str:
    if pref != "auto":
        return pref
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class Logger:
    """Experiment logger that uses W&B when available, else logs to a CSV.

    Falls back silently to a no-op/CSV logger when ``wandb`` is not installed,
    ``WANDB_API_KEY`` is unset, or ``backend='none'`` — so runs never block on
    an external account.
    """

    def __init__(
        self,
        backend: str = "auto",
        project: str = "crackfm",
        run_name: str | None = None,
        out_dir: str | os.PathLike = "runs",
        config: dict[str, Any] | None = None,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._csv_path = self.out_dir / "metrics.csv"
        self._csv_header_written = self._csv_path.exists()
        self._wandb = None

        want_wandb = backend in ("auto", "wandb")
        has_key = bool(os.environ.get("WANDB_API_KEY"))
        if want_wandb and (has_key or backend == "wandb"):
            try:
                import wandb

                mode = "online" if has_key else "offline"
                self._wandb = wandb.init(
                    project=project, name=run_name, config=config or {}, mode=mode
                )
            except Exception as exc:  # pragma: no cover - environment dependent
                print(f"[crackfm] W&B unavailable ({exc}); logging to CSV only.")
                self._wandb = None

    def log(self, metrics: dict[str, float], step: int | None = None) -> None:
        if self._wandb is not None:
            self._wandb.log(metrics, step=step)
        row = {"step": step if step is not None else "", **metrics}
        write_header = not self._csv_header_written
        with open(self._csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
                self._csv_header_written = True
            writer.writerow(row)

    def finish(self) -> None:
        if self._wandb is not None:
            self._wandb.finish()
