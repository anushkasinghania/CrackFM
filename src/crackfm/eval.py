"""Evaluation entrypoint: load a checkpoint and report metrics + severity.

Usage:
    python -m crackfm.eval ckpt=runs/best.pt data.root=./data/crack500
"""
from __future__ import annotations

import sys

import numpy as np
import torch

from .config import load_config
from .data import build_dataloaders
from .engine import evaluate
from .models import build_model
from .severity import quantify
from .utils import set_seed


def main(argv: list[str] | None = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    # Pull ckpt=... out of the dotlist (it is not part of the default schema).
    ckpt_path = None
    rest = []
    for a in argv:
        if a.startswith("ckpt="):
            ckpt_path = a.split("=", 1)[1]
        else:
            rest.append(a)
    if ckpt_path is None:
        raise SystemExit("Provide ckpt=<path/to/checkpoint.pt>")

    cfg = load_config(rest)
    set_seed(cfg.seed)

    _, val_loader = build_dataloaders(cfg)
    model = build_model(cfg).to(cfg.device)
    state = torch.load(ckpt_path, map_location=cfg.device)
    model.load_state_dict(state["model"])

    metrics = evaluate(model, val_loader, cfg.device, cfg.eval.boundary_tolerance)
    print("[crackfm] eval metrics:")
    for k, v in metrics.items():
        print(f"  {k:12s} {v:.4f}")

    # Severity report on the first validation batch as a demonstration.
    model.eval()
    with torch.no_grad():
        batch = next(iter(val_loader))
        logits = model(batch["image"].to(cfg.device))
        pred = (torch.sigmoid(logits) > 0.5).squeeze(1).cpu().numpy().astype(bool)
    thresholds = tuple(cfg.severity.width_thresholds)
    print("[crackfm] severity (first batch):")
    for i, m in enumerate(pred):
        s = quantify(m, cfg.severity.mm_per_pixel, thresholds)
        print(f"  sample {i}: severity={s.severity:6s} "
              f"max_width={s.max_width:.2f}{s.unit} length={s.length:.1f}{s.unit}")


if __name__ == "__main__":
    main()
