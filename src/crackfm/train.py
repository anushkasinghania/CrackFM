"""Training entrypoint.

Usage:
    python -m crackfm.train data.root=./data/crack500 model.name=unet trainer.epochs=30
"""
from __future__ import annotations

import sys

from .config import load_config
from .data import build_dataloaders
from .engine import Trainer
from .losses import CrackLoss
from .models import build_model
from .utils import Logger, set_seed


def main(argv: list[str] | None = None) -> None:
    cfg = load_config(argv if argv is not None else sys.argv[1:])
    set_seed(cfg.seed)
    print(f"[crackfm] device={cfg.device}  model={cfg.model.name}  root={cfg.data.root}")

    train_loader, val_loader = build_dataloaders(cfg)
    model = build_model(cfg)
    loss_fn = CrackLoss(
        dice=cfg.loss.dice,
        bce=cfg.loss.bce,
        boundary=cfg.loss.boundary,
        tversky_beta=cfg.loss.tversky_beta,
    )
    logger = Logger(
        backend=cfg.tracking.backend,
        project=cfg.tracking.project,
        run_name=cfg.tracking.run_name,
        out_dir=cfg.trainer.out_dir,
        config=dict(cfg),
    )
    trainer = Trainer(model, loss_fn, cfg, logger)
    best = trainer.fit(train_loader, val_loader)
    print("[crackfm] best:", "  ".join(f"{k}={v:.4f}" for k, v in best.items()))


if __name__ == "__main__":
    main()
