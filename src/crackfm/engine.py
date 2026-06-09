"""Plain-PyTorch training/evaluation engine.

Deliberately framework-light (no Lightning) so it runs reliably on Kaggle's free
tier with minimal dependencies. Handles AMP, grad clipping, checkpointing and
per-epoch validation with the CrackFM metric suite.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .metrics import MetricAccumulator
from .utils import Logger


def _to_mask(logits: torch.Tensor) -> np.ndarray:
    return (torch.sigmoid(logits) > 0.5).squeeze(1).cpu().numpy().astype(bool)


@torch.no_grad()
def evaluate(model, loader, device: str, boundary_tolerance: int = 2) -> dict[str, float]:
    model.eval()
    acc = MetricAccumulator()
    for batch in loader:
        images = batch["image"].to(device)
        gts = batch["mask"].squeeze(1).cpu().numpy().astype(bool)
        logits = model(images)
        preds = _to_mask(logits)
        for p, g in zip(preds, gts):
            acc.update(p, g, boundary_tolerance)
    return acc.compute()


class Trainer:
    def __init__(self, model, loss_fn, cfg, logger: Logger | None = None) -> None:
        self.cfg = cfg
        self.device = cfg.device
        self.model = model.to(self.device)
        self.loss_fn = loss_fn
        self.opt = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=cfg.trainer.lr,
            weight_decay=cfg.trainer.weight_decay,
        )
        self.use_amp = bool(cfg.trainer.amp) and self.device == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.logger = logger or Logger(backend="none", out_dir=cfg.trainer.out_dir)
        self.out_dir = Path(cfg.trainer.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.best = -1.0
        self._step = 0

    def _train_epoch(self, loader, epoch: int) -> None:
        self.model.train()
        for i, batch in enumerate(loader):
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)
            self.opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=self.use_amp):
                logits = self.model(images)
                loss = self.loss_fn(logits, masks)
            self.scaler.scale(loss).backward()
            if self.cfg.trainer.grad_clip:
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.trainer.grad_clip)
            self.scaler.step(self.opt)
            self.scaler.update()
            self._step += 1
            if i % self.cfg.trainer.log_every == 0:
                self.logger.log({"train/loss": float(loss.item()), "epoch": epoch}, self._step)

    def fit(self, train_loader, val_loader) -> dict[str, float]:
        metric_key = self.cfg.trainer.save_best_metric
        best_metrics: dict[str, float] = {}
        for epoch in range(self.cfg.trainer.epochs):
            self._train_epoch(train_loader, epoch)
            val = evaluate(self.model, val_loader, self.device, self.cfg.eval.boundary_tolerance)
            self.logger.log({f"val/{k}": v for k, v in val.items()} | {"epoch": epoch}, self._step)
            self._save("last.pt", epoch, val)
            if val.get(metric_key, -1.0) > self.best:
                self.best = val[metric_key]
                best_metrics = val
                self._save("best.pt", epoch, val)
            print(f"[epoch {epoch}] " + "  ".join(f"{k}={v:.4f}" for k, v in val.items()))
        self.logger.finish()
        return best_metrics

    def _save(self, name: str, epoch: int, metrics: dict[str, float]) -> None:
        torch.save(
            {
                "model": self.model.state_dict(),
                "epoch": epoch,
                "metrics": metrics,
                "config": self.cfg,
            },
            self.out_dir / name,
        )
