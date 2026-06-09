"""Config loading: defaults YAML + CLI dotlist overrides via OmegaConf."""
from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

_DEFAULT = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


def load_config(argv: list[str] | None = None, default_path: str | Path = _DEFAULT):
    """Load defaults and apply ``key=value`` overrides from ``argv``.

    Example: ``load_config(["data.root=./data/crack500", "trainer.epochs=10"])``.
    """
    cfg = OmegaConf.load(str(default_path))
    if argv:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(argv)))
    # Resolve device lazily here so downstream code gets a concrete string.
    from .utils import resolve_device

    cfg.device = resolve_device(cfg.device)
    return cfg
