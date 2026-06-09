"""Model factory."""
from __future__ import annotations


def build_model(cfg):
    """Construct a model from config. ``cfg.model.name`` in {unet, sam2_lora}."""
    name = cfg.model.name.lower()
    if name == "unet":
        from .unet import UNet

        return UNet(
            in_channels=cfg.model.in_channels,
            base_channels=cfg.model.base_channels,
        )
    if name in ("sam2", "sam2_lora"):
        from .sam2_wrapper import SAM2CrackSegmenter

        if not cfg.model.sam2_checkpoint:
            raise ValueError("model.sam2_checkpoint must be set for the sam2_lora model.")
        return SAM2CrackSegmenter(
            sam2_checkpoint=cfg.model.sam2_checkpoint,
            sam2_config=cfg.model.sam2_config,
            lora_rank=cfg.model.lora_rank,
            lora_alpha=cfg.model.lora_alpha,
            freeze_image_encoder=cfg.model.freeze_image_encoder,
        )
    raise ValueError(f"Unknown model.name: {cfg.model.name!r}")
