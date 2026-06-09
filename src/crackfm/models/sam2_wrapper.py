"""SAM 2 backbone wrapped for CrackFM.

This wraps the official SAM 2 image-segmentation model so it produces a single
crack logit map. The image encoder is frozen and adapted with LoRA; the mask
decoder is fine-tuned. Geometry-adaptive prompts (see ``crackfm.prompting``) can
be fed at inference.

SAM 2 is an *optional* dependency. If it is not installed, constructing this
class raises a clear error pointing at the install command — the rest of
CrackFM (U-Net path, prompting, severity, metrics) keeps working regardless.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .lora import inject_lora

_INSTALL_HINT = (
    "SAM 2 is not installed. Install with:\n"
    '  pip install "git+https://github.com/facebookresearch/sam2.git"\n'
    "and download a checkpoint from the SAM 2 model zoo."
)


class SAM2CrackSegmenter(nn.Module):
    def __init__(
        self,
        sam2_checkpoint: str,
        sam2_config: str = "sam2_hiera_s.yaml",
        lora_rank: int = 8,
        lora_alpha: int = 16,
        freeze_image_encoder: bool = True,
    ) -> None:
        super().__init__()
        try:
            from sam2.build_sam import build_sam2
        except Exception as exc:  # pragma: no cover - optional dep
            raise ImportError(_INSTALL_HINT) from exc

        self.sam2 = build_sam2(sam2_config, sam2_checkpoint)
        if freeze_image_encoder:
            for p in self.sam2.image_encoder.parameters():
                p.requires_grad_(False)
            n = inject_lora(self.sam2.image_encoder, lora_rank, lora_alpha)
            print(f"[crackfm] Injected LoRA into {n} linear layers of the SAM 2 image encoder.")

    def forward(self, images: torch.Tensor, prompts: dict | None = None) -> torch.Tensor:
        """Return single-channel logits at input resolution.

        ``prompts`` (optional) may carry ``point_coords`` / ``point_labels`` /
        ``boxes`` from ``crackfm.prompting``; when ``None`` the model is run in
        dense/automatic mode. The exact decoder plumbing depends on the SAM 2
        version and is intentionally isolated here so it is the only place to
        touch when upgrading SAM 2.
        """
        raise NotImplementedError(
            "Connect SAM 2's image_encoder + sam_mask_decoder here for your installed "
            "SAM 2 version. Encoder+LoRA wiring is done; decoder plumbing is the "
            "remaining integration step. Use model.name=unet to train end-to-end now."
        )
