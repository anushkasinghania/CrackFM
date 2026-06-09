"""Minimal LoRA for adapting a frozen foundation backbone.

A ``LoRALinear`` wraps an existing ``nn.Linear``, freezes it, and adds a
low-rank update ``B @ A`` (rank ``r``). ``inject_lora`` swaps matching linear
layers in-place so only the small A/B matrices train — the data-efficient
adaptation regime CrackFM relies on for small crack datasets.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = r
        self.scaling = alpha / r
        self.A = nn.Parameter(torch.zeros(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)
        # B stays zero so the adapted model starts identical to the base model.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scaling * (x @ self.A.t() @ self.B.t())


def inject_lora(
    module: nn.Module, r: int = 8, alpha: int = 16, target_substrings: tuple[str, ...] = ("proj", "qkv", "linear")
) -> int:
    """Replace matching ``nn.Linear`` layers with ``LoRALinear``. Returns count."""
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and any(s in name.lower() for s in target_substrings):
            setattr(module, name, LoRALinear(child, r, alpha))
            replaced += 1
        else:
            replaced += inject_lora(child, r, alpha, target_substrings)
    return replaced


def mark_only_lora_trainable(module: nn.Module) -> None:
    for n, p in module.named_parameters():
        p.requires_grad_(("A" in n.split(".")[-1]) or ("B" in n.split(".")[-1]))
