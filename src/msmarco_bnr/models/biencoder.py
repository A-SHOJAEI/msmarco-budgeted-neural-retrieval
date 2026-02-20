from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    # last_hidden_state: [B, T, H], attention_mask: [B, T]
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return summed / denom


@dataclass(frozen=True)
class BiEncoderConfig:
    encoder_name: str
    projection_dim: int
    normalize: bool
    pooling: str = "mean"


class BiEncoder(nn.Module):
    def __init__(self, cfg: BiEncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = AutoModel.from_pretrained(cfg.encoder_name)
        hidden = int(getattr(self.encoder.config, "hidden_size"))
        if cfg.projection_dim <= 0:
            raise ValueError("projection_dim must be positive")
        self.proj: nn.Module
        if cfg.projection_dim == hidden:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Linear(hidden, cfg.projection_dim, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        if self.cfg.pooling != "mean":
            raise ValueError(f"Unsupported pooling: {self.cfg.pooling}")
        emb = mean_pool(out.last_hidden_state, attention_mask)
        emb = self.proj(emb)
        if self.cfg.normalize:
            emb = F.normalize(emb, p=2, dim=-1)
        return emb

    @property
    def dim(self) -> int:
        if isinstance(self.proj, nn.Identity):
            return int(getattr(self.encoder.config, "hidden_size"))
        return int(self.cfg.projection_dim)

