from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@dataclass
class CrossEncoderTeacher:
    name: str
    device: torch.device
    max_length: int
    tokenizer: any
    model: any

    @classmethod
    def load(cls, name: str, device: torch.device, *, max_length: int) -> "CrossEncoderTeacher":
        tok = AutoTokenizer.from_pretrained(name, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(name)
        model.eval()
        model.to(device)
        return cls(name=name, device=device, max_length=max_length, tokenizer=tok, model=model)

    @torch.no_grad()
    def score_matrix(
        self,
        queries: Sequence[str],
        candidates: Sequence[str],
        *,
        batch_size: int = 32,
    ) -> torch.Tensor:
        """
        Score every (query_i, cand_j) pair, returning [B, C] float32 tensor on CPU.
        This is O(B*C); intended for small batches (smoke/dev).
        """
        q = list(queries)
        c = list(candidates)
        B = len(q)
        C = len(c)
        scores = torch.empty((B, C), dtype=torch.float32)

        # Flatten pairs to feed cross-encoder, then reshape.
        pairs_q: List[str] = []
        pairs_c: List[str] = []
        for qi in q:
            pairs_q.extend([qi] * C)
            pairs_c.extend(c)

        # Run in chunks to control memory.
        all_scores: List[torch.Tensor] = []
        for i in range(0, len(pairs_q), batch_size):
            qq = pairs_q[i : i + batch_size]
            cc = pairs_c[i : i + batch_size]
            enc = self.tokenizer(
                qq,
                cc,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            out = self.model(**enc)
            logits = out.logits
            # Many cross-encoders are single-regression head: [N,1] or [N]
            if logits.ndim == 2 and logits.shape[1] == 1:
                logits = logits[:, 0]
            all_scores.append(logits.detach().to("cpu", dtype=torch.float32))

        flat = torch.cat(all_scores, dim=0)
        if flat.numel() != B * C:
            raise RuntimeError("Teacher score shape mismatch")
        scores[:] = flat.view(B, C)
        return scores

    @torch.no_grad()
    def score_pairs(
        self,
        queries: Sequence[str],
        passages: Sequence[str],
        *,
        batch_size: int = 32,
    ) -> torch.Tensor:
        """
        Score aligned (query_i, passage_i) pairs. Returns [B] float32 on CPU.
        """
        q = list(queries)
        p = list(passages)
        if len(q) != len(p):
            raise ValueError("queries and passages must have same length")
        out_scores: List[torch.Tensor] = []
        for i in range(0, len(q), batch_size):
            qq = q[i : i + batch_size]
            pp = p[i : i + batch_size]
            enc = self.tokenizer(
                qq,
                pp,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            out = self.model(**enc)
            logits = out.logits
            if logits.ndim == 2 and logits.shape[1] == 1:
                logits = logits[:, 0]
            out_scores.append(logits.detach().to("cpu", dtype=torch.float32))
        return torch.cat(out_scores, dim=0)
