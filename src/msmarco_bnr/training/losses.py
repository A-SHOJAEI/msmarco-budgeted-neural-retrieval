from __future__ import annotations

import torch
import torch.nn.functional as F


def retrieval_ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Standard cross-entropy over candidate documents.
    logits: [B, C], labels: [B] with 0 <= labels[i] < C
    """
    return F.cross_entropy(logits, labels)


def distillation_kl(
    student_logits: torch.Tensor,
    teacher_scores: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    """
    KL( teacher || student ) with temperature scaling.
    Shapes: [B, C]
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    t = float(temperature)
    t_scores = teacher_scores / t
    s_scores = student_logits / t
    teacher_probs = F.softmax(t_scores, dim=-1)
    student_logprobs = F.log_softmax(s_scores, dim=-1)
    # Multiply by t^2 as in standard distillation.
    return F.kl_div(student_logprobs, teacher_probs, reduction="batchmean") * (t * t)

