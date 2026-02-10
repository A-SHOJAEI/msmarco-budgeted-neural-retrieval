from __future__ import annotations

import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from msmarco_bnr.data.msmarco import load_collection, load_queries, load_triples
from msmarco_bnr.models.biencoder import BiEncoder, BiEncoderConfig
from msmarco_bnr.training.losses import distillation_kl, retrieval_ce_loss
from msmarco_bnr.training.teacher import CrossEncoderTeacher
from msmarco_bnr.utils.io import ensure_dir, write_json
from msmarco_bnr.utils.repro import set_reproducibility


def _auto_device(cfg: Dict[str, Any]) -> torch.device:
    dev = cfg["run"].get("device", "auto")
    if dev == "auto":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return torch.device(dev)


class TriplesDataset(Dataset):
    def __init__(
        self,
        queries: Dict[int, str],
        passages: Dict[int, str],
        triples: List[Tuple[int, int, int]],
    ):
        self.queries = queries
        self.passages = passages
        self.triples = triples

    def __len__(self) -> int:
        return len(self.triples)

    def __getitem__(self, idx: int) -> Tuple[str, str, str]:
        qid, pos, neg = self.triples[idx]
        return self.queries[qid], self.passages[pos], self.passages[neg]


def _load_training_corpus(cfg: Dict[str, Any]) -> Tuple[Dict[int, str], Dict[int, str], List[Tuple[int, int, int]]]:
    ds = cfg["dataset"]
    root = Path(ds["root_dir"])
    if ds["type"] == "smoke_synthetic":
        # Use MS MARCO-like filenames produced by smoke generator.
        queries = load_queries(root / "queries.dev.small.tsv")
        triples = load_triples(root / "triples.train.small.tsv")
        passages = {pid: text for pid, text in load_collection(root / "collection.tsv")}
        return queries, passages, triples

    if ds["type"] == "msmarco_passage":
        triples_tsv = root / ds.get("triples_tsv", "triples.train.small.tsv")
        queries_tsv = root / ds.get("train_queries_tsv", "queries.train.tsv")
        max_triples = ds.get("max_triples")
        max_passages = ds.get("max_passages")
        queries = load_queries(queries_tsv)
        triples = load_triples(triples_tsv, max_triples=max_triples)
        passages = {pid: text for pid, text in load_collection(root / "collection.tsv", max_passages=max_passages)}
        return queries, passages, triples

    raise ValueError(f"Unknown dataset.type: {ds['type']}")


def _collate_batch(tokenizer, max_length: int):
    def fn(batch: List[Tuple[str, str, str]]):
        q, p, n = zip(*batch)
        q_enc = tokenizer(
            list(q),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        p_enc = tokenizer(
            list(p),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        n_enc = tokenizer(
            list(n),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return q_enc, p_enc, n_enc, list(q), list(p), list(n)

    return fn


def _save_checkpoint(model: BiEncoder, path: Path, extra: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    obj = {"model_state_dict": model.state_dict(), "extra": extra}
    torch.save(obj, path)


def _train_one(cfg: Dict[str, Any], *, distill_enabled: bool, run_id: str) -> Dict[str, Any]:
    device = _auto_device(cfg)
    seed = int(cfg["run"]["seed"])
    deterministic = bool(cfg["run"].get("deterministic", True))
    set_reproducibility(seed, deterministic=deterministic)

    artifacts_dir = Path(cfg["artifacts_dir"])
    ckpt_dir = artifacts_dir / "checkpoints" / run_id
    ensure_dir(ckpt_dir)

    queries, passages, triples = _load_training_corpus(cfg)

    model_cfg = BiEncoderConfig(
        encoder_name=str(cfg["model"]["encoder_name"]),
        projection_dim=int(cfg["model"]["projection_dim"]),
        normalize=bool(cfg["model"].get("normalize", True)),
        pooling=str(cfg["model"].get("pooling", "mean")),
    )
    model = BiEncoder(model_cfg).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.encoder_name, use_fast=True)

    train_cfg = cfg["train"]
    ds = TriplesDataset(queries, passages, triples)
    loader = DataLoader(
        ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=0,
        collate_fn=_collate_batch(tokenizer, int(train_cfg["max_length"])),
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    teacher = None
    dist_cfg = cfg.get("distillation", {})
    if distill_enabled:
        teacher = CrossEncoderTeacher.load(
            str(dist_cfg["teacher_name"]),
            device=device,
            max_length=int(dist_cfg.get("max_length", 256)),
        )

    temperature = float(dist_cfg.get("temperature", 1.0))
    alpha_kl = float(dist_cfg.get("alpha_kl", 0.5))

    log_every = int(train_cfg.get("log_every", 50))
    grad_clip = float(train_cfg.get("grad_clip_norm", 1.0))
    epochs = int(train_cfg.get("epochs", 1))

    step = 0
    losses: List[float] = []
    t0 = time.time()
    model.train()
    for epoch in range(epochs):
        for q_enc, p_enc, n_enc, q_texts, p_texts, n_texts in loader:
            step += 1
            q_enc = {k: v.to(device) for k, v in q_enc.items()}
            p_enc = {k: v.to(device) for k, v in p_enc.items()}
            n_enc = {k: v.to(device) for k, v in n_enc.items()}

            q_emb = model(**q_enc)
            p_emb = model(**p_enc)
            n_emb = model(**n_enc)
            cand = torch.cat([p_emb, n_emb], dim=0)  # [2B, D]
            logits = q_emb @ cand.t()  # [B, 2B]
            labels = torch.arange(q_emb.shape[0], device=device)
            loss = retrieval_ce_loss(logits, labels)

            if teacher is not None:
                # Distill only over aligned (pos, neg) for each query to avoid O(B^2) teacher cost.
                # This matches the plan's "soft scores over mined negatives" while staying scalable.
                tp = teacher.score_pairs(q_texts, p_texts).to(device)
                tn = teacher.score_pairs(q_texts, n_texts).to(device)
                teacher_local = torch.stack([tp, tn], dim=1)  # [B,2]
                student_local = torch.stack([(q_emb * p_emb).sum(dim=1), (q_emb * n_emb).sum(dim=1)], dim=1)  # [B,2]
                kl = distillation_kl(student_local, teacher_local, temperature=temperature)
                loss = (1.0 - alpha_kl) * loss + alpha_kl * kl

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            losses.append(float(loss.detach().cpu().item()))
            if step % log_every == 0:
                avg = float(np.mean(losses[-log_every:]))
                write_json(
                    ckpt_dir / "train_log.json",
                    {
                        "run_id": run_id,
                        "step": step,
                        "epoch": epoch,
                        "avg_loss_recent": avg,
                        "device": str(device),
                    },
                )

    dt = time.time() - t0
    final_path = ckpt_dir / "final.pt"
    _save_checkpoint(model, final_path, extra={"model_cfg": asdict(model_cfg), "run_id": run_id})
    return {
        "run_id": run_id,
        "distillation_enabled": bool(teacher is not None),
        "checkpoint_path": str(final_path),
        "num_triples": len(triples),
        "num_passages_loaded": len(passages),
        "train_seconds": float(dt),
        "steps": int(step),
        "device": str(device),
        "model": asdict(model_cfg),
    }


def train_biencoder(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Train:
    - distilled bi-encoder (if distillation.enabled)
    - ablation: no-distillation (contrastive-only), as specified in the plan.
    """
    dist_cfg = cfg.get("distillation", {})
    dist_enabled = bool(dist_cfg.get("enabled", False))

    outputs: Dict[str, Any] = {
        "runs": [],
    }
    if dist_enabled:
        outputs["runs"].append(_train_one(cfg, distill_enabled=True, run_id="dense_distill"))
        outputs["runs"].append(_train_one(cfg, distill_enabled=False, run_id="dense_nodistill"))
    else:
        outputs["runs"].append(_train_one(cfg, distill_enabled=False, run_id="dense_nodistill"))
    return outputs
