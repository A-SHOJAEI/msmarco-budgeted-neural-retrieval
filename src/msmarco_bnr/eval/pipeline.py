from __future__ import annotations

import os
import platform
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

# Optional heavy deps. Smoke runs should work without installing these.
try:
    import torch  # type: ignore
except Exception:  # noqa: BLE001
    torch = None  # type: ignore

try:
    from transformers import AutoTokenizer  # type: ignore
except Exception:  # noqa: BLE001
    AutoTokenizer = None  # type: ignore

from msmarco_bnr.bm25.pyserini import build_pyserini_msmarco_index, search_pyserini
from msmarco_bnr.bm25.rank_bm25 import build_rank_bm25, search_rank_bm25
from msmarco_bnr.data.msmarco import load_collection, load_qrels, load_queries
from msmarco_bnr.eval.metrics import compute_ir_metrics
from msmarco_bnr.index.faiss_index import FaissBuildSpec, build_faiss_index, save_faiss_index
from msmarco_bnr.utils.io import ensure_dir
from msmarco_bnr.utils.repro import set_reproducibility


def _auto_device(cfg: Dict[str, Any]):
    if torch is None:
        return "cpu"
    dev = cfg["run"].get("device", "auto")
    if dev == "auto":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return torch.device(dev)


def _load_eval_files(cfg: Dict[str, Any]) -> Tuple[Path, Path, Path]:
    ds = cfg["dataset"]
    root = Path(ds["root_dir"])
    queries_tsv = root / ds.get("eval_queries_tsv", "queries.dev.small.tsv")
    qrels_tsv = root / ds.get("eval_qrels_tsv", "qrels.dev.small.tsv")
    collection_tsv = root / "collection.tsv"
    for p in (queries_tsv, qrels_tsv, collection_tsv):
        if not p.exists():
            raise FileNotFoundError(p)
    return queries_tsv, qrels_tsv, collection_tsv


def _load_checkpoint(path: Path, device) -> Tuple["object", Dict[str, Any]]:
    if torch is None:
        raise RuntimeError("Dense evaluation requested but torch is not installed. Install `requirements-ml.txt`.")

    # Import lazily so smoke runs don't require torch/transformers.
    from msmarco_bnr.models.biencoder import BiEncoder, BiEncoderConfig

    obj = torch.load(path, map_location=device)
    extra = obj.get("extra", {})
    model_cfg = extra.get("model_cfg")
    if not model_cfg:
        raise RuntimeError(f"Missing model_cfg in checkpoint extra: {path}")
    cfg = BiEncoderConfig(
        encoder_name=str(model_cfg["encoder_name"]),
        projection_dim=int(model_cfg["projection_dim"]),
        normalize=bool(model_cfg.get("normalize", True)),
        pooling=str(model_cfg.get("pooling", "mean")),
    )
    model = BiEncoder(cfg)
    model.load_state_dict(obj["model_state_dict"])
    model.to(device)
    model.eval()
    return model, extra


def _encode_texts(
    model: "object",
    tokenizer,
    texts: List[str],
    *,
    device,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    if torch is None:
        raise RuntimeError("Dense evaluation requested but torch is not installed. Install `requirements-ml.txt`.")

    embs: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc).detach().to("cpu", dtype=torch.float32).numpy()
            embs.append(out)
    if not embs:
        dim = int(getattr(model, "dim"))
        return np.zeros((0, dim), dtype=np.float32)
    return np.concatenate(embs, axis=0)


def _dense_retrieve(
    qids: List[int],
    q_emb: np.ndarray,
    pid_to_text: Dict[int, str],
    passage_emb: np.ndarray,
    passage_ids: np.ndarray,
    *,
    index_spec: FaissBuildSpec,
    k: int,
    artifacts_dir: Path,
    run_id: str,
    index_tag: str,
    latency_max_queries: Optional[int],
) -> Tuple[Dict[int, List[int]], Dict[str, float], Dict[str, Any]]:
    index = build_faiss_index(passage_emb, passage_ids, index_spec)
    idx_path = artifacts_dir / "faiss" / run_id / f"{index_tag}.faiss"
    save_faiss_index(index, idx_path)
    index_bytes = idx_path.stat().st_size if idx_path.exists() else 0

    # Search. FAISS returns results aligned with query rows; map back to qids.
    if q_emb.dtype != np.float32:
        q_emb = q_emb.astype(np.float32, copy=False)

    # Time per-query latencies for a subset.
    Q = q_emb.shape[0]
    max_q = min(Q, int(latency_max_queries)) if latency_max_queries is not None else Q
    lat_ms: List[float] = []
    t0 = time.perf_counter()
    I_all: List[np.ndarray] = []
    for i in range(Q):
        if i < max_q:
            q0 = time.perf_counter()
            _, I = index.search(q_emb[i : i + 1], k)
            lat_ms.append((time.perf_counter() - q0) * 1000.0)
        else:
            _, I = index.search(q_emb[i : i + 1], k)
        I_all.append(I)
    t1 = time.perf_counter()
    I = np.concatenate(I_all, axis=0)  # [Q,k]
    retrieved: Dict[int, List[int]] = {}
    for row, qid in enumerate(qids):
        retrieved[qid] = [int(x) for x in I[row].tolist() if int(x) != -1]

    lat_arr = np.asarray(lat_ms, dtype=np.float64) if lat_ms else np.asarray([], dtype=np.float64)
    latency = {
        "p50_ms": float(np.percentile(lat_arr, 50)) if len(lat_arr) else float("nan"),
        "p95_ms": float(np.percentile(lat_arr, 95)) if len(lat_arr) else float("nan"),
        "qps": float(Q / max(1e-9, t1 - t0)),
    }

    meta = {
        "index_path": str(idx_path),
        "index_bytes": int(index_bytes),
        "num_passages": int(passage_emb.shape[0]),
        "dim": int(index_spec.dim),
        "index_type": index_spec.index_type,
        "ivf_nlist": int(index_spec.nlist) if index_spec.index_type == "ivfpq" else None,
        "pq_m": int(index_spec.m) if index_spec.index_type == "ivfpq" else None,
        "pq_nbits": int(index_spec.nbits) if index_spec.index_type == "ivfpq" else None,
        "nprobe": int(index_spec.nprobe) if index_spec.index_type == "ivfpq" else None,
    }
    return retrieved, latency, meta


def run_full_evaluation(cfg: Dict[str, Any]) -> Dict[str, Any]:
    artifacts_dir = Path(cfg["artifacts_dir"])
    ensure_dir(artifacts_dir)
    ensure_dir(artifacts_dir / "faiss")
    ensure_dir(artifacts_dir / "bm25")

    seed = int(cfg["run"]["seed"])
    deterministic = bool(cfg["run"].get("deterministic", True))
    set_reproducibility(seed, deterministic=deterministic)

    device = _auto_device(cfg) if torch is not None else "cpu"
    queries_tsv, qrels_tsv, collection_tsv = _load_eval_files(cfg)
    queries = load_queries(queries_tsv)
    qrels = load_qrels(qrels_tsv)

    k_values = cfg.get("eval", {}).get("k_values", [10, 100])
    eval_k_max = max(int(k) for k in k_values) if k_values else 10
    latency_max_queries = cfg.get("eval", {}).get("latency_max_queries")

    # Load passages (for smoke/dev). For large corpora, prefer pyserini BM25 and dense sharded encoding;
    # this pipeline keeps things simple and limits memory via optional max_passages.
    max_passages = cfg["dataset"].get("max_passages_eval") or cfg["dataset"].get("max_passages")
    passages_iter = load_collection(collection_tsv, max_passages=max_passages)
    pid_to_text: Dict[int, str] = {pid: text for pid, text in passages_iter}
    passage_ids = np.asarray(sorted(pid_to_text.keys()), dtype=np.int64)
    passage_texts = [pid_to_text[int(pid)] for pid in passage_ids.tolist()]

    runs: List[Dict[str, Any]] = []

    # Baseline BM25.
    bm25_backend = str(cfg.get("bm25", {}).get("backend", "rank_bm25"))
    bm25_k = max(int(cfg.get("bm25", {}).get("k", 10)), int(eval_k_max))
    bm25_retrieved: Dict[int, List[int]]
    bm25_latency: Dict[str, float]
    bm25_meta: Dict[str, Any] = {"backend": bm25_backend}
    if bm25_backend == "pyserini":
        index_dir = artifacts_dir / "bm25" / "pyserini_msmarco"
        try:
            idx = build_pyserini_msmarco_index(collection_root=collection_tsv.parent, index_dir=index_dir, threads=8)
            bm25_retrieved, bm25_latency = search_pyserini(idx, queries, k=bm25_k)
            bm25_meta["index_dir"] = str(index_dir)
        except Exception as e:  # noqa: BLE001
            # Keep smoke runs functional even without Java/pyserini.
            bm25_backend = "rank_bm25_fallback"
            bm25_meta["backend"] = bm25_backend
            idx = build_rank_bm25([(pid, pid_to_text[pid]) for pid in pid_to_text.keys()])
            bm25_retrieved, bm25_latency = search_rank_bm25(idx, queries, k=bm25_k)
            bm25_meta["warning"] = f"pyserini unavailable; fell back to rank_bm25 ({type(e).__name__})"
    else:
        idx = build_rank_bm25([(pid, pid_to_text[pid]) for pid in pid_to_text.keys()])
        bm25_retrieved, bm25_latency = search_rank_bm25(idx, queries, k=bm25_k)

    bm25_metrics = compute_ir_metrics(bm25_retrieved, qrels, k_values=k_values)
    runs.append(
        {
            "run_id": "bm25",
            "kind": "baseline",
            "method": "BM25",
            "backend": bm25_backend,
            "metrics": bm25_metrics,
            "latency": bm25_latency,
            "meta": bm25_meta,
        }
    )

    # Dense runs: distilled + ablation no-distillation.
    ckpt_root = artifacts_dir / "checkpoints"
    dense_run_ids = []
    if bool(cfg.get("faiss", {}).get("enabled", True)):
        if (ckpt_root / "dense_distill" / "final.pt").exists():
            dense_run_ids.append("dense_distill")
        if (ckpt_root / "dense_nodistill" / "final.pt").exists():
            dense_run_ids.append("dense_nodistill")

    faiss_cfg = cfg.get("faiss", {})
    index_types = faiss_cfg.get("index_types")
    if not index_types:
        t = str(faiss_cfg.get("index_type", "flatip"))
        index_types = [t]
        if t == "ivfpq":
            # Provide a built-in compression comparison (FlatIP vs IVF-PQ), as in the plan.
            index_types = ["flatip", "ivfpq"]

    for run_id in dense_run_ids:
        if torch is None or AutoTokenizer is None:
            # Skip dense evaluation if deps aren't installed; keep BM25 baseline functional.
            continue
        ckpt = ckpt_root / run_id / "final.pt"
        model, extra = _load_checkpoint(ckpt, device=device)
        tok = AutoTokenizer.from_pretrained(model.cfg.encoder_name, use_fast=True)

        # Encode queries/passages.
        qids = sorted(queries.keys())
        q_texts = [queries[qid] for qid in qids]
        q_emb = _encode_texts(
            model,
            tok,
            q_texts,
            device=device,
            batch_size=32,
            max_length=int(cfg["train"].get("max_length", 256)),
        )
        p_emb = _encode_texts(
            model,
            tok,
            passage_texts,
            device=device,
            batch_size=32,
            max_length=int(cfg["train"].get("max_length", 256)),
        )

        for index_type in index_types:
            if index_type == "flatip":
                spec = FaissBuildSpec(index_type="flatip", dim=int(p_emb.shape[1]))
                tag = "flatip"
            elif index_type == "ivfpq":
                spec = FaissBuildSpec(
                    index_type="ivfpq",
                    dim=int(p_emb.shape[1]),
                    nlist=int(faiss_cfg.get("nlist", 4096)),
                    m=int(faiss_cfg.get("m", 16)),
                    nbits=int(faiss_cfg.get("nbits", 8)),
                    nprobe=int(faiss_cfg.get("nprobe", 16)),
                )
                tag = f"ivfpq_nlist{spec.nlist}_m{spec.m}_nb{spec.nbits}_np{spec.nprobe}"
            else:
                raise ValueError(f"Unknown faiss index_type: {index_type}")

            retrieved, latency, meta = _dense_retrieve(
                qids,
                q_emb,
                pid_to_text,
                p_emb,
                passage_ids,
                index_spec=spec,
                k=max(int(faiss_cfg.get("k", 10)), int(eval_k_max)),
                artifacts_dir=artifacts_dir,
                run_id=run_id,
                index_tag=tag,
                latency_max_queries=latency_max_queries,
            )
            metrics = compute_ir_metrics(retrieved, qrels, k_values=k_values)
            kind = "ablation" if run_id == "dense_nodistill" else "model"
            runs.append(
                {
                    "run_id": f"{run_id}:{index_type}",
                    "kind": kind,
                    "method": "DenseBiEncoder",
                    "backend": "faiss",
                    "metrics": metrics,
                    "latency": latency,
                    "meta": meta,
                }
            )

    results = {
        "title": cfg.get("run", {}).get("name", "run"),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": cfg,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": (torch.__version__ if torch is not None else None),
            "cuda_available": (bool(torch.cuda.is_available()) if torch is not None else None),
            "device": str(device),
            "git_commit": _git_commit_or_none(artifacts_dir.parent),
        },
        "runs": runs,
    }
    return results


def _git_commit_or_none(repo_dir: Path) -> Optional[str]:
    import subprocess

    try:
        out = subprocess.check_output(["git", "-C", str(repo_dir), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:  # noqa: BLE001
        return None
