from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class FaissBuildSpec:
    index_type: str  # "flatip" | "ivfpq"
    dim: int
    nlist: int = 4096
    m: int = 16
    nbits: int = 8
    nprobe: int = 16


def _require_faiss():
    try:
        import faiss  # noqa: F401
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("faiss not available; ensure faiss-cpu is installed.") from e


def build_faiss_index(
    embeddings: np.ndarray,
    ids: np.ndarray,
    spec: FaissBuildSpec,
    *,
    train_size: int = 200_000,
) -> "object":
    _require_faiss()
    import faiss

    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32, copy=False)
    if ids.dtype != np.int64:
        ids = ids.astype(np.int64, copy=False)
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be [N,D]")
    if ids.ndim != 1 or ids.shape[0] != embeddings.shape[0]:
        raise ValueError("ids must be [N] aligned with embeddings")

    if spec.index_type == "flatip":
        base = faiss.IndexFlatIP(spec.dim)
        index = faiss.IndexIDMap2(base)
        index.add_with_ids(embeddings, ids)
        return index

    if spec.index_type == "ivfpq":
        quantizer = faiss.IndexFlatIP(spec.dim)
        index_ivf = faiss.IndexIVFPQ(quantizer, spec.dim, int(spec.nlist), int(spec.m), int(spec.nbits), faiss.METRIC_INNER_PRODUCT)
        index_ivf.nprobe = int(spec.nprobe)
        index = faiss.IndexIDMap2(index_ivf)
        # Train on a sample (FAISS requires training for IVF/PQ).
        n = embeddings.shape[0]
        n_train = min(n, int(train_size))
        train_x = embeddings[:n_train]
        index.train(train_x)
        index.add_with_ids(embeddings, ids)
        return index

    raise ValueError(f"Unknown FAISS index_type: {spec.index_type}")


def save_faiss_index(index: "object", path: Path) -> None:
    _require_faiss()
    import faiss

    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))


def load_faiss_index(path: Path) -> "object":
    _require_faiss()
    import faiss

    return faiss.read_index(str(path))


def search_faiss(
    index: "object",
    query_emb: np.ndarray,
    *,
    k: int,
    latency_max_queries: Optional[int] = None,
) -> Tuple[Dict[int, List[int]], Dict[str, float]]:
    if query_emb.dtype != np.float32:
        query_emb = query_emb.astype(np.float32, copy=False)
    if query_emb.ndim != 2:
        raise ValueError("query_emb must be [Q,D]")

    Q = query_emb.shape[0]
    max_q = min(Q, int(latency_max_queries)) if latency_max_queries is not None else Q

    # Measure per-query latencies using batch size 1 for stable percentiles (only for small eval).
    lat_ms: List[float] = []
    t0 = time.perf_counter()
    D_all: List[np.ndarray] = []
    I_all: List[np.ndarray] = []
    for i in range(Q):
        if i < max_q:
            q0 = time.perf_counter()
            D, I = index.search(query_emb[i : i + 1], k)
            lat_ms.append((time.perf_counter() - q0) * 1000.0)
        else:
            D, I = index.search(query_emb[i : i + 1], k)
        D_all.append(D)
        I_all.append(I)
    t1 = time.perf_counter()

    I = np.concatenate(I_all, axis=0)
    results: Dict[int, List[int]] = {i: [int(x) for x in I[i].tolist() if int(x) != -1] for i in range(Q)}

    stats = _latency_stats(lat_ms, total_time_s=t1 - t0, num_queries=Q)
    return results, stats


def _latency_stats(lat_ms: List[float], *, total_time_s: float, num_queries: int) -> Dict[str, float]:
    if not lat_ms:
        return {"p50_ms": float("nan"), "p95_ms": float("nan"), "qps": float(num_queries / max(1e-9, total_time_s))}
    arr = np.asarray(lat_ms, dtype=np.float64)
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "qps": float(num_queries / max(1e-9, total_time_s)),
    }

