from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
from rank_bm25 import BM25Okapi


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass(frozen=True)
class RankBm25Index:
    bm25: BM25Okapi
    doc_ids: List[int]
    tokenized_docs: List[List[str]]


def build_rank_bm25(passages: Iterable[Tuple[int, str]]) -> RankBm25Index:
    doc_ids: List[int] = []
    tokenized: List[List[str]] = []
    for pid, text in passages:
        doc_ids.append(int(pid))
        tokenized.append(_tokenize(text))
    bm25 = BM25Okapi(tokenized)
    return RankBm25Index(bm25=bm25, doc_ids=doc_ids, tokenized_docs=tokenized)


def search_rank_bm25(index: RankBm25Index, queries: Dict[int, str], *, k: int) -> Tuple[Dict[int, List[int]], Dict[str, float]]:
    t0 = time.perf_counter()
    results: Dict[int, List[int]] = {}
    lat_ms: List[float] = []
    for qid, q in queries.items():
        q_tok = _tokenize(q)
        q0 = time.perf_counter()
        scores = index.bm25.get_scores(q_tok)
        # Partial sort for top-k
        top_idx = np.argpartition(-scores, kth=min(k, len(scores) - 1))[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        results[qid] = [index.doc_ids[i] for i in top_idx.tolist()]
        lat_ms.append((time.perf_counter() - q0) * 1000.0)
    t1 = time.perf_counter()
    stats = _latency_stats(lat_ms, total_time_s=t1 - t0, num_queries=len(queries))
    return results, stats


def _latency_stats(lat_ms: List[float], *, total_time_s: float, num_queries: int) -> Dict[str, float]:
    if not lat_ms:
        return {"p50_ms": float("nan"), "p95_ms": float("nan"), "qps": 0.0}
    arr = np.asarray(lat_ms, dtype=np.float64)
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "qps": float(num_queries / max(1e-9, total_time_s)),
    }

