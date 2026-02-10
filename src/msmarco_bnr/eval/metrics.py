from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple


def _dcg(rels: Sequence[int]) -> float:
    s = 0.0
    for i, r in enumerate(rels, start=1):
        if r <= 0:
            continue
        s += (2.0**float(r) - 1.0) / math.log2(i + 1.0)
    return s


def ndcg_at_k(ranked_pids: Sequence[int], qrels: Dict[int, int], *, k: int) -> float:
    rels = [int(qrels.get(pid, 0)) for pid in ranked_pids[:k]]
    dcg = _dcg(rels)
    ideal = sorted([int(r) for r in qrels.values() if int(r) > 0], reverse=True)[:k]
    idcg = _dcg(ideal)
    if idcg <= 0:
        return 0.0
    return float(dcg / idcg)


def mrr_at_k(ranked_pids: Sequence[int], qrels: Dict[int, int], *, k: int) -> float:
    for i, pid in enumerate(ranked_pids[:k], start=1):
        if int(qrels.get(pid, 0)) > 0:
            return 1.0 / float(i)
    return 0.0


def recall_at_k(ranked_pids: Sequence[int], qrels: Dict[int, int], *, k: int) -> float:
    rel = {pid for pid, r in qrels.items() if int(r) > 0}
    if not rel:
        return 0.0
    hit = len(rel.intersection(set(ranked_pids[:k])))
    return float(hit / len(rel))


def compute_ir_metrics(
    retrieved: Dict[int, List[int]],
    qrels: Dict[int, Dict[int, int]],
    *,
    k_values: Iterable[int],
) -> Dict[str, float]:
    ks = sorted(set(int(k) for k in k_values))
    ndcg: Dict[int, float] = {k: 0.0 for k in ks}
    mrr: Dict[int, float] = {k: 0.0 for k in ks}
    rec: Dict[int, float] = {k: 0.0 for k in ks}

    qids = [qid for qid in qrels.keys() if qid in retrieved]
    if not qids:
        return {f"ndcg@{k}": 0.0 for k in ks} | {f"mrr@{k}": 0.0 for k in ks} | {f"recall@{k}": 0.0 for k in ks}

    for qid in qids:
        r = retrieved[qid]
        rels = qrels[qid]
        for k in ks:
            ndcg[k] += ndcg_at_k(r, rels, k=k)
            mrr[k] += mrr_at_k(r, rels, k=k)
            rec[k] += recall_at_k(r, rels, k=k)

    n = float(len(qids))
    out: Dict[str, float] = {}
    for k in ks:
        out[f"ndcg@{k}"] = float(ndcg[k] / n)
        out[f"mrr@{k}"] = float(mrr[k] / n)
        out[f"recall@{k}"] = float(rec[k] / n)
    return out

