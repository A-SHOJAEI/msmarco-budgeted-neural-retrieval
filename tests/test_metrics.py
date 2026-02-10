from __future__ import annotations

from msmarco_bnr.eval.metrics import compute_ir_metrics


def test_metrics_basic():
    qrels = {1: {10: 1}, 2: {20: 1}}
    retrieved = {1: [10, 11], 2: [30, 20]}
    m = compute_ir_metrics(retrieved, qrels, k_values=[1, 2, 10])
    assert 0.0 <= m["ndcg@10"] <= 1.0
    assert m["mrr@1"] == 0.5  # q1 hit@1 => 1.0, q2 miss@1 => 0.0 => avg 0.5
    assert m["recall@2"] == 1.0  # both relevant docs retrieved within top-2 across queries

