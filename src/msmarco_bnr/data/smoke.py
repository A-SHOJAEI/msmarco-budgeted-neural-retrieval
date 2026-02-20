"""Generate a tiny synthetic MS MARCO-like dataset for smoke testing."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict


_TOPICS = [
    "machine learning",
    "neural networks",
    "information retrieval",
    "natural language processing",
    "computer vision",
    "reinforcement learning",
    "speech recognition",
    "recommendation systems",
    "text classification",
    "question answering",
]


def generate_smoke_dataset(cfg: Dict[str, Any]) -> None:
    ds = cfg["dataset"]
    root = Path(ds["root_dir"])
    root.mkdir(parents=True, exist_ok=True)

    num_passages = int(ds.get("num_passages", 500))
    num_queries = int(ds.get("num_queries", 50))
    num_triples = int(ds.get("num_triples", 200))
    vocab_size = int(ds.get("vocab_size", 2000))
    seed = int(cfg.get("run", {}).get("seed", 42))

    rng = random.Random(seed)

    # Build a simple vocab
    words = [f"word{i}" for i in range(vocab_size)]

    def _random_text(min_len: int = 8, max_len: int = 30) -> str:
        length = rng.randint(min_len, max_len)
        topic = rng.choice(_TOPICS)
        return topic + " " + " ".join(rng.choices(words, k=length))

    # Generate passages
    passages = {}
    for pid in range(num_passages):
        passages[pid] = _random_text(10, 40)

    # Generate queries with assigned relevant passage
    queries = {}
    qrels = {}
    pids = list(passages.keys())
    for qid in range(num_queries):
        queries[qid] = _random_text(5, 15)
        # Assign a relevant passage
        qrels[qid] = rng.choice(pids)

    # Generate triples (qid, pos_pid, neg_pid)
    triples = []
    for _ in range(num_triples):
        qid = rng.choice(list(queries.keys()))
        pos_pid = qrels[qid]
        neg_pid = rng.choice(pids)
        while neg_pid == pos_pid:
            neg_pid = rng.choice(pids)
        triples.append((qid, pos_pid, neg_pid))

    # Write collection.tsv
    with open(root / "collection.tsv", "w", encoding="utf-8") as f:
        for pid in sorted(passages.keys()):
            f.write(f"{pid}\t{passages[pid]}\n")

    # Write queries (use as both train and eval queries for smoke)
    with open(root / "queries.dev.small.tsv", "w", encoding="utf-8") as f:
        for qid in sorted(queries.keys()):
            f.write(f"{qid}\t{queries[qid]}\n")

    # Write qrels (TREC format: qid 0 pid 1)
    with open(root / "qrels.dev.small.tsv", "w", encoding="utf-8") as f:
        for qid in sorted(qrels.keys()):
            f.write(f"{qid}\t0\t{qrels[qid]}\t1\n")

    # Write triples
    with open(root / "triples.train.small.tsv", "w", encoding="utf-8") as f:
        for qid, pos, neg in triples:
            f.write(f"{qid}\t{pos}\t{neg}\n")
