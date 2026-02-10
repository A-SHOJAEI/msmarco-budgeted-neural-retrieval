from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import sys


@dataclass(frozen=True)
class PyseriniIndex:
    index_dir: Path


def _require_pyserini() -> None:
    try:
        import pyserini  # noqa: F401
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "pyserini is not installed. Install it in the venv to use the Lucene BM25 baseline.\n"
            "Suggestion:\n"
            "  .venv/bin/pip install -r requirements-pyserini.txt\n"
            "You also need a working Java runtime (e.g., openjdk-17-jre)."
        ) from e


def build_pyserini_msmarco_index(collection_root: Path, index_dir: Path, *, threads: int = 8) -> PyseriniIndex:
    """
    Build a Lucene BM25 index over MS MARCO passages using pyserini.
    This is intentionally a thin wrapper around the official pyserini CLI.
    """
    _require_pyserini()
    index_dir = index_dir.resolve()
    if (index_dir / "segments.gen").exists() or any(index_dir.glob("segments_*")):
        return PyseriniIndex(index_dir=index_dir)

    index_dir.mkdir(parents=True, exist_ok=True)

    # pyserini expects the folder containing collection.tsv.
    cmd = [
        os.environ.get("PYTHON", sys.executable),
        "-m",
        "pyserini.index.lucene",
        "--collection",
        "MsMarcoCollection",
        "--input",
        str(collection_root),
        "--index",
        str(index_dir),
        "--generator",
        "DefaultLuceneDocumentGenerator",
        "--threads",
        str(threads),
        "--storePositions",
        "--storeDocvectors",
        "--storeRaw",
    ]
    subprocess.run(cmd, check=True)
    return PyseriniIndex(index_dir=index_dir)


def search_pyserini(index: PyseriniIndex, queries: Dict[int, str], *, k: int) -> Tuple[Dict[int, List[int]], Dict[str, float]]:
    _require_pyserini()
    from pyserini.search.lucene import LuceneSearcher

    searcher = LuceneSearcher(str(index.index_dir))
    searcher.set_bm25(k1=0.9, b=0.4)

    results: Dict[int, List[int]] = {}
    lat_ms: List[float] = []
    t0 = time.perf_counter()
    for qid, q in queries.items():
        q0 = time.perf_counter()
        hits = searcher.search(q, k=k)
        # MS MARCO passage ids are integer docids.
        results[qid] = [int(h.docid) for h in hits]
        lat_ms.append((time.perf_counter() - q0) * 1000.0)
    t1 = time.perf_counter()

    arr = np.asarray(lat_ms, dtype=np.float64)
    stats = {
        "p50_ms": float(np.percentile(arr, 50)) if len(arr) else float("nan"),
        "p95_ms": float(np.percentile(arr, 95)) if len(arr) else float("nan"),
        "qps": float(len(queries) / max(1e-9, t1 - t0)),
    }
    return results, stats
