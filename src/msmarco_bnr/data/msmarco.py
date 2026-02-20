"""MS MARCO-format TSV loaders and download helpers."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def load_collection(
    path: Path,
    max_passages: Optional[int] = None,
) -> Iterable[Tuple[int, str]]:
    """Yield (pid, text) from a two-column TSV (pid<TAB>text)."""
    count = 0
    with open(path, "r", encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                continue
            pid = int(row[0])
            text = row[1]
            yield pid, text
            count += 1
            if max_passages is not None and count >= max_passages:
                break


def load_queries(path: Path) -> Dict[int, str]:
    """Return {qid: text} from a two-column TSV."""
    queries: Dict[int, str] = {}
    with open(path, "r", encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                continue
            queries[int(row[0])] = row[1]
    return queries


def load_qrels(path: Path) -> Dict[int, Dict[int, int]]:
    """Return {qid: {pid: relevance}} from a qrels TSV.

    Accepts both TREC-style (4-column: qid 0 pid rel) and simple
    2-column (qid pid, implied rel=1) formats.
    """
    qrels: Dict[int, Dict[int, int]] = {}
    with open(path, "r", encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t")
        for row in reader:
            if len(row) >= 4:
                qid, pid, rel = int(row[0]), int(row[2]), int(row[3])
            elif len(row) >= 2:
                qid, pid, rel = int(row[0]), int(row[1]), 1
            else:
                continue
            qrels.setdefault(qid, {})[pid] = rel
    return qrels


def load_triples(
    path: Path,
    max_triples: Optional[int] = None,
) -> List[Tuple[int, int, int]]:
    """Return list of (qid, pos_pid, neg_pid) from a 3-column TSV."""
    triples: List[Tuple[int, int, int]] = []
    with open(path, "r", encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t")
        for row in reader:
            if len(row) < 3:
                continue
            triples.append((int(row[0]), int(row[1]), int(row[2])))
            if max_triples is not None and len(triples) >= max_triples:
                break
    return triples


def download_and_prepare_msmarco(cfg: Dict[str, Any]) -> None:
    """Download and prepare MS MARCO passage data. Placeholder for full pipeline."""
    raise NotImplementedError(
        "Full MS MARCO download not implemented in smoke mode. "
        "Use dataset.type=smoke_synthetic for testing."
    )
