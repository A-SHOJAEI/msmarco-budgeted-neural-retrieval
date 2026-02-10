from __future__ import annotations

from typing import Any, Dict, List


def _fmt(x: Any) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float):
        if x != x:  # NaN
            return "n/a"
        return f"{x:.4f}"
    return str(x)


def render_report_md(cfg: Dict[str, Any], results: Dict[str, Any]) -> str:
    runs: List[Dict[str, Any]] = list(results.get("runs", []))

    # Stable ordering: baseline first, then models, then ablations.
    kind_order = {"baseline": 0, "model": 1, "ablation": 2}
    runs.sort(key=lambda r: (kind_order.get(r.get("kind", ""), 9), r.get("run_id", "")))

    lines: List[str] = []
    lines.append(f"# Report: {results.get('title', 'run')}")
    lines.append("")
    lines.append(f"- Timestamp (UTC): `{results.get('timestamp_utc')}`")
    env = results.get("environment", {})
    lines.append(f"- Python: `{env.get('python')}`")
    lines.append(f"- Torch: `{env.get('torch')}` (CUDA available: `{env.get('cuda_available')}`)")
    lines.append(f"- Device: `{env.get('device')}`")
    lines.append("")

    # Table
    cols = [
        "run_id",
        "kind",
        "method",
        "backend",
        "ndcg@10",
        "mrr@10",
        "recall@10",
        "recall@100",
        "p50_ms",
        "p95_ms",
        "qps",
        "index_bytes",
    ]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for r in runs:
        m = r.get("metrics", {})
        l = r.get("latency", {})
        meta = r.get("meta", {})
        row = [
            r.get("run_id"),
            r.get("kind"),
            r.get("method"),
            r.get("backend"),
            m.get("ndcg@10"),
            m.get("mrr@10"),
            m.get("recall@10"),
            m.get("recall@100"),
            l.get("p50_ms"),
            l.get("p95_ms"),
            l.get("qps"),
            meta.get("index_bytes"),
        ]
        lines.append("| " + " | ".join(_fmt(x) for x in row) + " |")

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- `bm25` is the baseline. For full MS MARCO runs, set `bm25.backend: pyserini` and install `requirements-pyserini.txt`.")
    lines.append("- `dense_nodistill` is the ablation: contrastive-only bi-encoder (no distillation).")
    lines.append("- If `ivfpq` is enabled, the report typically includes both FlatIP and IVF-PQ to show the compression tradeoff.")
    return "\n".join(lines)

