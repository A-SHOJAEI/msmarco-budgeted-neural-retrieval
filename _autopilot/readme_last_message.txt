Overwrote `README.md` with a project-specific, code-faithful description pulled from `artifacts/results.json`, `artifacts/report.md`, and the implemented pipeline (`src/msmarco_bnr/*`). It includes:

- Clear problem statement + what “budget” means in this repo (latency percentiles, QPS, and FAISS index bytes)
- Dataset provenance for both `smoke_synthetic` (generated in `src/msmarco_bnr/data/smoke.py`) and MS MARCO Passage Ranking (exact URLs from `src/msmarco_bnr/data/msmarco.py`)
- Methodology grounded in the actual implementation (BM25 backends, bi-encoder architecture, contrastive loss, optional distillation, FAISS FlatIP vs IVF-PQ)
- Baseline/ablation naming exactly as emitted by `src/msmarco_bnr/eval/pipeline.py`
- Exact committed results table (Table 1) matching `artifacts/report.md`, plus the run timestamp `2026-02-10T06:34:02Z`
- Repro commands for smoke and MS MARCO dev.small, plus limitations and concrete next research steps