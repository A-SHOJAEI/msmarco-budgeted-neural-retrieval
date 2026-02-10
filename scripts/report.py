#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from msmarco_bnr.config import load_config
from msmarco_bnr.utils.io import read_json, write_text
from msmarco_bnr.utils.report import render_report_md


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    artifacts_dir = Path(cfg["artifacts_dir"])
    results_path = artifacts_dir / "results.json"
    if not results_path.exists():
        raise SystemExit(f"Missing {results_path}. Run `make eval` first.")

    results = read_json(results_path)
    md = render_report_md(cfg, results)
    write_text(artifacts_dir / "report.md", md)


if __name__ == "__main__":
    main()

