#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from msmarco_bnr.config import load_config
from msmarco_bnr.eval.pipeline import run_full_evaluation
from msmarco_bnr.utils.io import ensure_dir, write_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    artifacts_dir = Path(cfg["artifacts_dir"])
    ensure_dir(artifacts_dir)

    results = run_full_evaluation(cfg)
    write_json(artifacts_dir / "results.json", results)


if __name__ == "__main__":
    main()

