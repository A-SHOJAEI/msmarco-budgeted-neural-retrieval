#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from msmarco_bnr.config import load_config
from msmarco_bnr.utils.io import ensure_dir, write_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    artifacts_dir = Path(cfg["artifacts_dir"])
    ensure_dir(artifacts_dir)

    if not cfg.get("train", {}).get("enabled", True):
        return

    # Lazy import so smoke configs can run without heavy ML deps installed.
    from msmarco_bnr.training.train import train_biencoder

    out = train_biencoder(cfg)
    write_json(artifacts_dir / "train_summary.json", out)


if __name__ == "__main__":
    main()
