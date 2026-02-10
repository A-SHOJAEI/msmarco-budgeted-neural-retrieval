#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from msmarco_bnr.config import load_config
from msmarco_bnr.data.smoke import generate_smoke_dataset
from msmarco_bnr.data.msmarco import download_and_prepare_msmarco
from msmarco_bnr.utils.io import ensure_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    ds_type = cfg["dataset"]["type"]

    ensure_dir(Path(cfg["artifacts_dir"]))

    if ds_type == "smoke_synthetic":
        generate_smoke_dataset(cfg)
        return
    if ds_type == "msmarco_passage":
        download_and_prepare_msmarco(cfg)
        return

    raise SystemExit(f"Unknown dataset.type: {ds_type}")


if __name__ == "__main__":
    main()

