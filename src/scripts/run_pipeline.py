#!/usr/bin/env python3
"""CLI entry point for the full Temporal Leave-One-Out pipeline."""

import argparse
import json
import logging
import sys

sys.path.insert(0, "src")

from halulujah.config import load_config
from halulujah.pipeline.temporal_loo import run_temporal_leave_one_out


def main():
    parser = argparse.ArgumentParser(description="Run Temporal Leave-One-Out pipeline")
    parser.add_argument("--config", type=str, default="configs/experiment.yaml")
    parser.add_argument("--held-out-year", type=int, nargs="+", default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    overrides = {}
    if args.held_out_year:
        overrides["held_out_years"] = args.held_out_year

    cfg = load_config(args.config, overrides=overrides)
    results = run_temporal_leave_one_out(cfg)

    print("\n=== Pipeline Results ===")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
