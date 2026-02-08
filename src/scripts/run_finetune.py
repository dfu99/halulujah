#!/usr/bin/env python3
"""CLI entry point for SFT LoRA fine-tuning."""

import argparse
import sys

sys.path.insert(0, "src")

from halulujah.config import load_config
from halulujah.finetune.sft_lora import run_finetune


def main():
    parser = argparse.ArgumentParser(description="Run SFT LoRA fine-tuning")
    parser.add_argument("--config", type=str, default="configs/experiment.yaml")
    parser.add_argument("--held-out-year", type=int, nargs="+", default=None)
    parser.add_argument("--max-steps", type=int, default=None, help="Override num_train_epochs with max_steps for smoke testing")
    args = parser.parse_args()

    overrides = {}
    if args.max_steps is not None:
        overrides["sft"] = {"num_train_epochs": 1, "max_steps": args.max_steps}

    cfg = load_config(args.config, overrides=overrides)
    held_out = args.held_out_year or cfg.held_out_years
    path = run_finetune(cfg, held_out_years=held_out)
    print(f"Saved adapter to: {path}")


if __name__ == "__main__":
    main()
