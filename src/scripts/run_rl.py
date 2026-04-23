#!/usr/bin/env python3
"""CLI entry point for PPO RL training with Oracle reward."""

import argparse
import sys

sys.path.insert(0, "src")

from halulujah.config import load_config
from halulujah.oracle.index_builder import OracleIndex
from halulujah.oracle.retriever import OracleRetriever
from halulujah.oracle.reward import OracleRewardFunction
from halulujah.rl.ppo_trainer import run_rl_training


def main():
    parser = argparse.ArgumentParser(description="Run PPO RL training with Oracle reward")
    parser.add_argument("--config", type=str, default="configs/experiment.yaml")
    parser.add_argument("--sft-adapter", type=str, required=True, help="Path to SFT LoRA adapter")
    parser.add_argument("--oracle-dir", type=str, required=True, help="Path to saved Oracle index")
    parser.add_argument("--held-out-year", type=int, nargs="+", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    overrides = {}
    if args.max_steps is not None:
        overrides["ppo"] = {"max_steps": args.max_steps}

    cfg = load_config(args.config, overrides=overrides)
    held_out = args.held_out_year or cfg.held_out_years

    # Load Oracle
    oracle_index = OracleIndex(cfg.oracle)
    oracle_index.load(args.oracle_dir)
    retriever = OracleRetriever(oracle_index)
    reward_fn = OracleRewardFunction(retriever, cfg.oracle)

    result = run_rl_training(cfg, args.sft_adapter, held_out, reward_fn)
    if isinstance(result, tuple):
        path, stats = result
    else:
        path, stats = result, {}

    final_reward = float(stats.get("mean_reward", float("nan")))
    final_kl = float(stats.get("mean_kl", float("nan")))
    final_steps = int(stats.get("num_steps", 0))
    print(f"RL model saved to: {path}")
    print(
        f"FINAL_METRICS mean_reward={final_reward:.6f} mean_kl={final_kl:.6f} "
        f"num_steps={final_steps} adapter_path={path}"
    )


if __name__ == "__main__":
    main()
