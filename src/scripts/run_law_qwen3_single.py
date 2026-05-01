"""Single-rank training + verification for law specialist on Qwen3-1.7B.

Trains LoRA at rank=16 on CaseHOLD (4-option subset), then verifies on 3
MMLU law subjects + CaseHOLD test held-out.

Usage:
  python run_law_qwen3_single.py --rank 16
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--base", default="Qwen/Qwen3-1.7B")
    p.add_argument("--source", default="casehold")
    p.add_argument("--domain", default="law")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--max-train-examples", type=int, default=10000)
    p.add_argument("--mmlu-n", type=int, default=100)
    p.add_argument("--casehold-n", type=int, default=200)
    p.add_argument("--adapter-root",
                   default="/workspace/adapters_1p7b_ood/law_qwen3")
    p.add_argument("--results-dir",
                   default="results/specialist_verification/law_qwen3")
    p.add_argument("--log-dir", default="logs/law_qwen3")
    p.add_argument("--cache-dir", default="/workspace/hf_cache")
    args = p.parse_args()

    os.makedirs(args.adapter_root, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    rank_dir = Path(args.adapter_root) / f"r{args.rank}"
    train_log = Path(args.log_dir) / f"train_r{args.rank}.log"
    verify_log = Path(args.log_dir) / f"verify_r{args.rank}.log"
    output = Path(args.results_dir) / f"law_r{args.rank}.json"

    train_cmd = [
        sys.executable, str(ROOT / "src/scripts/train_specialist_ood.py"),
        "--model-name", args.base,
        "--source", args.source,
        "--domain", args.domain,
        "--rank", str(args.rank),
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
        "--adapter-dir", str(rank_dir),
        "--cache-dir", args.cache_dir,
        "--max-train-examples", str(args.max_train_examples),
    ]
    print(f"\n[train r={args.rank}] -> {train_log}")
    print(" ".join(train_cmd))
    with open(train_log, "w") as f:
        ret = subprocess.run(train_cmd, stdout=f, stderr=subprocess.STDOUT)
    if ret.returncode != 0:
        print(f"train failed; see {train_log}")
        return 1

    adapter_path = rank_dir / f"adapter_{args.domain}_{args.source}"
    verify_cmd = [
        sys.executable, str(ROOT / "src/scripts/verify_law_adapter.py"),
        "--base", args.base,
        "--adapter", str(adapter_path),
        "--mmlu-n", str(args.mmlu_n),
        "--casehold-n", str(args.casehold_n),
        "--cache-dir", args.cache_dir,
        "--output", str(output),
    ]
    print(f"\n[verify r={args.rank}] -> {verify_log}")
    print(" ".join(verify_cmd))
    with open(verify_log, "w") as f:
        ret = subprocess.run(verify_cmd, stdout=f, stderr=subprocess.STDOUT)
    if ret.returncode != 0:
        print(f"verify failed; see {verify_log}")
        return 1

    with open(output) as f:
        d = json.load(f)
    print(f"r={args.rank} verified={d['verified']} pass_count={d['pass_count']}")
    print(f"FINAL_METRICS verified={d['verified']} pass_count={d['pass_count']} "
          f"output={output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
