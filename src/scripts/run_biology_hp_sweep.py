"""Hyperparameter sweep for the biology specialist on Qwen3-1.7B.

Trains LoRA adapters at multiple ranks on PubMedQA pqa_artificial train
(subsampled), then verifies each on MMLU biology subjects + PubMedQA
pqa_labeled test held-out.

Usage:
  python run_biology_hp_sweep.py --ranks 8 16 32 64
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run_train(rank, base, source, domain, adapter_root, epochs, lr,
              max_train, cache_dir, log_dir):
    out_dir = Path(adapter_root) / f"r{rank}"
    cmd = [
        sys.executable, str(ROOT / "src/scripts/train_specialist_ood.py"),
        "--model-name", base,
        "--source", source,
        "--domain", domain,
        "--rank", str(rank),
        "--epochs", str(epochs),
        "--lr", str(lr),
        "--adapter-dir", str(out_dir),
        "--cache-dir", cache_dir,
        "--max-train-examples", str(max_train),
    ]
    log = Path(log_dir) / f"train_r{rank}.log"
    print(f"\n[train r={rank}] -> {log}")
    print(" ".join(cmd))
    with open(log, "w") as f:
        ret = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    if ret.returncode != 0:
        raise RuntimeError(f"train r={rank} failed; see {log}")
    return out_dir / f"adapter_{domain}_{source}"


def run_verify(rank, base, adapter_path, results_dir, mmlu_n, pubmedqa_n,
               cache_dir, log_dir):
    output = Path(results_dir) / f"biology_r{rank}.json"
    cmd = [
        sys.executable, str(ROOT / "src/scripts/verify_biology_adapter.py"),
        "--base", base,
        "--adapter", str(adapter_path),
        "--mmlu-n", str(mmlu_n),
        "--pubmedqa-n", str(pubmedqa_n),
        "--cache-dir", cache_dir,
        "--output", str(output),
    ]
    log = Path(log_dir) / f"verify_r{rank}.log"
    print(f"\n[verify r={rank}] -> {log}")
    print(" ".join(cmd))
    with open(log, "w") as f:
        ret = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    if ret.returncode != 0:
        raise RuntimeError(f"verify r={rank} failed; see {log}")
    return output


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ranks", type=int, nargs="+", default=[8, 16, 32, 64])
    p.add_argument("--base", default="Qwen/Qwen3-1.7B")
    p.add_argument("--source", default="pubmedqa")
    p.add_argument("--domain", default="biology")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--max-train-examples", type=int, default=10000)
    p.add_argument("--mmlu-n", type=int, default=100)
    p.add_argument("--pubmedqa-n", type=int, default=200)
    p.add_argument("--adapter-root",
                   default="/workspace/adapters_1p7b_ood/biology_sweep")
    p.add_argument("--results-dir",
                   default="results/specialist_verification/biology_sweep")
    p.add_argument("--log-dir",
                   default="logs/biology_sweep")
    p.add_argument("--cache-dir", default="/workspace/hf_cache")
    args = p.parse_args()

    os.makedirs(args.adapter_root, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    summary = {}
    for r in args.ranks:
        print(f"\n========== rank {r} ==========")
        try:
            adapter_path = run_train(
                r, args.base, args.source, args.domain,
                args.adapter_root, args.epochs, args.lr,
                args.max_train_examples, args.cache_dir, args.log_dir)
            res_path = run_verify(
                r, args.base, adapter_path, args.results_dir,
                args.mmlu_n, args.pubmedqa_n, args.cache_dir, args.log_dir)
            with open(res_path) as f:
                d = json.load(f)
            summary[f"r{r}"] = {
                "verified": d["verified"],
                "pass_count": d["pass_count"],
                "result_path": str(res_path),
                "adapter_path": str(adapter_path),
            }
            print(f"r={r} verified={d['verified']} pass_count={d['pass_count']}")
        except Exception as e:
            print(f"rank {r} failed: {e}")
            summary[f"r{r}"] = {"error": str(e)}

    summary_path = Path(args.results_dir) / "sweep_summary.json"
    json.dump(summary, open(summary_path, "w"), indent=2)
    print(f"\nFINAL_METRICS sweep_summary={summary_path}")


if __name__ == "__main__":
    main()
