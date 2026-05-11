"""Workstation-side wrapper: per-LoRA-adapter MMLU 5-shot eval on the pod.

Mirrors run_mmlu_5shot_pod_wrapper_4b.py but for LoRA adapters. For each
(rank, domain) pair under --ckpt-base, this script:

  1. rsyncs the (small ~50 MB) adapter dir UP to /workspace/active_adapter/
  2. ssh-runs run_mmlu_5shot_baseline.py with --model-name=Qwen/Qwen3-4B
     and --adapter-path=/workspace/active_adapter
  3. scp's the resulting JSON back
  4. ssh deletes /workspace/active_adapter/

Usage:
  python -m src.scripts.run_mmlu_5shot_pod_wrapper_4b_lora \\
    --ckpt-base /media/dan/WD_BLACK/halulujah_4b_lora \\
    --output-dir results/full_ft_4b_streaming/mmlu_5shot_lora \\
    --pod-host root@69.30.85.238 --pod-port 22192
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DOMAINS = ["medicine", "math", "biology", "law", "physics"]
POD_ACTIVE_DIR = "/workspace/active_adapter"
POD_BENCH_OUT = "/workspace/bench_out.json"
BASE_MODEL = "Qwen/Qwen3-4B"


def find_adapters(ckpt_base: Path) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for rank_dir in sorted(ckpt_base.glob("r*")):
        if not rank_dir.is_dir():
            continue
        rank = rank_dir.name  # e.g. "r8"
        for d in DOMAINS:
            ad = rank_dir / d
            if (ad / "adapter_config.json").exists():
                out.append((f"{rank}_{d}-final", ad))
    return out


def ssh_args(args: argparse.Namespace) -> list[str]:
    return ["ssh", "-p", args.pod_port, "-i", args.pod_key,
            "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=10",
            args.pod_host]


def ssh_run(args: argparse.Namespace, cmd: str) -> tuple[int, str, str]:
    full = ssh_args(args) + [cmd]
    res = subprocess.run(full, capture_output=True, text=True)
    return res.returncode, res.stdout, res.stderr


def evaluate_adapter(name: str, path: Path,
                     args: argparse.Namespace) -> bool:
    out_local = Path(args.output_dir) / "per_ckpt" / f"{name}.json"
    if out_local.exists() and not args.force:
        print(f"[skip] {name} already done", flush=True)
        return True
    out_local.parent.mkdir(parents=True, exist_ok=True)

    print(f"[{name}] rsync UP {path} -> {POD_ACTIVE_DIR}", flush=True)
    ssh_run(args, f"rm -rf {POD_ACTIVE_DIR}; mkdir -p {POD_ACTIVE_DIR}")
    rsync_cmd = [
        "rsync", "-rL", "--no-owner", "--no-group", "--no-perms",
        "-e", f"ssh -p {args.pod_port} -i {args.pod_key} "
              f"-o StrictHostKeyChecking=no",
        str(path) + "/",
        f"{args.pod_host}:{POD_ACTIVE_DIR}/",
    ]
    res = subprocess.run(rsync_cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"  rsync FAIL rc={res.returncode}: "
              f"{res.stderr.strip()[:200]}", flush=True)
        return False

    print(f"[{name}] running 5-shot eval (LoRA) on pod...", flush=True)
    eval_cmd = (
        f"cd /workspace/halulujah && "
        f"HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 "
        f"python src/scripts/run_mmlu_5shot_baseline.py "
        f"--model-name '{BASE_MODEL}' "
        f"--adapter-path '{POD_ACTIVE_DIR}' "
        f"--n-questions {args.n_questions} "
        f"--n-shot {args.n_shot} "
        f"--output {POD_BENCH_OUT} --gpu 2>&1 | tail -20"
    )
    rc, stdout, stderr = ssh_run(args, eval_cmd)
    print(stdout)
    if rc != 0:
        print(f"  eval FAIL rc={rc}: {stderr.strip()[:200]}", flush=True)
        return False

    scp_cmd = [
        "scp", "-P", args.pod_port, "-i", args.pod_key,
        "-o", "StrictHostKeyChecking=no",
        f"{args.pod_host}:{POD_BENCH_OUT}",
        str(out_local),
    ]
    res = subprocess.run(scp_cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"  scp FAIL: {res.stderr.strip()[:200]}", flush=True)
        return False
    ssh_run(args, f"rm -rf {POD_ACTIVE_DIR} {POD_BENCH_OUT}")
    print(f"[{name}] done -> {out_local}", flush=True)
    return True


def aggregate(args: argparse.Namespace) -> None:
    out_path = Path(args.output_dir) / "scan.json"
    summary: dict = {"per_ckpt": {}}
    per_ckpt_dir = Path(args.output_dir) / "per_ckpt"
    for j in sorted(per_ckpt_dir.glob("*.json")):
        try:
            data = json.loads(j.read_text())
            domain_accs = {
                d: data["per_domain"][d]["accuracy"]
                for d in DOMAINS
                if d in data.get("per_domain", {})
            }
            mean_acc = (sum(domain_accs.values()) / len(domain_accs)
                        if domain_accs else 0.0)
            summary["per_ckpt"][j.stem] = {
                "domains": domain_accs,
                "mean_acc": mean_acc,
            }
        except Exception as e:
            print(f"[err] {j}: {e}")
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote summary {out_path} ({len(summary['per_ckpt'])} adapters)")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-base",
                   default="/media/dan/WD_BLACK/halulujah_4b_lora")
    p.add_argument("--output-dir",
                   default="results/full_ft_4b_streaming/mmlu_5shot_lora")
    p.add_argument("--n-questions", type=int, default=50)
    p.add_argument("--n-shot", type=int, default=5)
    p.add_argument("--pod-host", default="root@69.30.85.238")
    p.add_argument("--pod-port", default="22192")
    p.add_argument("--pod-key", default=str(Path.home() / ".ssh/runpod_key"))
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    adapters = find_adapters(Path(args.ckpt_base))
    print(f"=== plan ===")
    print(f"Found {len(adapters)} adapters to evaluate")
    for i, (name, path) in enumerate(adapters):
        print(f"  [{i:>2d}] {name} -> {path}")

    n_ok = 0
    for name, path in adapters:
        if evaluate_adapter(name, path, args):
            n_ok += 1

    aggregate(args)
    print(f"FINAL_METRICS bench_complete={n_ok == len(adapters)} "
          f"n_ok={n_ok}/{len(adapters)}")
    return 0 if n_ok == len(adapters) else 1


if __name__ == "__main__":
    sys.exit(main())
