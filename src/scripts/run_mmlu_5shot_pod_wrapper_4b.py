"""Workstation-side wrapper: per-ckpt MMLU 5-shot eval on the pod.

For each ckpt (base model + 5 finals + 24 per-step ckpts), this script:
  1. rsyncs the ckpt UP from WD_BLACK to /workspace/active_ckpt/ on pod
  2. ssh-runs run_mmlu_5shot_baseline.py on pod against that path
  3. scp's the resulting JSON back to local results dir
  4. ssh deletes /workspace/active_ckpt/ on pod (frees quota)
  5. proceeds to next ckpt

Each ckpt cycle: ~5 min (rsync 3.4 GB up + ~3 min eval + delete).
30 ckpts × 5 min ≈ 2.5 h total.

Usage:
  python -m src.scripts.run_mmlu_5shot_pod_wrapper \\
    --ckpt-base /media/dan/WD_BLACK/halulujah_full_ft_streaming \\
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
POD_ACTIVE_DIR = "/workspace/active_ckpt"
POD_BENCH_OUT = "/workspace/bench_out.json"


def find_ckpts(ckpt_base: Path, include_base: bool = True) -> list[tuple[str, Path | None]]:
    out: list[tuple[str, Path | None]] = []
    if include_base:
        out.append(("base", None))  # None means use HF model name
    for d in DOMAINS:
        domain_root = ckpt_base / d
        if not domain_root.exists():
            continue
        if (domain_root / "config.json").exists():
            out.append((f"{d}-final", domain_root))
        for ckpt in sorted(domain_root.glob("checkpoint-*"),
                           key=lambda p: int(p.name.split("-")[1])):
            step = ckpt.name.split("-")[1]
            out.append((f"{d}-step{step}", ckpt))
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


def evaluate_ckpt(name: str, path: Path | None,
                  args: argparse.Namespace) -> bool:
    out_local = Path(args.output_dir) / "per_ckpt" / f"{name}.json"
    if out_local.exists() and not args.force:
        print(f"[skip] {name} already done", flush=True)
        return True
    out_local.parent.mkdir(parents=True, exist_ok=True)

    if path is None:
        # base — load directly from HF
        pod_model_arg = "Qwen/Qwen3-4B"
    else:
        # rsync up to pod. For finals (path is a domain root), exclude
        # checkpoint-* subdirs (they're 5 × 3.4 GB = 17 GB and would
        # blow the moosefs ~21 GB user/group quota even if we deleted
        # them later). The per-step evals point directly at checkpoint-N
        # dirs and don't have this problem.
        is_final = (path.parent.name == "halulujah_full_ft_streaming"
                    or path.parent.name == "halulujah_4b_full_ft")
        rsync_extras = ["--exclude=checkpoint-*"] if is_final else []
        print(f"[{name}] rsync UP {path} -> {POD_ACTIVE_DIR}"
              f"{' (excluding ckpts)' if is_final else ''}", flush=True)
        rsync_cmd = [
            "rsync", "-rL", "--no-owner", "--no-group", "--no-perms",
            *rsync_extras,
            "-e", f"ssh -p {args.pod_port} -i {args.pod_key} "
                  f"-o StrictHostKeyChecking=no",
            str(path) + "/",
            f"{args.pod_host}:{POD_ACTIVE_DIR}/",
        ]
        ssh_run(args, f"rm -rf {POD_ACTIVE_DIR}; mkdir -p {POD_ACTIVE_DIR}")
        res = subprocess.run(rsync_cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"  rsync FAIL rc={res.returncode}: "
                  f"{res.stderr.strip()[:200]}", flush=True)
            return False
        pod_model_arg = POD_ACTIVE_DIR

    # Run eval on pod
    print(f"[{name}] running 5-shot eval on pod...", flush=True)
    eval_cmd = (
        f"cd /workspace/halulujah && "
        f"python src/scripts/run_mmlu_5shot_baseline.py "
        f"--model-name '{pod_model_arg}' "
        f"--n-questions {args.n_questions} "
        f"--n-shot {args.n_shot} "
        f"--output {POD_BENCH_OUT} --gpu 2>&1 | tail -20"
    )
    rc, stdout, stderr = ssh_run(args, eval_cmd)
    print(stdout)
    if rc != 0:
        print(f"  eval FAIL rc={rc}: {stderr.strip()[:200]}", flush=True)
        return False

    # scp back the JSON
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

    # Cleanup pod active dir
    if path is not None:
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
    print(f"\nWrote summary {out_path} ({len(summary['per_ckpt'])} ckpts)")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-base",
                   default="/media/dan/WD_BLACK/halulujah_full_ft_streaming")
    p.add_argument("--output-dir",
                   default="results/full_ft_streaming/mmlu_5shot")
    p.add_argument("--n-questions", type=int, default=50)
    p.add_argument("--n-shot", type=int, default=5)
    p.add_argument("--pod-host", default="root@69.30.85.238")
    p.add_argument("--pod-port", default="22192")
    p.add_argument("--pod-key", default=str(Path.home() / ".ssh/runpod_key"))
    p.add_argument("--force", action="store_true")
    p.add_argument("--start-from", type=int, default=0)
    args = p.parse_args()

    ckpts = find_ckpts(Path(args.ckpt_base))
    print(f"=== plan ===")
    print(f"Found {len(ckpts)} ckpts to evaluate")
    for i, (name, _) in enumerate(ckpts):
        marker = " (skip)" if i < args.start_from else ""
        print(f"  [{i:>2d}] {name}{marker}")

    n_ok = 0
    n_run = 0
    for i, (name, path) in enumerate(ckpts):
        if i < args.start_from:
            continue
        n_run += 1
        if evaluate_ckpt(name, path, args):
            n_ok += 1

    aggregate(args)
    print(f"FINAL_METRICS bench_complete={n_ok == n_run} n_ok={n_ok}/{n_run}")
    return 0 if n_ok == n_run else 1


if __name__ == "__main__":
    sys.exit(main())
