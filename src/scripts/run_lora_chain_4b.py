"""Sequential LoRA training chain for Qwen3-4B specialists.

For each (domain, rank) pair, this script:
  1. ssh-launches train_specialist_lora.py on the pod (setsid'd to
     survive SSH disconnects).
  2. Polls for adapter_config.json in the output dir on the pod.
  3. rsyncs the small adapter dir (~50 MB) down to WD_BLACK.
  4. Deletes the pod-side adapter dir.
  5. Moves to next (domain, rank).

Unlike the Full FT chain, LoRA adapters are ~50 MB each, so quota /
streaming-cleanup is not a concern. Each domain × rank cycle: ~25 min
(no rsync overhead). 5 domains × 2 ranks = 10 cycles ≈ 4 h total.

Usage:
  python -m src.scripts.run_lora_chain_4b \\
      --pod-host root@69.30.85.238 --pod-port 22192 \\
      --pod-key ~/.ssh/runpod_key \\
      --pod-base /workspace/adapters_4b_lora \\
      --local-base /media/dan/WD_BLACK/halulujah_4b_lora \\
      --ranks 8 64
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

# (domain, source dataset)
DOMAINS = [
    ("medicine", "medqa"),
    ("math", "gsm8k"),
    ("biology", "pubmedqa"),
    ("law", "casehold"),
    ("physics", "sciq"),
]

MODEL_NAME = "Qwen/Qwen3-4B"
TRAIN_TIMEOUT_S = 60 * 60  # 1 hour per (domain, rank)
POLL_INTERVAL_S = 30


def ssh_run(host, port, key, cmd, dry=False):
    full = ["ssh", "-p", str(port), "-i", key,
            "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=10",
            host, cmd]
    if dry:
        print(f"  DRY: ssh ... '{cmd}'", flush=True)
        return 0, "", ""
    res = subprocess.run(full, capture_output=True, text=True)
    return res.returncode, res.stdout, res.stderr


def get_pod_disk_free(args):
    rc, out, _ = ssh_run(args.pod_host, args.pod_port, args.pod_key,
                         "df -B1 /workspace | tail -1 | awk '{print $4}'")
    if rc == 0 and out.strip().isdigit():
        return f"{int(out.strip()) / 1e9:.1f} GB"
    return "?"


def launch_training(args, domain, source, rank):
    """setsid-launch train_specialist_lora.py on the pod (background)."""
    pod_out = f"{args.pod_base}_r{rank}/{domain}"
    log_path = f"/tmp/lora_train_{domain}_r{rank}.log"
    train_cmd = (
        f"cd /workspace/halulujah && "
        f"HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 "
        f"setsid nohup python src/scripts/train_specialist_lora.py "
        f"--model-name '{MODEL_NAME}' "
        f"--source {source} --domain {domain} --rank {rank} "
        f"--output-dir {pod_out} "
        f"--epochs 3 --max-train-examples 5000 "
        f"--save-steps 99999 "
        f"--batch-size 1 --grad-accum 8 "
        f"> {log_path} 2>&1 < /dev/null &"
    )
    print(f"[{domain} r={rank}] launching training (pod log: {log_path})",
          flush=True)
    rc, _, err = ssh_run(args.pod_host, args.pod_port, args.pod_key,
                          f"mkdir -p {args.pod_base}_r{rank}/", args.dry_run)
    if rc != 0:
        print(f"  ssh mkdir FAIL: {err.strip()[:200]}", flush=True)
        return False
    rc, _, err = ssh_run(args.pod_host, args.pod_port, args.pod_key,
                          train_cmd, args.dry_run)
    if rc != 0:
        print(f"  launch FAIL: {err.strip()[:200]}", flush=True)
        return False
    return True


def wait_for_completion(args, domain, rank):
    """Poll for adapter_config.json or pid disappearance."""
    pod_out = f"{args.pod_base}_r{rank}/{domain}"
    deadline = time.time() + TRAIN_TIMEOUT_S
    last_log_size = -1
    while time.time() < deadline:
        _, out, _ = ssh_run(
            args.pod_host, args.pod_port, args.pod_key,
            f"test -f {pod_out}/adapter_config.json && echo DONE || "
            f"ps -ef | grep 'train_specialist_lora.*--domain {domain}.*--rank {rank}' | "
            f"grep -v grep | wc -l")
        out = out.strip()
        if out == "DONE":
            print(f"  [{domain} r={rank}] training done", flush=True)
            return True
        if out == "0":
            _, sz, _ = ssh_run(
                args.pod_host, args.pod_port, args.pod_key,
                f"tail -40 /tmp/lora_train_{domain}_r{rank}.log "
                f"2>/dev/null | tr '\\n' '|'")
            print(f"  [{domain} r={rank}] training process gone, no adapter. "
                  f"Tail: {sz.strip()[:500]}", flush=True)
            return False
        _, sz, _ = ssh_run(
            args.pod_host, args.pod_port, args.pod_key,
            f"stat -c %s /tmp/lora_train_{domain}_r{rank}.log 2>/dev/null || echo 0")
        sz = int((sz.strip() or "0").split()[0])
        if sz != last_log_size:
            last_log_size = sz
            print(f"  [{domain} r={rank}] training in progress "
                  f"(log: {sz} bytes)", flush=True)
        time.sleep(POLL_INTERVAL_S)
    print(f"  [{domain} r={rank}] training TIMEOUT", flush=True)
    return False


def pull_and_clean(args, domain, rank):
    """rsync adapter down, delete on pod."""
    pod_dir = f"{args.pod_base}_r{rank}/{domain}"
    local_dir = Path(args.local_base) / f"r{rank}" / domain
    local_dir.mkdir(parents=True, exist_ok=True)
    rsync_cmd = [
        "rsync", "-rL", "--no-owner", "--no-group", "--no-perms",
        "-e", f"ssh -p {args.pod_port} -i {args.pod_key} "
              f"-o StrictHostKeyChecking=no",
        f"{args.pod_host}:{pod_dir}/",
        str(local_dir) + "/",
    ]
    print(f"  rsync {pod_dir} -> {local_dir}", flush=True)
    if args.dry_run:
        print(f"  DRY: {' '.join(rsync_cmd)}", flush=True)
        return True
    res = subprocess.run(rsync_cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"  rsync FAIL: {res.stderr.strip()[:200]}", flush=True)
        return False
    if not (local_dir / "adapter_config.json").exists():
        print(f"  ERROR: pulled but no adapter_config.json at {local_dir}",
              flush=True)
        return False
    # Cleanup on pod
    ssh_run(args.pod_host, args.pod_port, args.pod_key,
            f"rm -rf {pod_dir}", args.dry_run)
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pod-host", default="root@69.30.85.238")
    p.add_argument("--pod-port", default="22192")
    p.add_argument("--pod-key", default=str(Path.home() / ".ssh/runpod_key"))
    p.add_argument("--pod-base", default="/workspace/adapters_4b_lora")
    p.add_argument("--local-base",
                   default="/media/dan/WD_BLACK/halulujah_4b_lora")
    p.add_argument("--ranks", type=int, nargs="+", default=[8, 64])
    p.add_argument("--start-from-domain", default=None)
    p.add_argument("--start-from-rank", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    Path(args.local_base).mkdir(parents=True, exist_ok=True)
    print(f"LoRA chain; pod {args.pod_host}:{args.pod_port}", flush=True)
    print(f"  pod_base = {args.pod_base}_r<rank>", flush=True)
    print(f"  local_base = {args.local_base}", flush=True)
    print(f"  ranks = {args.ranks}", flush=True)
    print(f"  pod disk: {get_pod_disk_free(args)}", flush=True)

    results = []
    started = (args.start_from_rank is None
               and args.start_from_domain is None)
    for rank in args.ranks:
        for domain, source in DOMAINS:
            if not started:
                if (args.start_from_rank is None or rank == args.start_from_rank) \
                   and (args.start_from_domain is None
                        or domain == args.start_from_domain):
                    started = True
                else:
                    print(f"[{domain} r={rank}] skipped (--start-from)",
                          flush=True)
                    continue

            print(f"\n=== [{domain} r={rank}] (source={source}) ===",
                  flush=True)
            local_dir = Path(args.local_base) / f"r{rank}" / domain
            if (local_dir / "adapter_config.json").exists():
                print(f"  already exists locally at {local_dir}; skipping",
                      flush=True)
                results.append((domain, rank, "REUSED"))
                continue

            if not launch_training(args, domain, source, rank):
                results.append((domain, rank, "launch_failed"))
                continue
            if not wait_for_completion(args, domain, rank):
                results.append((domain, rank, "training_timeout"))
                continue
            if not pull_and_clean(args, domain, rank):
                results.append((domain, rank, "cleanup_failed"))
                continue
            results.append((domain, rank, "OK"))
            print(f"  pod disk: {get_pod_disk_free(args)}", flush=True)

    print("\n=== LoRA chain summary ===", flush=True)
    for d, r, status in results:
        print(f"  {d} r={r}: {status}", flush=True)
    success_states = {"OK", "REUSED"}
    n_ok = sum(1 for _, _, s in results if s in success_states)
    print(f"FINAL_METRICS chain_complete={n_ok == len(results)} "
          f"n_ok={n_ok}/{len(results)}", flush=True)
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
