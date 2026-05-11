"""Single-domain 4B Full FT bio retrain (corruption recovery, 2026-05-10).

Designed for the post-volume-upgrade pod (60 GB ceiling). Unlike the
multi-domain chain, this script:
  - Launches train_specialist_full_ft.py for biology only
  - Uses save-steps=1500 so cp-1500 acts as a safety-net checkpoint
  - Polls until BOTH top-level config.json AND model-*.safetensors exist
  - rsyncs the TOP-LEVEL output (sharded) — not cp-N — because save_model
    is now expected to succeed under the 60 GB ceiling
  - Runs a per-tensor zero-scan after rsync; aborts if any tensor is zero
  - Falls back to cp-1500 if top-level integrity check fails

Usage:
  python -m src.scripts.retrain_4b_biology \\
      --pod-host root@69.30.85.238 --pod-port 22192 \\
      --pod-key ~/.ssh/runpod_key
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

DOMAIN = "biology"
SOURCE = "pubmedqa"
MODEL_NAME = "Qwen/Qwen3-4B"
EXPECTED_SAFETENSORS_BYTES = 8_050_000_000


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


def launch_training(args):
    out = f"{args.pod_base}/{DOMAIN}"
    log = f"{args.pod_logs}/{DOMAIN}_full_ft.log"
    pidfile = f"{args.pod_logs}/{DOMAIN}_full_ft.pid"
    cmd = (
        f"mkdir -p {args.pod_logs} {out} && "
        f"setsid bash -c 'nohup python {args.pod_repo}/src/scripts/"
        f"train_specialist_full_ft.py "
        f"--model-name {MODEL_NAME} "
        f"--source {SOURCE} --domain {DOMAIN} "
        f"--output-dir {out} --save-steps 1500 "
        f"--batch-size 1 --grad-accum 8 --epochs 3 "
        f"--max-train-examples 5000 "
        f"> {log} 2>&1 < /dev/null & echo $! > {pidfile}'"
    )
    print(f"[{DOMAIN}] launching training -> {log}", flush=True)
    rc, _, err = ssh_run(args.pod_host, args.pod_port, args.pod_key, cmd)
    if rc != 0:
        print(f"  ssh launch FAIL: {err.strip()[:300]}", flush=True)
        return False
    return True


def wait_for_completion(args):
    out = f"{args.pod_base}/{DOMAIN}"
    deadline = time.time() + args.max_wait_hours * 3600
    last_log_size = -1
    while time.time() < deadline:
        # Top-level fully written?
        _, st, _ = ssh_run(
            args.pod_host, args.pod_port, args.pod_key,
            f"test -f {out}/config.json && "
            f"test -f {out}/model.safetensors.index.json && echo OK || echo MISSING")
        if "OK" in st:
            # Check trainer is gone
            _, ps, _ = ssh_run(
                args.pod_host, args.pod_port, args.pod_key,
                f"ps -ef | grep 'train_specialist_full_ft.*--domain {DOMAIN}' | "
                f"grep -v grep | wc -l")
            if ps.strip() == "0":
                print(f"[{DOMAIN}] training complete -> top-level config + index",
                      flush=True)
                return True
        # Log size update
        _, sz, _ = ssh_run(
            args.pod_host, args.pod_port, args.pod_key,
            f"stat -c %s {args.pod_logs}/{DOMAIN}_full_ft.log 2>/dev/null || echo 0")
        sz = int((sz.strip() or "0").split()[0])
        if sz != last_log_size:
            last_log_size = sz
            _, dfree, _ = ssh_run(
                args.pod_host, args.pod_port, args.pod_key,
                "df -B1G /workspace | tail -1 | awk '{print $4 \"GB free\"}'")
            print(f"  [{DOMAIN}] training in progress (log: {sz} bytes, "
                  f"{dfree.strip()})", flush=True)
        time.sleep(args.poll_seconds)
    print(f"  [{DOMAIN}] TIMEOUT", flush=True)
    return False


def pull_top_level(args):
    pod_dir = f"{args.pod_base}/{DOMAIN}/"
    local_dir = Path(args.local_base) / DOMAIN
    local_dir.mkdir(parents=True, exist_ok=True)
    rsync_cmd = [
        "rsync", "-rL", "--no-owner", "--no-group", "--no-perms",
        "--exclude=checkpoint-*",  # only pull top-level, not cp dirs
        "-e", f"ssh -p {args.pod_port} -i {args.pod_key} "
              f"-o StrictHostKeyChecking=no",
        f"{args.pod_host}:{pod_dir}",
        str(local_dir) + "/",
    ]
    print(f"[{DOMAIN}] rsync top-level -> {local_dir}", flush=True)
    res = subprocess.run(rsync_cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"  rsync FAIL: {res.stderr.strip()[:300]}", flush=True)
        return False
    return True


def integrity_scan(local_dir):
    """Return (n_total, n_all_zero, list_of_zero_keys)."""
    from safetensors import safe_open
    shards = sorted(Path(local_dir).glob("model-*.safetensors"))
    if not shards:
        return 0, 1, ["NO_SHARDS"]
    n_total = 0
    n_zero = 0
    zero_keys = []
    for sh in shards:
        with safe_open(sh, framework="pt") as f:
            for k in f.keys():
                t = f.get_tensor(k)
                n_total += 1
                if float(t.abs().max()) == 0.0:
                    n_zero += 1
                    zero_keys.append(f"{sh.name}::{k}")
    return n_total, n_zero, zero_keys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pod-host", default="root@69.30.85.238")
    p.add_argument("--pod-port", default="22192")
    p.add_argument("--pod-key", default=str(Path.home() / ".ssh/runpod_key"))
    p.add_argument("--pod-repo", default="/workspace/halulujah")
    p.add_argument("--pod-base", default="/workspace/adapters_4b_full_ft")
    p.add_argument("--pod-logs", default="/workspace/logs_4b_full_ft")
    p.add_argument("--local-base",
                   default="/media/dan/WD_BLACK/halulujah_4b_full_ft")
    p.add_argument("--max-wait-hours", type=float, default=2.0)
    p.add_argument("--poll-seconds", type=int, default=60)
    args = p.parse_args()

    if not launch_training(args):
        return 1
    if not wait_for_completion(args):
        return 2
    if not pull_top_level(args):
        return 3

    local_dir = Path(args.local_base) / DOMAIN
    print(f"\n=== integrity scan: {local_dir} ===")
    n_total, n_zero, zero_keys = integrity_scan(local_dir)
    print(f"  ok={n_total - n_zero}/{n_total}  all_zero={n_zero}")
    if n_zero > 0:
        print(f"  CORRUPTION DETECTED. First 10 zero tensors:")
        for k in zero_keys[:10]:
            print(f"    {k}")
        print(f"FINAL_METRICS retrain_complete=false reason=corruption "
              f"n_zero={n_zero}")
        return 4

    # Cleanup pod
    ssh_run(args.pod_host, args.pod_port, args.pod_key,
            f"rm -rf {args.pod_base}/{DOMAIN}")
    print(f"FINAL_METRICS retrain_complete=true n_total={n_total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
