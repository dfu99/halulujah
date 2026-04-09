#!/usr/bin/env python3
"""Monitor RunPod experiment progress via SSH.

Checks fine-tuning and collaboration phases, reports status,
and pulls results when complete.

Usage:
    python src/scripts/monitor_runpod.py
    python src/scripts/monitor_runpod.py --interval 300  # check every 5 min
    python src/scripts/monitor_runpod.py --pull           # pull results now
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

SSH_CMD = "ssh -i ~/.ssh/runpod_key -o StrictHostKeyChecking=no -o ConnectTimeout=10"
HOST = "root@69.30.85.178"
PORT = "22090"
REMOTE_DIR = "/workspace/halulujah"
REMOTE_LOG = f"{REMOTE_DIR}/logs/finetune.log"
REMOTE_RESULTS = f"{REMOTE_DIR}/results/revision"
LOCAL_RESULTS = Path("results/runpod_revision")


def ssh(cmd, timeout=30):
    """Run a command on the RunPod pod."""
    full = f"{SSH_CMD} {HOST} -p {PORT} \"{cmd}\""
    try:
        result = subprocess.run(
            full, shell=True, capture_output=True, text=True, timeout=timeout,
        )
        return result.stdout.strip(), result.returncode
    except subprocess.TimeoutExpired:
        return "[SSH timeout]", 1
    except Exception as e:
        return f"[Error: {e}]", 1


def check_process():
    """Check if the experiment process is still running."""
    out, _ = ssh("ps aux | grep 'run_domain_experiment' | grep -v grep | head -3")
    return bool(out.strip())


def check_finetune_progress():
    """Parse fine-tuning log for domain progress."""
    out, rc = ssh(
        "grep -E '(Fine-tuning domain|Adapter saved|FINETUNE COMPLETE)' "
        f"{REMOTE_LOG} 2>/dev/null"
    )
    if rc != 0:
        return None, "No log found"

    lines = out.strip().split("\n") if out.strip() else []
    domains_started = [l for l in lines if "Fine-tuning domain" in l]
    domains_saved = [l for l in lines if "Adapter saved" in l]
    complete = any("FINETUNE COMPLETE" in l for l in lines)

    return {
        "started": len(domains_started),
        "saved": len(domains_saved),
        "complete": complete,
        "domains_started": domains_started,
    }, None


def check_phase_progress():
    """Check which phases have completed."""
    out, _ = ssh(
        "grep -E '(Phase [234] |=== Protocol:|FINETUNE COMPLETE|"
        "Phase [234] done|ALL COMPLETE|Collab .+:)' "
        f"{REMOTE_LOG} 2>/dev/null | tail -30"
    )
    return out


def check_collab_results():
    """Check for collaboration result files."""
    out, _ = ssh(
        f"ls -la {REMOTE_RESULTS}/collaboration_*/collab_summary.json 2>/dev/null; "
        f"ls -la {REMOTE_RESULTS}/collaboration/collab_summary.json 2>/dev/null"
    )
    return out


def check_adapters():
    """Check how many adapters exist."""
    out, _ = ssh(f"ls -d {REMOTE_RESULTS}/adapter_* 2>/dev/null | wc -l")
    try:
        return int(out.strip())
    except ValueError:
        return 0


def check_gpu():
    """Check GPU utilization."""
    out, _ = ssh("nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total "
                 "--format=csv,noheader,nounits 2>/dev/null")
    return out


def get_last_log_lines(n=5):
    """Get the tail of the experiment log."""
    # Filter out progress bars, just get meaningful lines
    out, _ = ssh(
        f"grep -v '^\\ ' {REMOTE_LOG} 2>/dev/null | "
        f"grep -v '|█\\||▊\\||▋\\||▌\\||▍\\||▎\\||▏\\|| \\|it/s]' | "
        f"tail -{n}"
    )
    return out


def pull_results():
    """SCP results from RunPod to local machine."""
    LOCAL_RESULTS.mkdir(parents=True, exist_ok=True)
    print(f"Pulling results to {LOCAL_RESULTS}/...")

    cmd = (
        f"scp -r -i ~/.ssh/runpod_key -P {PORT} "
        f"{HOST}:{REMOTE_RESULTS}/collaboration* "
        f"{HOST}:{REMOTE_RESULTS}/domain_manifest.json "
        f"{HOST}:{REMOTE_RESULTS}/cross_eval "
        f"{HOST}:{REMOTE_RESULTS}/domain_distance* "
        f"{LOCAL_RESULTS}/ 2>&1"
    )
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
    if result.returncode == 0:
        print("Results pulled successfully.")
    else:
        print(f"Pull failed: {result.stderr}")
    return result.returncode == 0


def format_status():
    """Build a full status report."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [f"\n{'='*60}", f"  RunPod Experiment Status — {now}", f"{'='*60}"]

    # Process alive?
    running = check_process()
    lines.append(f"\n  Process: {'RUNNING' if running else 'NOT RUNNING'}")

    # GPU
    gpu = check_gpu()
    if gpu:
        parts = gpu.split(", ")
        if len(parts) == 3:
            lines.append(f"  GPU: {parts[0]}% util, {parts[1]}MB / {parts[2]}MB VRAM")
        else:
            lines.append(f"  GPU: {gpu}")

    # Adapters
    n_adapters = check_adapters()
    lines.append(f"  Adapters: {n_adapters}/10")

    # Fine-tuning
    ft, err = check_finetune_progress()
    if ft:
        status = "COMPLETE" if ft["complete"] else f"{ft['started']}/10 started, {ft['saved']}/10 saved"
        lines.append(f"  Fine-tuning: {status}")
    elif err:
        lines.append(f"  Fine-tuning: {err}")

    # Phase progress
    phases = check_phase_progress()
    if phases:
        lines.append(f"\n  Recent activity:")
        for line in phases.split("\n")[-8:]:
            lines.append(f"    {line.strip()}")

    # Collaboration results
    collab = check_collab_results()
    if collab:
        lines.append(f"\n  Result files found:")
        for line in collab.strip().split("\n"):
            lines.append(f"    {line.strip()}")

    # Last log lines
    last = get_last_log_lines(3)
    if last:
        lines.append(f"\n  Last log entries:")
        for line in last.split("\n")[-3:]:
            lines.append(f"    {line.strip()[:100]}")

    lines.append(f"\n{'='*60}")
    return "\n".join(lines)


def notify_slack(message):
    """Stage a Slack notification."""
    slack_dir = Path("/tmp/mc-slack-out/halulujah")
    slack_dir.mkdir(parents=True, exist_ok=True)
    (slack_dir / "attachments").mkdir(exist_ok=True)

    (slack_dir / "response.md").write_text(
        f"> Re: RunPod experiment monitor\n\n{message}"
    )
    (slack_dir / "metadata.json").write_text(json.dumps({
        "project": "halulujah",
        "response_file": "response.md",
        "attachments": [],
        "timestamp": datetime.now().strftime("%Y-%m-%d"),
    }))


def main():
    parser = argparse.ArgumentParser(description="Monitor RunPod experiment")
    parser.add_argument("--interval", type=int, default=600,
                        help="Check interval in seconds (default: 600 = 10 min)")
    parser.add_argument("--pull", action="store_true",
                        help="Pull results and exit")
    parser.add_argument("--once", action="store_true",
                        help="Check once and exit")
    parser.add_argument("--notify", action="store_true",
                        help="Send Slack notification on completion")
    args = parser.parse_args()

    if args.pull:
        pull_results()
        return

    print(f"Monitoring RunPod experiment (checking every {args.interval}s)")
    print(f"Pod: {HOST}:{PORT}")
    print(f"Press Ctrl+C to stop\n")

    last_adapters = 0
    start_time = datetime.now()

    while True:
        try:
            status = format_status()
            print(status)

            # Check for completion
            phases = check_phase_progress() or ""
            if "ALL COMPLETE" in phases:
                elapsed = datetime.now() - start_time
                print(f"\n  EXPERIMENT COMPLETE (monitored for {elapsed})")
                print("  Pulling results...")
                if pull_results():
                    print(f"  Results saved to {LOCAL_RESULTS}/")
                if args.notify:
                    notify_slack(
                        "*RunPod experiment COMPLETE*\n"
                        f"Runtime: {elapsed}\n"
                        "Results pulled to `results/runpod_revision/`"
                    )
                break

            # Check if process died
            if not check_process():
                ft, _ = check_finetune_progress()
                if ft and ft.get("complete"):
                    # Fine-tuning done but no ALL COMPLETE — check if collab started
                    collab = check_collab_results()
                    if not collab:
                        print("\n  WARNING: Process died after fine-tuning, before collaboration")
                        print("  You may need to restart Phase 4 manually")
                elif not ft or not ft.get("complete"):
                    print("\n  WARNING: Process not running and fine-tuning not complete")
                    print("  Experiment may have crashed — check log:")
                    print(f"    ssh -i ~/.ssh/runpod_key {HOST} -p {PORT} "
                          f"\"tail -30 {REMOTE_LOG}\"")

            # Progress notification
            n_adapters = check_adapters()
            if n_adapters > last_adapters:
                print(f"  [+] New adapter trained ({n_adapters}/10)")
                last_adapters = n_adapters

            if args.once:
                break

            time.sleep(args.interval)

        except KeyboardInterrupt:
            print("\nStopped monitoring.")
            break


if __name__ == "__main__":
    main()
