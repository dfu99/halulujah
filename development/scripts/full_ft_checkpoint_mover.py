"""Continuously mirror Full FT checkpoints from the A40 pod to WD_BLACK
and delete them from the pod, so saves do not pile up on /workspace and
trip moosefs disk-quota errors mid-write.

Polls every --poll-seconds. For each domain dir, finds completed
checkpoints (model.safetensors >= 95% expected size, mtime older than
--skip-newer-than seconds), rsyncs to WD_BLACK, deletes from pod.
Always keeps the most-recent checkpoint on the pod.

Run on the local workstation (not the pod). Loops until killed.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

POD_HOST = "root@69.30.85.238"
POD_PORT = "22192"
POD_KEY = os.path.expanduser("~/.ssh/runpod_key")
POD_BASE = "/workspace/adapters_1p7b_full_ft"
LOCAL_BASE = Path("/media/dan/WD_BLACK/halulujah_full_ft_streaming")

DOMAINS = ["medicine", "math", "biology", "law", "physics"]
EXPECTED_SAFETENSORS_BYTES = 3_446_000_000  # ~3.4 GB for Qwen3-1.7B bf16
SAFETENSORS_TOLERANCE = 0.95


def ssh_run(cmd):
    args = ["ssh", "-p", POD_PORT, "-i", POD_KEY,
            "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
            POD_HOST, cmd]
    res = subprocess.run(args, capture_output=True, text=True)
    return res.returncode, res.stdout, res.stderr


def list_pod_checkpoints(domain):
    cmd = (f"for d in {POD_BASE}/{domain}/checkpoint-*; do "
           f"if [ -d \"$d\" ] && [ -f \"$d/model.safetensors\" ]; then "
           f"  SIZE=$(stat -c%s \"$d/model.safetensors\"); "
           f"  MTIME=$(stat -c%Y \"$d/model.safetensors\"); "
           f"  echo \"$(basename $d) $MTIME $SIZE\"; "
           f"fi; done")
    rc, out, _ = ssh_run(cmd)
    if rc != 0:
        return []
    rows = []
    for line in out.strip().splitlines():
        parts = line.split()
        if len(parts) == 3:
            rows.append((parts[0], int(parts[1]), int(parts[2])))
    return sorted(rows, key=lambda r: r[1])


def is_complete(size):
    return size >= EXPECTED_SAFETENSORS_BYTES * SAFETENSORS_TOLERANCE


def pull_and_delete(domain, ckpt_name):
    src = f"{POD_HOST}:{POD_BASE}/{domain}/{ckpt_name}/"
    dst_dir = LOCAL_BASE / domain / ckpt_name
    dst_dir.mkdir(parents=True, exist_ok=True)
    rsync_cmd = [
        "rsync", "-rL", "--no-owner", "--no-group", "--no-perms",
        "-e", f"ssh -p {POD_PORT} -i {POD_KEY} -o StrictHostKeyChecking=no",
        src, str(dst_dir) + "/",
    ]
    print(f"  pull {domain}/{ckpt_name} -> {dst_dir}", flush=True)
    res = subprocess.run(rsync_cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"  rsync FAIL: {res.stderr.strip()[:200]}", flush=True)
        return False
    pulled = dst_dir / "model.safetensors"
    if not pulled.exists() or not is_complete(pulled.stat().st_size):
        print(f"  pulled file incomplete; not deleting from pod", flush=True)
        return False
    rc, _, err = ssh_run(f"rm -rf {POD_BASE}/{domain}/{ckpt_name}")
    if rc != 0:
        print(f"  pod rm FAIL: {err.strip()[:200]}", flush=True)
        return False
    print(f"  deleted {domain}/{ckpt_name} on pod", flush=True)
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--poll-seconds", type=int, default=180)
    p.add_argument("--skip-newer-than", type=int, default=120)
    args = p.parse_args()

    LOCAL_BASE.mkdir(parents=True, exist_ok=True)
    print(f"mover up; poll {args.poll_seconds}s; dest {LOCAL_BASE}", flush=True)
    while True:
        now = time.time()
        for domain in DOMAINS:
            ckpts = list_pod_checkpoints(domain)
            if len(ckpts) <= 1:
                continue  # keep at least one (the newest) on pod
            for name, mtime, size in ckpts[:-1]:
                if (now - mtime) < args.skip_newer_than:
                    continue
                if not is_complete(size):
                    continue
                pull_and_delete(domain, name)
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    sys.exit(main())
