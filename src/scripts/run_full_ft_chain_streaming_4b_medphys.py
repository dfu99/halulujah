"""Streaming-cleanup wrapper for the 5-domain Full FT queue (Option A).

Local-side workstation script. For each domain in
medicine -> math -> biology -> law -> physics:

  1. ssh-launch train_specialist_full_ft.py on the pod (setsid'd so it
     survives SSH disconnects per the 2026-05-04 lesson).
  2. Poll for completion (top-level config.json in the output dir).
  3. rsync all completed checkpoints from pod -> WD_BLACK.
  4. Verify each pulled model.safetensors is >= 95% expected size
     (~3.4 GB for Qwen3-1.7B bf16).
  5. ssh: delete completed checkpoints on pod (keep the top-level
     final adapter dir; the per-step checkpoints are no longer
     needed once mirrored).
  6. Move to next domain.

This addresses the 2026-05-06 PI direction: train one, back it up,
clear storage, move to next. Replaces the no-cleanup chain at
src/scripts/run_full_ft_chain.py whose 4-of-5 truncation was
caused by accumulated checkpoints overflowing the 50 GB container
volume.

Peak disk per domain with this pattern: ~3.4 GB × 4 ckpts +
intermediate optimizer state during training ≈ 14 GB. Plus 28 GB
HF cache + 1.3 GB src ≈ 43 GB peak. Fits the default RunPod
container without volume increase.

Usage (run on the workstation, NOT the pod):
  python -m src.scripts.run_full_ft_chain_streaming \\
      --pod-host root@<ip> --pod-port <port> \\
      --pod-key ~/.ssh/runpod_key \\
      --pod-base /workspace/adapters_1p7b_full_ft \\
      --local-base /media/dan/WD_BLACK/halulujah_full_ft_streaming \\
      [--dry-run]

Pre-flight: requires GPU access on the pod and the source code +
adapters already synced to the pod via `mc runpod sync halulujah`.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# (domain, training source dataset)
DOMAINS = [
    # Medicine + physics 4B FT specialists do NOT exist on WD_BLACK
    # (the original run_4b_full_ft.py trained-evaluated-deleted them
    # ephemerally, retaining only solo + base-helper eval JSONs).
    # Retraining here for the full 4B pair-grid.
    ("medicine", "medqa"),
    ("physics", "sciq"),
]

EXPECTED_SAFETENSORS_BYTES = 8_050_000_000  # ~8.05 GB for Qwen3-4B bf16
MODEL_NAME = "Qwen/Qwen3-4B"


def _build_safetensors_index(local_dir, shards, idx_path):
    """Reconstruct model.safetensors.index.json from sharded files."""
    import json as _json
    from safetensors import safe_open as _safe_open
    weight_map = {}
    total_size = 0
    for shard in shards:
        with _safe_open(shard, framework="pt") as f:
            for k in f.keys():
                weight_map[k] = shard.name
                t = f.get_tensor(k)
                total_size += t.element_size() * t.numel()
    idx_path.write_text(_json.dumps({
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }, indent=2))
SAFETENSORS_TOLERANCE = 0.95


def ssh_args(pod_host: str, pod_port: str, pod_key: str) -> list[str]:
    return [
        "ssh", "-p", pod_port, "-i", pod_key,
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=10",
        "-o", "ServerAliveInterval=30",
        pod_host,
    ]


def ssh_run(pod_host: str, pod_port: str, pod_key: str, cmd: str,
            dry_run: bool = False) -> tuple[int, str, str]:
    args = ssh_args(pod_host, pod_port, pod_key) + [cmd]
    if dry_run:
        print(f"  DRY: {' '.join(args)}", flush=True)
        return 0, "", ""
    res = subprocess.run(args, capture_output=True, text=True)
    return res.returncode, res.stdout, res.stderr


def has_reusable_local(args: argparse.Namespace, domain: str) -> bool:
    """Return True if local_base/<domain>/model.safetensors is already complete."""
    local_st = Path(args.local_base) / domain / "model.safetensors"
    if not local_st.exists():
        return False
    size = local_st.stat().st_size
    if size < EXPECTED_SAFETENSORS_BYTES * SAFETENSORS_TOLERANCE:
        return False
    return True


def push_local_to_pod(args: argparse.Namespace, domain: str) -> bool:
    """rsync a complete local adapter dir up to the pod (skips retraining)."""
    src_dir = Path(args.local_base) / domain
    pod_dir = f"{args.pod_base}/{domain}"
    print(f"[{domain}] reusing local checkpoint; rsync up -> {pod_dir}",
          flush=True)
    # ensure parent on pod exists
    rc, _, err = ssh_run(args.pod_host, args.pod_port, args.pod_key,
                          f"mkdir -p {args.pod_base}", args.dry_run)
    if rc != 0:
        print(f"  ssh mkdir FAIL: {err.strip()[:200]}", flush=True)
        return False
    rsync_cmd = [
        "rsync", "-rL", "--no-owner", "--no-group", "--no-perms",
        "-e", f"ssh -p {args.pod_port} -i {args.pod_key} "
              f"-o StrictHostKeyChecking=no",
        str(src_dir) + "/",
        f"{args.pod_host}:{pod_dir}/",
    ]
    if args.dry_run:
        print(f"  DRY: {' '.join(rsync_cmd)}", flush=True)
        return True
    res = subprocess.run(rsync_cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"  rsync up FAIL rc={res.returncode}: "
              f"{res.stderr.strip()[:300]}", flush=True)
        return False
    print(f"[{domain}] local adapter mirrored to pod ✓", flush=True)
    return True


def launch_training(args: argparse.Namespace, domain: str, source: str) -> bool:
    """setsid-launch train_specialist_full_ft.py on the pod (background)."""
    out = f"{args.pod_base}/{domain}"
    log = f"{args.pod_logs}/{domain}_full_ft.log"
    pidfile = f"{args.pod_logs}/{domain}_full_ft.pid"
    # 4B params: --batch-size 1 --grad-accum 8 (vs 1.7B's batch 1 grad-accum 4),
    # --gradient-checkpointing on, --max-train-examples 5000 (down from 10k for time),
    # --epochs 3.
    # 4B params: --batch-size 1 --grad-accum 8, gradient_checkpointing on,
    # --max-train-examples 5000 (3 epochs × 5000 / 8 = 1875 steps),
    # --save-steps very high (≥ total steps) to skip per-step ckpts
    # and avoid the moosefs ~21 GB user quota with 8 GB-per-ckpt models.
    # Final adapter (8 GB) saved at training end + rsync'd to WD_BLACK
    # before next domain starts.
    cmd = (
        f"mkdir -p {args.pod_logs} {out} && "
        f"setsid bash -c 'nohup python {args.pod_repo}/src/scripts/"
        f"train_specialist_full_ft.py "
        f"--model-name {MODEL_NAME} "
        f"--source {source} --domain {domain} "
        f"--output-dir {out} --save-steps 99999 "
        f"--batch-size 1 --grad-accum 8 --epochs 3 "
        f"--max-train-examples 5000 "
        f"> {log} 2>&1 < /dev/null & echo $! > {pidfile}'"
    )
    print(f"[{domain}] launching training -> {log}", flush=True)
    rc, _, err = ssh_run(args.pod_host, args.pod_port, args.pod_key, cmd,
                          args.dry_run)
    if rc != 0:
        print(f"  ssh launch FAIL: {err.strip()[:300]}", flush=True)
        return False
    return True


def wait_for_completion(args: argparse.Namespace, domain: str) -> bool:
    """Poll for either (a) top-level config.json or (b) checkpoint-N/config.json
    signalling training complete.

    For 4B Full FT, the trainer's final save_model() to top-level often
    fails with quota (see audit lessons.md): 8 GB cp-N already on disk,
    save_model tries to write another 8 GB to top-level → quota error.
    The cp-N has the trained model. We accept it as final."""
    out = f"{args.pod_base}/{domain}"
    deadline = time.time() + args.max_wait_hours * 3600
    poll = args.poll_seconds
    while time.time() < deadline:
        if args.dry_run:
            return True
        # Option A: top-level config.json (trainer.save_model succeeded)
        _, stdout_a, _ = ssh_run(
            args.pod_host, args.pod_port, args.pod_key,
            f"test -f {out}/config.json && echo OK || echo MISSING",
            args.dry_run,
        )
        if "OK" in stdout_a:
            print(f"[{domain}] training complete -> top-level config.json",
                  flush=True)
            return True
        # Option B: any checkpoint-N/model*.safetensors (final-step ckpt)
        _, stdout_b, _ = ssh_run(
            args.pod_host, args.pod_port, args.pod_key,
            f"ls {out}/checkpoint-*/model*.safetensors 2>/dev/null | head -1",
            args.dry_run,
        )
        if stdout_b.strip().endswith(".safetensors"):
            # Also check trainer process is gone (so save is complete)
            _, stdout_p, _ = ssh_run(
                args.pod_host, args.pod_port, args.pod_key,
                f"ps -ef | grep 'train_specialist_full_ft.*--domain {domain}' | "
                f"grep -v grep | head -1",
                args.dry_run,
            )
            if not stdout_p.strip():
                print(f"[{domain}] training complete -> checkpoint-N detected"
                      f" + trainer exited", flush=True)
                return True
        # Progress: most-recent step from log
        _, tail, _ = ssh_run(args.pod_host, args.pod_port, args.pod_key,
                             f"tail -n 1 {args.pod_logs}/{domain}_full_ft.log "
                             f"2>/dev/null || echo '(no log yet)'",
                             args.dry_run)
        print(f"[{domain}] waiting; tail: {tail.strip()[:160]}", flush=True)
        time.sleep(poll)
    print(f"[{domain}] TIMEOUT", flush=True)
    return False


def pull_and_clean(args: argparse.Namespace, domain: str) -> bool:
    """rsync the trained model from pod to WD_BLACK; verify; delete from pod.

    For 4B Full FT, the trainer's final save_model() to top-level often
    fails on quota; cp-N (deepest checkpoint-N dir) has the trained
    model and is what we pull. The model is sharded
    (model-00001-of-00002.safetensors etc.). We also pull tokenizer
    files from the Qwen3-4B base cache and reconstruct
    model.safetensors.index.json locally if missing."""
    pod_dir = f"{args.pod_base}/{domain}/"
    local_dir = Path(args.local_base) / domain
    local_dir.mkdir(parents=True, exist_ok=True)
    # Find latest cp-N on pod
    rc, stdout, _ = ssh_run(
        args.pod_host, args.pod_port, args.pod_key,
        f"ls -d {pod_dir}/checkpoint-* 2>/dev/null | "
        f"sort -V | tail -1",
        args.dry_run,
    )
    cp_dir = stdout.strip() if not args.dry_run else f"{pod_dir}checkpoint-1875"
    src = f"{args.pod_host}:{cp_dir}/"
    # Pull cp-N (sharded model + config files)
    rsync_cmd = [
        "rsync", "-rL", "--no-owner", "--no-group", "--no-perms",
        "-e", f"ssh -p {args.pod_port} -i {args.pod_key} "
              f"-o StrictHostKeyChecking=no",
        src, str(local_dir) + "/",
    ]
    print(f"[{domain}] rsync {cp_dir} -> {local_dir}", flush=True)
    if args.dry_run:
        print(f"  DRY: {' '.join(rsync_cmd)}", flush=True)
    else:
        res = subprocess.run(rsync_cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"  rsync FAIL rc={res.returncode}: "
                  f"{res.stderr.strip()[:300]}", flush=True)
            return False

    # Pull tokenizer files from base Qwen3-4B cache (single cache snapshot)
    rc, snap_out, _ = ssh_run(
        args.pod_host, args.pod_port, args.pod_key,
        "ls /workspace/hf_cache/models--Qwen--Qwen3-4B/snapshots/ 2>/dev/null | head -1",
        args.dry_run,
    )
    snap = snap_out.strip() if not args.dry_run else "DRY"
    if snap and not args.dry_run:
        cache_src = (
            f"/workspace/hf_cache/models--Qwen--Qwen3-4B/snapshots/{snap}/"
        )
        tok_files = [
            "tokenizer.json", "tokenizer_config.json",
            "special_tokens_map.json", "added_tokens.json",
            "merges.txt", "vocab.json", "chat_template.jinja",
        ]
        rsync_tok = [
            "rsync", "-aL", "--no-owner", "--no-group", "--no-perms",
            "--ignore-missing-args",
            "-e", f"ssh -p {args.pod_port} -i {args.pod_key} "
                  f"-o StrictHostKeyChecking=no",
        ]
        for f in tok_files:
            rsync_tok.append(f"{args.pod_host}:{cache_src}{f}")
        rsync_tok.append(str(local_dir) + "/")
        subprocess.run(rsync_tok, capture_output=True, text=True)
        print(f"  pulled tokenizer files from base snapshot {snap[:8]}",
              flush=True)

    # Verify total model size; reconstruct safetensors index if missing
    if not args.dry_run:
        shards = sorted(local_dir.glob("model-*.safetensors"))
        single = local_dir / "model.safetensors"
        if shards:
            total = sum(s.stat().st_size for s in shards)
            if total < EXPECTED_SAFETENSORS_BYTES * SAFETENSORS_TOLERANCE:
                print(f"  sharded model only {total/1e9:.2f} GB; expected "
                      f">={EXPECTED_SAFETENSORS_BYTES*SAFETENSORS_TOLERANCE/1e9:.2f} GB. "
                      f"ABORT.", flush=True)
                return False
            print(f"  verified {len(shards)} shards, {total/1e9:.2f} GB total ✓",
                  flush=True)
            # Build index.json if missing
            idx = local_dir / "model.safetensors.index.json"
            if not idx.exists():
                _build_safetensors_index(local_dir, shards, idx)
                print(f"  reconstructed {idx.name}", flush=True)
        elif single.exists():
            size = single.stat().st_size
            if size < EXPECTED_SAFETENSORS_BYTES * SAFETENSORS_TOLERANCE:
                print(f"  single model.safetensors only {size/1e9:.2f} GB; "
                      f"ABORT.", flush=True)
                return False
            print(f"  verified single safetensors {size/1e9:.2f} GB ✓",
                  flush=True)
        else:
            print(f"  no model files found at {local_dir}; ABORT.",
                  flush=True)
            return False

    # Aggressive cleanup: delete the ENTIRE <domain>/ dir from pod after
    # successful rsync to WD_BLACK. The PI's 2026-05-07 directive was
    # "Whenever we produce a new checkpoint, move it to WD_BLACK to keep
    # the Pod volume open." Per-domain rm keeps cumulative pod usage at
    # ~10-15 GB peak (just current-domain training + HF cache + 1 prior
    # final at most), well below the moosefs ~21 GB user/group quota.
    rm_cmd = f"rm -rf {args.pod_base}/{domain}"
    rc, _, err = ssh_run(args.pod_host, args.pod_port, args.pod_key, rm_cmd,
                          args.dry_run)
    if rc != 0:
        print(f"  ssh rm FAIL: {err.strip()[:200]}", flush=True)
        return False
    print(f"[{domain}] aggressive cleanup done; pod {domain}/ removed entirely",
          flush=True)
    return True


def get_pod_disk_free(args: argparse.Namespace) -> str:
    _, stdout, _ = ssh_run(args.pod_host, args.pod_port, args.pod_key,
                            "df -BG /workspace | tail -n 1",
                            args.dry_run)
    if args.dry_run:
        return "(dry)"
    return stdout.strip()


def main() -> int:
    p = argparse.ArgumentParser(
        description="Streaming-cleanup chain: train + backup + clean per domain"
    )
    p.add_argument("--pod-host", required=True,
                   help="e.g. root@69.30.85.238")
    p.add_argument("--pod-port", required=True)
    p.add_argument("--pod-key", default=os.path.expanduser("~/.ssh/runpod_key"))
    p.add_argument("--pod-base", default="/workspace/adapters_4b_full_ft")
    p.add_argument("--pod-repo", default="/workspace/halulujah")
    p.add_argument("--pod-logs", default="/workspace/halulujah/logs")
    p.add_argument("--local-base", required=True,
                   help="WD_BLACK destination root for the streaming run")
    p.add_argument("--save-steps", type=int, default=1500)
    p.add_argument("--poll-seconds", type=int, default=120)
    p.add_argument("--max-wait-hours", type=int, default=8)
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands without executing on pod")
    p.add_argument("--start-from", default=None,
                   help="Resume from this domain (skip earlier domains)")
    p.add_argument("--reuse-local", action="store_true",
                   help="If --local-base/<domain>/model.safetensors is already "
                        "complete, rsync it up to the pod and skip training. "
                        "Useful for resuming a partial chain.")
    args = p.parse_args()

    Path(args.local_base).mkdir(parents=True, exist_ok=True)

    print(f"streaming chain up; pod {args.pod_host}:{args.pod_port}",
          flush=True)
    print(f"  pod_base = {args.pod_base}", flush=True)
    print(f"  local_base = {args.local_base}", flush=True)
    print(f"  pod disk: {get_pod_disk_free(args)}", flush=True)

    started = (args.start_from is None)
    results = []
    for domain, source in DOMAINS:
        if not started:
            if domain == args.start_from:
                started = True
            else:
                print(f"[{domain}] skipped (--start-from {args.start_from})",
                      flush=True)
                continue

        print(f"\n=== [{domain}] begin (source={source}) ===", flush=True)

        # If the local adapter dir already has a complete model.safetensors,
        # mirror it up to the pod and skip training. Saves ~1h45m per domain.
        if args.reuse_local and has_reusable_local(args, domain):
            if push_local_to_pod(args, domain):
                results.append((domain, "REUSED"))
                print(f"  pod disk after reuse: {get_pod_disk_free(args)}",
                      flush=True)
                continue
            else:
                print(f"[{domain}] reuse failed; will retrain", flush=True)

        if not launch_training(args, domain, source):
            results.append((domain, "launch_failed"))
            continue
        if not wait_for_completion(args, domain):
            results.append((domain, "training_timeout"))
            continue
        if not pull_and_clean(args, domain):
            results.append((domain, "cleanup_failed"))
            continue
        results.append((domain, "OK"))
        print(f"  pod disk after cleanup: {get_pod_disk_free(args)}",
              flush=True)

    print("\n=== streaming chain summary ===", flush=True)
    for d, status in results:
        print(f"  {d}: {status}", flush=True)
    success_states = {"OK", "REUSED"}
    all_ok = all(s in success_states for _, s in results)
    n_ok = sum(1 for _, s in results if s in success_states)
    n_reused = sum(1 for _, s in results if s == "REUSED")
    print(f"FINAL_METRICS chain_complete={all_ok} "
          f"n_ok={n_ok}/{len(results)} n_reused={n_reused}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
