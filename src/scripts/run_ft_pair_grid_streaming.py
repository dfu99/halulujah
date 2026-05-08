"""Streaming pair-grid wrapper: workstation-side orchestrator.

Iterates the 5x5 (primary, helper) matrix on the pod by rsync'ing one
ckpt at a time into /workspace/pair_grid_ckpts/. Solves the moosefs
~21 GB user quota that prevents holding all 5 FT ckpts (3.4 GB each)
+ hf_cache + base on disk simultaneously.

For each cell:
  1. ensure primary ckpt staged in /workspace/pair_grid_ckpts/primary/
  2. ensure helper ckpt staged in /workspace/pair_grid_ckpts/helper/
     (or symlink primary if same)
  3. ssh-run cell_pair_grid_eval.py on pod with both paths
  4. scp the resulting condition JSON back to local results dir
  5. cleanup pod state

Cells:
  - solo_<d>          : primary alone on its domain (model_b ignored)
  - pair_<a>_<b>      : primary=a, helper=b, on a's domain
  - pair_<a>_base     : primary=a, helper=base
  - base_solo_<d>     : base alone (reused from LoRA grid if --reuse)
  - base_pair_<d>     : base+base (reused from LoRA grid if --reuse)

Resume: each cell writes its own JSON; existing JSONs are skipped.

Usage:
  python -m src.scripts.run_ft_pair_grid_streaming \
    --pod-host root@69.30.85.238 --pod-port 22192 \
    --ckpt-base /media/dan/WD_BLACK/halulujah_full_ft_streaming \
    --domains medicine math biology law physics \
    --n-questions 50 --n-rounds 3 \
    --output-dir results/ft_pair_grid_2026-05-08
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

POD_BASE = "/workspace/pair_grid_ckpts"
POD_PRIMARY_DIR = f"{POD_BASE}/primary"
POD_HELPER_DIR = f"{POD_BASE}/helper"
POD_OUT = "/workspace/cell_out.json"


def ssh_args(args):
    return ["ssh", "-p", args.pod_port, "-i", args.pod_key,
            "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
            args.pod_host]


def ssh_run(args, cmd, capture=True):
    full = ssh_args(args) + [cmd]
    if capture:
        res = subprocess.run(full, capture_output=True, text=True)
        return res.returncode, res.stdout, res.stderr
    return subprocess.run(full).returncode, "", ""


def rsync_to_pod(args, src_dir: Path, pod_dest: str,
                 exclude_ckpts: bool = True) -> bool:
    extras = ["--exclude=checkpoint-*"] if exclude_ckpts else []
    cmd = [
        "rsync", "-rL", "--no-owner", "--no-group", "--no-perms",
        "--delete",
        *extras,
        "-e", f"ssh -p {args.pod_port} -i {args.pod_key} "
              f"-o StrictHostKeyChecking=no",
        str(src_dir).rstrip("/") + "/",
        f"{args.pod_host}:{pod_dest}/",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"  rsync FAIL rc={res.returncode}: "
              f"{res.stderr.strip()[:300]}", flush=True)
        return False
    return True


def stage_primary(args, domain: str, current: dict[str, str | None]) -> bool:
    if current.get("primary") == domain:
        return True
    src = Path(args.ckpt_base) / domain
    if not src.exists():
        print(f"  [stage] primary {domain}: missing source {src}", flush=True)
        return False
    print(f"  [stage] primary <- {domain}", flush=True)
    ssh_run(args, f"mkdir -p {POD_PRIMARY_DIR}")
    if not rsync_to_pod(args, src, POD_PRIMARY_DIR):
        return False
    current["primary"] = domain
    return True


def stage_helper(args, domain: str, current: dict[str, str | None]) -> bool:
    if domain == "base":
        # Base helper means use HF model name; clear helper dir
        if current.get("helper") != "base":
            ssh_run(args, f"rm -rf {POD_HELPER_DIR}")
            current["helper"] = "base"
        return True
    if current.get("helper") == domain:
        return True
    src = Path(args.ckpt_base) / domain
    if not src.exists():
        print(f"  [stage] helper {domain}: missing source", flush=True)
        return False
    print(f"  [stage] helper <- {domain}", flush=True)
    ssh_run(args, f"mkdir -p {POD_HELPER_DIR}")
    if not rsync_to_pod(args, src, POD_HELPER_DIR):
        return False
    current["helper"] = domain
    return True


def run_cell(args, cell_id: str, mode: str, domain: str,
             primary_path: str, helper_path: str | None) -> dict | None:
    """ssh-run cell_pair_grid_eval.py for a single cell, fetch result."""
    out_local = Path(args.output_dir) / "cells" / f"{cell_id}.json"
    if out_local.exists() and not args.force:
        print(f"  [skip] {cell_id} already done", flush=True)
        return json.loads(out_local.read_text())
    out_local.parent.mkdir(parents=True, exist_ok=True)

    # Build pod command
    cmd_parts = [
        "cd /workspace/halulujah && PYTHONPATH=/workspace/halulujah/src",
        "python src/scripts/cell_pair_grid_eval.py",
        f"--mode {mode}",
        f"--domain {domain}",
        f"--n-questions {args.n_questions}",
        f"--n-rounds {args.n_rounds}",
        f"--cache-dir /workspace/hf_cache",
        f"--primary-path {primary_path}",
        f"--out {POD_OUT}",
    ]
    if helper_path is not None:
        cmd_parts.append(f"--helper-path {helper_path}")
    cmd_parts.append("2>&1")
    pod_cmd = " ".join(cmd_parts)

    print(f"  [run] {cell_id} ...", flush=True)
    t0 = time.time()
    rc, out, err = ssh_run(args, pod_cmd)
    dt = time.time() - t0
    if rc != 0:
        print(f"  [FAIL] {cell_id} rc={rc} dt={dt:.0f}s "
              f"err={err.strip()[:200]} stdout_tail={out.strip()[-300:]}",
              flush=True)
        return None

    # scp result back
    scp = subprocess.run([
        "scp", "-P", args.pod_port, "-i", args.pod_key,
        "-o", "StrictHostKeyChecking=no",
        f"{args.pod_host}:{POD_OUT}", str(out_local),
    ], capture_output=True, text=True)
    if scp.returncode != 0:
        print(f"  [FAIL] scp {cell_id}: {scp.stderr.strip()[:200]}",
              flush=True)
        return None

    data = json.loads(out_local.read_text())
    n = data.get("summary", {}).get("n", 0)
    acc = data.get("summary", {}).get("accuracy", 0.0)
    print(f"  [done] {cell_id} n={n} acc={acc:.3f} dt={dt:.0f}s",
          flush=True)
    return data


def aggregate(out_dir: Path) -> None:
    matrix = {"conditions": {}}
    cells_dir = out_dir / "cells"
    for j in sorted(cells_dir.glob("*.json")):
        try:
            d = json.loads(j.read_text())
            cid = d.get("cell_id") or j.stem
            matrix["conditions"][cid] = d.get("summary", {})
            # Also keep per_q in conditions for downstream telemetry
            matrix["conditions"][cid]["per_q"] = d.get("per_q", [])
        except Exception as e:
            print(f"  [agg] error reading {j}: {e}", flush=True)
    out = out_dir / "matrix_results.json"
    out.write_text(json.dumps(matrix, indent=2))
    print(f"\n[agg] wrote {out} ({len(matrix['conditions'])} conditions)",
          flush=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-base",
                   default="/media/dan/WD_BLACK/halulujah_full_ft_streaming")
    p.add_argument("--domains", nargs="+",
                   default=["medicine", "math", "biology", "law", "physics"])
    p.add_argument("--include-base-helper", action="store_true",
                   help="Also run pair_<d>_base for each domain")
    p.add_argument("--include-base-conditions", action="store_true",
                   help="Run base_solo and base_pair (else relies on LoRA grid reuse)")
    p.add_argument("--n-questions", type=int, default=50)
    p.add_argument("--n-rounds", type=int, default=3)
    p.add_argument("--output-dir", default="results/ft_pair_grid_2026-05-08")
    p.add_argument("--pod-host", default="root@69.30.85.238")
    p.add_argument("--pod-port", default="22192")
    p.add_argument("--pod-key",
                   default=str(Path.home() / ".ssh/runpod_key"))
    p.add_argument("--force", action="store_true")
    p.add_argument("--skip-failed", action="store_true",
                   help="Continue past cell failures rather than abort")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    (out_dir / "cells").mkdir(parents=True, exist_ok=True)

    # Build cell list. Order: solo_<d>, pair_<a>_<b>, pair_<a>_base
    cells = []
    for d in args.domains:
        cells.append(("solo", f"solo_{d}", d, d, None))
    for a in args.domains:
        for b in args.domains:
            cells.append(("pair", f"pair_{a}_{b}", a, a, b))
    if args.include_base_helper:
        for a in args.domains:
            cells.append(("pair", f"pair_{a}_base", a, a, "base"))
    if args.include_base_conditions:
        for d in args.domains:
            cells.append(("base_solo", f"base_solo_{d}", d, "base", None))
            cells.append(("base_pair", f"base_pair_{d}", d, "base", "base"))

    print(f"[plan] {len(cells)} cells; "
          f"{args.n_questions} q × {args.n_rounds} rounds each", flush=True)

    # Setup pod dirs
    ssh_run(args, f"mkdir -p {POD_BASE}")

    current: dict[str, str | None] = {"primary": None, "helper": None}
    for mode, cell_id, domain, primary, helper in cells:
        out_local = Path(args.output_dir) / "cells" / f"{cell_id}.json"
        if out_local.exists() and not args.force:
            print(f"[skip] {cell_id}", flush=True)
            continue
        # Stage primary
        if primary != "base":
            if not stage_primary(args, primary, current):
                if args.skip_failed:
                    continue
                return 1
            primary_path = POD_PRIMARY_DIR
        else:
            primary_path = "Qwen/Qwen3-1.7B"
        # Stage helper
        if helper is None:
            helper_path = None
        elif helper == "base":
            helper_path = "Qwen/Qwen3-1.7B"
        else:
            if not stage_helper(args, helper, current):
                if args.skip_failed:
                    continue
                return 1
            helper_path = POD_HELPER_DIR
        result = run_cell(args, cell_id, mode, domain, primary_path,
                          helper_path)
        if result is None and not args.skip_failed:
            return 2

    # Cleanup pod ckpts
    ssh_run(args, f"rm -rf {POD_BASE}")
    aggregate(out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
