"""Runs MMLU 5-shot per checkpoint × per domain for forgetting curves.

For each ckpt path (base model + 5 finals + 25 per-step ckpts), run
the per-domain 5-shot MMLU eval. Output: 31 JSONs covering 5 domains
each, plus a combined summary.

Usage on pod:
  python -m src.scripts.run_mmlu_5shot_per_ckpt \\
      --root /workspace/full_ft_evals \\
      --ckpt-base /workspace/halulujah_full_ft_streaming \\
      --n-questions 50

Output: results/full_ft_streaming/mmlu_5shot/
  scan.json  — combined summary (per domain × per ckpt accuracy)
  per_ckpt/{name}.json — full per-question breakdown per ckpt
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DOMAINS = ["medicine", "math", "biology", "law", "physics"]


def find_ckpts(ckpt_base: Path) -> list[tuple[str, Path]]:
    """Yield (name, path) for base model + each domain's final + per-step ckpts."""
    out: list[tuple[str, Path]] = []
    out.append(("base", Path("Qwen/Qwen3-1.7B")))  # passed by name, not path
    for d in DOMAINS:
        domain_root = ckpt_base / d
        if not domain_root.exists():
            print(f"[WARN] {domain_root} missing", file=sys.stderr)
            continue
        # Final adapter (top of dir)
        if (domain_root / "config.json").exists():
            out.append((f"{d}-final", domain_root))
        # Per-step ckpts
        for ckpt in sorted(domain_root.glob("checkpoint-*"),
                           key=lambda p: int(p.name.split("-")[1])):
            step = ckpt.name.split("-")[1]
            out.append((f"{d}-step{step}", ckpt))
    return out


def run_one(ckpt_name: str, ckpt_path: Path, args: argparse.Namespace) -> bool:
    out_path = Path(args.output_dir) / "per_ckpt" / f"{ckpt_name}.json"
    if out_path.exists() and not args.force:
        print(f"[skip] {ckpt_name} already done at {out_path}")
        return True
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(ROOT / "src/scripts/run_mmlu_5shot_baseline.py"),
        "--model-name", str(ckpt_path),
        "--n-questions", str(args.n_questions),
        "--n-shot", str(args.n_shot),
        "--output", str(out_path),
        "--gpu",
    ]
    print(f"[run] {ckpt_name} -> {out_path}")
    res = subprocess.run(cmd, capture_output=False)
    return res.returncode == 0


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
            print(f"[err] reading {j}: {e}")
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote summary {out_path} ({len(summary['per_ckpt'])} ckpts)")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-base",
                   default="/media/dan/WD_BLACK/halulujah_full_ft_streaming",
                   help="Root containing per-domain dirs with finals + ckpts")
    p.add_argument("--output-dir",
                   default="results/full_ft_streaming/mmlu_5shot")
    p.add_argument("--n-questions", type=int, default=50)
    p.add_argument("--n-shot", type=int, default=5)
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing eval JSONs")
    p.add_argument("--start-from", type=int, default=0,
                   help="Skip the first N ckpts (resume)")
    args = p.parse_args()

    ckpts = find_ckpts(Path(args.ckpt_base))
    print(f"Found {len(ckpts)} checkpoints to evaluate")
    for i, (name, _) in enumerate(ckpts):
        marker = "..." if i < args.start_from else ""
        print(f"  [{i}] {name} {marker}")

    n_ok = 0
    for i, (name, path) in enumerate(ckpts):
        if i < args.start_from:
            continue
        if run_one(name, path, args):
            n_ok += 1

    aggregate(args)
    print(f"FINAL_METRICS bench_complete=True n_ok={n_ok}/{len(ckpts) - args.start_from}")
    return 0 if n_ok == len(ckpts) - args.start_from else 1


if __name__ == "__main__":
    sys.exit(main())
