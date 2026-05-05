"""Select the Full FT checkpoint that matches the LoRA solo accuracy.

For each FT-trained domain on WD_BLACK, load each
`checkpoint-{1500,3000,4500,6000}` and run the same 50-mixed-MMLU
3-CoT-round solo protocol used by `run_verified_pair_grid.py`. Pick
the checkpoint whose solo accuracy is closest to the verified-LoRA
solo accuracy within a tolerance (default +-2 pp).

This is the matched-solo-accuracy control that lets the LoRA-vs-FT
pair-grid comparison claim equal pre-collab competence between the
two training methods.

Usage (CPU dry-run, prints checkpoint inventory and target deltas):
    python -m src.scripts.select_matched_ft_checkpoint --dry-run

Usage (GPU, picks matched checkpoints):
    python -m src.scripts.select_matched_ft_checkpoint \
        --ft-root /media/dan/WD_BLACK/halulujah_2026-05-04_full_ft_checkpoints \
        --n-questions 50 --n-rounds 3 --tolerance-pp 2.0

Outputs:
    results/specialist_verification/full_ft_matched/manifest.json
        - per-domain matched checkpoint path, FT solo acc, LoRA solo acc,
          delta, within-tolerance flag, all-checkpoint scan results
    results/specialist_verification/full_ft_matched/<domain>_scan.json
        - per-checkpoint solo accuracy curve

Audit context: tasks/audit-2026-05-05.md, follow-up #3.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("matched_ft_select")

DEFAULT_FT_ROOT = "/media/dan/WD_BLACK/halulujah_2026-05-04_full_ft_checkpoints"
LORA_MATRIX = ROOT / "results/verified_pair_grid_qwen3_1p7b/matrix_results.json"

DEFAULT_DOMAINS = ["medicine", "math", "biology", "physics"]


def lora_solo_accuracies() -> dict[str, float]:
    """Read solo_<domain> accuracies from the verified-LoRA pair-grid."""
    m = json.load(open(LORA_MATRIX))
    return {
        d: m["conditions"][f"solo_{d}"]["accuracy"]
        for d in DEFAULT_DOMAINS + ["law"]
    }


def list_checkpoints(ft_root: Path, domain: str) -> list[Path]:
    d = ft_root / domain
    if not d.exists():
        return []
    return sorted(
        [p for p in d.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")],
        key=lambda p: int(p.name.split("-")[-1]),
    )


def evaluate_checkpoint(
    ckpt_path: Path,
    domain: str,
    n_questions: int,
    n_rounds: int,
    cache_dir: str,
    device: str,
    base_tok_name: str,
) -> tuple[float, list[dict]]:
    """Run the pair-grid solo protocol on a single FT checkpoint."""
    # Imports are inside the function so --dry-run works without GPU deps.
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from halulujah.domain.collab_eval import solo_reasoning
    from halulujah.domain.cross_eval import extract_answer_letter

    sys.path.insert(0, str(ROOT / "src/scripts"))
    from run_verified_pair_grid import load_domain_questions  # noqa: E402

    logger.info(f"  loading checkpoint {ckpt_path.name}")
    tok = AutoTokenizer.from_pretrained(
        base_tok_name, trust_remote_code=True, cache_dir=cache_dir,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(ckpt_path),
        dtype=torch.bfloat16,
        trust_remote_code=True,
        cache_dir=cache_dir,
    ).to(device)
    model.eval()

    questions = load_domain_questions(
        domain, n_questions, cache_dir=cache_dir, seed=42,
    )
    per_q = []
    correct = 0
    for i, entry in enumerate(questions):
        final, _ = solo_reasoning(
            model, tok, entry["question"], domain,
            n_rounds=n_rounds, device=device, temperature=0.7,
        )
        pred = extract_answer_letter(final)
        ok = (pred == entry["answer_letter"])
        per_q.append({
            "idx": i, "subject": entry["subject"],
            "expected": entry["answer_letter"],
            "predicted": pred, "correct": ok,
        })
        correct += int(ok)
    acc = correct / max(len(questions), 1)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return acc, per_q


def select_matched(scan: dict[str, float], target: float,
                   tolerance_pp: float) -> tuple[str | None, float]:
    """Return (checkpoint_name, abs_delta_pp) for the best match within tolerance."""
    best_name, best_delta = None, float("inf")
    for name, acc in scan.items():
        delta_pp = abs(acc - target) * 100.0
        if delta_pp < best_delta:
            best_name, best_delta = name, delta_pp
    if best_name is None:
        return None, float("inf")
    if best_delta > tolerance_pp:
        logger.warning(
            f"    no checkpoint within +-{tolerance_pp} pp of target {target:.3f}; "
            f"best is {best_name} at delta {best_delta:.2f} pp"
        )
    return best_name, best_delta


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ft-root", default=DEFAULT_FT_ROOT,
                   help="WD_BLACK path with per-domain Full FT checkpoints")
    p.add_argument("--domains", nargs="+", default=DEFAULT_DOMAINS,
                   help="domains to consider (must have FT checkpoints)")
    p.add_argument("--n-questions", type=int, default=50)
    p.add_argument("--n-rounds", type=int, default=3)
    p.add_argument("--tolerance-pp", type=float, default=2.0,
                   help="solo accuracy tolerance in percentage points")
    p.add_argument("--cache-dir", default="/workspace/hf_cache")
    p.add_argument("--base-tok-name", default="Qwen/Qwen3-1.7B")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dry-run", action="store_true",
                   help="print inventory + targets, do not load checkpoints")
    p.add_argument("--out-dir",
                   default=str(ROOT / "results/specialist_verification/full_ft_matched"))
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ft_root = Path(args.ft_root)
    targets = lora_solo_accuracies()

    inventory: dict[str, dict] = {}
    for d in args.domains:
        ckpts = list_checkpoints(ft_root, d)
        inventory[d] = {
            "checkpoints": [p.name for p in ckpts],
            "lora_solo_target": round(targets.get(d, float("nan")), 4),
        }

    logger.info("=" * 60)
    logger.info("Inventory + LoRA solo targets")
    logger.info("=" * 60)
    for d, info in inventory.items():
        logger.info(f"  {d:10s} target={info['lora_solo_target']:.3f}  "
                    f"checkpoints={info['checkpoints']}")

    if args.dry_run:
        logger.info("\n--dry-run set; not loading any checkpoint.")
        manifest = {
            "mode": "dry-run",
            "ft_root": str(ft_root),
            "tolerance_pp": args.tolerance_pp,
            "n_questions": args.n_questions,
            "n_rounds": args.n_rounds,
            "domains": inventory,
        }
        out_file = out_dir / "manifest_dry_run.json"
        with open(out_file, "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info(f"wrote {out_file}")
        return 0

    if args.device != "cuda":
        logger.error(
            "GPU required for actual evaluation; pass --dry-run for "
            "inventory-only mode."
        )
        return 2

    manifest_entries: dict[str, dict] = {}
    for d in args.domains:
        ckpts = list_checkpoints(ft_root, d)
        if not ckpts:
            logger.warning(f"[{d}] no checkpoints under {ft_root / d}; skipping")
            manifest_entries[d] = {"status": "no_checkpoints"}
            continue
        target = targets.get(d)
        if target is None:
            logger.warning(f"[{d}] no LoRA solo target; skipping")
            manifest_entries[d] = {"status": "no_lora_target"}
            continue

        logger.info(f"\n[{d}] target solo acc = {target:.3f}; scanning {len(ckpts)} ckpts")
        scan: dict[str, float] = {}
        scan_per_q: dict[str, list[dict]] = {}
        for c in ckpts:
            t0 = time.time()
            acc, per_q = evaluate_checkpoint(
                c, d, args.n_questions, args.n_rounds,
                args.cache_dir, args.device, args.base_tok_name,
            )
            dt = time.time() - t0
            logger.info(f"    {c.name}: solo acc {acc:.3f} ({dt/60:.1f} min)")
            scan[c.name] = acc
            scan_per_q[c.name] = per_q

        chosen, delta_pp = select_matched(scan, target, args.tolerance_pp)
        manifest_entries[d] = {
            "lora_solo_target": round(target, 4),
            "scan": {k: round(v, 4) for k, v in scan.items()},
            "chosen_checkpoint": chosen,
            "abs_delta_pp": round(delta_pp, 3),
            "within_tolerance": delta_pp <= args.tolerance_pp,
            "tolerance_pp": args.tolerance_pp,
            "matched_checkpoint_path": str(ft_root / d / chosen) if chosen else None,
        }
        with open(out_dir / f"{d}_scan.json", "w") as f:
            json.dump({"scan": scan, "per_q": scan_per_q}, f, indent=2)
        logger.info(f"  -> chosen: {chosen} (delta {delta_pp:.2f} pp)")

    manifest = {
        "ft_root": str(ft_root),
        "tolerance_pp": args.tolerance_pp,
        "n_questions": args.n_questions,
        "n_rounds": args.n_rounds,
        "domains": manifest_entries,
    }
    out_file = out_dir / "manifest.json"
    with open(out_file, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"\nwrote {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
