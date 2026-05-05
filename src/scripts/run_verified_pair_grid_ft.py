"""5x5 Full FT pair-grid collaboration on verified Qwen3-1.7B specialists.

Mirror of run_verified_pair_grid.py but loads each domain's matched
Full FT checkpoint (from select_matched_ft_checkpoint.py manifest)
as the *entire model* instead of layering a LoRA adapter onto the
shared Qwen3-1.7B base.

Loads each FT checkpoint and runs all 45 conditions:
  - solo (5)         : FT specialist alone on its domain
  - base_solo (5)    : base alone on each domain   (reuses LoRA grid value)
  - same_pair (5)    : FT specialist deliberating with itself
  - cross_pair (20)  : FT_a + FT_b deliberating on a's domain
  - mixed_pair (5)   : FT_specialist + base deliberating
  - base_pair (5)    : two bases deliberating       (reuses LoRA grid value)

For each condition: N=50 questions, 3 deliberation rounds, full-cot
(reasoning-preserved). Outputs:
  results/verified_pair_grid_qwen3_1p7b_full_ft/matrix_results.json

Usage:
  python -m src.scripts.run_verified_pair_grid_ft \
      --manifest results/specialist_verification/full_ft_matched/manifest.json \
      --n-questions 50 --n-rounds 3

Audit context: tasks/audit-2026-05-05.md follow-up #5.
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
logger = logging.getLogger("ft_pair_grid")


def load_ft_model(ckpt_path: str | None, cache_dir: str, device: str,
                  base_name: str = "Qwen/Qwen3-1.7B"):
    """Load either an FT checkpoint as a full model or the base.

    If `ckpt_path` is None, returns the un-modified Qwen3-1.7B base.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        base_name, trust_remote_code=True, cache_dir=cache_dir,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if ckpt_path is None:
        model = AutoModelForCausalLM.from_pretrained(
            base_name,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            cache_dir=cache_dir,
        ).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            ckpt_path,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            cache_dir=cache_dir,
        ).to(device)
    model.eval()
    return model, tok


def free(*models):
    for m in models:
        del m
    gc.collect()
    torch.cuda.empty_cache()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True,
                   help="JSON from select_matched_ft_checkpoint.py "
                        "(per-domain matched_checkpoint_path)")
    p.add_argument("--n-questions", type=int, default=50)
    p.add_argument("--n-rounds", type=int, default=3)
    p.add_argument("--base", default="Qwen/Qwen3-1.7B")
    p.add_argument("--cache-dir", default="/workspace/hf_cache")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir",
                   default=str(ROOT / "results/verified_pair_grid_qwen3_1p7b_full_ft"))
    p.add_argument(
        "--lora-matrix",
        default=str(ROOT / "results/verified_pair_grid_qwen3_1p7b/matrix_results.json"),
        help="Optional: re-use base_solo and base_pair conditions from "
             "the LoRA grid since they don't depend on the FT specialist.",
    )
    p.add_argument("--reuse-base-conditions", action="store_true",
                   help="Skip base_solo and base_pair runs and copy from --lora-matrix")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "matrix_results.json"

    manifest = json.load(open(args.manifest))
    ft_paths: dict[str, str | None] = {}
    for d, info in manifest.get("domains", {}).items():
        ft_paths[d] = info.get("matched_checkpoint_path")

    valid_domains = [d for d, p_ in ft_paths.items() if p_]
    skipped = [d for d in ft_paths if d not in valid_domains]
    logger.info(f"FT-trained domains: {valid_domains}")
    if skipped:
        logger.warning(f"skipped (no matched checkpoint): {skipped}")

    if args.device != "cuda":
        logger.error("GPU required.")
        return 2

    # Defer the heavy imports until after argparse.
    sys.path.insert(0, str(ROOT / "src/scripts"))
    from run_verified_pair_grid import (  # noqa: E402
        load_domain_questions,
        evaluate_solo,
        evaluate_collab,
        summarize,
    )

    # Load the question pool once per domain (deterministic seed).
    test = {
        d: load_domain_questions(d, args.n_questions, args.cache_dir)
        for d in valid_domains
    }

    # Reuse base_solo / base_pair from the LoRA grid if available.
    reused: dict[str, dict] = {}
    if args.reuse_base_conditions and Path(args.lora_matrix).exists():
        lora = json.load(open(args.lora_matrix))
        for d in valid_domains:
            for prefix in ("base_solo_", "base_pair_"):
                k = f"{prefix}{d}"
                if k in lora["conditions"]:
                    reused[k] = lora["conditions"][k]
                    logger.info(f"reusing {k} from LoRA matrix")

    # Resume support: if the output already exists, pick up the partials.
    conditions: dict[str, dict] = {}
    if out_file.exists():
        prior = json.load(open(out_file))
        conditions = prior.get("conditions", {})
        logger.info(f"resuming from {len(conditions)} existing conditions")

    # Apply reused values if not already present.
    for k, v in reused.items():
        conditions.setdefault(k, v)

    config = {
        "base": args.base,
        "matched_ft": ft_paths,
        "domains": valid_domains,
        "n_questions": args.n_questions,
        "n_rounds": args.n_rounds,
        "training": "full_ft",
    }

    def write():
        with open(out_file, "w") as f:
            json.dump({"conditions": conditions, "config": config}, f, indent=2)

    # SOLO conditions: FT specialist alone on its domain
    solo_acc: dict[str, float] = {}
    for d in valid_domains:
        cid = f"solo_{d}"
        if cid in conditions:
            solo_acc[d] = conditions[cid]["accuracy"]
            logger.info(f"  [{cid}] cached acc={solo_acc[d]:.3f}")
            continue
        logger.info(f"loading FT specialist {d} for solo")
        model, tok = load_ft_model(ft_paths[d], args.cache_dir, args.device, args.base)
        t0 = time.time()
        per_q = evaluate_solo(model, tok, test[d], d, args.n_rounds, args.device)
        s = summarize(per_q)
        s["elapsed_s"] = time.time() - t0
        s["per_q"] = per_q
        conditions[cid] = s
        solo_acc[d] = s["accuracy"]
        write()
        free(model)

    # SAME-PAIR: FT_d + FT_d on d
    for d in valid_domains:
        cid = f"pair_{d}_{d}"
        if cid in conditions:
            continue
        logger.info(f"loading FT specialist {d} (twice) for same_pair")
        model_a, tok = load_ft_model(ft_paths[d], args.cache_dir, args.device, args.base)
        model_b, _ = load_ft_model(ft_paths[d], args.cache_dir, args.device, args.base)
        t0 = time.time()
        per_q = evaluate_collab(model_a, model_b, tok, test[d], d, d,
                                args.n_rounds, args.device)
        s = summarize(per_q, solo_acc=solo_acc.get(d))
        s["elapsed_s"] = time.time() - t0
        s["per_q"] = per_q
        conditions[cid] = s
        write()
        free(model_a, model_b)

    # CROSS-PAIR: FT_a primary + FT_b helper, on a's domain
    for primary in valid_domains:
        for helper in valid_domains:
            if primary == helper:
                continue
            cid = f"pair_{primary}_{helper}"
            if cid in conditions:
                continue
            logger.info(f"loading FT_{primary} + FT_{helper} for cross_pair")
            model_a, tok = load_ft_model(ft_paths[primary], args.cache_dir,
                                         args.device, args.base)
            model_b, _ = load_ft_model(ft_paths[helper], args.cache_dir,
                                       args.device, args.base)
            t0 = time.time()
            per_q = evaluate_collab(model_a, model_b, tok, test[primary],
                                    primary, helper, args.n_rounds, args.device)
            s = summarize(per_q, solo_acc=solo_acc.get(primary))
            s["elapsed_s"] = time.time() - t0
            s["per_q"] = per_q
            conditions[cid] = s
            write()
            free(model_a, model_b)

    # MIXED-PAIR: FT_d primary + base helper
    for d in valid_domains:
        cid = f"pair_{d}_base"
        if cid in conditions:
            continue
        logger.info(f"loading FT_{d} + base for mixed_pair")
        model_a, tok = load_ft_model(ft_paths[d], args.cache_dir, args.device, args.base)
        model_b, _ = load_ft_model(None, args.cache_dir, args.device, args.base)
        t0 = time.time()
        per_q = evaluate_collab(model_a, model_b, tok, test[d], d, "base",
                                args.n_rounds, args.device)
        s = summarize(per_q, solo_acc=solo_acc.get(d))
        s["elapsed_s"] = time.time() - t0
        s["per_q"] = per_q
        conditions[cid] = s
        write()
        free(model_a, model_b)

    # base_solo and base_pair conditions are skipped if --reuse-base-conditions
    if not args.reuse_base_conditions:
        # base_solo
        for d in valid_domains:
            cid = f"base_solo_{d}"
            if cid in conditions:
                continue
            logger.info(f"loading base for base_solo {d}")
            model, tok = load_ft_model(None, args.cache_dir, args.device, args.base)
            t0 = time.time()
            per_q = evaluate_solo(model, tok, test[d], d, args.n_rounds, args.device)
            s = summarize(per_q)
            s["elapsed_s"] = time.time() - t0
            s["per_q"] = per_q
            conditions[cid] = s
            write()
            free(model)
        # base_pair
        for d in valid_domains:
            cid = f"base_pair_{d}"
            if cid in conditions:
                continue
            logger.info(f"loading base+base for base_pair {d}")
            model_a, tok = load_ft_model(None, args.cache_dir, args.device, args.base)
            model_b, _ = load_ft_model(None, args.cache_dir, args.device, args.base)
            t0 = time.time()
            per_q = evaluate_collab(model_a, model_b, tok, test[d], d, "base",
                                    args.n_rounds, args.device)
            s = summarize(per_q, solo_acc=conditions.get(f"base_solo_{d}", {}).get("accuracy"))
            s["elapsed_s"] = time.time() - t0
            s["per_q"] = per_q
            conditions[cid] = s
            write()
            free(model_a, model_b)

    logger.info(f"\nwrote {out_file} ({len(conditions)} conditions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
