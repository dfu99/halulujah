"""Single-cell pair-grid evaluator — runs ONE cell at a time.

Designed to be invoked on the pod by run_ft_pair_grid_streaming.py
(workstation orchestrator) with paths to a primary and (optionally)
helper checkpoint dir.

Mode:
  solo       : primary alone on `domain`
  pair       : primary + helper alternating on `domain` (full-cot)
  base_solo  : base alone on `domain`
  base_pair  : base+base alternating on `domain` (full-cot)

Output JSON:
  {
    "cell_id": "<id>",
    "mode": "...",
    "domain": "...",
    "summary": {accuracy, n, c2w, w2c, ratio, ...},
    "per_q": [<traces>],
    "primary_path": ..., "helper_path": ...
  }
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
logger = logging.getLogger("cell_pair_grid")


def load_model(path: str, cache_dir: str, device: str,
               base_name: str = "Qwen/Qwen3-1.7B"):
    """Load a checkpoint. Three layouts supported:
      - HF base name (path == base_name or None): just load base
      - Full FT directory (has config.json): load from path
      - LoRA adapter directory (has adapter_config.json): load base, apply adapter
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        base_name, trust_remote_code=True, cache_dir=cache_dir,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if path == base_name or path is None:
        model = AutoModelForCausalLM.from_pretrained(
            base_name, torch_dtype=torch.bfloat16,
            trust_remote_code=True, cache_dir=cache_dir,
        ).to(device)
    elif (Path(path) / "adapter_config.json").exists():
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(
            base_name, torch_dtype=torch.bfloat16,
            trust_remote_code=True, cache_dir=cache_dir,
        ).to(device)
        model = PeftModel.from_pretrained(base, path).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16,
            trust_remote_code=True, cache_dir=cache_dir,
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
    p.add_argument("--mode", required=True,
                   choices=["solo", "pair", "base_solo", "base_pair"])
    p.add_argument("--domain", required=True)
    p.add_argument("--primary-path", required=True,
                   help="Path to primary ckpt OR HF model name (Qwen/Qwen3-1.7B)")
    p.add_argument("--helper-path", default=None,
                   help="Path to helper ckpt OR HF model name (Qwen/Qwen3-1.7B); "
                        "ignored for solo modes")
    p.add_argument("--n-questions", type=int, default=50)
    p.add_argument("--n-rounds", type=int, default=3)
    p.add_argument("--cache-dir", default="/root/hf_cache")
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", required=True)
    p.add_argument("--cell-id", default=None,
                   help="Override the auto-derived cell id "
                        "(orchestrator passes this for clarity)")
    p.add_argument("--base-name", default="Qwen/Qwen3-1.7B",
                   help="HF base model name (default Qwen3-1.7B; for 4B "
                        "specialists or 4B LoRA pass Qwen/Qwen3-4B)")
    args = p.parse_args()

    if args.device != "cuda":
        logger.error("GPU required for cell evals")
        return 2

    sys.path.insert(0, str(ROOT / "src/scripts"))
    from run_verified_pair_grid import (  # noqa: E402
        load_domain_questions,
        evaluate_solo,
        evaluate_collab,
        summarize,
    )
    from halulujah.domain.cross_eval import (  # noqa: E402
        assert_letter_extraction_quality,
    )

    logger.info(f"loading question pool for {args.domain} (n={args.n_questions})")
    questions = load_domain_questions(
        args.domain, args.n_questions, args.cache_dir,
    )

    cell_id = args.cell_id or f"{args.mode}_{args.domain}"
    if args.cell_id is None and args.mode == "pair":
        helper_name = (
            "base"
            if args.helper_path == args.base_name
            else Path(args.helper_path).name
        )
        cell_id = f"pair_{args.domain}_{helper_name}"

    t0 = time.time()
    if args.mode in ("solo", "base_solo"):
        model, tok = load_model(args.primary_path, args.cache_dir,
                                args.device, args.base_name)
        per_q = evaluate_solo(model, tok, questions, args.domain,
                              args.n_rounds, args.device)
        summary = summarize(per_q)
        free(model)
    else:  # pair / base_pair
        if args.helper_path is None:
            logger.error("--helper-path required for pair modes")
            return 1
        model_a, tok = load_model(args.primary_path, args.cache_dir,
                                  args.device, args.base_name)
        model_b, _ = load_model(args.helper_path, args.cache_dir,
                                args.device, args.base_name)
        per_q = evaluate_collab(model_a, model_b, tok, questions,
                                args.domain, args.domain,
                                args.n_rounds, args.device)
        # We don't have solo_acc handy here — fill in post-hoc by aggregator
        summary = summarize(per_q)
        free(model_a, model_b)
    elapsed = time.time() - t0

    x_rate = assert_letter_extraction_quality(
        per_q,
        max_x_rate=0.05,
        context=cell_id,
    )

    out = {
        "cell_id": cell_id,
        "mode": args.mode,
        "domain": args.domain,
        "primary_path": args.primary_path,
        "helper_path": args.helper_path,
        "n_rounds": args.n_rounds,
        "n_questions": args.n_questions,
        "elapsed_s": elapsed,
        "summary": summary,
        "x_rate": x_rate,
        "per_q": per_q,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    logger.info(f"wrote {args.out} acc={summary.get('accuracy', 0):.3f} "
                f"n={summary.get('n', 0)} dt={elapsed:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
