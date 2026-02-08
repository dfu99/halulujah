"""End-to-end Temporal Leave-One-Out pipeline orchestration."""

import logging
import os
from pathlib import Path
from typing import List

from ..config import ExperimentConfig
from ..data.loader import load_dataset_by_name, load_egnivia, split_by_year
from ..eval.grade_exam import grade_exam_dir
from ..eval.take_exam import run_exam
from ..finetune.sft_lora import run_finetune
from ..oracle.index_builder import OracleIndex
from ..oracle.retriever import OracleRetriever
from ..oracle.reward import OracleRewardFunction
from ..rl.ppo_trainer import run_rl_training

logger = logging.getLogger(__name__)


def _find_pdfs_for_years(pdf_dir: str, years: List[int]) -> List[str]:
    """Find 10-K PDF files matching the given years."""
    pdf_dir = Path(pdf_dir)
    pdfs = []
    for year in years:
        # Try common naming patterns
        for pattern in [f"*{year}*10-K*.pdf", f"*{year}*.pdf", f"*10-K*{year}*.pdf"]:
            pdfs.extend(str(p) for p in pdf_dir.glob(pattern))
    # Deduplicate preserving order
    seen = set()
    unique = []
    for p in pdfs:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    if not unique:
        # Fall back to all PDFs in the directory
        unique = sorted(str(p) for p in pdf_dir.glob("*.pdf"))
        logger.warning("No year-specific PDFs found for %s, using all %d PDFs", years, len(unique))
    return unique


def run_temporal_leave_one_out(cfg: ExperimentConfig) -> dict:
    """Run the full Temporal Leave-One-Out pipeline for each held-out year.

    Supports both EGNIVIA (PDF-based Oracle) and TempLAMA (history-based
    context) datasets via ``cfg.dataset``.

    For each held-out year:
    1. SFT: Fine-tune with LoRA on all other years
    2. Build Oracle (EGNIVIA only): FAISS index on held-out year's 10-K PDFs
    3. Pre-RL eval: Temperature sweeps on held-out year, grade
    4. RL: PPO training on held-out year questions with Oracle reward
    5. Post-RL eval: Temperature sweeps again
    6. Compare: pre-RL vs post-RL sweet spots

    Returns:
        Dict with results per held-out year.
    """
    use_templama = cfg.dataset == "templama"

    # Load data based on dataset selection
    if use_templama:
        data = load_dataset_by_name("templama", str(cfg.paths.resolve("project_root") / "data"))
    else:
        data = load_egnivia(str(cfg.paths.resolve("data_json")))

    all_results = {}

    for year in cfg.held_out_years:
        logger.info("=" * 60)
        logger.info("Temporal LOO — held-out year: %d (dataset: %s)", year, cfg.dataset)
        logger.info("=" * 60)

        held_out = [year]
        _, held_out_data = split_by_year(data, held_out)

        # --- Step 1: SFT ---
        logger.info("[Step 1] Fine-tuning with held-out year %d", year)
        sft_path = run_finetune(cfg, held_out_years=held_out)

        # --- Step 2: Build Oracle / prepare context ---
        retriever = None
        reward_fn = None

        if use_templama:
            # TempLAMA: context is built from entity history (already in entries).
            # No PDF-based Oracle needed.  For RL reward, use simple_verifier
            # directly against the known answers.
            logger.info("[Step 2] TempLAMA — context from entity history (no Oracle needed)")
        else:
            logger.info("[Step 2] Building Oracle FAISS index for year %d", year)
            pdf_dir = str(cfg.paths.resolve("pdf_dir"))
            pdfs = _find_pdfs_for_years(pdf_dir, held_out)
            oracle_index = OracleIndex(cfg.oracle)
            oracle_index.build_from_pdfs(pdfs)

            oracle_save_dir = os.path.join(
                cfg.paths.resolve("output_dir"), f"oracle_{year}"
            )
            oracle_index.save(oracle_save_dir)

            retriever = OracleRetriever(oracle_index)
            reward_fn = OracleRewardFunction(retriever, cfg.oracle)

        # --- Step 3: Pre-RL evaluation ---
        logger.info("[Step 3] Pre-RL evaluation sweep")
        pre_rl_dir = os.path.join(
            cfg.paths.resolve("output_dir"), f"eval_pre_rl_{year}"
        )
        run_exam(cfg, sft_path, held_out_data, output_dir=pre_rl_dir, retriever=retriever)
        pre_rl_graded = grade_exam_dir(pre_rl_dir)

        # --- Step 4: RL ---
        logger.info("[Step 4] PPO training with Oracle reward")
        if reward_fn is None and use_templama:
            # For TempLAMA, build a lightweight reward function that
            # compares against the known answer(s) directly.
            from ..oracle.reward import TemplamaRewardFunction
            reward_fn = TemplamaRewardFunction()
            reward_fn.set_answer_map(held_out_data)
        rl_path = run_rl_training(cfg, sft_path, held_out, reward_fn, retriever=retriever)

        # --- Step 5: Post-RL evaluation ---
        logger.info("[Step 5] Post-RL evaluation sweep")
        post_rl_dir = os.path.join(
            cfg.paths.resolve("output_dir"), f"eval_post_rl_{year}"
        )
        run_exam(cfg, rl_path, held_out_data, output_dir=post_rl_dir, retriever=retriever)
        post_rl_graded = grade_exam_dir(post_rl_dir)

        # --- Step 6: Summary ---
        all_results[year] = {
            "sft_path": sft_path,
            "rl_path": rl_path,
            "pre_rl_graded_dir": pre_rl_graded,
            "post_rl_graded_dir": post_rl_graded,
        }
        logger.info("Year %d complete. Pre-RL: %s, Post-RL: %s",
                     year, pre_rl_graded, post_rl_graded)

    return all_results
