"""Grade exam results using simple_verifier (no OpenAI API)."""

import json
import logging
import os
from pathlib import Path
from typing import List

from ..rl.verifier import extract_cot_answer, multi_answer_verifier, simple_verifier

logger = logging.getLogger(__name__)


def grade_exam_file(file_path: str) -> List[dict]:
    """Grade a single exam JSON file using simple_verifier.

    Returns the list of entries with 'score' field populated.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    for entry in results:
        expected = entry.get("expected_answer", "")
        predicted = entry.get("model_answer", "")
        predicted = extract_cot_answer(predicted)
        # Use multi-answer verifier when aliases are available (e.g., TempLAMA)
        answers_all = entry.get("expected_answers_all")
        if answers_all:
            score_val = multi_answer_verifier(answers_all, predicted)
        else:
            score_val = simple_verifier(expected, predicted)
        # Map to categorical labels for compatibility with make_results
        if score_val >= 0.6:
            entry["score"] = "Correct"
        else:
            entry["score"] = "Incorrect"
        entry["score_numeric"] = score_val

    return results


def grade_exam_dir(input_dir: str, output_dir: str = None) -> str:
    """Grade all exam JSON files in a directory.

    Args:
        input_dir: Directory containing exam result JSONs.
        output_dir: Where to write graded files. Defaults to input_dir + '_graded'.

    Returns:
        Path to the graded output directory.
    """
    if output_dir is None:
        output_dir = input_dir.rstrip("/") + "_graded"
    os.makedirs(output_dir, exist_ok=True)

    json_files = sorted(Path(input_dir).rglob("*.json"))
    graded_count = 0

    for jf in json_files:
        # Preserve subdirectory structure
        rel = jf.relative_to(input_dir)
        out_path = Path(output_dir) / rel

        if out_path.exists():
            logger.info("Skipping already graded: %s", rel)
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)
        graded = grade_exam_file(str(jf))
        with open(out_path, "w") as f:
            json.dump(graded, f, indent=4)
        graded_count += 1

    logger.info("Graded %d files → %s", graded_count, output_dir)
    return output_dir
