#!/usr/bin/env python3
"""CLI entry point for evaluation (exam generation + grading + visualization)."""

import argparse
import sys

sys.path.insert(0, "src")

from halulujah.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Run evaluation: exam, grading, and/or visualization")
    sub = parser.add_subparsers(dest="command", required=True)

    # --- exam ---
    exam_p = sub.add_parser("exam", help="Generate exam responses with sweeps")
    exam_p.add_argument("--config", type=str, default="configs/experiment.yaml")
    exam_p.add_argument("--model-path", type=str, required=True)
    exam_p.add_argument("--output-dir", type=str, default=None)
    exam_p.add_argument("--held-out-year", type=int, nargs="+", default=None)

    # --- grade ---
    grade_p = sub.add_parser("grade", help="Grade exam results with simple_verifier")
    grade_p.add_argument("--input-dir", type=str, required=True)
    grade_p.add_argument("--output-dir", type=str, default=None)

    # --- plot ---
    plot_p = sub.add_parser("plot", help="Generate heatmap visualization")
    plot_p.add_argument("folder", help="Folder with graded JSON files")
    plot_p.add_argument("--out", default="qa_map.png")
    plot_p.add_argument("--title", default="QA Correctness Map")

    args = parser.parse_args()

    if args.command == "exam":
        from halulujah.data.loader import load_egnivia, split_by_year
        from halulujah.eval.take_exam import run_exam

        cfg = load_config(args.config)
        held_out = args.held_out_year or cfg.held_out_years
        data = load_egnivia(str(cfg.paths.resolve("data_json")))
        _, exam_data = split_by_year(data, held_out)
        path = run_exam(cfg, args.model_path, exam_data, output_dir=args.output_dir)
        print(f"Exam results saved to: {path}")

    elif args.command == "grade":
        from halulujah.eval.grade_exam import grade_exam_dir

        path = grade_exam_dir(args.input_dir, output_dir=args.output_dir)
        print(f"Graded results saved to: {path}")

    elif args.command == "plot":
        from halulujah.eval.make_results import make_heatmap

        path = make_heatmap(args.folder, out_path=args.out, title=args.title)
        print(f"Plot saved to: {path}")


if __name__ == "__main__":
    main()
