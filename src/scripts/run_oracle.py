#!/usr/bin/env python3
"""CLI entry point for building the Oracle FAISS index from PDFs."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, "src")

from halulujah.config import load_config
from halulujah.oracle.index_builder import OracleIndex


def main():
    parser = argparse.ArgumentParser(description="Build Oracle FAISS index from 10-K PDFs")
    parser.add_argument("--config", type=str, default="configs/experiment.yaml")
    parser.add_argument("--pdf-dir", type=str, default=None, help="Override PDF directory")
    parser.add_argument("--save-dir", type=str, default=None, help="Where to save the index")
    args = parser.parse_args()

    cfg = load_config(args.config)
    pdf_dir = args.pdf_dir or str(cfg.paths.resolve("pdf_dir"))
    save_dir = args.save_dir or str(Path(cfg.paths.resolve("output_dir")) / "oracle")

    pdfs = sorted(str(p) for p in Path(pdf_dir).glob("*.pdf"))
    print(f"Found {len(pdfs)} PDFs in {pdf_dir}")

    oracle = OracleIndex(cfg.oracle)
    oracle.build_from_pdfs(pdfs)
    oracle.save(save_dir)
    print(f"Oracle index saved to {save_dir}")


if __name__ == "__main__":
    main()
