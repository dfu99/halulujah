#!/usr/bin/env python3
import argparse
import glob
import json
import os
import re
from typing import List, Tuple

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np


FILENAME_RE = re.compile(
    r"(?:^|_)t(?P<t>\d+(?:\.\d+)?)_p(?P<p>\d+(?:\.\d+)?)_k(?P<k>\d+)(?:_|\.|$)"
)

def parse_args():
    ap = argparse.ArgumentParser(description="Plot QA correctness map across JSON files.")
    ap.add_argument("folder", help="Folder containing JSON files.")
    ap.add_argument("--out", default="qa_map.png", help="Output image path (default: qa_map.png).")
    ap.add_argument("--title", default="QA Correctness Map", help="Plot title.")
    return ap.parse_args()

def extract_tpk(filename: str) -> Tuple[float, float, int]:
    """
    Extract (t, p, k) from a filename like 'exam_t2.0_p0.8_k50.json'.
    Returns (t, p, k). Raises ValueError if not found.
    """
    m = FILENAME_RE.search(filename)
    if not m:
        raise ValueError(f"Could not parse t/p/k from filename: {filename}")
    t = float(m.group("t"))
    p = float(m.group("p"))
    k = int(m.group("k"))
    return t, p, k

def score_to_code(score: str) -> int:
    """
    Map score string -> int code for colormap.
    0: Correct (navy), 1: Incorrect (maroon), 2: Other (black)
    """
    if isinstance(score, str):
        s = score.strip().lower()
        if s == "correct":
            return 0
        if s == "incorrect":
            return 1
    return 2

def load_file_scores(path: str) -> List[int]:
    """
    Load JSON file and convert its 100 elements' 'score' fields to integer codes.
    If fewer than 100 entries, pad with 'Other' (black). If more, truncate to 100.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"{os.path.basename(path)} does not contain a top-level list.")

    codes = [score_to_code(item.get("score")) for item in data[:100]]
    if len(codes) < 100:
        codes.extend([2] * (100 - len(codes)))  # pad as 'Other'
    return codes

def main():
    args = parse_args()
    folder = os.path.abspath(args.folder)
    json_paths = sorted(glob.glob(os.path.join(folder, "*.json")))
    if not json_paths:
        raise SystemExit(f"No JSON files found in: {folder}")

    rows = []
    labels = []
    tpk_list = []

    # Load and parse each file
    for jp in json_paths:
        fname = os.path.basename(jp)
        try:
            t, p, k = extract_tpk(fname)
        except ValueError:
            # Skip files that don't match the pattern
            continue

        codes = load_file_scores(jp)  # length 100
        rows.append(codes)
        labels.append(f"t={t}, p={p}, k={k}")
        tpk_list.append((t, p, k, fname))

    if not rows:
        raise SystemExit("No files with recognizable t/p/k pattern were found.")

    # Sort by (t, p, k) for a consistent Y-axis ordering
    # Keep rows/labels in the same order
    sort_idx = sorted(range(len(tpk_list)), key=lambda i: (tpk_list[i][0], tpk_list[i][1], tpk_list[i][2]))
    rows = [rows[i] for i in sort_idx]
    labels = [labels[i] for i in sort_idx]
    tpk_list = [tpk_list[i] for i in sort_idx]

    # Build a (num_files, 100) matrix
    M = np.array(rows, dtype=int)

    # Colormap: 0 -> navy (Correct), 1 -> maroon (Incorrect), 2 -> black (Other)
    cmap = ListedColormap(["navy", "maroon", "black"])
    norm = BoundaryNorm(boundaries=[-0.5, 0.5, 1.5, 2.5], ncolors=cmap.N)

    # Plot
    fig, ax = plt.subplots(figsize=(16, max(4, 0.5 * len(labels))))
    im = ax.imshow(M, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest", origin="upper")

    # Axes and ticks
    ax.set_title(args.title)
    ax.set_xlabel("Question #")
    ax.set_ylabel("Run (t, p, k)")

    # X: questions 1..100 (show a reasonable subset of ticks to avoid clutter)
    ax.set_xticks([0, 24, 49, 74, 99])
    ax.set_xticklabels(["1", "25", "50", "75", "100"])

    # Y: one tick per file
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)

    # Grid lines (optional for readability)
    ax.set_xticks(np.arange(-0.5, 100, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="lightgray", linewidth=0.25)
    ax.tick_params(axis="both", which="both", length=0)

    # Legend proxy
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(color="navy", label="Correct"),
        Patch(color="maroon", label="Incorrect"),
        Patch(color="black", label="Other"),
    ]
    ax.legend(handles=legend_handles, loc="upper right")

    plt.tight_layout()
    fig.savefig(args.out, dpi=200)
    print(f"Saved plot to: {args.out}")

if __name__ == "__main__":
    main()
