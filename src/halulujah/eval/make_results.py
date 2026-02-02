"""Generate heatmap visualization of exam results."""

import argparse
import glob
import json
import os
import re
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

FILENAME_RE = re.compile(
    r"(?:^|_)t(?P<t>\d+(?:\.\d+)?)_p(?P<p>\d+(?:\.\d+)?)_k(?P<k>\d+)(?:_|\.|$)"
)


def extract_tpk(filename: str) -> Tuple[float, float, int]:
    """Extract (t, p, k) from a filename like 'exam_t2.0_p0.8_k50.json'."""
    m = FILENAME_RE.search(filename)
    if not m:
        raise ValueError(f"Could not parse t/p/k from filename: {filename}")
    return float(m.group("t")), float(m.group("p")), int(m.group("k"))


def score_to_code(score) -> int:
    """Map score string → int code: 0=Correct, 1=Incorrect, 2=Other."""
    if isinstance(score, str):
        s = score.strip().lower()
        if s == "correct":
            return 0
        if s == "incorrect":
            return 1
    return 2


def load_file_scores(path: str, max_questions: int = 100) -> List[int]:
    """Load JSON file and convert score fields to integer codes."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    codes = [score_to_code(item.get("score")) for item in data[:max_questions]]
    if len(codes) < max_questions:
        codes.extend([2] * (max_questions - len(codes)))
    return codes


def make_heatmap(
    folder: str,
    out_path: str = "qa_map.png",
    title: str = "QA Correctness Map",
) -> str:
    """Generate a heatmap PNG from graded exam JSON files.

    Returns the path to the saved image.
    """
    json_paths = sorted(glob.glob(os.path.join(folder, "*.json")))
    if not json_paths:
        raise FileNotFoundError(f"No JSON files found in: {folder}")

    rows, labels, tpk_list = [], [], []
    for jp in json_paths:
        fname = os.path.basename(jp)
        try:
            t, p, k = extract_tpk(fname)
        except ValueError:
            continue
        codes = load_file_scores(jp)
        rows.append(codes)
        labels.append(f"t={t}, p={p}, k={k}")
        tpk_list.append((t, p, k))

    if not rows:
        raise ValueError("No files with recognizable t/p/k pattern found.")

    # Sort by (t, p, k)
    sort_idx = sorted(range(len(tpk_list)), key=lambda i: tpk_list[i])
    rows = [rows[i] for i in sort_idx]
    labels = [labels[i] for i in sort_idx]

    M = np.array(rows, dtype=int)
    cmap = ListedColormap(["navy", "maroon", "black"])
    norm = BoundaryNorm(boundaries=[-0.5, 0.5, 1.5, 2.5], ncolors=cmap.N)

    fig, ax = plt.subplots(figsize=(16, max(4, 0.5 * len(labels))))
    ax.imshow(M, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest", origin="upper")

    ax.set_title(title)
    ax.set_xlabel("Question #")
    ax.set_ylabel("Run (t, p, k)")

    n_q = M.shape[1]
    tick_positions = [0, n_q // 4, n_q // 2, 3 * n_q // 4, n_q - 1]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(i + 1) for i in tick_positions])

    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)

    ax.set_xticks(np.arange(-0.5, n_q, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="lightgray", linewidth=0.25)
    ax.tick_params(axis="both", which="both", length=0)

    legend_handles = [
        Patch(color="navy", label="Correct"),
        Patch(color="maroon", label="Incorrect"),
        Patch(color="black", label="Other"),
    ]
    ax.legend(handles=legend_handles, loc="upper right")

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Plot QA correctness heatmap.")
    ap.add_argument("folder", help="Folder containing graded JSON files.")
    ap.add_argument("--out", default="qa_map.png", help="Output image path.")
    ap.add_argument("--title", default="QA Correctness Map", help="Plot title.")
    args = ap.parse_args()

    path = make_heatmap(args.folder, out_path=args.out, title=args.title)
    print(f"Saved plot to: {path}")


if __name__ == "__main__":
    main()
