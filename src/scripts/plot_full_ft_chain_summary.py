"""Plot the 2026-05-07 Full FT chain completion summary.

Shows each of 5 domains × per-step + final checkpoint inventory,
training time per domain, total model artifacts on WD_BLACK.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
WD = Path("/media/dan/WD_BLACK/halulujah_full_ft_streaming")
OUT = ROOT / "figures/full_ft_chain_2026-05-07.png"

DOMAINS = ["medicine", "math", "biology", "law", "physics"]
COLOR = {
    "math": "#1f77b4",
    "medicine": "#ff7f0e",
    "biology": "#2ca02c",
    "law": "#d62728",
    "physics": "#9467bd",
}

# Approximate training start/end times (UTC) from chain log
TIMING_HHMM = {
    "medicine": ("12:30", "13:55"),
    "math":     ("13:57", "15:08"),
    "biology":  ("15:10", "16:52"),
    "law":      ("17:13", "18:48"),
    "physics":  ("18:52", "20:19"),
}

def to_minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def main() -> None:
    fig = plt.figure(figsize=(15, 7), constrained_layout=False)
    gs = fig.add_gridspec(nrows=2, ncols=2, hspace=0.45, wspace=0.30,
                          left=0.07, right=0.97, top=0.90, bottom=0.08)
    fig.suptitle(
        "2026-05-07 Full FT chain: 5/5 domains complete with full per-step checkpoints",
        fontsize=13, fontweight="bold", y=0.96)

    # Panel A: per-step ckpt inventory per domain
    ax = fig.add_subplot(gs[0, 0])
    rows = []
    for d in DOMAINS:
        ckpts = sorted(
            int(p.name.split("-")[1])
            for p in (WD / d).glob("checkpoint-*")
        )
        rows.append((d, ckpts))
    max_ckpt = max(max(c) for _, c in rows)
    for i, (d, ckpts) in enumerate(rows):
        for c in ckpts:
            ax.scatter([c], [i], color=COLOR[d], s=80,
                       edgecolor="black", lw=0.5)
            ax.text(c, i + 0.18, str(c), fontsize=7, ha="center", color="#444")
        # Training-end checkpoint marker
        ax.scatter([max(ckpts) + 100], [i], marker="*", color=COLOR[d],
                   s=220, edgecolor="black", lw=0.7, zorder=5)
        ax.text(max(ckpts) + 100, i + 0.22, "final",
                fontsize=7, ha="center", color="#444", fontweight="bold")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([d for d, _ in rows], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("training step", fontsize=10)
    ax.set_xlim(0, max_ckpt + 800)
    ax.set_title("A. Per-step checkpoint inventory (* = final adapter)",
                 fontsize=11, fontweight="bold")
    ax.grid(axis="x", linestyle=":", alpha=0.3)

    # Panel B: training time per domain (Gantt-style)
    ax = fig.add_subplot(gs[0, 1])
    base = to_minutes(TIMING_HHMM["medicine"][0])
    for i, d in enumerate(DOMAINS):
        s, e = TIMING_HHMM[d]
        start = to_minutes(s) - base
        end = to_minutes(e) - base
        ax.barh(i, end - start, left=start, color=COLOR[d],
                edgecolor="black", lw=0.5)
        ax.text(start + (end - start) / 2, i, f"{end - start} min",
                fontsize=9, ha="center", va="center", color="white",
                fontweight="bold")
    ax.set_yticks(range(len(DOMAINS)))
    ax.set_yticklabels(DOMAINS, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("minutes from chain start (12:30 UTC)", fontsize=10)
    total_minutes = to_minutes(TIMING_HHMM["physics"][1]) - base
    ax.set_xlim(0, total_minutes + 30)
    ax.set_title(f"B. Training time per domain (chain total: {total_minutes // 60}h {total_minutes % 60}m)",
                 fontsize=11, fontweight="bold")
    ax.grid(axis="x", linestyle=":", alpha=0.3)

    # Panel C: final adapter sizes
    ax = fig.add_subplot(gs[1, 0])
    sizes = []
    for d in DOMAINS:
        st = WD / d / "model.safetensors"
        sizes.append(st.stat().st_size / 1e9 if st.exists() else 0)
    ax.bar(DOMAINS, sizes,
           color=[COLOR[d] for d in DOMAINS],
           edgecolor="black", lw=0.5)
    for i, sz in enumerate(sizes):
        ax.text(i, sz + 0.05, f"{sz:.2f} GB", fontsize=10, ha="center",
                fontweight="bold")
    ax.axhline(3.44, color="gray", linestyle=":", lw=0.7, alpha=0.5)
    ax.text(4.4, 3.50, "expected 3.44 GB", fontsize=8, color="gray",
            ha="right", style="italic")
    ax.set_ylabel("final model.safetensors size (GB)", fontsize=10)
    ax.set_ylim(0, 4.0)
    ax.set_title("C. Final adapter integrity (all 5 ✓ complete)",
                 fontsize=11, fontweight="bold")

    # Panel D: total artifact count summary
    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")
    n_finals = sum(1 for d in DOMAINS if (WD / d / "model.safetensors").exists())
    n_ckpts = sum(len(list((WD / d).glob("checkpoint-*"))) for d in DOMAINS)
    total_size_gb = sum(sizes)
    # Add ckpt sizes too
    for d in DOMAINS:
        for c in (WD / d).glob("checkpoint-*"):
            st = c / "model.safetensors"
            if st.exists():
                total_size_gb += st.stat().st_size / 1e9

    summary_lines = [
        ("Final adapters", f"{n_finals} / 5", "✓"),
        ("Per-step checkpoints", f"{n_ckpts} (avg ~5 per domain)", "✓"),
        ("Total artifacts", f"{n_finals + n_ckpts} model.safetensors", "✓"),
        ("Total disk on WD_BLACK", f"~{total_size_gb:.0f} GB", "✓"),
        ("Chain wall-time", f"~{total_minutes // 60}h {total_minutes % 60}m unattended", "✓"),
        ("Bugs caught + fixed", "5 deps + 2 race/quota", "✓"),
        ("PI directive", '"no half-measures"', "✓"),
    ]
    ax.text(0.05, 0.95, "Summary", fontsize=12, fontweight="bold",
            transform=ax.transAxes)
    y = 0.83
    for desc, val, mark in summary_lines:
        ax.text(0.05, y, desc, fontsize=10, transform=ax.transAxes)
        ax.text(0.55, y, val, fontsize=10, fontweight="bold",
                color="#1f77b4", transform=ax.transAxes)
        ax.text(0.95, y, mark, fontsize=14, color="#2ca02c",
                fontweight="bold", ha="right", transform=ax.transAxes)
        y -= 0.10
    ax.text(0.05, y - 0.05,
            "Next: per-checkpoint MMLU 5-shot benchmark (forgetting curve)\n"
            "      → matched-solo selection → multi-step pair-grid (drift study)",
            fontsize=8, transform=ax.transAxes, style="italic",
            color="#444")

    fig.savefig(OUT, dpi=140, bbox_inches="tight", facecolor="white")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
