"""Audit follow-up #23: effect sizes (Cohen's f, ω²) on the hard-only ANOVA.

§6w computed effect sizes on the full §6r ANOVA: f_primary = 0.27 (medium),
f_helper = 0.06 (trivial), f_interaction = 0.11 (small).
§6dd reran the ANOVA on the hard-only subset: F_primary jumped from 26.55
to 31.22; F_helper dropped from 0.96 to 0.50.

This script standardizes the §6dd F-stats into Cohen's f and ω² for direct
comparability with §6w.

Output: results/verified_pair_grid_qwen3_1p7b/effect_sizes_hard_only.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HARD_ANOVA_PATH = ROOT / "results/verified_pair_grid_qwen3_1p7b/anova_replicates_hard_only.json"
FULL_ES_PATH = ROOT / "results/verified_pair_grid_qwen3_1p7b/effect_sizes.json"
OUT = ROOT / "results/verified_pair_grid_qwen3_1p7b/effect_sizes_hard_only.json"


def cohen_f_label(f: float) -> str:
    if f < 0.10:
        return "trivial"
    if f < 0.25:
        return "small"
    if f < 0.40:
        return "medium"
    return "large"


def main() -> None:
    av = json.loads(HARD_ANOVA_PATH.read_text())
    full_es = json.loads(FULL_ES_PATH.read_text())

    ss = av["ss"]
    df = av["df"]
    F = av["F"]
    p = av["p"]

    ss_total = ss["total"]
    ss_within = ss["within"]
    df_within = df["within"]
    ms_within = ss_within / df_within if df_within else 0.0

    out = {"sources": {}}
    for src in ("primary_A", "helper_B", "interaction_AB"):
        ss_src = ss[src]
        df_src = df[src]
        F_src = F[src]
        p_src = p[src]
        eta_sq = ss_src / ss_total if ss_total else 0.0
        eta_sq_partial = ss_src / (ss_src + ss_within) if (ss_src + ss_within) else 0.0
        omega_sq = (ss_src - df_src * ms_within) / (ss_total + ms_within) if (ss_total + ms_within) else 0.0
        cohens_f = float(np.sqrt(eta_sq_partial / (1 - eta_sq_partial))) if eta_sq_partial < 1 else float("inf")

        # Comparison to full-grid §6w
        full_src = full_es["sources"][src]

        out["sources"][src] = {
            "F": F_src,
            "p": p_src,
            "df_num": df_src,
            "df_den": df_within,
            "ss": ss_src,
            "ss_frac_of_total": float(eta_sq),
            "eta_squared": float(eta_sq),
            "eta_squared_partial": float(eta_sq_partial),
            "omega_squared": float(max(0.0, omega_sq)),
            "cohens_f": float(cohens_f),
            "cohens_f_label": cohen_f_label(cohens_f),
            "full_grid_baseline": {
                "F": full_src["F"],
                "cohens_f": full_src["cohens_f"],
                "cohens_f_label": full_src["cohens_f_label"],
                "omega_squared": full_src["omega_squared"],
            },
            "delta_vs_full": {
                "F_ratio": float(F_src / full_src["F"]) if full_src["F"] else float("inf"),
                "cohens_f_delta": float(cohens_f - full_src["cohens_f"]),
                "omega_squared_delta": float(max(0.0, omega_sq) - full_src["omega_squared"]),
                "cohens_f_label_change": (
                    cohen_f_label(cohens_f) if cohen_f_label(cohens_f) != full_src["cohens_f_label"]
                    else "unchanged"
                ),
            },
        }

    OUT.write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUT}")
    print()
    print("=" * 72)
    print("§6ff. Hard-only effect sizes (Cohen's f, ω²) — comparison to §6w full grid")
    print("=" * 72)
    print(f"{'source':16s}  {'F':>8s} {'f_full':>8s} {'f_hard':>8s} {'Δf':>7s} {'ω²_full':>9s} {'ω²_hard':>9s} {'label':>16s}")
    for src, label in [
        ("primary_A", "Primary"),
        ("helper_B", "Helper"),
        ("interaction_AB", "Interaction"),
    ]:
        s = out["sources"][src]
        f_h = s["cohens_f"]
        f_f = s["full_grid_baseline"]["cohens_f"]
        ow_h = s["omega_squared"]
        ow_f = s["full_grid_baseline"]["omega_squared"]
        change = s["delta_vs_full"]["cohens_f_label_change"]
        if change == "unchanged":
            change_str = f"{s['cohens_f_label']:s}"
        else:
            change_str = f"{s['full_grid_baseline']['cohens_f_label']}→{s['cohens_f_label']}"
        print(f"{label:16s}  {s['F']:>8.2f} {f_f:>8.3f} {f_h:>8.3f} {(f_h - f_f):>+7.3f} "
              f"{ow_f:>9.3f} {ow_h:>9.3f} {change_str:>16s}")


if __name__ == "__main__":
    main()
