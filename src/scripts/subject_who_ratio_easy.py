"""Audit follow-up #38: Easy-subset subject-stratified WHO ratio (§6uu).

Mirrors §6ss on the easy subset (primary solo correct). The cell-mean
metric on easy is C2W rate inverted: pair_acc on easy = (1 − C2W),
so cell-mean delta = pair_acc − solo_acc = pair_acc − 1 = −C2W.

§6ee reported easy WHO 1.50× on primary × helper; §6cc easy primary
effect 24% of variance. This script tests whether subject-stratification
on easy gives a similar (small) ratio, confirming that the WHO-asymmetry
is a hard-only phenomenon at both row factor levels.

Filter to subjects with n_easy >= 4. (n_easy = primary's solo-correct
count restricted to the subject.)

Output: results/verified_pair_grid_qwen3_1p7b/subject_who_ratio_easy.json
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = ROOT / "results/verified_pair_grid_qwen3_1p7b/matrix_results.json"
OUT = ROOT / "results/verified_pair_grid_qwen3_1p7b/subject_who_ratio_easy.json"

DOMAINS = ["math", "medicine", "biology", "law", "physics"]
HELPERS = ["base"] + DOMAINS

N_MIN_EASY = 4


def two_way_anova_decomposition(
    cell_means: np.ndarray,
    cell_ns: np.ndarray | None = None,
) -> dict:
    """Two-way ANOVA SS decomposition on cell means."""
    if cell_ns is None:
        weights = np.ones_like(cell_means)
    else:
        weights = cell_ns.astype(float)
    valid = ~np.isnan(cell_means)
    cm = np.where(valid, cell_means, 0.0)
    w = np.where(valid, weights, 0.0)
    grand = (cm * w).sum() / w.sum() if w.sum() > 0 else 0.0
    row_w = w.sum(axis=1)
    row_mean = np.where(
        row_w > 0,
        (cm * w).sum(axis=1) / np.where(row_w > 0, row_w, 1.0),
        grand,
    )
    col_w = w.sum(axis=0)
    col_mean = np.where(
        col_w > 0,
        (cm * w).sum(axis=0) / np.where(col_w > 0, col_w, 1.0),
        grand,
    )
    pred_additive = row_mean[:, None] + col_mean[None, :] - grand
    interaction = cm - pred_additive
    ss_row = (w * ((row_mean[:, None] - grand) ** 2)).sum()
    ss_col = (w * ((col_mean[None, :] - grand) ** 2)).sum()
    ss_inter = (w * (interaction ** 2)).sum()
    ss_total = ss_row + ss_col + ss_inter
    return {
        "grand_mean": float(grand),
        "row_means": row_mean.tolist(),
        "col_means": col_mean.tolist(),
        "SS_row": float(ss_row),
        "SS_col": float(ss_col),
        "SS_interaction": float(ss_inter),
        "SS_total": float(ss_total),
        "frac_row": float(ss_row / ss_total) if ss_total > 0 else 0.0,
        "frac_col": float(ss_col / ss_total) if ss_total > 0 else 0.0,
        "frac_interaction": float(ss_inter / ss_total) if ss_total > 0 else 0.0,
    }


def main() -> None:
    matrix = json.loads(MATRIX_PATH.read_text())
    cond = matrix["conditions"]

    subject_to_primary: dict[str, str] = {}
    primary_subject_easy_idxs: dict[tuple[str, str], list[int]] = defaultdict(list)

    for p in DOMAINS:
        solo_pq = cond[f"solo_{p}"]["per_q"]
        for q in solo_pq:
            s = q["subject"]
            subject_to_primary[s] = p
            if q["correct"]:
                primary_subject_easy_idxs[(p, s)].append(q["idx"])

    subjects = sorted(
        subject_to_primary.keys(),
        key=lambda s: (DOMAINS.index(subject_to_primary[s]), s),
    )

    n_subj = len(subjects)
    # On easy, cell-mean delta = pair_acc - 1 (since solo on easy is 1
    # by construction); equivalently, -C2W rate
    cell_delta = np.full((n_subj, len(HELPERS)), np.nan)
    cell_n_easy = np.zeros((n_subj, len(HELPERS)), dtype=int)

    for i, s in enumerate(subjects):
        p = subject_to_primary[s]
        idxs = primary_subject_easy_idxs[(p, s)]
        if not idxs:
            continue
        for j, h in enumerate(HELPERS):
            pair_pq = cond[f"pair_{p}_{h}"]["per_q"]
            pair_correct = np.array([pair_pq[idx]["correct"] for idx in idxs])
            cell_delta[i, j] = pair_correct.mean() - 1.0  # solo_acc = 1 on easy
            cell_n_easy[i, j] = len(idxs)

    n_min = N_MIN_EASY
    mask = np.array([cell_n_easy[i, 0] >= n_min for i in range(n_subj)])
    filt_subjects = [s for s, m in zip(subjects, mask) if m]
    filt_idx = [i for i, m in enumerate(mask) if m]
    filt_delta = cell_delta[filt_idx, :]
    filt_n = cell_n_easy[filt_idx, :]

    # Reference primary × helper easy
    primary_helper_delta = np.zeros((len(DOMAINS), len(HELPERS)))
    primary_helper_n = np.zeros((len(DOMAINS), len(HELPERS)), dtype=int)
    for i, p in enumerate(DOMAINS):
        solo_pq = cond[f"solo_{p}"]["per_q"]
        easy_idxs = [q["idx"] for q in solo_pq if q["correct"]]
        if not easy_idxs:
            continue
        for j, h in enumerate(HELPERS):
            pair_pq = cond[f"pair_{p}_{h}"]["per_q"]
            pa = np.mean([pair_pq[idx]["correct"] for idx in easy_idxs])
            primary_helper_delta[i, j] = pa - 1.0
            primary_helper_n[i, j] = len(easy_idxs)

    anova_primary = two_way_anova_decomposition(primary_helper_delta, primary_helper_n)
    anova_full_uniform = two_way_anova_decomposition(cell_delta)
    anova_full_weighted = two_way_anova_decomposition(cell_delta, cell_n_easy)
    anova_filt_uniform = two_way_anova_decomposition(filt_delta)
    anova_filt_weighted = two_way_anova_decomposition(filt_delta, filt_n)

    a = anova_filt_weighted
    f_subject = float(np.sqrt(a["frac_row"] / (1 - a["frac_row"]))) if a["frac_row"] < 1 else float("inf")
    f_helper = float(np.sqrt(a["frac_col"] / (1 - a["frac_col"]))) if a["frac_col"] < 1 else float("inf")
    a_p = anova_primary
    f_primary_easy = float(np.sqrt(a_p["frac_row"] / (1 - a_p["frac_row"]))) if a_p["frac_row"] < 1 else float("inf")
    f_helper_primary_easy = float(np.sqrt(a_p["frac_col"] / (1 - a_p["frac_col"]))) if a_p["frac_col"] < 1 else float("inf")

    subject_row_means_easy = []
    for i, s in enumerate(subjects):
        valid = ~np.isnan(cell_delta[i, :])
        if valid.any():
            subject_row_means_easy.append(
                (s, subject_to_primary[s],
                 float(cell_delta[i, valid].mean()),
                 int(cell_n_easy[i, 0]))
            )

    out = {
        "subjects": subjects,
        "filtered_subjects": filt_subjects,
        "subject_to_primary": subject_to_primary,
        "helpers": HELPERS,
        "n_min_easy": n_min,
        "cell_delta_full": cell_delta.tolist(),
        "cell_n_easy_full": cell_n_easy.tolist(),
        "cell_delta_filtered": filt_delta.tolist(),
        "cell_n_easy_filtered": filt_n.tolist(),
        "anova_full_uniform": anova_full_uniform,
        "anova_full_weighted": anova_full_weighted,
        "anova_filtered_uniform": anova_filt_uniform,
        "anova_filtered_weighted": anova_filt_weighted,
        "anova_primary_helper_easy": anova_primary,
        "cohen_f_subject_filtered_weighted": f_subject,
        "cohen_f_helper_filtered_weighted": f_helper,
        "cohen_f_primary_easy": f_primary_easy,
        "cohen_f_helper_primary_easy": f_helper_primary_easy,
        "subject_row_means_easy": subject_row_means_easy,
        "headline": {
            "subject_helper_ratio_filtered_weighted": (
                a["frac_row"] / a["frac_col"] if a["frac_col"] > 0 else float("inf")
            ),
            "primary_helper_ratio_easy": (
                a_p["frac_row"] / a_p["frac_col"] if a_p["frac_col"] > 0 else float("inf")
            ),
        },
    }

    OUT.write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUT}")
    print()
    print("=" * 80)
    print("§6uu. Easy-subset subject-stratified WHO ratio")
    print("=" * 80)

    a_p = anova_primary
    print(f"\nReference primary × helper easy:")
    print(f"  SS_primary = {a_p['SS_row']:.4f} ({a_p['frac_row']*100:.1f}%)")
    print(f"  SS_helper  = {a_p['SS_col']:.4f} ({a_p['frac_col']*100:.1f}%)")
    print(f"  SS_inter   = {a_p['SS_interaction']:.4f} ({a_p['frac_interaction']*100:.1f}%)")
    print(f"  primary/helper ratio = {out['headline']['primary_helper_ratio_easy']:.2f}×")
    print(f"  Cohen's f primary = {f_primary_easy:.3f}, helper = {f_helper_primary_easy:.3f}")

    print(f"\n§6uu subject × helper easy (full {n_subj}, n-weighted):")
    a = anova_full_weighted
    print(f"  SS_subject = {a['SS_row']:.4f} ({a['frac_row']*100:.1f}%)")
    print(f"  SS_helper  = {a['SS_col']:.4f} ({a['frac_col']*100:.1f}%)")
    print(f"  SS_inter   = {a['SS_interaction']:.4f} ({a['frac_interaction']*100:.1f}%)")

    print(f"\n§6uu subject × helper easy (FILTERED n_easy>={n_min}, n-weighted):")
    a = anova_filt_weighted
    print(f"  SS_subject = {a['SS_row']:.4f} ({a['frac_row']*100:.1f}%)")
    print(f"  SS_helper  = {a['SS_col']:.4f} ({a['frac_col']*100:.1f}%)")
    print(f"  SS_inter   = {a['SS_interaction']:.4f} ({a['frac_interaction']*100:.1f}%)")
    print(f"  subject/helper ratio = {out['headline']['subject_helper_ratio_filtered_weighted']:.2f}×")
    print(f"  Cohen's f subject = {f_subject:.3f}, helper = {f_helper:.3f}")

    print(f"\nFiltered subjects (n_easy >= {n_min}, easy delta = pair_acc - 1):")
    for s, p, m, n in subject_row_means_easy:
        if s in filt_subjects:
            print(f"  {p:10s} {s:32s} n_easy={n:>2d}  delta={m*100:+5.1f} pp  C2W={-m*100:.1f}%")

    print(f"\nExcluded (n_easy < {n_min}):")
    for s, p, m, n in subject_row_means_easy:
        if s not in filt_subjects:
            print(f"  {p:10s} {s:32s} n_easy={n}")

    print()
    print("COMPARISON (all four cells of the WHO ratio family):")
    print(f"  §6m   primary × helper full :  22.1×  (SS_primary 83.3%, SS_helper 3.8%)")
    print(f"  §6rr  subject × helper full :  21.7×  (SS_subject 72.1%, SS_helper 3.3%)")
    print(f"  §6cc  primary × helper hard :  50.3×  (SS_primary 87.3%, SS_helper 1.7%)")
    print(f"  §6ss  subject × helper hard :  56.5×  (SS_subject 80.2%, SS_helper 1.4%)")
    print(f"  ----- primary × helper easy :  {out['headline']['primary_helper_ratio_easy']:.1f}×  "
          f"(SS_primary {anova_primary['frac_row']*100:.1f}%, SS_helper {anova_primary['frac_col']*100:.1f}%)")
    print(f"  §6uu  subject × helper easy :  {out['headline']['subject_helper_ratio_filtered_weighted']:.1f}×  "
          f"(SS_subject {anova_filt_weighted['frac_row']*100:.1f}%, SS_helper {anova_filt_weighted['frac_col']*100:.1f}%)")


if __name__ == "__main__":
    main()
