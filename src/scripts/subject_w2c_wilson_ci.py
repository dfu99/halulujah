"""Audit follow-up #45: Wilson closed-form CIs on §6tt point W2C rates (§6bbb).

§6tt placed bootstrap 95% CIs around per-subject hard W2C rates.
This script computes the closed-form Wilson score interval on the
same data and compares the two CI methods. If they agree closely,
the §6tt bootstrap is well-behaved. If Wilson is materially tighter,
the bootstrap is conservative; if wider, the bootstrap underestimates
uncertainty.

Wilson interval (Wilson 1927) for a proportion p̂ = k/n with confidence
1-α is:
    p̂ + z²/(2n) ± z·sqrt(p̂(1-p̂)/n + z²/(4n²))
    -------------------------------------------
                  1 + z²/n
where z = z_{1-α/2}.

This is more accurate than the normal-approximation interval at small
n and avoids the boundary issues that bootstrap percentile can have
when point estimates are at 0% or 100%.

Output: results/verified_pair_grid_qwen3_1p7b/subject_w2c_wilson_ci.json
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = ROOT / "results/verified_pair_grid_qwen3_1p7b/matrix_results.json"
BOOT_PATH = ROOT / "results/verified_pair_grid_qwen3_1p7b/subject_w2c_hard_bootstrap.json"
OUT = ROOT / "results/verified_pair_grid_qwen3_1p7b/subject_w2c_wilson_ci.json"

DOMAINS = ["math", "medicine", "biology", "law", "physics"]
HELPERS = ["base"] + DOMAINS

N_MIN_HARD = 4
Z = 1.959963984540054  # 1-α/2 = 0.975 for 95% CI


def wilson_ci(k: int, n: int, z: float = Z) -> tuple[float, float, float]:
    """Wilson score interval for k successes out of n trials.

    Returns (point_estimate, ci_lo, ci_hi).
    """
    if n == 0:
        return float("nan"), 0.0, 1.0
    p_hat = k / n
    denom = 1 + z ** 2 / n
    center = (p_hat + z ** 2 / (2 * n)) / denom
    margin = z * np.sqrt(p_hat * (1 - p_hat) / n + z ** 2 / (4 * n ** 2)) / denom
    return p_hat, max(0.0, center - margin), min(1.0, center + margin)


def main() -> None:
    matrix = json.loads(MATRIX_PATH.read_text())
    cond = matrix["conditions"]
    bootstrap = json.loads(BOOT_PATH.read_text())

    subject_to_primary: dict[str, str] = {}
    primary_subject_hard_idxs: dict[tuple[str, str], list[int]] = defaultdict(list)
    for p in DOMAINS:
        solo_pq = cond[f"solo_{p}"]["per_q"]
        for q in solo_pq:
            s = q["subject"]
            subject_to_primary[s] = p
            if not q["correct"]:
                primary_subject_hard_idxs[(p, s)].append(q["idx"])

    # Per-subject: count successes (pair_correct=1) over (n_hard × 6 helpers)
    # trials and compute Wilson CI
    out = {"subjects": {}, "comparison": []}

    for (p, s), idxs in primary_subject_hard_idxs.items():
        if len(idxs) < N_MIN_HARD:
            continue
        n_trials = len(idxs) * len(HELPERS)
        k_success = 0
        for h in HELPERS:
            pair_pq = cond[f"pair_{p}_{h}"]["per_q"]
            for idx in idxs:
                if pair_pq[idx]["correct"]:
                    k_success += 1
        p_hat, lo, hi = wilson_ci(k_success, n_trials)
        boot_b = bootstrap["subject_summary"].get(s, {})
        out["subjects"][s] = {
            "primary": p,
            "n_hard": len(idxs),
            "n_trials": n_trials,
            "k_success": k_success,
            "wilson_point": p_hat,
            "wilson_ci_lo": lo,
            "wilson_ci_hi": hi,
            "wilson_ci_width": hi - lo,
            "boot_point": boot_b.get("point_w2c", float("nan")),
            "boot_ci_lo": boot_b.get("ci_lo", float("nan")),
            "boot_ci_hi": boot_b.get("ci_hi", float("nan")),
            "boot_ci_width": boot_b.get("ci_hi", 0) - boot_b.get("ci_lo", 0),
        }

    subjects_sorted = sorted(
        out["subjects"].keys(),
        key=lambda s: out["subjects"][s]["wilson_point"],
        reverse=True,
    )

    print(f"Wrote {OUT}")
    print()
    print("=" * 100)
    print("§6bbb. Wilson closed-form CI vs bootstrap CI for per-subject hard W2C")
    print("=" * 100)
    print()
    print(f"{'Subject':32s} {'primary':10s} {'n_tri':>5s} {'k':>4s} {'point':>7s} "
          f"{'Wilson 95% CI':>22s} {'Wilson w':>9s}  {'boot 95% CI':>22s} {'boot w':>8s}  {'Δw':>7s}")
    for s in subjects_sorted:
        b = out["subjects"][s]
        wilson_ci_str = f"[{b['wilson_ci_lo']*100:.1f}, {b['wilson_ci_hi']*100:.1f}]"
        boot_ci_str = f"[{b['boot_ci_lo']*100:.1f}, {b['boot_ci_hi']*100:.1f}]"
        delta_w = (b['wilson_ci_width'] - b['boot_ci_width']) * 100
        print(f"  {s:30s} {b['primary']:10s} {b['n_trials']:>5d} {b['k_success']:>4d} "
              f"{b['wilson_point']*100:>6.1f}% {wilson_ci_str:>22s} {b['wilson_ci_width']*100:>8.1f}%  "
              f"{boot_ci_str:>22s} {b['boot_ci_width']*100:>7.1f}%  {delta_w:>+6.1f}")

    # Aggregate comparison
    n_subjects = len(out["subjects"])
    n_wilson_tighter = sum(
        1 for s in out["subjects"]
        if out["subjects"][s]["wilson_ci_width"] < out["subjects"][s]["boot_ci_width"]
    )
    n_wilson_wider = sum(
        1 for s in out["subjects"]
        if out["subjects"][s]["wilson_ci_width"] > out["subjects"][s]["boot_ci_width"]
    )
    mean_delta_w = np.mean([
        out["subjects"][s]["wilson_ci_width"] - out["subjects"][s]["boot_ci_width"]
        for s in out["subjects"]
    ])
    out["comparison"] = {
        "n_subjects": n_subjects,
        "n_wilson_tighter_than_bootstrap": int(n_wilson_tighter),
        "n_wilson_wider_than_bootstrap": int(n_wilson_wider),
        "mean_delta_width_wilson_minus_boot": float(mean_delta_w),
    }

    OUT.write_text(json.dumps(out, indent=2))
    print()
    print(f"Aggregate: of {n_subjects} subjects,")
    print(f"  Wilson CI tighter than bootstrap: {n_wilson_tighter}")
    print(f"  Wilson CI wider than bootstrap:   {n_wilson_wider}")
    print(f"  Mean Δ width (Wilson − boot):     {mean_delta_w*100:+.2f} pp")


if __name__ == "__main__":
    main()
