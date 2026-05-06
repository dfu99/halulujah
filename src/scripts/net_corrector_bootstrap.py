"""Audit follow-up #26: Bootstrap CIs on per-primary net corrector score (§6ii).

§6hh introduced the net corrector score (W2C rate − C2W rate) per
primary, with point estimates: biology +51.7, physics +17.9,
medicine +16.7, math +3.2, law −8.1 pp. This script bootstraps
question-clustered within each primary to give CIs and test:
  - Does law's CI exclude zero (i.e., is law a significant net
    distractor)?
  - Does biology's CI exclude law's CI (do they differ
    significantly)?
  - What is the bootstrap probability that math's score is positive
    (only +3.2 pp at point)?

Method:
  For each primary p, resample n questions with replacement (preserving
  alignment across helpers via the seed=42 idx-based join). For each
  iteration:
    - Recompute mean C2W rate (across 6 helpers) on the resampled set
    - Recompute mean W2C rate (across 6 helpers) on the resampled set
    - Net = W2C_mean − C2W_mean
  Report per-primary 95% CI on net score. Also pairwise comparisons.

Output: results/verified_pair_grid_qwen3_1p7b/net_corrector_bootstrap.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = ROOT / "results/verified_pair_grid_qwen3_1p7b/matrix_results.json"
OUT = ROOT / "results/verified_pair_grid_qwen3_1p7b/net_corrector_bootstrap.json"

DOMAINS = ["math", "medicine", "biology", "law", "physics"]
HELPERS = ["base"] + DOMAINS

N_ITER = 2000
SEED = 2026


def load_data() -> tuple[dict, dict]:
    """Returns (post_correct_per_idx, solo_correct_per_idx).

    post_correct[primary][helper] = length-50 array, 0/1.
    solo_correct[primary] = length-50 array, 0/1.
    """
    matrix = json.loads(MATRIX_PATH.read_text())
    cond = matrix["conditions"]
    post_correct: dict[str, dict[str, np.ndarray]] = {p: {} for p in DOMAINS}
    solo_correct: dict[str, np.ndarray] = {}
    for p in DOMAINS:
        solo_correct[p] = np.array(
            [int(q["correct"]) for q in cond[f"solo_{p}"]["per_q"]],
            dtype=np.int8,
        )
        for h in HELPERS:
            post_correct[p][h] = np.array(
                [int(q["correct"]) for q in cond[f"pair_{p}_{h}"]["per_q"]],
                dtype=np.int8,
            )
    return post_correct, solo_correct


def net_corrector_score(
    post_correct: dict[str, dict[str, np.ndarray]],
    solo_correct: dict[str, np.ndarray],
    primary: str,
    sample_idx: np.ndarray,
) -> tuple[float, float, float]:
    """Returns (mean_c2w_rate, mean_w2c_rate, net_score) for a primary
    on the resampled question set."""
    sc = solo_correct[primary][sample_idx]
    easy_mask = sc == 1
    hard_mask = sc == 0
    n_easy = int(easy_mask.sum())
    n_hard = int(hard_mask.sum())
    if n_easy == 0 or n_hard == 0:
        return float("nan"), float("nan"), float("nan")
    c2w_rates = []
    w2c_rates = []
    for h in HELPERS:
        pc = post_correct[primary][h][sample_idx]
        c2w = ((sc == 1) & (pc == 0)).sum() / n_easy
        w2c = ((sc == 0) & (pc == 1)).sum() / n_hard
        c2w_rates.append(c2w)
        w2c_rates.append(w2c)
    mean_c2w = float(np.mean(c2w_rates))
    mean_w2c = float(np.mean(w2c_rates))
    return mean_c2w, mean_w2c, mean_w2c - mean_c2w


def summarize(arr: list[float]) -> dict:
    a = np.array([x for x in arr if np.isfinite(x)])
    if len(a) == 0:
        return {"n_finite": 0}
    pcts = np.percentile(a, [2.5, 5, 50, 95, 97.5])
    return {
        "n_finite": int(len(a)),
        "mean": float(a.mean()),
        "median": float(pcts[2]),
        "p2.5": float(pcts[0]),
        "p5": float(pcts[1]),
        "p95": float(pcts[3]),
        "p97.5": float(pcts[4]),
    }


def main() -> None:
    post_correct, solo_correct = load_data()
    rng = np.random.default_rng(SEED)

    n_per_primary = {p: len(solo_correct[p]) for p in DOMAINS}
    print(f"n_per_primary: {n_per_primary}")

    # Point estimates
    point = {}
    for p in DOMAINS:
        idx_full = np.arange(n_per_primary[p])
        c2w, w2c, net = net_corrector_score(post_correct, solo_correct, p, idx_full)
        point[p] = {
            "mean_c2w_rate_easy": c2w,
            "mean_w2c_rate_hard": w2c,
            "net_corrector_score": net,
        }
        print(f"  {p:10s}: c2w={c2w*100:5.1f}%  w2c={w2c*100:5.1f}%  net={net*100:+6.1f}pp")

    # Bootstrap
    boot_per_primary: dict[str, dict[str, list[float]]] = {
        p: {"c2w": [], "w2c": [], "net": []} for p in DOMAINS
    }
    boot_pairwise_diffs: dict[str, list[float]] = {}

    for _ in range(N_ITER):
        # Resample independently per primary (each primary's question set is its own)
        per_iter_net: dict[str, float] = {}
        for p in DOMAINS:
            n = n_per_primary[p]
            idx = rng.integers(0, n, size=n)
            c2w, w2c, net = net_corrector_score(post_correct, solo_correct, p, idx)
            boot_per_primary[p]["c2w"].append(c2w)
            boot_per_primary[p]["w2c"].append(w2c)
            boot_per_primary[p]["net"].append(net)
            per_iter_net[p] = net
        # Pairwise differences: biology vs law, biology vs math, math vs law,
        # plus all pairwise
        for i, p1 in enumerate(DOMAINS):
            for p2 in DOMAINS[i + 1:]:
                key = f"{p1}_vs_{p2}"
                boot_pairwise_diffs.setdefault(key, []).append(
                    per_iter_net[p1] - per_iter_net[p2]
                )

    # Summarize
    out = {
        "n_iter": N_ITER,
        "seed": SEED,
        "point_estimates": point,
        "bootstrap_per_primary": {
            p: {
                "c2w": summarize(boot_per_primary[p]["c2w"]),
                "w2c": summarize(boot_per_primary[p]["w2c"]),
                "net": summarize(boot_per_primary[p]["net"]),
            }
            for p in DOMAINS
        },
        "pairwise_diffs": {},
        "tests": {},
    }
    # P(net > 0) per primary — i.e., is each primary a *real* corrector
    for p in DOMAINS:
        nets = np.array(boot_per_primary[p]["net"])
        nets = nets[np.isfinite(nets)]
        p_net_gt_0 = float((nets > 0).mean())
        p_net_lt_0 = float((nets < 0).mean())
        out["tests"][f"P(net_{p} > 0)"] = p_net_gt_0
        out["tests"][f"P(net_{p} < 0)"] = p_net_lt_0

    # Pairwise comparisons
    for key, diffs in boot_pairwise_diffs.items():
        d = np.array([x for x in diffs if np.isfinite(x)])
        if len(d) == 0:
            continue
        out["pairwise_diffs"][key] = {
            "summary": summarize(diffs),
            "p_diff_gt_0": float((d > 0).mean()),
        }

    OUT.write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUT}")
    print()
    print("=" * 78)
    print(f"§6ii. Bootstrap CIs on net corrector score (n_iter = {N_ITER})")
    print("=" * 78)
    print(f"{'primary':10s} {'c2w%':>10s} {'w2c%':>10s} {'net (pt)':>10s}  {'net 95% CI':>22s}  {'P(net>0)':>10s}")
    for p in DOMAINS:
        sn = out["bootstrap_per_primary"][p]["net"]
        pn = point[p]["net_corrector_score"] * 100
        ci = f"[{sn['p2.5']*100:+5.1f}, {sn['p97.5']*100:+5.1f}] pp"
        p_gt = out["tests"][f"P(net_{p} > 0)"]
        print(f"{p:10s} {point[p]['mean_c2w_rate_easy']*100:>9.1f}% {point[p]['mean_w2c_rate_hard']*100:>9.1f}% "
              f"{pn:>+9.1f}pp  {ci:>22s}  {p_gt*100:>9.1f}%")
    print()
    print("Pairwise net-score differences (95% CIs, P(diff > 0)):")
    print(f"{'pair':28s} {'point diff':>12s} {'95% CI':>26s} {'P(diff>0)':>10s}")
    # Show selected key pairs
    key_pairs = ["biology_vs_law", "biology_vs_math", "math_vs_law",
                 "biology_vs_medicine", "physics_vs_law", "physics_vs_math"]
    for kp in key_pairs:
        if kp not in out["pairwise_diffs"]:
            continue
        s = out["pairwise_diffs"][kp]["summary"]
        p_gt = out["pairwise_diffs"][kp]["p_diff_gt_0"]
        p1, p2 = kp.split("_vs_")
        pt_diff = (point[p1]["net_corrector_score"] - point[p2]["net_corrector_score"]) * 100
        ci = f"[{s['p2.5']*100:+6.1f}, {s['p97.5']*100:+6.1f}] pp"
        print(f"{kp:28s} {pt_diff:>+11.1f}pp {ci:>26s}  {p_gt*100:>9.1f}%")


if __name__ == "__main__":
    main()
