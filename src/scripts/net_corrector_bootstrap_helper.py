"""Audit follow-up #27: Per-helper net corrector score bootstrap (§6jj).

§6ii bootstrapped per primary on the net corrector score. This is the
symmetric per-helper analysis: for each helper h, compute the mean
W2C and C2W rates across the 5 primary rows, and net = W2C − C2W.

Per-helper point estimates (from §6hh):
  base       net = +4.4 pp  (W2C 35.0 − C2W 30.6)
  math       net = +26.1 pp (W2C 40.7 − C2W 14.6)
  medicine   net = +23.5 pp (W2C 41.1 − C2W 17.6)
  biology    net = +12.5 pp (W2C 38.2 − C2W 25.7)
  law        net = +19.8 pp (W2C 39.7 − C2W 19.9)
  physics    net = +11.4 pp (W2C 38.5 − C2W 27.1)

Bootstrap design: question-cluster within each primary independently.
For each primary, resample the 50 questions with replacement; reuse
that single resample across all 6 helpers (preserving alignment).
Then re-aggregate per-helper across the 5 primary rows.

Hypothesis: Helper net-score spread (~22 pp) is real but the per-helper
CIs overlap substantially. Pairwise differences are mostly not
significant. Compare to per-primary spread (~60 pp from §6ii) — primary
spread / helper spread should still be ~3×.

Output: results/verified_pair_grid_qwen3_1p7b/net_corrector_bootstrap_helper.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = ROOT / "results/verified_pair_grid_qwen3_1p7b/matrix_results.json"
PRIMARY_BOOT_PATH = ROOT / "results/verified_pair_grid_qwen3_1p7b/net_corrector_bootstrap.json"
OUT = ROOT / "results/verified_pair_grid_qwen3_1p7b/net_corrector_bootstrap_helper.json"

DOMAINS = ["math", "medicine", "biology", "law", "physics"]
HELPERS = ["base"] + DOMAINS

N_ITER = 2000
SEED = 2026


def load_data() -> tuple[dict, dict]:
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


def helper_net_scores(
    post_correct: dict,
    solo_correct: dict,
    sample_idx_per_primary: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    """For each helper h, compute mean C2W and W2C rates across primaries
    on the resampled question sets. Returns per-helper rate dict."""
    # Per (primary, helper) compute c2w_rate and w2c_rate, then average over
    # primaries per helper.
    cell_c2w = np.zeros((len(DOMAINS), len(HELPERS)))
    cell_w2c = np.zeros((len(DOMAINS), len(HELPERS)))
    for i, p in enumerate(DOMAINS):
        idx = sample_idx_per_primary[p]
        sc = solo_correct[p][idx]
        easy_mask = sc == 1
        hard_mask = sc == 0
        n_easy = int(easy_mask.sum())
        n_hard = int(hard_mask.sum())
        if n_easy == 0 or n_hard == 0:
            cell_c2w[i, :] = float("nan")
            cell_w2c[i, :] = float("nan")
            continue
        for j, h in enumerate(HELPERS):
            pc = post_correct[p][h][idx]
            cell_c2w[i, j] = ((sc == 1) & (pc == 0)).sum() / n_easy
            cell_w2c[i, j] = ((sc == 0) & (pc == 1)).sum() / n_hard

    # Mean across primaries per helper
    per_helper = {}
    for j, h in enumerate(HELPERS):
        c2w_vals = cell_c2w[:, j]
        w2c_vals = cell_w2c[:, j]
        c2w_vals = c2w_vals[np.isfinite(c2w_vals)]
        w2c_vals = w2c_vals[np.isfinite(w2c_vals)]
        if len(c2w_vals) == 0 or len(w2c_vals) == 0:
            per_helper[h] = {"c2w": float("nan"), "w2c": float("nan"), "net": float("nan")}
            continue
        c2w_mean = float(c2w_vals.mean())
        w2c_mean = float(w2c_vals.mean())
        per_helper[h] = {"c2w": c2w_mean, "w2c": w2c_mean, "net": w2c_mean - c2w_mean}
    return per_helper


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

    # Point estimates
    full_idx = {p: np.arange(len(solo_correct[p])) for p in DOMAINS}
    point_per_helper = helper_net_scores(post_correct, solo_correct, full_idx)
    print("Point estimates:")
    for h in HELPERS:
        s = point_per_helper[h]
        print(f"  {h:10s}: c2w={s['c2w']*100:5.1f}%  w2c={s['w2c']*100:5.1f}%  net={s['net']*100:+6.1f}pp")

    # Bootstrap
    boot_per_helper: dict[str, dict[str, list[float]]] = {
        h: {"c2w": [], "w2c": [], "net": []} for h in HELPERS
    }
    boot_pairwise_diffs: dict[str, list[float]] = {}

    for _ in range(N_ITER):
        sample_idx = {
            p: rng.integers(0, len(solo_correct[p]), size=len(solo_correct[p]))
            for p in DOMAINS
        }
        per_h = helper_net_scores(post_correct, solo_correct, sample_idx)
        per_iter_net: dict[str, float] = {}
        for h in HELPERS:
            boot_per_helper[h]["c2w"].append(per_h[h]["c2w"])
            boot_per_helper[h]["w2c"].append(per_h[h]["w2c"])
            boot_per_helper[h]["net"].append(per_h[h]["net"])
            per_iter_net[h] = per_h[h]["net"]
        # Pairwise
        for i, h1 in enumerate(HELPERS):
            for h2 in HELPERS[i + 1:]:
                key = f"{h1}_vs_{h2}"
                diff = per_iter_net[h1] - per_iter_net[h2]
                if np.isfinite(diff):
                    boot_pairwise_diffs.setdefault(key, []).append(diff)

    # Build output
    out = {
        "n_iter": N_ITER,
        "seed": SEED,
        "point_estimates": point_per_helper,
        "bootstrap_per_helper": {
            h: {
                "c2w": summarize(boot_per_helper[h]["c2w"]),
                "w2c": summarize(boot_per_helper[h]["w2c"]),
                "net": summarize(boot_per_helper[h]["net"]),
            }
            for h in HELPERS
        },
        "tests": {},
        "pairwise_diffs": {},
    }
    for h in HELPERS:
        nets = np.array(boot_per_helper[h]["net"])
        nets = nets[np.isfinite(nets)]
        out["tests"][f"P(net_{h} > 0)"] = float((nets > 0).mean())
        out["tests"][f"P(net_{h} < 0)"] = float((nets < 0).mean())

    n_pairs_sig = 0
    for key, diffs in boot_pairwise_diffs.items():
        d = np.array([x for x in diffs if np.isfinite(x)])
        if len(d) == 0:
            continue
        s = summarize(diffs)
        p_gt = float((d > 0).mean())
        is_sig = p_gt >= 0.975 or p_gt <= 0.025
        if is_sig:
            n_pairs_sig += 1
        out["pairwise_diffs"][key] = {
            "summary": s,
            "p_diff_gt_0": p_gt,
            "is_95_sig": is_sig,
        }
    out["n_pairs_95_sig"] = n_pairs_sig
    out["n_pairs_total"] = len(out["pairwise_diffs"])

    # Comparison to per-primary bootstrap
    if PRIMARY_BOOT_PATH.exists():
        primary_boot = json.loads(PRIMARY_BOOT_PATH.read_text())
        primary_nets = [primary_boot["point_estimates"][p]["net_corrector_score"] * 100
                        for p in DOMAINS]
        helper_nets = [point_per_helper[h]["net"] * 100 for h in HELPERS]
        out["comparison_to_primary_axis"] = {
            "primary_net_range_pp": float(max(primary_nets) - min(primary_nets)),
            "helper_net_range_pp": float(max(helper_nets) - min(helper_nets)),
            "primary_spread_to_helper_spread": float(
                (max(primary_nets) - min(primary_nets)) /
                (max(helper_nets) - min(helper_nets))
            ),
        }

    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {OUT}")
    print()
    print("=" * 78)
    print(f"§6jj. Per-helper net corrector bootstrap (n_iter = {N_ITER})")
    print("=" * 78)
    print(f"{'helper':10s} {'c2w%':>10s} {'w2c%':>10s} {'net (pt)':>10s}  {'net 95% CI':>22s}  {'P(net>0)':>10s}")
    for h in HELPERS:
        sn = out["bootstrap_per_helper"][h]["net"]
        pn = point_per_helper[h]["net"] * 100
        ci = f"[{sn['p2.5']*100:+5.1f}, {sn['p97.5']*100:+5.1f}] pp"
        p_gt = out["tests"][f"P(net_{h} > 0)"] * 100
        print(f"{h:10s} {point_per_helper[h]['c2w']*100:>9.1f}% {point_per_helper[h]['w2c']*100:>9.1f}% "
              f"{pn:>+9.1f}pp  {ci:>22s}  {p_gt:>9.1f}%")
    print()
    print(f"Pairwise comparisons: {out['n_pairs_95_sig']} of {out['n_pairs_total']} are 95% sig")
    print()
    print("All 15 pairwise diffs (P(diff>0)):")
    for key, v in out["pairwise_diffs"].items():
        s = v["summary"]
        marker = " ★" if v["is_95_sig"] else ""
        h1, h2 = key.split("_vs_")
        pt_diff = (point_per_helper[h1]["net"] - point_per_helper[h2]["net"]) * 100
        print(f"  {key:24s}  pt={pt_diff:+5.1f}pp  CI [{s['p2.5']*100:+5.1f}, {s['p97.5']*100:+5.1f}]  "
              f"P={v['p_diff_gt_0']*100:5.1f}%{marker}")
    print()
    if "comparison_to_primary_axis" in out:
        c = out["comparison_to_primary_axis"]
        print(f"Comparison to per-primary axis (§6ii):")
        print(f"  primary net range: {c['primary_net_range_pp']:.1f} pp")
        print(f"  helper net range:  {c['helper_net_range_pp']:.1f} pp")
        print(f"  primary / helper spread ratio: {c['primary_spread_to_helper_spread']:.2f}×")


if __name__ == "__main__":
    main()
