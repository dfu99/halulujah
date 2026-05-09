"""Audit §14 + §15 verification: cross-check the Full FT pair-grid
and drift-study headline numbers in tasks/audit-2026-05-05.md
against the source JSONs.

Headlines to check:
  §14b: 5-shot mean accuracies per ckpt (base + 5 finals)
  §14b: gate verdict (PASS/FAIL per specialist)
  §14c: 5×6 cell accuracies (per-row, per-helper means; matrix entries)
  §14d: WHO-asymmetry ratio under Full FT (52.37×) and LoRA (16.93×)
  §14e: c2w / w2c totals (LoRA 82/625, Full FT 135/574)

Outputs a pass/fail line per headline. Run after §14 is updated to
detect any drift between the audit prose and the underlying data.

Source-of-truth files:
  results/full_ft_streaming/mmlu_5shot/scan.json           (§14b)
  results/full_ft_streaming/verification_gate.json         (§14b gate)
  results/ft_pair_grid_2026-05-08/matrix_results_v2.json   (§14c-§14e)
  results/verified_pair_grid_qwen3_1p7b/matrix_results.json (LoRA cmp)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCAN = ROOT / "results/full_ft_streaming/mmlu_5shot/scan.json"
GATE = ROOT / "results/full_ft_streaming/verification_gate.json"
FT_MAT = ROOT / "results/ft_pair_grid_2026-05-08/matrix_results_v2.json"
LORA_MAT = ROOT / "results/verified_pair_grid_qwen3_1p7b/matrix_results.json"

DOMAINS = ["medicine", "math", "biology", "law", "physics"]
HELPERS = ["medicine", "math", "biology", "law", "physics", "base"]


def check_close(name: str, claimed: float, observed: float,
                tol: float = 0.02) -> bool:
    """Tolerance ~2% relative (matches verify_audit_headlines pattern)."""
    if claimed == 0:
        ok = abs(observed) < tol
    else:
        ok = abs(observed - claimed) / abs(claimed) < tol
    flag = "✓" if ok else "✗"
    print(f"  {flag} {name:50s} claimed={claimed:>8.3f}  observed={observed:>8.3f}")
    return ok


def check_int(name: str, claimed: int, observed: int) -> bool:
    ok = claimed == observed
    flag = "✓" if ok else "✗"
    print(f"  {flag} {name:50s} claimed={claimed:>4d}     observed={observed:>4d}")
    return ok


def check_str(name: str, claimed: str, observed: str) -> bool:
    ok = claimed == observed
    flag = "✓" if ok else "✗"
    print(f"  {flag} {name:50s} claimed={claimed!r:>10s}  observed={observed!r:>10s}")
    return ok


def matrix_view(grid: dict) -> np.ndarray:
    out = np.full((len(DOMAINS), len(HELPERS)), np.nan)
    for i, p in enumerate(DOMAINS):
        for j, h in enumerate(HELPERS):
            c = grid.get("conditions", {}).get(f"pair_{p}_{h}", {})
            v = c.get("accuracy")
            if v is not None:
                out[i, j] = v
    return out


def main() -> None:
    results = []

    print("=" * 90)
    print("§14 (Full FT pair-grid extension) headline verification")
    print("=" * 90)

    if not all(p.exists() for p in (SCAN, GATE, FT_MAT)):
        missing = [str(p) for p in (SCAN, GATE, FT_MAT) if not p.exists()]
        print(f"Missing source files: {missing}")
        return

    scan = json.loads(SCAN.read_text())
    gate = json.loads(GATE.read_text())
    ft = json.loads(FT_MAT.read_text())
    lora = json.loads(LORA_MAT.read_text()) if LORA_MAT.exists() else {}

    # §14b: 5-shot ckpt accuracies (claimed in §14b table, row 'mean' column)
    print("\n[§14b: per-ckpt 5-shot mean accuracies]")
    claimed_means = {
        "base": 0.564, "medicine-final": 0.592, "math-final": 0.516,
        "biology-final": 0.556, "law-final": 0.504, "physics-final": 0.580,
    }
    for k, claimed in claimed_means.items():
        observed = scan["per_ckpt"].get(k, {}).get("mean_acc", 0)
        results.append(check_close(f"5-shot mean {k}", claimed, observed))

    # §14b: gate verdicts
    print("\n[§14b: +5pp OOD verification gate]")
    claimed_gate = {"medicine": "pass", "math": "fail", "biology": "pass",
                    "law": "fail", "physics": "pass"}
    for d, claimed in claimed_gate.items():
        observed = gate["per_specialist"][d]["status"]
        results.append(check_str(f"gate verdict {d}", claimed, observed))

    # §14c: per-primary mean (Full FT)
    print("\n[§14c: per-primary mean across helpers (Full FT)]")
    arr = matrix_view(ft)
    primary_means = np.nanmean(arr, axis=1)
    claimed_pm = {"medicine": 0.667, "math": 0.273, "biology": 0.723,
                  "law": 0.437, "physics": 0.507}
    for i, d in enumerate(DOMAINS):
        results.append(check_close(f"primary mean {d}", claimed_pm[d],
                                    primary_means[i]))

    # §14c: per-helper mean (Full FT)
    print("\n[§14c: per-helper mean across primaries (Full FT)]")
    helper_means = np.nanmean(arr, axis=0)
    claimed_hm = {"medicine": 0.528, "math": 0.544, "biology": 0.512,
                  "law": 0.552, "physics": 0.504, "base": 0.488}
    for j, h in enumerate(HELPERS):
        results.append(check_close(f"helper mean {h}", claimed_hm[h],
                                    helper_means[j]))

    # §14d: WHO-asymmetry ratio
    print("\n[§14d: WHO-asymmetry ratio]")
    pv_f = float(np.nanvar(primary_means))
    hv_f = float(np.nanvar(helper_means))
    ratio_f = pv_f / hv_f if hv_f > 0 else float("inf")
    results.append(check_close("Full FT primary_var", 0.02616, pv_f, tol=0.05))
    results.append(check_close("Full FT helper_var", 0.00050, hv_f, tol=0.05))
    results.append(check_close("Full FT ratio", 52.37, ratio_f, tol=0.03))
    if lora:
        lora_arr = matrix_view(lora)
        pv_l = float(np.nanvar(np.nanmean(lora_arr, axis=1)))
        hv_l = float(np.nanvar(np.nanmean(lora_arr, axis=0)))
        ratio_l = pv_l / hv_l if hv_l > 0 else float("inf")
        results.append(check_close("LoRA ratio", 16.93, ratio_l, tol=0.03))

    # §14e: switch-type totals
    print("\n[§14e: switch-type totals across 30 pair cells]")
    def total_switches(grid: dict) -> tuple[int, int]:
        c2w = w2c = 0
        for p in DOMAINS:
            for h in HELPERS:
                c = grid.get("conditions", {}).get(f"pair_{p}_{h}", {})
                c2w += c.get("c2w", 0)
                w2c += c.get("w2c", 0)
        return c2w, w2c
    ft_c, ft_w = total_switches(ft)
    results.append(check_int("Full FT c2w total", 135, ft_c))
    results.append(check_int("Full FT w2c total", 574, ft_w))
    if lora:
        lora_c, lora_w = total_switches(lora)
        results.append(check_int("LoRA c2w total", 82, lora_c))
        results.append(check_int("LoRA w2c total", 625, lora_w))

    # §14d: cluster-respecting bootstrap (run separately via
    # clustered_bootstrap_who_ft.py)
    boot_path = ROOT / "results/ft_pair_grid_2026-05-08/clustered_bootstrap.json"
    if boot_path.exists():
        boot = json.loads(boot_path.read_text())
        print("\n[§14d: clustered bootstrap (delta-cell aggregator)]")
        results.append(check_close("FT spread ratio point",
                                    8.54, boot["point"]["spread_ratio_§6b"],
                                    tol=0.01))
        results.append(check_close("FT variance ratio point",
                                    84.71, boot["point"]["variance_ratio_§6m"],
                                    tol=0.01))
        ci_low = boot["bootstrap_variance_ratio"]["p2.5"]
        ci_high = boot["bootstrap_variance_ratio"]["p97.5"]
        results.append(check_close("FT variance ratio 95% lo",
                                    18.64, ci_low, tol=0.05))
        results.append(check_close("FT variance ratio 95% hi",
                                    259.32, ci_high, tol=0.05))
        p_gt_5 = boot["bootstrap_variance_ratio"]["p_gt_5"]
        results.append(check_close("FT P(variance > 5)",
                                    1.000, p_gt_5, tol=0.001))

    # §15 drift study checks
    drift_path = ROOT / "results/ft_pair_grid_2026-05-08/drift_summary.json"
    if drift_path.exists():
        drift = json.loads(drift_path.read_text())
        print("\n[§15b: drift summary numbers]")
        results.append(check_close("global mean Δ acc",
                                    0.065, drift["global_mean_delta_acc"]))
        results.append(check_close("global mean Δ w2c",
                                    2.73, drift["global_mean_delta_w2c"]))
        results.append(check_close("global mean Δ c2w",
                                    0.20, drift["global_mean_delta_c2w"], tol=0.10))
        results.append(check_close("drift WHO-asym ratio",
                                    0.23, drift["drift_who_asymmetry_ratio"], tol=0.05))
        # Per-helper (base = 0.0 is the headline finding)
        per_h = drift["per_helper_mean_delta_acc"]
        results.append(check_close("Δ acc base helper",
                                    0.000, per_h["base"], tol=0.005))
        results.append(check_close("Δ acc law helper",
                                    0.108, per_h["law"]))
        # Per-primary
        per_p = drift["per_primary_mean_delta_acc"]
        results.append(check_close("Δ acc medicine primary",
                                    0.093, per_p["medicine"]))
        results.append(check_close("Δ acc math primary",
                                    0.063, per_p["math"]))

    # Summary
    print()
    print("=" * 90)
    n_pass = sum(results)
    n_total = len(results)
    print(f"§14 + §15 VERIFICATION SUMMARY: {n_pass} / {n_total} headlines consistent")
    print("=" * 90)
    if n_pass == n_total:
        print("All §14 headline numbers match the source JSONs ✓")
    else:
        print(f"WARNING: {n_total - n_pass} drift(s) between §14 prose and data ✗")


if __name__ == "__main__":
    main()
