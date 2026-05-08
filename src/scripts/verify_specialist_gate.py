"""Verification gate for FT specialists.

Per PI memory feedback_specialist_verification_gate: each specialist
must show >= +5pp absolute accuracy improvement over the base model
on at least ONE out-of-domain MMLU 5-shot benchmark before that
specialist is allowed into a pair-grid run.

Reads results/full_ft_streaming/mmlu_5shot/scan.json
Writes results/full_ft_streaming/verification_gate.json (per-domain
PASS/FAIL with deltas).
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCAN = ROOT / "results/full_ft_streaming/mmlu_5shot/scan.json"
OUT = ROOT / "results/full_ft_streaming/verification_gate.json"

DOMAINS = ["medicine", "math", "biology", "law", "physics"]
THRESHOLD_PP = 5.0  # +5pp on at least 1 OOD


def main() -> None:
    data = json.loads(SCAN.read_text())
    per = data["per_ckpt"]
    base_acc = per["base"]["domains"]
    base_mean = per["base"]["mean_acc"]

    report = {
        "threshold_pp": THRESHOLD_PP,
        "base_accuracies": base_acc,
        "base_mean": base_mean,
        "per_specialist": {},
    }

    print(f"Verification gate: must be >= +{THRESHOLD_PP:.0f}pp over base "
          f"on >= 1 OOD bench")
    print(f"Base mean: {base_mean:.3f}; per-domain: "
          f"{', '.join(f'{d}={base_acc[d]:.2f}' for d in DOMAINS)}")
    print()

    for spec in DOMAINS:
        key = f"{spec}-final"
        if key not in per:
            print(f"  {key:18s}  MISSING from scan")
            report["per_specialist"][spec] = {
                "status": "missing", "passed": False,
            }
            continue
        accs = per[key]["domains"]
        deltas_pp = {d: (accs[d] - base_acc[d]) * 100.0 for d in DOMAINS}
        ood = {d: dp for d, dp in deltas_pp.items() if d != spec}
        max_ood_domain = max(ood, key=lambda d: ood[d])
        max_ood_pp = ood[max_ood_domain]
        passed = max_ood_pp >= THRESHOLD_PP
        in_dom_pp = deltas_pp[spec]
        deltas_str = " ".join(
            f"{d[:3]}={dp:+.1f}" for d, dp in deltas_pp.items()
        )
        status = "PASS" if passed else "FAIL"
        in_dom_marker = "*" if in_dom_pp < 0 else " "
        print(f"  {key:18s}  {status}  in-dom{in_dom_marker}={in_dom_pp:+.1f}pp  "
              f"max-OOD={max_ood_pp:+.1f}pp ({max_ood_domain})  | {deltas_str}")
        report["per_specialist"][spec] = {
            "status": status.lower(),
            "passed": passed,
            "in_domain_delta_pp": in_dom_pp,
            "max_ood_delta_pp": max_ood_pp,
            "max_ood_domain": max_ood_domain,
            "all_deltas_pp": deltas_pp,
        }

    n_pass = sum(1 for v in report["per_specialist"].values()
                 if v.get("passed"))
    n_total = len(report["per_specialist"])
    report["summary"] = {"n_pass": n_pass, "n_total": n_total}
    print(f"\nGate: {n_pass}/{n_total} specialists pass")
    pass_list = [s for s, v in report["per_specialist"].items()
                 if v.get("passed")]
    fail_list = [s for s, v in report["per_specialist"].items()
                 if not v.get("passed")]
    print(f"  PASS: {pass_list}")
    print(f"  FAIL: {fail_list}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
