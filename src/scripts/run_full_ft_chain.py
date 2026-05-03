"""Sequential chain for the remaining 4 Full FT trainings on the A40.

Polls for medicine Full FT completion (config.json at top of output dir),
then trains math -> biology -> law -> physics in sequence.

Each train is invoked as a subprocess so any single failure does not
abort the rest of the chain. Logs go to logs/<domain>_full_ft.log.

Usage:
  setsid bash -c 'nohup python run_full_ft_chain.py > logs/full_ft_chain.log 2>&1 < /dev/null &'
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DOMAINS = [
    ("math", "gsm8k"),
    ("biology", "pubmedqa"),
    ("law", "casehold"),
    ("physics", "sciq"),
]


def wait_for_medicine(adapter_root, poll_seconds=60, max_wait_hours=6):
    """Wait until /workspace/adapters_1p7b_full_ft/medicine/config.json exists."""
    target = Path(adapter_root) / "medicine" / "config.json"
    deadline = time.time() + max_wait_hours * 3600
    while not target.exists():
        if time.time() > deadline:
            print(f"timeout waiting for {target}")
            return False
        print(f"waiting for medicine training to complete -> {target} not found",
              flush=True)
        time.sleep(poll_seconds)
    print(f"medicine done -> {target} exists", flush=True)
    return True


def train_one(domain, source, adapter_root, log_dir, save_steps):
    out = Path(adapter_root) / domain
    # Consider domain "done" if either top-level config.json exists OR a
    # complete checkpoint at the final step is present.
    if (out / "config.json").exists():
        print(f"[{domain}] already done -> {out}/config.json", flush=True)
        return True
    if out.exists():
        ckpts = sorted([d for d in out.iterdir()
                        if d.is_dir() and d.name.startswith("checkpoint-")])
        if ckpts:
            last = ckpts[-1]
            if (last / "config.json").exists() and (
                    last / "model.safetensors").exists():
                print(f"[{domain}] already done -> {last}", flush=True)
                return True
    log = Path(log_dir) / f"{domain}_full_ft.log"
    cmd = [
        sys.executable,
        str(ROOT / "src/scripts/train_specialist_full_ft.py"),
        "--source", source,
        "--domain", domain,
        "--output-dir", str(out),
        "--save-steps", str(save_steps),
    ]
    print(f"[{domain}] training -> {log}", flush=True)
    print(" ".join(cmd), flush=True)
    with open(log, "w") as f:
        ret = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    if ret.returncode != 0:
        print(f"[{domain}] FAILED rc={ret.returncode}; see {log}", flush=True)
        return False
    print(f"[{domain}] done", flush=True)
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter-root", default="/workspace/adapters_1p7b_full_ft")
    p.add_argument("--log-dir", default="/workspace/halulujah/logs")
    p.add_argument("--save-steps", type=int, default=1500)
    p.add_argument("--skip-wait", action="store_true",
                   help="Don't wait for medicine; assume already done")
    args = p.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)

    if not args.skip_wait:
        if not wait_for_medicine(args.adapter_root):
            print("medicine wait timed out; aborting chain")
            return 1

    results = []
    for domain, source in DOMAINS:
        ok = train_one(domain, source, args.adapter_root, args.log_dir,
                       args.save_steps)
        results.append((domain, ok))

    print("\n=== chain summary ===", flush=True)
    for d, ok in results:
        print(f"  {d}: {'OK' if ok else 'FAILED'}", flush=True)
    print(f"FINAL_METRICS chain_complete={all(r[1] for r in results)}")
    return 0 if all(r[1] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
