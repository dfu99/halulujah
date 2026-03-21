# Lessons — halulujah

_Hard-won lessons, gotchas, and things that broke before._
_This file is append-mostly. Only remove entries proven wrong._

## General

- README should never expose internal plans, next directions, or full methodology. Keep it to: question, answer, setup, usage, result.
- Store datasets and fine-tuned models on PACE scratch (`~/scratch/`), NOT locally. Not enough local disk space. PACE scratch path: `/storage/home/hcoda1/6/dfu71/scratch/`.
- Blog Authorship Corpus CSV has NUL bytes and fields >131KB. Must set `csv.field_size_limit(sys.maxsize)` and strip `\x00` before parsing.
- PACE repo is at `~/scratch/halulujah/repo/` — always `git pull` before `sbatch`.

## Literature

- Machine unlearning (Harry Potter, TOFU) erases content post-hoc but doesn't train behavioral responses to complexity. Our "trained confusion" framing is distinct.
- Personality measurement papers mostly use prompting, not fine-tuning. Per-human LoRA + distributional measurement is an open lane.
- PERSIST (AAAI 2026) measures personality instability but doesn't isolate temperature as a variable — that's our specific angle for Pivot A.
