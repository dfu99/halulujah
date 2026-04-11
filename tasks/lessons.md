# Lessons — halulujah

_Hard-won lessons, gotchas, and things that broke before._
_This file is append-mostly. Only remove entries proven wrong._

## General

- README should never expose internal plans, next directions, or full methodology. Keep it to: question, answer, setup, usage, result.
- Store datasets and fine-tuned models on PACE scratch (`~/scratch/`), NOT locally. Not enough local disk space. PACE scratch path: `/storage/home/hcoda1/6/dfu71/scratch/`.
- Blog Authorship Corpus CSV has NUL bytes and fields >131KB. Must set `csv.field_size_limit(sys.maxsize)` and strip `\x00` before parsing.
- PACE repo is at `~/scratch/halulujah/repo/` — always `git pull` before `sbatch`.
- PACE venv does NOT have flash-attn installed. Use `attn_implementation="sdpa"` (PyTorch native) instead of `"flash_attention_2"`.

## Methodology

- KL divergence is a reliable measure of distributional distance between fine-tuned models. Validated on persona adapters: ~29 nats vs base, 7-8 nats between personas. Reusable for domain-specialist comparison.
- Embedding-level separation (~1.05 ratio) is coarse but persistent — survives even high-temperature erosion. Token-level KL is the more sensitive and informative signal.
- Pivot A was the measurement validation; Pivot B (domain collaboration) is the research goal. Don't lose sight of the purpose behind tooling work.

## RunPod Operations

- RunPod shared pods run multiple projects' jobs. Always check `nvidia-smi` AND `ps aux | grep python` before launching GPU work — another project may have started using the GPU since your last check.
- Python logging to nohup files is heavily buffered. Log output may not appear until process exits. Use `ps -p PID -o stat,time` to confirm process is alive.
- RunPod key is at `~/.ssh/runpod_key` (not id_ed25519). Connection: `ssh root@<ip> -p <port> -i ~/.ssh/runpod_key`.
- Base Qwen3-1.7B uses ~4.4GB VRAM in bf16. Two models (for collaboration) need ~9-10GB + KV cache overhead. RTX A4500 (20GB) fits both comfortably when GPU is clear.

## Methodology (continued)

- Base Qwen3-1.7B scores 15.4% on MMLU MCQ — below random chance (25%). This is NOT a capable model without LoRA fine-tuning. The 35pp gap to specialists confirms LoRA is essential, not optional.
- Our collaboration experiment measures MCQ accuracy changes (correct→wrong, wrong→correct), NOT hallucination detection. We use "hallucination" loosely but the data only supports "accuracy erosion under collaboration." To measure actual hallucination would need CoT chain analysis for fabrication content.

## Literature

- Machine unlearning (Harry Potter, TOFU) erases content post-hoc but doesn't train behavioral responses to complexity. Our "trained confusion" framing is distinct.
- Personality measurement papers mostly use prompting, not fine-tuning. Per-human LoRA + distributional measurement is an open lane.
- PERSIST (AAAI 2026) measures personality instability but doesn't isolate temperature as a variable — that's our specific angle for Pivot A.
