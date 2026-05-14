# Cron Check-In Prompt — halulujah

This file is the prompt that `bin/mc-cron-checkin.sh` sends to this
project's tmux session every 6 hours (4x/day, at HH:00–HH:30 staggered by
RunPod priority). **You own this file.** When you finish a cron turn,
edit it to steer what the next tick (~6h later) will ask you to do.

## Current focus (edit me each turn)

A chemistry specialist (Qwen3-1.7B + LoRA r=16 on SciQ) was launched on
the RunPod A40 at 2026-05-14 03:50 UTC via the runpod-idle-pinger cron.
Expected wall time ~2 h training + ~15 min verification. By the time
this prompt fires (next 6h tick), the run should have completed.

Concrete next step:

1. Pull results with `mc runpod fetch halulujah` then inspect
   `results/specialist_verification/chemistry_qwen3/chemistry_r16.json`
   for the `verified` flag and per-benchmark deltas (mmlu
   high_school_chemistry, college_chemistry, mmlu_pro_chem, sciq-test).
2. If `verified == true`: add chemistry to the verified-specialist
   roster in `tasks/planning.md`, log an objective entry with a small
   bar-chart figure (`figures/chemistry_verification_2026-05-14.png`),
   and stage the adapter on WD_BLACK. The 6-primary pair-grid
   (math/medicine/biology/law/physics + chemistry) becomes the next
   paper-side experiment.
3. If `verified == false`: examine `logs/chemistry_qwen3/verify_r16.log`
   for the per-subject base-vs-spec deltas. Two recovery branches:
   (a) re-train at r=8 or r=64 to bracket; (b) note the negative result
   in lessons.md and move on — chemistry-via-SciQ may simply not transfer.

Also pending from yesterday's parser audit: the 1.7B LoRA pair-grid
re-run with the patched `final_raw` schema (so it can be parser-recovered
like 1.7B Full FT was). If the chemistry result is clean and the pod is
still idle, kick that off next — highest paper-impact remaining item.

## Standing rules

- Re-read `CLAUDE.md` and `tasks/planning.md` if you need to ground.
- Per the autonomy rule (`~/.claude/CLAUDE.md` § Autonomy), do not ask
  the PI for direction on reversible in-repo work — pick the higher-EV
  option and execute, defend after.
- Per the evals rule (`~/.claude/CLAUDE.md` § Evals), do not unwire the
  tripwire hook in `.claude/settings.json`.
- If you need RunPod GPU and it is occupied, subscribe via
  `mc runpod subscribe halulujah "<note>"` and continue on a CPU-friendly
  sub-task in the meantime. (RunPod priority: halulujah > FIND-SNP > others.)
- Visualize results before claiming done (rule 6 in global CLAUDE.md).
