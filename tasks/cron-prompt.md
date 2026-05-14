# Cron Check-In Prompt — halulujah

This file is the prompt that `bin/mc-cron-checkin.sh` sends to this
project's tmux session every 6 hours (4x/day, at HH:00–HH:30 staggered by
RunPod priority). **You own this file.** When you finish a cron turn,
edit it to steer what the next tick (~6h later) will ask you to do.

## Current focus (edit me each turn)

**Pod stopped by PI 2026-05-14 ~02:40 UTC** ("not getting my money's
worth out of it"). The runpod-idle-pinger cron has been disabled in
the user crontab. Do not attempt `mc runpod` calls — they will fail.

What landed before stop:
- Chemistry specialist verified (commit `5a604ea`, gate PASS via
  MMLU-Pro chemistry +14.6 pp).
- Three pod compat bugs codified into eval tests:
  `tests/test_peft_grad_ckpt_compat.py` + `tests/test_from_pretrained_torch_dtype.py`.
- Pre-stop backup at `/media/dan/WD_BLACK/halulujah_2026-05-14_pre_pod_stop/`
  (262 MB; chemistry adapter, 4B LoRA r=8 law, logs, README).

CPU-only next steps for cron-tick work:
- *Paper-side.* The bootstrap CI on the recovered 1.7B FT WHO ratio
  is now committed (`d5c521b`, point 54.74× / 95% CI [12.4, 159.2])
  with the FT-vs-LoRA bootstrap-overlap caveat. Propagate that into
  `paper/claim_evidence_map.md` C9 row + the abstract's
  rank-amplification sentence — currently the abstract still asserts
  the point-vs-point separation as if it were CI-supported.
- *Audit closure.* The audit-2026-05-05.md document still references
  the 16.93× / 52.37× point-comparison without the bootstrap CI;
  add a §15.5 closure block citing the v3 cluster bootstrap and
  the FT/LoRA CI overlap.
- *Visualisation backlog.* `figures/clustered_bootstrap_ft_recovered_2026-05-13.png`
  is the figure-1 candidate for the WHO-asymmetry section but the
  paper doesn't reference it yet.

Wait for the PI to re-authorize the pod before touching the LoRA
pair-grid re-run or the chemistry 6-primary extension. The next-tick
prompt should not initiate any GPU job.

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
