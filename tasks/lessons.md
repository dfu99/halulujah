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

- **MANDATORY GPU memory gate before any RunPod launch (2026-04-19):**
  RunPod has NO scheduler — multiple projects share the pod and can OOM each
  other. Before ANY GPU job:
  1. Estimate peak VRAM: `params * bytes + optimizer states + activations`.
     For Qwen3-1.7B bf16 inference ≈ 4.4 GB; two-model collab ≈ 10 GB;
     Qwen3-4B bf16 ≈ 9 GB; full-FT training adds ≥ 4× param memory for
     optimizer state + activations.
  2. `mc runpod check` → reads free VRAM as JSON.
  3. `mc runpod fits <gb>` → exit code 0 means it is safe to launch now.
  4. `mc runpod await <gb> --timeout 60` → blocks until memory is free,
     with a timeout to avoid deadlock.
  5. `mc runpod sync <project>` → push code; run the job; `mc runpod
     fetch <project>` → pull results.
  6. NEVER bypass this gate. If `fits` returns non-zero, queue a non-GPU
     task while you wait — you are not blocked, you are polite.
- RunPod shared pods run multiple projects' jobs. Always check `nvidia-smi` AND `ps aux | grep python` before launching GPU work — another project may have started using the GPU since your last check. (Complementary to the `mc runpod` gate above; use both.)
- Python logging to nohup files is heavily buffered. Log output may not appear until process exits. Use `ps -p PID -o stat,time` to confirm process is alive.
- RunPod key is at `~/.ssh/runpod_key` (not id_ed25519). Connection: `ssh root@<ip> -p <port> -i ~/.ssh/runpod_key`.
- Base Qwen3-1.7B uses ~4.4GB VRAM in bf16. Two models (for collaboration) need ~9-10GB + KV cache overhead. RTX A4500 (20GB) fits both comfortably when GPU is clear.

## Methodology (continued)

- Base Qwen3-1.7B scores 15.4% on MMLU MCQ — below random chance (25%). This is NOT a capable model without LoRA fine-tuning. The 35pp gap to specialists confirms LoRA is essential, not optional.
- Our collaboration experiment measures MCQ accuracy changes (correct→wrong, wrong→correct), NOT hallucination detection. We use "hallucination" loosely but the data only supports "accuracy erosion under collaboration." To measure actual hallucination would need CoT chain analysis for fabrication content.

## Epistemic Rigidity & Collaboration

- Training entropy is the rigidity proxy: r=4 entropy 0.865, r=8 entropy 0.851, r=32 entropy 0.749. Higher rank → lower entropy → more confident model. Monotonic and clean signal.
- Switch rate correlates with rank: r=4 54%, r=8 48%, r=32 38%. Higher rank → fewer switches → more stubborn. But C2W/W2C RATIO is stable (~1.5x) across all ranks with base helper.
- Rank controls switch QUANTITY; helper's training controls switch QUALITY. Base helper → ~1.5x C2W/W2C (mild). Cross-domain specialist → 3.4x. Mediator → 7.1x. The helper, not the specialist, determines whether switches are harmful.
- Mix ratio is irrelevant: 9 ratios (90/10 to 10/90) all produce identical damage patterns. Damage is from LoRA training itself, not from data composition.
- When launching RunPod rank ablation, include ALL ranks in the --ranks flag even if the adapter already exists (training gets skipped). Otherwise the eval loop misses existing adapters.
- RP mediators fix the catastrophe (old -32.8pp → new +4.0pp) but C2W/W2C ratio remains elevated (3.6x). Reasoning preservation is necessary but not sufficient for fully safe collaboration. The best pair (phys+math, 1.1x) suggests domain similarity matters — closely related domains collaborate more safely.
- Shared RunPod GPU contention is a real hazard for evaluation jobs. Training survived but evaluation OOM'd because loading specialist + mediator simultaneously requires ~7GB, and another process was using 16.5GB. Always use `--skip-training` flag when relaunching after crashes to avoid redundant work.

## Polluted-data hygiene (added 2026-04-27)

- *When a fix to a known pathology is committed (e.g. the 75/25 mix +
  loss_scale="ignore_empty_think" fix to the empty-think-tag failure
  mode), retroactively audit and ARCHIVE every downstream analysis
  pipeline output that used pre-fix data.* Do not let the old numbers
  continue living in results/, figures/, and paper drafts. We let the
  PACE 10-domain n=20 numbers (with empty-think-tag chains) carry the
  paper's 22x asymmetry headline for ~10 days before catching that the
  underlying chains were artifacts of broken-thinking specialists. The
  fix was easy (move to archive/, regenerate after RP run lands); the
  cost was a partial paper rewrite and a credibility hit if a reviewer
  had caught it first.
- *When the trace data is the load-bearing claim, sample 5-10 actual
  chain transcripts before locking the abstract.* If the chains show
  agents copying each other's verbatim answers instead of reasoning,
  the deliberation analysis is measuring artifacts not reasoning.
- *Comparing accuracy numbers to "the literature" without specifying
  protocol mismatches is dangerous.* Our base 1.7B reads 15.4% on
  domain clusters under our alternating-CoT protocol; Qwen3 reports
  62-66% under standard MMLU 5-shot. Both are correct measurements,
  measuring different things. Always show both columns when comparing
  to published baselines.

## Reasoning Preservation During Fine-tuning

- Qwen3 auto-inserts `<think>` tags via chat template. Training on bare-answer data teaches the model to produce empty `<think></think>` blocks — this is NOT catastrophic forgetting, it's a training format bug.
- **loss_scale="ignore_empty_think"** (ms-swift): Masks loss on empty think tokens so the model never learns "empty thinking = correct." For HuggingFace/trl, implement as a custom loss mask on think token IDs.
- **75/25 data mix** (Unsloth recommendation): 75% reasoning data (e.g., open-math-reasoning) + 25% domain data. Reasoning examples keep the thinking pathway alive while domain examples add knowledge. Training on 100% domain data + bolted-on CoT DOES NOT WORK — the CoT signal overwhelms domain signal (14% solo vs 66% original).
- **`/no_think` suffix**: Adding to training queries signals non-reasoning mode. Model learns domain facts without touching reasoning pathway.
- For two-mode models (thinking + non-thinking), include BOTH modes in training data at ~2:1 ratio max.

## Collaboration Dynamics

- LoRA specialization destroys "collaborativeness" (Together AI term). Base models get +29pp from deliberation; specialists get +0.5pp. The training that gives domain knowledge constrains reasoning flexibility needed for collaboration.
- Base pair deliberation is the strongest condition across all experiments. Two untrained models deliberating (48.8%) crush solo specialists (29.2% = MoE routing baseline). This is pure deliberation value that MoE cannot capture.
- Composite questions: collaboration helps only when BOTH specialists are weak on the task. When one specialist is already decent, the weaker partner drags it down. Pattern: structurally necessary collaboration (+40pp for phys+math where both solo at 10%) vs optional collaboration (+0.5pp cross-domain on single-domain questions).
- RunPod shared pods have high system load (load avg 13+). Long-running training jobs can get OOM-killed silently (no Python traceback). Always use checkpointing and resumability in experiment scripts.
- The multi-agent literature (arXiv:2604.02460, April 2026) confirms: single agents outperform multi-agent under equal compute budgets. Multi-agent value is in sequential task decomposition (Einstein Arena, ChatDev), not parallel deliberation.

## Literature (Harry Potter, TOFU) erases content post-hoc but doesn't train behavioral responses to complexity. Our "trained confusion" framing is distinct.
- Personality measurement papers mostly use prompting, not fine-tuning. Per-human LoRA + distributional measurement is an open lane.
- PERSIST (AAAI 2026) measures personality instability but doesn't isolate temperature as a variable — that's our specific angle for Pivot A.

## Specialist verification (2026-04-29)

- **Off-the-shelf "code specialists" are NOT MMLU-CS knowledge specialists.**
  Qwen2.5-Coder-1.5B-Instruct loses on every MMLU-CS subject vs Qwen2.5-1.5B-Instruct
  base (mean -4.5 pp, 0/4 pass at +5 pp gate). Code-instruction tuning trades MCQ
  knowledge for code generation. Implication: `<task>-specialist != <domain>-specialist`.
  Always verify on the *exact eval format* you intend to deploy.
- **Off-the-shelf math specialists DO transfer to MMLU-math + GSM8K.**
  Qwen2.5-Math-1.5B-Instruct passes 2/5 (abstract_algebra +14, GSM8K +15.5).
  Math instruction-tuning preserves enough MCQ ability that the specialist
  still beats base on subject-overlap MCQ. Useful baseline: math is unusually
  amenable to fine-tuning (abstract symbolic content, deterministic answers).
- **Verification gate must use per-subject MMLU, not custom domain clusters.**
  Our prior "medicine cluster" (4 MMLU subjects mashed into one) was not
  comparable to the standard MMLU baselines reported by other labs. Keep the
  per-subject breakdown in all reported numbers.
- **MedQA OOD test is the diagnostic that catches MMLU-format pattern-matchers.**
  An adapter trained on MMLU-style data may pass on MMLU-medicine but fail on
  MedQA-USMLE (same MCQ format, different distribution). Verification gate must
  include >=1 OOD benchmark per domain.
- **LoRA fine-tuning shifts MCQ math UP while shifting GSM8K free-form math
  DOWN, even when trained on GSM8K data.** Qwen3-1.7B + LoRA r=16 on
  GSM8K-train: +6 to +9 pp on MMLU MCQ math subjects but -5.5 pp on GSM8K-test.
  Off-the-shelf Qwen2.5-Math-1.5B-Instruct gains +15.5 on GSM8K. LoRA produces
  a characteristically different specialist profile from instruction-tuned
  full domain LMs: format-aligned MMLU gains, format-misaligned eval regresses.
  Useful contrast for the paper.
- **Single-rank training is sufficient when the rank sweep is flat.** Medicine
  HP sweep showed all 4 ranks {8, 16, 32, 64} pass the gate. Biology likewise
  (1/3 each rank). Saves ~7h GPU per domain to launch single-rank r=16 only,
  unless rank turns out to matter (which we will learn from medicine's
  3/7 vs 1/7 spread).
- **trl 1.x breaks transformers 4.46 SFT pipeline on Qwen3-1.7B Full FT**:
  the pod re-bootstrapped 2026-05-07 arrived with `trl==1.3.0` and
  `transformers==4.46.0`. Importing `from trl import SFTConfig, SFTTrainer`
  raised `AttributeError: type object '_BaseConfig' has no attribute
  '_VALID_DICT_FIELDS'` because trl 1.x expects a newer `_BaseConfig`
  API than 4.46 provides. Symptom: the train log shows hundreds of
  identical traceback lines and the chain never gets past data loading.
  Fix: `pip install 'trl==0.13.0'` on the pod. The project's pyproject
  only pins `trl>=0.8` (too loose); pin to `trl==0.13.0` whenever a
  pod is freshly bootstrapped, or upgrade train_specialist_full_ft.py
  to the trl 1.x API.
- **`nohup ... &` is not enough on RunPod**: a Full FT training launched via
  `ssh ... "nohup python ... &"` died at step ~1000 when the SSH session
  closed. The nohup'd child was reaped despite the `&`. Use `setsid` to
  fully detach the process from any controlling terminal:
  `setsid bash -c 'nohup python ... > log 2>&1 < /dev/null &'`. This
  survives SSH disconnects. Verified by `ps -p <pid>` showing the
  process running 1+ minute after SSH command returned.
- **Full FT checkpoints at fp32 default to ~10 GB each**: includes model
  + optimizer state. Set `save_only_model=True` in SFTConfig to drop the
  optimizer state and keep checkpoints at ~3.4 GB (just bf16 model + tokenizer).
  Saves ~80% disk per checkpoint. Optimizer state is only useful for resuming
  interrupted training; for matched-solo-accuracy selection we only need the
  model weights at each checkpoint.
- **rsync -a fails on RunPod moosefs `/workspace`**: the filesystem disallows
  arbitrary chown even for root, so every entry fails with `Operation not
  permitted` and rsync returns code 23 with most files actually transferred
  but errors flooding the output. Fix: drop ownership preservation with
  `--no-owner --no-group --no-perms`. Use `rsync -rL --no-owner --no-group
  --no-perms ...` instead of `rsync -a ...`.
- **Predict-then-verify before writing predictions into an audit doc.** The
  2026-05-05 audit §6f wrote *"question-clustering would widen the CI, not
  narrow it. So the 95% CI [1.89, 8.39] is optimistic."* The clustered
  bootstrap (audit follow-up #9) gave [2.20, 7.45] — *narrower* on both
  ends. Intuition was based on "clustering correctly accounts for cross-
  cell dependence → should be more uncertain"; the empirical result is
  that within-cell over-injects independent cell-level noise that the
  clustered version suppresses. Lesson: in an audit explicitly calling
  for empirical rigor, do not embed un-tested predictions about what the
  empirical result will be — run the test first, then write the
  paragraph. Audit docs should never anticipate their own conclusions.
- **Raw-vs-delta confound in matrix-grid swap analyses.** First version
  of audit §6j used raw pair-accuracy axes for the pair_X_Y vs pair_Y_X
  swap; mid-session it was caught that the two cells use DIFFERENT
  primary-domain question sets, so raw accuracy mixes domain difficulty
  with help-asymmetry. Legitimate single-pair WHO-asymmetry comparison
  subtracts each primary's solo on its own questions
  (`delta_pair_X_Y = pair_X_Y - solo_X`). Mean |swap| went 20pp → 25.6pp
  on deltas (deltas EXPOSE more asymmetry, not less, when the
  domain-difficulty confound is removed). Lesson: any cell-vs-cell
  comparison in this matrix needs to verify whether the cells share a
  question set; if not, work in deltas.
- **Parsing artifacts can dominate behavioral metrics.** The audit §6a
  "rate ratio 0.70 net-helpful" claim used `pre_a_correct` as the gate
  for "started correct vs started wrong." 51.6% of `pre_a` records are
  'X' (parsing failures), and 67.5% of all W2C events are X→letter
  parsing recoveries (agent emitted no parseable letter on the 1-pass
  extraction, valid letter after deliberation). These are mechanically
  guaranteed to land in W2C and are not behavioral updates from peer
  pressure. Letter-only pooled rate ratio is 0.92, far closer to 1.0
  than 0.70. Lesson: rate-ratio claims derived from regex-extracted
  answers must control for parsing-failure pre-states. Also: capture
  the un-truncated raw response (`raw_full`) alongside the regex-
  extracted letter, so post-hoc re-extraction with a more permissive
  parser is possible without re-running the experiment.
- **Auto-recovery can mark queue items "completed" before the work is
  done.** During the 2026-05-05 audit deepening session, queue.yaml had
  been pre-edited by the auto-requeue/recovery system: follow-ups #10
  and #11 were marked `completed` with timestamps and notes saying
  *"Recovered completion from done marker after stale session"* despite
  the actual implementation not yet existing in code. Lesson: trust git
  commits and `figures/` artifacts as ground truth, not queue.yaml
  status flags. Overwrite the placeholder completion notes with
  accurate session records when the work actually lands.
- **datasets 4.x dropped script-based dataset support**: `casehold/casehold`
  ships as a `casehold.py` script in HuggingFace's repo, which datasets 4.x
  refuses to load (`RuntimeError: Dataset scripts are no longer supported,
  but found casehold.py`) even with `trust_remote_code=True`. Fix: pin
  `datasets<4` (3.6.0 works). The casehold loader code is unchanged; only
  the `datasets` library version matters.
- **trl 0.13 + Qwen3 + casehold fine-tune SIGKILLs at first per-step
  checkpoint save (deterministic, no traceback)**: two consecutive Full
  FT runs on law (casehold) died at exactly step 1500 = save_steps,
  right after `checkpoint-1500/` was written. Process exit was silent
  (no traceback in the log; pid gone from `ps`). Math (gsm8k) on the
  identical SFTConfig saved fine at step 1500/3000/4500. Workaround
  used 2026-05-07: pass `--save-steps 99999` for law so no per-step
  save fires; rely on the final adapter save at training end. Cost:
  no per-step checkpoints for law's matched-solo-accuracy selection,
  but the final adapter is still produced. If save-time investigation
  is ever needed, `attempt1.log` and `attempt2.log` on the pod under
  `/workspace/halulujah/logs/` preserve both crashes.
- **moosefs user/group quota fires deterministically when /workspace/
  adapters dir reaches ~21 GB (2026-05-07 RunPod state)**: even though
  /workspace's moosefs total has 148 TB free, individual write operations
  fail with `Disk quota exceeded` once accumulated user-space data on
  /workspace exceeds ~20-21 GB. Symptom for our chain: law training
  writes its 3.44 GB model.safetensors (succeeds), then can't write
  small subsequent files (config.json copy, tokenizer.json copy from
  another dir) which is what makes the post-train save_model() flow
  silently SIGKILL — trl never traceback-prints the EIO. Fix when
  hitting this: either delete some pod-side adapters (the 4 per-domain
  finals are ~3.4 GB each, deleting 1 frees 3.4 GB), or run domains
  one-at-a-time with full pod-cleanup between them. The streaming
  chain's "keep all final adapters on pod" pattern only works if the
  combined size stays under ~17 GB.
- **HuggingFace cais/mmlu API "Internal Error" intermittently fires
  during long-running pair-grid jobs (2026-05-08)**: the wrapper hit
  HF API outages on cells that needed to resolve specific subjects
  (`college_physics`, `astronomy`, `conceptual_physics`). load_dataset
  appears to validate via the API even when the dataset is fully
  cached. Fix: prepend `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1` to
  the SSH-run command (per-cell exec context). Requires that the
  dataset has been cached at least once. Updated
  run_ft_pair_grid_streaming.py to set both env vars by default,
  since the 33/35 cells' worth of cache is sufficient for any
  retry. Side benefit: marginally faster cell launches (skip the
  network roundtrip).
- **A FT model.safetensors can be partially-zeroed without raising an
  error at load time (2026-05-08 bio-final corruption)**: bio's
  /media/dan/WD_BLACK/halulujah_full_ft_streaming/biology/model.safetensors
  loaded fine via AutoModelForCausalLM.from_pretrained, ran inference
  fine, but emitted gibberish (all None preds at MMLU 5-shot, 0.0 acc
  on every domain). Per-tensor scan revealed: model.norm.weight = 0,
  model.layers.17+ input_layernorm = 0, model.layers.25 mlp.down_proj
  = 0. The first ~15 layers and embed_tokens were intact. Cause:
  rsync from pod to WD_BLACK during the streaming chain caught the
  file mid-write while trl's save_model was still flushing. The other
  4 finals were clean. Detection (run as paranoia after any big
  rsync chain): for each safetensors, sample model.norm.weight,
  model.layers.{0,17,25}.input_layernorm, model.embed_tokens; flag if
  abs.mean is 0 on any non-bias. Fix when caught: replace the
  corrupted final with the highest checkpoint-N/model.safetensors
  (saved seconds before the post-train final, same training step).
  md5 will then match. Add an integrity check to any future streaming
  pipeline by running the per-tensor scan after rsync completes.
- **datasets 3.6 vs 4.0 conflict for our 5-domain mix**: GBaker/MedQA-USMLE-4-options
  metadata uses `List` feature type (added in datasets 4.x); casehold/casehold
  is a script-based loader (dropped in datasets 4.x except for cached data).
  Resolution 2026-05-07: pin `datasets==4.0.0` AND pre-cache casehold via a
  one-time load on a 3.x install. The cached casehold then loads fine under
  4.x via the local cache layer. After the cache is established, we never
  need 3.x again. *If a fresh pod is bootstrapped:* either (a) install
  datasets==3.6 first, run a one-line `load_dataset("casehold/casehold", ...)`
  to fill the cache, then upgrade to 4.0.0; OR (b) host the casehold parquet
  on our own HF mirror and switch the load_dataset call to that path.
