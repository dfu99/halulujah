# Full FT chain launch runbook (1.7B, A40)

Authoritative sequence for re-launching the 5-domain Full FT queue
under the streaming-cleanup pattern (Option A from PI Slack 2026-05-06).

## Pre-flight (must pass before launch)

1. **Pod reachable.** `mc runpod active` shows a registered host. `mc runpod check` returns a non-error response with the A40 listed and free VRAM ≥ 15 GB. If unreachable, ask PI to re-register or spin up a new A40 instance:
   ```
   mc runpod register root@<host> --port <N> --key ~/.ssh/runpod_key
   ```

2. **GPU free.** `mc runpod check` should show `processes: []` or only project-`halulujah` processes. If another project is occupying VRAM, wait or use `mc runpod await 15` to block until free.

3. **Source synced.** Push the latest repo to the pod:
   ```
   mc sync halulujah
   ```
   Verify on the pod: `ls /workspace/halulujah/src/scripts/run_full_ft_chain_streaming.py` exists.

4. **HF cache primed.** First domain training will download Qwen3-1.7B base if not cached. This adds ~5 min one-time. Subsequent domains reuse the cache. To pre-pull:
   ```
   ssh <pod> "python -c 'from transformers import AutoModel; AutoModel.from_pretrained(\"Qwen/Qwen3-1.7B\")'"
   ```

5. **Local-base populated** (only if reusing existing adapters). For medicine and physics (already complete on WD_BLACK):
   ```
   mkdir -p /media/dan/WD_BLACK/halulujah_full_ft_streaming
   cp -r /media/dan/WD_BLACK/models/halulujah_full_ft_models/full_ft_medicine \
         /media/dan/WD_BLACK/halulujah_full_ft_streaming/medicine
   cp -r /media/dan/WD_BLACK/models/halulujah_full_ft_models/full_ft_physics \
         /media/dan/WD_BLACK/halulujah_full_ft_streaming/physics
   ```

## Launch

From the workstation (NOT from the pod):

```
python -m src.scripts.run_full_ft_chain_streaming \
    --pod-host $(mc runpod active | awk '/Host:/ {print $2}') \
    --pod-port $(mc runpod active | awk '/Port:/ {print $2}') \
    --pod-key ~/.ssh/runpod_key \
    --local-base /media/dan/WD_BLACK/halulujah_full_ft_streaming \
    --reuse-local \
    2>&1 | tee logs/full_ft_chain_streaming.log
```

The script blocks while training each domain on the pod, then rsyncs and cleans up before starting the next. Total wall-time:

| Domains to train | Wall-time |
|---|---:|
| 5 fresh (no reuse) | ~9 h |
| 3 fresh (medicine + physics REUSED) | ~5.5 h |
| 1 fresh (only law missing) | ~2 h |

## Monitoring

Watch the first 30 min after launch to verify the medicine launch + pull cycle works on a real pod (the `--dry-run` only validated the command structure, not actual ssh/rsync against a live pod).

In another terminal:

```
# Watch pod disk usage
watch -n 60 'ssh <pod> "df -BG /workspace | tail -1"'

# Watch the medicine training log on the pod
ssh <pod> "tail -f /workspace/halulujah/logs/medicine_full_ft.log"

# Watch local rsync output
tail -f logs/full_ft_chain_streaming.log
```

Expected sequence per domain:
1. `[<domain>] launching training -> /workspace/halulujah/logs/<domain>_full_ft.log`
2. `[<domain>] waiting; tail: ...` (every 2 min, training progress)
3. `[<domain>] training complete -> .../config.json present`
4. `[<domain>] rsync pod -> /media/dan/WD_BLACK/...`
5. `verified final model.safetensors 3.44 GB ✓`
6. `[<domain>] cleanup done; pod per-step ckpts removed`

If a domain shows `REUSED` instead of `OK`, the local adapter was already complete and was just rsync'd UP to the pod (skipping training).

## Common failure modes

- **SSH "Connection refused"**: pod terminated. Re-register or spin up a new instance.
- **Training timeout (`--max-wait-hours 4` exceeded)**: training likely OOM'd or hung. Check the pod log; common cause is another project's process taking VRAM (use `mc runpod check` to verify).
- **Rsync incomplete**: `final model.safetensors only X.YZ GB; expected ≥ 3.27`. Cleanup is aborted to avoid losing data; investigate rsync error in the log, then re-run with `--start-from <domain>`.
- **Pod disk fills mid-training**: should not happen with the streaming pattern, but if it does, the chain will halt at the next rsync. The `--reuse-local` flag lets you resume from any partial point.

## Post-chain

1. Verify all 5 final adapters exist on WD_BLACK:
   ```
   for d in medicine math biology law physics; do
     du -h /media/dan/WD_BLACK/halulujah_full_ft_streaming/$d/model.safetensors
   done
   ```
   All 5 should be ~3.4 GB.

2. Run the verification gate (≥+5 pp OOD) for math, biology, law (the freshly trained ones). Medicine and physics already passed during the original chain.

3. Run matched-solo-accuracy selection per domain using `src/scripts/select_matched_ft_checkpoint.py`.

4. Run `src/scripts/run_verified_pair_grid_ft.py` with FT checkpoints in place of LoRA adapters to produce the LoRA-vs-Full-FT pair-grid comparison. Output:
   `results/verified_pair_grid_qwen3_1p7b_full_ft/matrix_results.json`.

5. Run `verify_data_integrity.py` against the FT pair-grid to confirm 45 conditions, N=50, alignment.

6. Run `verify_audit_headlines.py` and `verify_paper_consistency.py` to confirm no drift.

## Audit references

- Audit follow-up #4 (1.7B Full FT for `law`): blocked on pod; will be unblocked by this chain.
- Audit follow-up #5 (verified pair-grid with FT checkpoints): scaffolded in `run_verified_pair_grid_ft.py`; runs after chain completes.
- §13 closure capstone documents this as the audit's primary remaining gap; the streaming chain is the unblock path.
