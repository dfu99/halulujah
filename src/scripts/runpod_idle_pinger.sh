#!/usr/bin/env bash
# runpod_idle_pinger.sh — silent cron sentry that keeps the shared RunPod
# A40 busy. Wired into the user crontab on a 10-minute cadence; produces
# no Slack output unless it decides to act.
#
# What it does each tick:
#   1. Skip if no pod is registered.
#   2. Skip if the pod is busy (any process, or any GPU using >200 MiB).
#   3. Skip until idleness has been sustained for ~15 minutes (avoids
#      false-positive on the gap between two jobs).
#   4. Skip if it already pinged within the last hour (avoids hammering
#      Slack when the agent takes a while to start the next job).
#   5. Otherwise: compose a creative prompt that (a) tells the agent the
#      pod is idle, (b) hands it a menu of "expand scope" directions
#      (larger LoRA r, new domain training sets, ablations) so the
#      canonical queue running out does not mean the pod goes cold, and
#      (c) routes to the halulujah project via `mc send`. That send hits
#      the halulujah agent's tmux session, which is the same window that
#      reads from Slack channel #C0AL62V03FT.
#
# Logs go to ~/Documents/code/halulujah/logs/runpod-idle-pinger.log so
# the user can inspect cron behavior without it spamming Slack.
#
# Manual smoke test:
#   /home2/Documents/code/halulujah/src/scripts/runpod_idle_pinger.sh --dry-run
#
# Install (one-time, idempotent — see install block at the bottom):
#   /home2/Documents/code/halulujah/src/scripts/runpod_idle_pinger.sh --install

set -uo pipefail

REPO="/home2/Documents/code/halulujah"
MC="/home/dan/Documents/code/development/bin/mc"
LOG_DIR="$REPO/logs"
LOG="$LOG_DIR/runpod-idle-pinger.log"
IDLE_TRACKER="/tmp/halulujah-runpod-idle-since"
PING_TRACKER="/tmp/halulujah-runpod-idle-lastping"
IDLE_REQUIRED_SECS=900     # ~15 min sustained idleness before pinging
PING_COOLDOWN_SECS=3600    # do not ping more than once per hour

mkdir -p "$LOG_DIR"
ts() { date -Iseconds; }
log() { echo "[$(ts)] $*" >> "$LOG"; }

DRY_RUN=0
if [[ "${1:-}" == "--install" ]]; then
    line="*/10 * * * * /home2/Documents/code/halulujah/src/scripts/runpod_idle_pinger.sh"
    if crontab -l 2>/dev/null | grep -qF "runpod_idle_pinger.sh"; then
        echo "Already in crontab."
    else
        (crontab -l 2>/dev/null; echo "$line") | crontab -
        echo "Installed: $line"
    fi
    exit 0
fi
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
fi

# 1. Is there an active pod registration?
if ! ACTIVE=$("$MC" runpod active 2>/dev/null) || [[ -z "$ACTIVE" ]]; then
    exit 0  # no pod — silent
fi

# 2. Snapshot pod state. Treat any failure as "skip this tick" — never
#    spam Slack on transient mc/ssh errors.
STATE=$("$MC" runpod check 2>/dev/null) || exit 0

IDLE=$(printf '%s' "$STATE" | python3 -c '
import json, sys
try:
    s = json.load(sys.stdin)
    if s.get("processes"):
        print("no"); sys.exit(0)
    for g in s.get("gpus", []):
        if g.get("used_mib", 0) > 200:
            print("no"); sys.exit(0)
    print("yes")
except Exception:
    print("error")
') || IDLE="error"

NOW=$(date +%s)

if [[ "$IDLE" != "yes" ]]; then
    # Reset the sustained-idleness tracker if pod went busy.
    rm -f "$IDLE_TRACKER"
    exit 0
fi

# 3. Sustained-idleness gate. First idle tick just records the timestamp.
if [[ ! -f "$IDLE_TRACKER" ]]; then
    echo "$NOW" > "$IDLE_TRACKER"
    exit 0
fi
IDLE_SINCE=$(cat "$IDLE_TRACKER" 2>/dev/null || echo "$NOW")
IDLE_FOR=$((NOW - IDLE_SINCE))
if (( IDLE_FOR < IDLE_REQUIRED_SECS )); then
    exit 0
fi

# 4. Cooldown — don't ping more than once per PING_COOLDOWN_SECS.
if [[ -f "$PING_TRACKER" ]]; then
    LAST_PING=$(cat "$PING_TRACKER" 2>/dev/null || echo 0)
    if (( NOW - LAST_PING < PING_COOLDOWN_SECS )); then
        exit 0
    fi
fi

# 5. Inspect halulujah queue to decide whether the canonical queue is
#    empty (out-of-jobs → expand scope) or has pending work.
QUEUE_FILE="$REPO/tasks/queue.yaml"
N_QUEUED=$(python3 - "$QUEUE_FILE" <<'PY' 2>/dev/null
import sys, yaml
try:
    with open(sys.argv[1]) as f:
        d = yaml.safe_load(f) or {}
    print(sum(1 for x in (d.get("queue") or []) if x.get("status") == "queued"))
except Exception:
    print(0)
PY
)
N_QUEUED=${N_QUEUED:-0}

IDLE_MIN=$(( IDLE_FOR / 60 ))

# Pick the prompt theme. Rotation keeps the agent from being told the
# same thing every hour when the queue stays empty.
THEMES=(
    "rank-sweep|Train a Qwen3-1.7B LoRA r=128 specialist on one of the under-covered domains (philosophy via SEP, history via Wikipedia-history, economics via FiQA). The current rank ladder is {4, 8, 16, 32, 64}; r=128 extends the rank-amplification claim. Run the verification gate (>= base + 5 pp on >= 1 OOD benchmark) before adding to pair-grid roster."
    "new-domain|Train a new domain specialist that is NOT in the current 5-primary roster (math/medicine/biology/law/physics). Top candidates by data availability: economics (FiQA + econ textbooks), philosophy (SEP, MoralChoice — careful overlap), chemistry (SciBench, ChemBench, MMLU-Pro chem). LoRA r=16 with the existing training scaffold. Verification gate required."
    "protocol-ablation|Re-train one existing specialist (suggest medicine — the strongest current row primary) with answer-only training data (strip CoT scaffolding from the prompt template). Then run a small pair-grid (3 helpers x 30 q) to test whether the WHO-asymmetry survives protocol-stripping. Tests reviewer concern about CoT being the load-bearing variable."
    "lora-vs-ft-matched|If the 1.7B LoRA + 4B FT pair-grids still need re-runs (parser audit 2026-05-13 flagged them), use the idle pod for those — they are the highest paper-impact items. Otherwise: run the matched-solo-accuracy ckpt selection for 1.7B FT-vs-LoRA to land C9's strongest framing."
    "broader-coop|Probe the *cooperation* axis directly: take two existing specialists and run a triple-agent variant (primary + helper-1 + helper-2) where the second helper is from a DIFFERENT domain than helper-1. Does cross-domain helper addition close more of the §6x oracle gap? Pilot on biology primary (the bootstrap-firmest row) with 30 q."
)
IDX=$(( ($(date +%s) / 3600) % ${#THEMES[@]} ))
THEME_LINE="${THEMES[$IDX]}"
THEME_TAG="${THEME_LINE%%|*}"
THEME_BODY="${THEME_LINE#*|}"

if (( N_QUEUED > 0 )); then
    PROMPT="RunPod is idle for ${IDLE_MIN} min (auto-detected by cron). The canonical queue at tasks/queue.yaml has ${N_QUEUED} queued item(s) — start the highest-priority one. Verify it has a matched-solo-accuracy contract and a parser X-rate gate before kicking off pair-grid evaluation."
else
    PROMPT="RunPod is idle for ${IDLE_MIN} min (auto-detected by cron) and the canonical queue at tasks/queue.yaml is empty. Expand scope to keep the pod warm. This rotation's direction (theme=${THEME_TAG}): ${THEME_BODY} Overall objective remains: build broad expertise in how LLMs learn domain knowledge and how domain specialists cooperate. If this theme doesn't fit current state, pick a different direction from tasks/planning.md and document the choice."
fi

if (( DRY_RUN )); then
    echo "[DRY-RUN] idle_for=${IDLE_FOR}s queued=${N_QUEUED} theme=${THEME_TAG}"
    echo "PROMPT:"
    echo "$PROMPT"
    exit 0
fi

# 6. Push. mc send routes to the halulujah tmux session, which is the
#    same place the #C0AL62V03FT Slack channel feeds into.
if "$MC" send halulujah "$PROMPT" >> "$LOG" 2>&1; then
    log "PUSH theme=${THEME_TAG} queued=${N_QUEUED} idle_for=${IDLE_FOR}s"
    echo "$NOW" > "$PING_TRACKER"
    # Reset idle tracker so the next ping requires a fresh sustained-idle
    # window — prevents back-to-back pings if the agent reads the prompt
    # but takes a few minutes to actually start GPU work.
    rm -f "$IDLE_TRACKER"
else
    log "PUSH-FAILED theme=${THEME_TAG} queued=${N_QUEUED} idle_for=${IDLE_FOR}s"
fi
