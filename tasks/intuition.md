# Project Intuition

## One-line claim (refined after N=200 paper sweep, 2026-04-19)

At matched solo accuracy, LoRA domain specialists lose the ability to benefit
from collaboration while full fine-tuning preserves it — at both Qwen3-1.7B and
Qwen3-4B scales. The mechanism is LoRA's rank constraint, not specialization
per se: compressing domain knowledge into a low-rank subspace creates rigidity
that blocks information flow during multi-agent deliberation.

## Why we believe it (N=200 evidence)

1. *Full FT vs LoRA r=128 at 4B, same solo accuracy (84%):*
   - Full FT medicine + base helper: 89% (+5pp, *1.4× C2W/W2C*)
   - LoRA r=128 medicine + base helper: 85.5% (+1.5pp, *19× C2W/W2C*)
   - 13× better switching quality, 3× larger collaboration delta — at same
     solo knowledge.

2. *Deliberation has genuine value beyond compute.* Base-model pair deliberation
   adds +21pp mean across 5 domains *above a compute-matched single agent
   (6 rounds)*. Compute alone adds +15pp. Deliberation ≈ 1.4× compute scaling.

3. *Base pair beats FT solo on 4/5 domains.* Untrained models deliberating
   (mean 50.6%) outperform trained specialists alone (mean 46.7%) — with
   C2W/W2C ratios near 0.1× (overwhelmingly helpful switches).

4. *Trained mediators uniformly fail.* 2-agent (specialist + trained mediator)
   -3 to -3.5pp. 3-agent bridge with FT mediator: -1 to -2pp. Only base-model
   mediator helps (+6.5pp on medicine, -8.5pp on physics — asymmetric).

## Historical context (what we originally thought)

The original 10-domain PACE study at n=20 showed net-harmful collaboration
(-9.2pp mean, LoRA cross-domain). That study had ±18pp CI half-widths —
the effect was real but the mechanism unclear. The N=200 sweep refined the
story: it's not *collaboration* that's harmful, it's *LoRA-specific rigidity*
that blocks collaboration. Full FT models at same solo accuracy collaborate
healthily.

## What would falsify the refined claim

- *Collapse at higher rank:* if LoRA r=256 or r=512 fully recovers full FT's
  collaboration delta and C2W/W2C ratio, the rank-constraint hypothesis fails.
  Our r=128 data shows no recovery (still 19× ratio vs full FT's 1.4×).
- *Scale invariance failure:* if the same comparison at 7B shows LoRA r=128
  = full FT, the constraint is specific to 1.7B/4B and the paper's scaling
  argument collapses.
- *GPT-5 judge contamination:* if held-out human evaluation on medicine
  questions shows the grader systematically penalizes LoRA output style,
  the metric is invalid.
- *Compute confound:* if equalizing inference-time compute between LoRA
  (more rounds) and full FT eliminates the delta gap, deliberation is really
  just compute and the rank story is wrong.

## Target panel and venue

- *Panel*: see `tasks/review-panel.yaml`
- *Venue*: ACL 2026 (main conference) or COLM 2026
- *Why this venue*: ACL reaches the multi-agent debate and PEFT communities
  who will challenge both the protocol and the rank-constraint hypothesis.
  COLM is the natural home for mechanistic findings about small-LLM behavior.

## Open questions

- *Mechanism*: is the rank constraint actually about *representational
  rigidity* (fewer directions to move in response to new input) or about
  *epistemic overconfidence* (LoRA imprints high-confidence predictions that
  block updating)? We lack direct evidence distinguishing these — entropy
  analysis per turn (Reviewer D) would help.
- *Domain asymmetry*: base-as-mediator helps medicine (+6.5pp) but hurts
  physics (-8.5pp). Why?
- *Scaling prediction*: does full FT's advantage widen or shrink at 7B?
- *Composite question pattern*: collaboration helps (+40pp phys+math) when
  both solo specialists are weak (<15% each). Is this a general rule or a
  medicine-physics artifact?
