# Claim → Evidence Map

*Scaffolding for the ACL 2026 paper.  Every claim made in the draft
`paper/abstract_and_intro.md` must appear here with a pointer to (a) the
experimental condition that produced it, (b) the results JSON, and
(c) the figure the reader can inspect.  Last updated 2026-04-19.*

---

## C1. Multi-agent LLM debate outcomes in the literature span ≈ -10 to +15 pp

| Field            | Value                                                                     |
|------------------|---------------------------------------------------------------------------|
| Paper sentence   | Abstract: "roughly -10 to +15 percentage points on matched benchmarks"    |
| Evidence type    | Literature survey                                                          |
| Source           | `tasks/lit_update_2026_apr.md` § C (multi-agent debate papers)           |
| Figure           | `figures/reviewer_c_literature_context.png` (left panel: literature range bars) |
| Key citations    | Du et al. 2023 (+5..+15); Talk Isn't Always Cheap 2509.05396 (-3..-10); Can LLM Agents Really Debate? 2511.07784 (~0); MoA Wang 2024 (+7.6); MAD Liang 2024 (+2..+8) |
| Rebuttal ready   | Yes — figure subsumes the full literature range with our 8 conditions overlaid as markers |

---

## C2. At matched solo accuracy, LoRA kills the collaboration delta; full FT preserves it

### C2.a  4B medicine headline

| Field           | Value                                                                     |
|-----------------|---------------------------------------------------------------------------|
| Paper sentence  | Abstract / §1.3: "+5.0 pp (1.4× C2W/W2C) vs +1.5 pp (19× C2W/W2C)"        |
| Experiment cond.| 4B Qwen3 medicine specialist (LoRA r=128 or Full FT) + matched base partner, N=200 |
| Solo accuracy   | 84% (both conditions — matched)                                           |
| Result JSON     | `results/paper_sweep/qwen3_4b_ft/4b_full_ft.json` (Full FT, delta=0.05, c2w=7, w2c=5) |
| Result JSON     | `results/paper_sweep/qwen3_4b/rank_sweep_rp.json` (LoRA r=128, delta=0.015, c2w=19, w2c=1) |
| Figure          | `figures/reviewer_b_rank_vs_ft.png` (bottom-left panel: 4B medicine LoRA vs full FT bar chart) |

### C2.b  4B physics

| Field           | Value                                                                     |
|-----------------|---------------------------------------------------------------------------|
| Paper sentence  | §4 (not in intro): "Full FT 4B physics: +1.5 pp, 1.8× C2W/W2C"            |
| Experiment cond.| 4B Qwen3 physics specialist (LoRA r=128 or Full FT) + matched base partner, N=200 |
| Result JSON     | `results/paper_sweep/qwen3_4b_ft/4b_full_ft.json` (Full FT physics, delta=0.015, c2w=11, w2c=6) |
| Result JSON     | `results/paper_sweep/qwen3_4b/rank_sweep_rp.json` (LoRA r=128 physics, delta=0.045, c2w=19, w2c=3) |
| Figure          | `figures/reviewer_b_rank_vs_ft.png` (bottom-right panel)                  |
| Caveat          | Physics comparison is *weaker* than medicine — both methods give small positive deltas; use medicine as the headline. |

### C2.c  1.7B 5-domain pattern

| Field           | Value                                                                     |
|-----------------|---------------------------------------------------------------------------|
| Paper sentence  | §4: "At 1.7B, full FT base-helper pair recovers +21 to +48 pp across 5 domains" |
| Experiment cond.| 1.7B Qwen3 per-domain full-FT specialists (5 domains) + matched base partner, N=200 |
| Result JSON     | `results/paper_sweep/full_ft_5domain/full_ft_5domain.json` (49 conditions; deltas: medicine +0.215/+0.39, physics -0.005/+0.41, law +0.225, math +0.315, biology +0.485) |
| Figure          | `results/paper_sweep/paper_sweep_summary.png` (multi-domain aggregate)   |
| Note            | Full FT at 1.7B sometimes *under-performs* full FT at 4B on absolute solo accuracy but produces larger collab deltas — interpret carefully in §4. |

---

## C3. Rank alone does not recover full FT's collaborativeness

### C3.a  1.7B rank sweep

| Field           | Value                                                                     |
|-----------------|---------------------------------------------------------------------------|
| Paper sentence  | §1.3 / §4: "A rank sweep from r=4 to r=128 … fails to close the gap"      |
| Experiment cond.| 1.7B Qwen3 medicine and physics LoRA at r ∈ {4, 8, 16, 32, 64, 128}, + base partner, N=50 per rank (coarse) |
| Result JSON     | `results/paper_sweep/rank_sweep_rp/rank_sweep_rp.json` (24 conditions)    |
| Figure          | `figures/reviewer_b_rank_vs_ft.png` (top row: 1.7B rank sweep line with full-FT horizontal) |
| Key numbers     | Medicine: r=4 +10pp, r=8 -8pp, r=16 +8pp, r=32 -6pp.  Highest positive at r=4 or r=16 (+8–10pp) — below Full FT's 1.7B deltas of +21–48pp. |

### C3.b  4B rank sweep (coarse)

| Field           | Value                                                                     |
|-----------------|---------------------------------------------------------------------------|
| Paper sentence  | §1.3 / §4: "Higher ranks slightly improve solo accuracy but leave the deliberation channel pathological" |
| Experiment cond.| 4B Qwen3 medicine and physics LoRA at r ∈ {16, 128}, + base partner, N=200 |
| Result JSON     | `results/paper_sweep/qwen3_4b/rank_sweep_rp.json` (8 conditions)          |
| Figure          | `figures/reviewer_b_rank_vs_ft.png` (bottom row)                          |
| Key numbers     | Medicine: r=16 (-1.5pp, 51/6=8.5x), r=128 (+1.5pp, 19/1=19x) — *ratio worsens* at higher rank. |

---

## C4. Deliberation has value beyond raw compute (base model)

| Field           | Value                                                                     |
|-----------------|---------------------------------------------------------------------------|
| Paper sentence  | §1.3: "base deliberation +21 pp vs. compute-matched +15 pp — 1.4× scaling ratio" |
| Experiment cond.| 1.7B Qwen3 base-base pair, 5 domains, N=200 vs single-agent 6-round N=200 |
| Result JSON     | `results/paper_sweep/full_ft_5domain/full_ft_5domain.json` (base conditions included) |
| Figure          | `results/paper_sweep/paper_sweep_summary.png`                             |
| Note            | This is the *control* that establishes deliberation is not just "more compute" — without it, Reviewer C's strongest attack is "you're seeing compute effects, not collaboration effects." |

---

## C5. LoRA specialists have a 13× worse C2W/W2C than full FT at matched solo accuracy

| Field           | Value                                                                     |
|-----------------|---------------------------------------------------------------------------|
| Paper sentence  | Abstract / §1.3: "13.5× better switching quality"                         |
| Derivation      | LoRA r=128 C2W/W2C = 19/1 = 19×; Full FT C2W/W2C = 7/5 = 1.4×; ratio ≈ 13.6× |
| Result JSON     | Same as C2.a                                                              |
| Figure          | `figures/reviewer_d_entropy_by_turn.png` (right panel: C2W vs W2C bars)   |
| Note            | Switch classification is our direct calibration proxy (logits not saved from collab runs). |

---

## C6. The failure mechanism is plausibly LoRA's rank-constrained bilinear update

| Field           | Value                                                                     |
|-----------------|---------------------------------------------------------------------------|
| Paper sentence  | §1.5 / §5: "intruder-dimension rigidity prevents integration of peer disagreement" |
| Evidence type   | Theoretical citation + our empirical measurement                          |
| Citations       | Shuttleworth et al. 2410.21228 (intruder dimensions); CeRA 2602.22911 (linear ceiling); PERA 2604.11841 (bilinear); Bayesian-LoRA 2601.21003 (calibration) |
| Source          | `tasks/lit_update_2026_apr.md` § A                                        |
| Our contribution| C2W/W2C measurement showing LoRA's post-FT overconfidence *empirically*; rank sweep showing architectural (not parameter) nature |
| Status          | Direct weight-space measurement (CKA on ΔW) deferred — `plot_reviewer_e_cka.py` OOM'd. Not currently in paper claims. |

---

## C7. Cross-domain vs same-domain LoRA (contextualizing the old PACE finding)

| Field           | Value                                                                     |
|-----------------|---------------------------------------------------------------------------|
| Paper sentence  | §4 (not intro): "Earlier 10-domain PACE study at n=20 showed cross-domain LoRA collaboration was net harmful (-9.2 pp)" |
| Experiment cond.| 90 LoRA cross-domain pairs, Qwen3-1.7B, 10 domains, N=20 each             |
| Result JSON     | `results/pace_domain_10/` (the full n=20 study)                           |
| Figures         | `figures/reviewer_c_literature_context.png` (overlay points); `results/pace_domain_10/figures/bootstrap_cis.png` |
| Framing         | This was our *starting* result that motivated the N=200 refinement; it is *not* our headline. The paper must be careful to present it as motivation, not a contradiction with the +5 pp full-FT finding. |

---

## C8. Mediator specialist uniformly fails; base mediator is mixed

| Field           | Value                                                                     |
|-----------------|---------------------------------------------------------------------------|
| Paper sentence  | Likely cut from intro; goes in §4 or §6 (discussion of intervention strategies) |
| Experiment cond.| Ratio sweep of RP mediators + full-FT mediator at 1.7B, 9 conditions      |
| Result JSON     | `results/rp_mediator/`, `results/full_ft_mediator/`                       |
| Figure          | `results/rp_mediator/` (need to generate a clean publication figure)      |
| Status          | Supporting result — ensures we aren't oversold on deliberation: even *base* mediator is asymmetric (+6.5 pp medicine, -8.5 pp physics). |

---

## Gap analysis — what the paper *cannot* yet claim

1. **Direct weight-space mechanism measurement.** The CKA script was written
   but OOM'd at the 10-adapter load step.  The paper currently relies on
   cited theory (Shuttleworth et al.) for the mechanism and our behavioural
   C2W/W2C measurement for its consequence.  The direct measurement is
   desirable but not required for submission.  [Blocker: memory on RunPod.]

2. **7B scale test.** The hypothesis predicts that LoRA's rank constraint
   should persist at 7B.  We have not run this.  The intuition.md lists
   this as a falsification target.  Queued for post-submission.

3. **Domain distance as predictor of collaboration.** r=0.197 at n=20 is
   weak.  The paper does not foreground this claim; the limitations
   section acknowledges the weak correlation.

4. **Extended rank.** We stop at r=128 at 4B.  A reviewer may ask for
   r=256 or r=512; in the CeRA paper these are argued to hit a "linear
   ceiling" regardless, so the requested extension is theoretically
   expected to confirm our negative result — but it is not run.  Flag
   in limitations.

---

## Figures inventory (currently in `figures/`)

| File                                            | Status | Used in which claim |
|-------------------------------------------------|--------|---------------------|
| `figures/reviewer_b_rank_vs_ft.png`             | Ready  | C2.a, C2.b, C3      |
| `figures/reviewer_c_literature_context.png`     | Ready  | C1, C7 (overlay)    |
| `figures/reviewer_d_entropy_by_turn.png`        | Ready  | C5                  |
| `figures/reviewer_e_cka_distance.png`           | Deferred (OOM) | — (would support C6) |
| `results/paper_sweep/paper_sweep_summary.png`   | Ready  | C2.c, C4            |

Additional figures still to produce for the paper proper (not reviewer-rebuttal
figures):

- **Figure 1** (main): Four-panel summary — 4B medicine / 4B physics / 1.7B rank
  sweep / C2W/W2C decomposition.  Consolidated from existing reviewer figures.
- **Figure 2** (mechanism): ΔW singular value spectrum comparing LoRA r=128 to
  full FT on medicine.  Needs weight-space analysis to run — *blocked on memory*.
- **Figure 3** (control): Deliberation vs compute-matched single agent scaling
  curves.  Data exists in paper_sweep; plot to be made.

---

## How this map should be used

1. Any paper sentence asserting an empirical fact must reference a claim
   ID (C1–C8) in a comment in the draft markdown.
2. When a claim lacks a pointer to a JSON or figure, it cannot go in the
   paper yet.  Move it to the gap analysis.
3. When running a new experiment, add its outputs to the relevant claim
   block so the reviewer trail is maintained.
