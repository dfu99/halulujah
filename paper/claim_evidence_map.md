# Claim → Evidence Map

*Scaffolding for the ACL 2026 paper.  Every claim made in the draft
`paper/abstract_and_intro.md` must appear here with a pointer to (a) the
experimental condition that produced it, (b) the results JSON, and
(c) the figure the reader can inspect.  Last updated 2026-05-05.*

---

## 2026-05-05 audit revision

The 2026-05-05 audit (`tasks/audit-2026-05-05.md`, §6a) found that the
**raw C2W:W2C count ratio is base-rate confounded** by solo accuracy.
At high solo accuracy (where most "before" answers are correct) the
count ratio mechanically inflates toward C2W; at low solo accuracy
it deflates toward W2C. Re-running the analysis on the verified-LoRA
1.7B 5×5 pair-grid gives a pooled count ratio of **0.13** (W2C
dominates) — directly *opposite* to the 4B Full FT pooled ratio of
**1.76** (C2W dominates) — even though both settings are "more
sycophantic than baseline" in the headline-claim sense.

The defensible substitute is the **conditional rate ratio**:

> P(C2W | started correct)  ÷  P(W2C | started wrong)

Both 1.7B verified LoRA (mean 0.70) and 4B Full FT (pooled 0.63) lie
*below 1.0* under this measure. So:

- **The direction of asymmetric switching is preserved by rank-constrained adaptation, not destroyed by it.** Both LoRA and Full FT specialists are *more likely to flip from wrong to correct than from correct to wrong* in the rank-normalized sense.
- **The amplitude of switching differs.** LoRA flips more often in *both* directions than Full FT (32.7% vs 14.7% C2W|C; 46.7% vs 23.3% W2C|W). This is the cleaner statement of the rank-constraint effect.

All claims marked **(REVISED 2026-05-05)** below have been rewritten
under this framing. The original count-ratio claims are preserved in
*deprecated subsections* (C5.dep, C2.dep) so reviewers can audit the
trajectory of the work.

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

## C2. At matched solo accuracy, LoRA flips more eagerly than Full FT; the *amplitude* of update is rank-amplified  *(REVISED 2026-05-05)*

### C2.a  4B medicine headline

| Field           | Value                                                                     |
|-----------------|---------------------------------------------------------------------------|
| Paper sentence  | Abstract / §1.3 (REVISED): "Full FT preserves a small positive delta (+5.0 pp); LoRA gives a smaller delta (+1.5 pp) with much higher per-question switch rates in both directions." |
| Experiment cond.| 4B Qwen3 medicine specialist (LoRA r=128 or Full FT) + matched base partner, N=200 |
| Solo accuracy   | 84% (both conditions — matched)                                           |
| Result JSON     | `results/paper_sweep/qwen3_4b_ft/4b_full_ft.json` (Full FT, delta=0.05, c2w=7, w2c=5) |
| Result JSON     | `results/paper_sweep/qwen3_4b/rank_sweep_rp.json` (LoRA r=128, delta=0.015, c2w=19, w2c=1) |
| Figure          | `figures/reviewer_b_rank_vs_ft.png` (bottom-left panel: 4B medicine LoRA vs full FT bar chart) |
| Caveat (NEW)    | *Avoid* citing C2W:W2C as 1.4× vs 19×.  At medicine's 84% solo accuracy the W2C denominator is mechanically small (16 wrong out of 100). The headline should be the **delta** (+5 vs +1.5 pp); the switching numbers should be cited as conditional rates per C5 below. |

### C2.b  4B physics

| Field           | Value                                                                     |
|-----------------|---------------------------------------------------------------------------|
| Paper sentence  | §4 (REVISED): "Full FT 4B physics: +1.5 pp delta; LoRA 4B physics +4.5 pp delta — at this solo-accuracy point the delta is not the strongest discriminator; report switch rates instead." |
| Experiment cond.| 4B Qwen3 physics specialist (LoRA r=128 or Full FT) + matched base partner, N=200 |
| Result JSON     | `results/paper_sweep/qwen3_4b_ft/4b_full_ft.json` (Full FT physics, delta=0.015, c2w=11, w2c=6) |
| Result JSON     | `results/paper_sweep/qwen3_4b/rank_sweep_rp.json` (LoRA r=128 physics, delta=0.045, c2w=19, w2c=3) |
| Figure          | `figures/reviewer_b_rank_vs_ft.png` (bottom-right panel)                  |
| Caveat          | Physics comparison is *weaker* than medicine — both methods give small positive deltas. Per the audit, do NOT use this as a headline; it is a secondary reproduction. |

### C2.c  1.7B 5-domain pattern

| Field           | Value                                                                     |
|-----------------|---------------------------------------------------------------------------|
| Paper sentence  | §4: "At 1.7B, full FT base-helper pair recovers +21 to +48 pp across 5 domains" |
| Experiment cond.| 1.7B Qwen3 per-domain full-FT specialists (5 domains) + matched base partner, N=200 |
| Result JSON     | `results/paper_sweep/full_ft_5domain/full_ft_5domain.json` (49 conditions; deltas: medicine +0.215/+0.39, physics -0.005/+0.41, law +0.225, math +0.315, biology +0.485) |
| Figure          | `results/paper_sweep/paper_sweep_summary.png` (multi-domain aggregate)   |
| Note            | Full FT at 1.7B sometimes *under-performs* full FT at 4B on absolute solo accuracy but produces larger collab deltas — interpret carefully in §4. |
| Status (2026-05-05) | These deltas come from the *pre-verification* 1.7B Full FT roster (the polluted PACE-derived corpus). The audit recommends NOT citing C2.c numbers in the abstract until the post-verification 1.7B Full FT pair-grid (audit follow-up #5) has been run. |

### C2.dep  *(deprecated count-ratio interpretation, retained for trail)*

The original C2.a / C2.b numbers (1.4× vs 19× C2W:W2C count ratios)
are mathematically derivable from the cell counts above but are
**no longer the headline interpretation**. The audit shows that
count ratios depend on solo accuracy in a base-rate-driven way and
flip direction across plausible verified-LoRA settings. They remain
true descriptive statistics of the cell counts; they should not be
used as a *behavioural* claim about LoRA's switching tendency.

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

## C5. LoRA specialists update on peer answers more eagerly than Full FT, in *both* directions  *(REVISED 2026-05-05)*

| Field           | Value                                                                     |
|-----------------|---------------------------------------------------------------------------|
| Paper sentence  | Abstract / §1.3 (REVISED): "Across the verified roster, the LoRA specialist's per-question conditional switch rate is roughly 2× the Full FT specialist's, in both the C2W (correct→wrong) and W2C (wrong→correct) directions." |
| Verified-LoRA evidence | `/tmp/halulujah_audit/conditional_rates.json`: 1.7B verified LoRA pair-grid (30 cells) mean C2W&#124;C = 32.7%, mean W2C&#124;W = 46.7%, rate ratio = 0.70. |
| 4B Full FT evidence (approximate) | Same JSON: 4B FT (3 cells, +base helper, N=200) pooled C2W&#124;C ≈ 14.7%, W2C&#124;W ≈ 23.3%, rate ratio ≈ 0.63. **Caveat**: the 4B FT runner did not save per_q, so the denominator is solo accuracy, not actual pre_a-correct count. The math-feasible range for the 4B FT rate ratio is **0.115–14.5**; the most plausible central estimate is ~1.6 if pre_a tracks solo at ~70% (which is what we observe in the 1.7B grid). Audit follow-up #7 patched `src/scripts/run_4b_full_ft.py` to capture per_q for next run. |
| Magnitude difference | LoRA / FT C2W&#124;C ≈ 32.7 / 14.7 = ~2× and LoRA / FT W2C&#124;W ≈ 46.7 / 23.3 = ~2× — but both 4B FT numbers are approximations, see caveat above. |
| Direction       | The 1.7B LoRA rate ratio (0.70) is exact and below 1.0. The 4B FT rate ratio is between 0.6 and 1.6 depending on the unobserved pre_a accuracy. The headline claim is therefore: *the 1.7B verified-LoRA grid shows net-helpful switching when rank-normalized* — extending this to 4B FT requires the per_q-capturing rerun. |
| Caveat — solo accuracy not matched | The 1.7B LoRA cells span 4–14% pre-collab accuracy (low) while 4B FT cells span 61–87% (high). The audit explicitly flags that the LoRA-vs-FT magnitude comparison is *not* yet at matched solo accuracy. Audit follow-up #5 (1.7B Full FT pair-grid) is required to close this. |
| Figure          | `figures/audit-2026-05-05.png` (Panel C: conditional switch rates LoRA vs FT) |
| Source          | `tasks/audit-2026-05-05.md` §6a                                           |

### C5.dep  *(deprecated count-ratio interpretation)*

| Field           | Value                                                                     |
|-----------------|---------------------------------------------------------------------------|
| Old paper sentence | Abstract / §1.3: "13.5× better switching quality"                      |
| Old derivation  | LoRA r=128 C2W:W2C = 19/1 = 19×; Full FT C2W:W2C = 7/5 = 1.4×; ratio ≈ 13.6× |
| Why deprecated  | Both numbers are count ratios in regimes where the C and W denominators are unequal. Recomputing on the 1.7B verified-LoRA pair-grid gives a pooled ratio of 0.13 — the inverse direction — confirming the count ratio is dominated by solo accuracy. |
| Status          | Retained here for audit trail only; do *not* cite in the paper. |
| Earlier figure  | `figures/reviewer_d_entropy_by_turn.png` (right panel: raw C2W vs W2C bars; legend should be updated to clarify these are counts, not rates). |

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

5. **(NEW 2026-05-05) Matched-solo-accuracy LoRA-vs-FT comparison at 1.7B.**
   The current §C2 evidence pairs 4B Full FT (medicine, physics) with
   either pre-verification 1.7B Full FT (C2.c, polluted) or 4B LoRA at
   r=128. There is no *post-verification* matched-solo-accuracy LoRA-vs-FT
   pair-grid at 1.7B. Tracked as audit follow-ups #3 (matched-checkpoint
   selector), #4 (1.7B FT for `law`), #5 (run pair-grid). Until #5 lands,
   do not cite C2 numbers in the abstract.

6. **(NEW 2026-05-05) Conditional rate measurement on 4B FT pair-grid.**
   §C5 uses solo accuracy as the denominator approximation for the 4B FT
   conditional rate. Per audit follow-up #7, the per-question chain data
   should be re-aggregated for an exact rate ratio. The 0.63 number is
   accurate to ~±2 pp.

7. **(NEW 2026-05-05, REVISED LATER 2026-05-05) WHO-asymmetry headline ratio.**
   Reconciled with obj-040 via `src/scripts/compute_who_asymmetry.py`. The
   canonical aggregator uses **delta cells** (`pair_acc - solo_primary_acc`)
   on the full 5×5 specialist roster + base helper column, giving
   **4.47×** exactly (matches obj-040 commit `7047905`).
   Roster-sensitivity envelope: **3.05× (4×4 minus law) → 6.27×
   (5×5 specialists-only, no base helper col)**. Authoritative summary at
   `results/verified_pair_grid_qwen3_1p7b/who_summary.json`. The §C7
   PACE 22.3× ratio is suspended per obj-043 and should not appear in
   the abstract.

---

## Figures inventory (currently in `figures/`)

| File                                            | Status | Used in which claim |
|-------------------------------------------------|--------|---------------------|
| `figures/reviewer_b_rank_vs_ft.png`             | Ready  | C2.a, C2.b, C3      |
| `figures/reviewer_c_literature_context.png`     | Ready  | C1, C7 (overlay)    |
| `figures/reviewer_d_entropy_by_turn.png`        | Ready  | C5                  |
| `figures/reviewer_e_cka_distance.png`           | Deferred (OOM) | — (would support C6) |
| `results/paper_sweep/paper_sweep_summary.png`   | Ready  | C2.c, C4            |
| `figures/audit-2026-05-05.png`                  | Ready  | C5 (Panel C: conditional rates), §6a / §6b audit context |

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
