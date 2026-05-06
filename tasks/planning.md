# Planning — halulujah

## Current State (2026-05-05)

**Active research direction**: WHO-asymmetry in multi-agent LLM collaboration with
*verified* domain specialists. Earlier MMLU-cluster results were retracted after
the 2026-04-27 audit (empty-think-tag chains) and the 2026-04-28 OOD test (medicine
specialist underperformed base on MedQA, indicating MMLU-format pattern matching).

**2026-05-05 audit (`tasks/audit-2026-05-05.md`, figure `figures/audit-2026-05-05.png`)**
re-derived the verified-LoRA pair-grid numbers. Three findings drive the
paper-rewrite gates:

1. **WHO-asymmetry ratio = 4.47× (95% CI 2.20–7.45 question-clustered).**
   Canonical aggregator: delta cells, full 5×5 + base helper, mean-row
   vs mean-col (audit §6b). Roster-sensitivity envelope **3.05×–6.27×**
   (audit §6c). Question-clustered bootstrap on existing data using
   the deterministic seed=42 question alignment (audit §6f revised,
   §12 resolved): 95% CI **[2.20, 7.45]**, P(ratio > 1) = 100%,
   P(ratio > 2) = 99.1%. The earlier prediction "clustering widens
   the CI" was wrong; clustering tightens it.

2. **§6a "rate ratio 0.70 net-helpful" is mostly an X-parsing artifact
   (audit §6g — critical correction).** 51.6% of pre_a records are
   parsing failures (X), and 67.5% of all W2C events are X→letter
   parsing recoveries, not genuine peer-induced updates. Letter-only
   pooled rate ratio is **0.92** (essentially balanced); 3 of 5
   primaries flip > 1.0 once X-parsing is controlled (math 1.54,
   law 2.04, physics 1.24). The "4 of 5 primaries net-helpful" claim
   does not survive. The honest paper sentence is now: *amplitude of
   switching is rank-amplified; direction of switching is roughly
   balanced once parsing artifacts are excluded*. Follow-up #10
   (re-run with `pre_a_full` capture) still pending for exact
   numbers.

3. **Stickiness vs recovery is highly anti-correlated (audit §6i).**
   Held-rate vs W2C|W: Spearman ρ = -0.92 across 30 cells. Held-rate
   vs C2W|C: ρ = +0.08. LoRA specialists that "stick to their guns"
   abandon recovery without preserving correctness — rank-amplification
   is asymmetric in *direction*, not *magnitude*.

**PI directive 2026-04-29**: do not run more collaboration experiments until each
specialist passes a verification gate (>= base + 5 pp on >=1 OOD benchmark).

## Current Priorities (in order)

### 1. Verified specialists pipeline (active)

Per-domain status:

| Domain | Strategy | Status | Pass? |
|---|---|---|---|
| math | off-the-shelf Qwen2.5-Math-1.5B-Instruct | DONE | yes (2/5 at +5pp) |
| math-Qwen3 | in-house Qwen3-1.7B + LoRA r=16 on GSM8K-train | DONE | yes (3/5 at +5pp on MCQ; GSM8K -5.5 pp) |
| CS | off-the-shelf Qwen2.5-Coder-1.5B-Instruct | DONE | NO (0/4, mean -4.5 pp) |
| medicine | in-house HP sweep on Qwen3-1.7B + MedQA-USMLE-train | DONE | yes (r=64 best, 3/7 at +5pp; college_med +9, MedQA-test +6.5) |
| biology | in-house HP sweep on Qwen3-1.7B + PubMedQA-train (10K) | DONE | weakly (r=16 best, 1/3 at +5pp; hs_bio +7) |
| chemistry | TBD (SciBench, ChemBench, MMLU-Pro chem) | queued | — |
| physics | in-house Qwen3-1.7B + LoRA r=16 on SciQ-train | DONE | yes (1/5: college_physics +6, SciQ-test +4) |
| law | in-house Qwen3-1.7B + LoRA r=16 on CaseHOLD | DONE | yes (1/4: CaseHOLD-test +24, but MMLU law -25/-13/-7; severe overfit) |
| philosophy | TBD (SEP, MoralChoice — careful overlap) | queued | — |
| history | TBD (Wikipedia history, HistorySocialScienceQA) | queued | — |
| economics | TBD (FiQA, econ textbooks) | queued | — |

### 2. Medicine HP sweep (DONE 2026-04-30)

All 4 ranks pass the verification gate. Best rank = r=64 (3/7 benchmarks at +5pp).

| benchmark | base | r=64 spec | delta |
|---|---|---|---|
| anatomy | 56.0 | 53.0 | -3.0 |
| clinical_knowledge | 66.0 | 60.0 | -6.0 |
| college_medicine | 62.0 | 71.0 | **+9.0** |
| medical_genetics | 72.0 | 71.0 | -1.0 |
| professional_medicine | 59.0 | 64.0 | **+5.0** |
| virology | 51.0 | 47.0 | -4.0 |
| MedQA-test | 45.0 | 51.5 | **+6.5** |

Pattern: training on real MedQA-USMLE-train transfers to MedQA-test (held out)
and clinical-format MMLU subjects (college_med, professional_med). Trades
breadth for clinical depth. Adapter at
`/workspace/adapters_1p7b_ood/medicine_sweep/r64/adapter_medicine_medqa`.

### 3. Biology HP sweep (running, launched 2026-04-30 14:31 UTC, ~10 h)

- Pod: same A4500 (medicine sweep finished, freed GPU)
- Script: `src/scripts/run_biology_hp_sweep.py`
- Training: Qwen3-1.7B + LoRA on PubMedQA pqa_artificial (10K subsample),
  3 epochs, lr=5e-5, ranks {8, 16, 32, 64} sequential
- Verifier: `src/scripts/verify_biology_adapter.py` — MMLU college_biology
  + high_school_biology + PubMedQA pqa_labeled held-out
- Output: `results/specialist_verification/biology_sweep/biology_r{rank}.json`

### 4. Next domains (queued)

If best rank passes → use that adapter as the verified medicine specialist.
If no rank passes → escalate (different training data / longer epochs / scrap medicine).

Then:
- Hunt off-the-shelf for biology/chem/physics/law (parallel CPU work).
- For each domain that has no off-the-shelf candidate, run the same HP sweep template.
- After 4+ verified specialists exist, run pair-grid collaboration at N=200 on
  verified specialists only (NOT on the polluted PACE adapters).

### 4. Paper rewrite

Hold all paper claims until verified specialists exist. Replace PACE-derived numbers
with verified-specialist N=200 numbers. Keep WHO-asymmetry as the spine (per memory).

**2026-05-05 update**: `paper/claim_evidence_map.md` §C2 and §C5
are revised under the conditional-rate framing, but §6g now
*deprecates* the rate-ratio finding in turn — the X-parsing
correction means the letter-only pooled ratio (0.92) is too close
to 1.0 to support the "net-helpful" framing. Until audit follow-up
#10 (re-run with `pre_a_full` capture) lands, do not cite the
abstract-level rate-ratio claim. The defensible WHO-asymmetry
headline is now **4.47× (95% CI 2.20–7.45 question-clustered;
roster envelope 3.05×–6.27×)** (audit §6f revised, §6c).

### 5. Audit follow-ups (2026-05-05, see `tasks/queue.yaml`)

DONE: #1 paper map; #2 canonical WHO aggregator; #3 matched-FT
checkpoint selector scaffold; #4 1.7B FT for `law`; #5 verified
pair-grid with FT checkpoints (scaffolded); #6 restricted-roster
WHO sensitivity; #7 4B FT rate-bound; #8 4B FT medicine+physics;
#9 question-clustered bootstrap [2.20, 7.45]; #10 pre_a_full
capture + permissive parser; #11 pair-swap figure-1;
#12 subject-stratified WHO ratio (rows-of-total drops 83.3% → 65.2%
on subject grid; 21.8% of total var is within-primary subject mix);
**#13 replicate-aware 2-way ANOVA — only primary main effect is
significant (F(4, 1470) = 26.55, p < 1e-10); helper and interaction
NOT significant; 92% of var is question-level noise; this session;
#14 claim_evidence_map.md updated with §6o/§6q/§6r power audit
and new C9 WHO-asymmetry row consolidating all the audit evidence;
this session**.

DONE: #1–#14, plus this session's deepening:
- §6n/6o/6p (subject heterogeneity / Wilson CI 17/30 / helper col_std)
- §6q follow-up #12 (subject-stratified WHO 65.2% rows)
- §6r follow-up #13 (replicate-aware ANOVA F=26.55 p<1e-10 primary)
- §6s/6t (per-q cross-helper agreement / strict permutation p=0.001)
- §6u follow-up #15 (cluster permutation: within-row p=0.65 helper
  not sig, within-column p<0.0001 primary highly sig — confirms
  §6r decomposition non-parametrically)

DONE this session also: §6v (follow-up #16) — per-q helper-correctness
vs helper self-solo: pooled rho = +0.064, P(rho > 0) = 49.6% (chance).
Helper structure is absent at the per-q level. Reconciles §6h
col-mean r = +0.51 (aggregation) with §6r/§6u helper insignificance
(cell-mean / within-row).

DONE this session also: #17 (§10 abstract directive fourth revision)
— locked-in 6-clause paragraph at top of §10 plus updated C9 row in
claim_evidence_map.md with all five helper-effect tests (none reject
H0) and four primary-effect tests (all reject H0). Defensible
one-paragraph abstract sentence locked in for the paper.

ALSO this session, §6w/6x/6y deepening:
- §6w effect sizes: primary Cohen f=0.27 (medium), ω²=0.06; helper
  Cohen f=0.06 (trivial), ω²=0.00; interaction Cohen f=0.11 (small).
- §6x oracle ceiling: best-of-6 helpers leaves +21.4 pp pooled gap
  over actual mean (51% → 72.4% oracle); physics has the biggest
  oracle-over-solo gap (+58 pp).
- §6y specialist-jackknife on WHO ratio: LOO range 10.37–31.33×;
  law specialist has Δ=-11.74 max leverage, confirming §6c/§6o/§6v
  converging diagnosis that law dominates magnitude.

Figure expanded to 10×3 = 30 panels.

DONE THIS SESSION ALSO: #18 (Tukey-style pairwise) — 0/30 cells
significant at any threshold; six converging interaction tests now
agree that the additive model fits the data perfectly. Audit §6z
written.

DONE THIS SESSION ALSO: #19 (helper-aware orchestration predictor)
— best-by-col-mean rule closes 33% of the §6x oracle gap (+7.0 pp);
majority-vote-of-6 closes 20% (+4.2 pp); always-base, always-self-match,
best-self-solo, subject-aware all UNDERPERFORM random. Striking
pattern: 4 of 5 primaries' best helper is CROSS-DOMAIN (only math
primary best with base; medicine/biology with law; law with medicine;
physics with biology). Self-match is never best. Reconciles §6r/§6u
"helper main effect not significant" with §6x oracle gap: helper
effect IS exploitable conditional on primary, not as a global ranking.
Audit §6aa written.

DONE THIS SESSION ALSO: #20 (helper-as-corrector + difficulty-stratified)
— §6bb classifies each of 30 cells as corrector (w2c > c2w) /
distractor (c2w > w2c) / neutral; result: 24/30 (80%) corrector,
5/30 distractor, 1/30 neutral. ALL 6 non-corrector cells are in the
law primary's row (0/6 corrector for law); the four other primaries
are 6/6 corrector each. Every helper has the same 4-corrector +
1-non-corrector pattern with the non-corrector cell always being
law-primary. §6cc splits per_q records by primary solo correctness
and recomputes WHO ratio: easy=1.50× (n=74, primary 24% / helper 16%),
hard=67.69× (n=176, primary 88.4%). The pooled 22.11× is a weighted
average of these two regimes; the WHO-asymmetry is concentrated on
hard questions where collaboration matters most. Audit §6bb + §6cc
written; §11 reviewer K (difficulty) + L (cell-role) added; §10
abstract directive sixth revision locked in. Figure expanded to 11×3
= 33 panels (added AC/AD/AE).

DONE THIS SESSION ALSO: #21 (hard-only replicate-aware ANOVA) —
§6dd restricts §6r ANOVA to hard-question replicates (n_total = 1056,
unbalanced across primaries: math 32, medicine 35, biology 31, law 34,
physics 44). RESULT: F_primary = 31.22, p < 1e-23 (vs §6r baseline 26.55;
ratio 1.18×); F_helper = 0.50, p = 0.78 (vs 0.96; ratio 0.52×);
F_interaction = 0.79, p = 0.73 (unchanged). Hypothesis "F_primary
jumps to 60+" partially falsified — the increase is modest because
the §6cc cell-mean ratio of 67.69× measures *pure between-cell-mean*
variance while the F-statistic measures aggregate vs within-cell
question-level noise (which remains 87.8% of total SS on hard subset).
SS_primary fraction did rise +62% relative (6.6% → 10.7%). F_helper
actually dropped further, strengthening the no-helper-main-effect
finding. Nine converging primary-effect tests now (added §6dd hard-only
F=31.22). Figure expanded to 12×3 = 36 panels (added AF/AG/AH:
F-stat comparison, SS-percentage breakdown, nine-test triangulation
card).

DONE THIS SESSION ALSO: #22 (difficulty-stratified clustered bootstrap)
— §6ee question-clusters bootstrap within each subset (easy n=74,
hard n=176) preserving the seed=42 alignment. RESULT: easy WHO ratio
95% CI [0.23, 8.36] median 1.57 (point 1.50, CI INCLUDES 1.0); hard
WHO ratio 95% CI [10.80, 198.14] median 36.05 (point 67.69, CI
EXCLUDES 1.0 by ~10×); hard/easy ratio of ratios 95% CI [2.97, 304.57]
median 23.79 (point 45.04); **P(hard > easy) = 99.9%** (1998/2000
iterations). Three findings: (i) easy CI includes 1.0 — primary
effect not stat larger than helper on easy questions; (ii) hard CI
firmly excludes 1.0 by ~10×; (iii) difficulty stratification is
99.9% bootstrap-firm. Ten converging primary-effect tests now.
Figure expanded to 13×3 = 39 panels (added AI/AJ/AK: CI bars,
bootstrap distribution histograms, ten-test triangulation card).

DONE THIS SESSION ALSO: #23 (hard-only effect sizes) —
§6ff standardizes the §6dd ANOVA F-stats into Cohen's f and ω².
RESULT: Primary Cohen's f = 0.349 (medium edging toward large)
vs full grid 0.269 (medium); ω²_primary = 10.3% vs 6.4% (+60%
relative). Helper f stayed trivial (0.05); interaction f stayed
small (0.12). Four-metric reconciliation: cell-mean ratio (+3.06×),
F-stat (+1.18×), Cohen's f (+1.30×), ω² (+1.60×) — different
magnitudes because each metric folds within-cell noise differently.
Eleven statistical-test+effect-size frames now agree on primary
dominance and its hard-question amplification. Figure expanded to
14×3 = 42 panels (added AL Cohen's f comparison, AM ω² comparison,
AN defensible-headline summary card with all 12 paper-ready
disclosures).

DONE THIS SESSION ALSO: #24 (hard-only specialist-jackknife) —
§6gg LOO specialist-drop on §6cc hard subset. RESULT: hard LOO
range 15.58–177.14× (mean 71.34, std 59.72) vs full grid
10.37–31.33× (mean 17.67). Hard has 3.8× more LOO sensitivity.
Max-leverage specialist FLIPS from law (full, Δ=-11.74) to
medicine (hard, Δ=+109.45 positive — medicine inclusion attenuates
hard ratio). Sign flip is mathematical artifact of LOO ratio.
Law leverage amplifies 4.14× on hard (-11.74 → -48.57) but loses
#1 spot. All 5 LOO replicates on hard remain firmly above 15×.
Twelve converging primary-effect tests now. Figure expanded to
15×3 = 45 panels (added AO leverage comparison, AP LOO ratios
bar chart, AQ twelve-test triangulation card).

DONE THIS SESSION ALSO: #25 (conditional rates by difficulty) —
§6hh per-cell P(post wrong | solo correct) on easy and P(post
correct | solo wrong) on hard. Per-primary W2C rates on hard:
biology 64.0%, physics 48.5%, medicine 40.0%, law 21.1%, math
20.8% (lowest). Per-primary C2W rates on easy: physics 30.6%
(highest), law 29.2%, medicine 23.3%, math 17.6%, biology 12.3%.
Net corrector score (W2C − C2W): biology +51.7 pp, physics +17.9,
medicine +16.7, math +3.2 (barely positive), law -8.1 (net
distractor — only negative row). Per-helper W2C spread is just
6.1 pp (35-41% across all 6); per-helper C2W spread is 16.0 pp
(15-31%). Primary spread / helper spread on W2C = 7.07× — primary
drives recovery 7× more than helper. The recovery-rate framing is
the most paper-readable mechanistic disclosure. Thirteen converging
primary-effect tests now. Figure expanded to 16×3 = 48 panels
(added AR per-primary conditional rates with net score, AS per-helper
conditional rates showing flatness, AT W2C-on-hard heatmap).

DONE THIS SESSION ALSO: §10 abstract directive seventh revision
(post-§6hh) — folded §6dd, §6ee, §6ff, §6gg, §6hh findings into
the canonical paper paragraph. Defensible one-paragraph abstract
sentence locked in: "Conversation outcome is determined by the
primary's willingness to update from wrong, not by helper identity."
Thirteen-test triangulation table updated.

DONE THIS SESSION ALSO: #26 (net corrector score bootstrap) —
§6ii question-clusters bootstrap within each primary on the §6hh
net corrector score. RESULT: biology is the only primary with
95% CI excluding 0 ([+34.4, +68.1], P(>0)=100%); medicine
borderline [-0.2, +34.1] P=97.4%; physics likely positive [-10.2,
+42.2] P=91%; math NOT robust [-9.1, +16.1] P=67.8% (CI crosses 0);
law likely negative but CI crosses 0 [-25.8, +7.9] P(<0)=81.2%.
PAIRWISE (10 comparisons): biology robustly higher than every
other primary (4/4 at 98.8-100% sig); medicine vs law borderline
(97.6%); math vs law NOT sig (85%); other 5 pairs NOT sig. The
§6bb count-based "0/6 corrector for law" is the categorically
sharp statement; §6ii shows the row-level magnitude uncertainty.
Fourteen converging primary-effect tests now. Figure expanded to
17×3 = 51 panels (added AU per-primary CIs, AV pairwise diff
matrix, AW P(>0)/P(<0) per-primary).

DONE THIS SESSION ALSO: #27 (per-helper net corrector bootstrap)
— §6jj symmetric to §6ii on helper axis. RESULT: 3 specialist
helpers (math, medicine, law) firmly positive (P(>0) >= 99.9%);
biology and physics borderline (95-97%); BASE HELPER is the lone
CI-crosses-0 case ([-8.4, +18.0], P(>0) = 75.9%). Pairwise: only
3/15 pairs are 95% sig (base vs math/medicine; math vs physics);
12/15 pairs cross zero. Base helper is the lone helper-side outlier
— specialist helpers are essentially fungible. Primary/helper
spread on net score = 2.76× (60 pp / 22 pp). Fifteen converging
tests now. Figure expanded to 18×3 = 54 panels (added AX per-helper
CIs, AY pairwise diff matrix heatmap, AZ side-by-side primary vs
helper bootstrap envelope comparison).

DONE THIS SESSION ALSO: #28 (multiple-comparison corrections) —
§6kk applies Bonferroni, Holm step-down, and BH-FDR to the 25
pairwise tests (10 primary + 15 helper). RESULT: 8 uncorrected
significant; 5 BH-FDR; 2 Bonferroni; 2 Holm. The 2 ironclad pairs
that survive even Bonferroni: biology > law (+59.8 pp, p~0.001)
and biology > math (-48.5 pp, p~0.001) — both involve biology as
upper outlier paired against the two lowest-W2C primaries. Under
BH-FDR adds biology > medicine, base < math, base < medicine.
Bootstrap precision floor (n_iter=2000) gives two-tailed p~0.001
for any 0%/100% case — at the edge of Bonferroni-25 significance.
Sixteen converging tests now. Figure expanded to 19×3 = 57 panels
(added BA correction-survival bars, BB top-10 pairs by two-tailed
p, BC sixteen-test final triangulation card).

DONE THIS SESSION ALSO: #29 (best-helper bootstrap stability) —
§6ll question-clusters the §6aa best-by-col-mean orchestration.
RESULT: stability scores per primary: physics 60.7%, biology 54.8%,
math 53.0% (3 stable); law 42.9%, medicine 33.1% (2 unstable).
Average stability 48.9% (vs 16.7% chance — much better than random
but far from 100%). Cross-domain pairing is DIRECTIONALLY ROBUST:
top-2 candidates per primary are cross-domain specialists for 4/5
primaries; only math primary has its top-2 as base + math
(neutral + self-match). Specific best-helper assignments stable for
3/5 primaries. Seventeen converging tests now. Figure expanded to
20×3 = 60 panels (added BD stacked-bar distribution, BE per-cell
delta forest plot, BF top-3 cross-domain marker card).

DONE THIS SESSION ALSO: #30 (best-helper LOO-CV) — §6mm leave-one-
question-out cross-validation directly measures out-of-sample
generalization of §6aa's best-by-col-mean rule. STRIKING RESULT:
in-sample +7.0 pp lift (33% gap closure) drops to LOO-CV +1.8 pp
(8% gap closure). Overfitting penalty: 5.2 pp. The LOO-picked
helper matches in-sample best at 90-100% per primary (rule is
stable in WHICH helper it picks), but picked helper accuracy on
held-out questions is lower than in-sample col-mean due to
small-N noise + selection bias. Per-primary LOO acc: biology 74%
(best), physics 60%, medicine 46%, math 42%, law 42%. Cross-domain
pairing remains directionally robust (LOO picks cross-domain at
90-100% for 4/5 primaries). The §6aa orchestration claim was
substantially overstated; honest paper-readable framing is "+1.8 pp
out-of-sample (8% of gap)". Eighteen converging tests now. Figure
expanded to 21×3 = 63 panels (added BG in-sample vs LOO-CV vs
oracle, BH per-primary LOO accuracy, BI eighteen-test card).

DONE THIS SESSION ALSO: #31 (difficulty-stratified LOO-CV) —
§6nn restricts §6mm to easy or hard subsets separately. STRIKING:
LOO gap closure is 30% on easy (+6.4 pp) but only 4% on hard
(+0.9 pp). Hard has larger oracle gap (+22.5 pp) but orchestration
realizes almost none of it under honest evaluation. Reconciliation:
(1) hard questions are intrinsically hard, recovery requires per-
question helper-knowledge coupling that §6v showed is absent;
(2) easy questions benefit because helper choice preserves correct
answers (low C2W rate); (3) the §6cc WHO-asymmetry on hard
(67.69×) is a PRIMARY-LEVEL finding (between-primary recovery
differences), NOT a helper-orchestration opportunity. Per-primary
LOO on hard: math 19% (32 qs, only 6 correct), law 24%, medicine
34%, physics 57%, biology 61%. Math/law primaries essentially
can't recover from hard questions even with best-by-col-mean.
Nineteen converging tests now. Figure expanded to 22×3 = 66
panels (added BJ subset gap-closure comparison, BK per-primary
LOO accuracy by subset, BL nineteen-test triangulation card).

DONE THIS SESSION ALSO: #32 (per-cell net corrector CI bootstrap)
— §6oo most granular bootstrap: 30 cells × n=2000. RESULT: 8/30
cells robust positive (CI excludes 0), 0/30 robust negative, 22/30
uncertain (CI crosses 0). Robust corrector cells: ALL 6 biology-
primary cells (range +40 to +68 pp) plus medicine×law (+33.3 pp)
and math×math (+22.6 pp). Biology row uniquely robust at every
aggregation level. NO cell is robustly distractor — §6bb's
"5 distractor in law row" is direction-correct but doesn't survive
bootstrap at the cell level. By primary: biology 6/6 robust,
medicine 1/6, math 1/6, physics 0/6, law 0/6. By helper: math and
law each contribute 2 of 8 robust cells. Twenty converging
primary-effect tests now. Figure expanded to 23×3 = 69 panels
(added BM heatmap with sig markers, BN forest plot of all 30
cells sorted by point net, BO twenty-test triangulation card).

DONE THIS SESSION ALSO: #33 (helper agreement on hard questions)
— §6pp provides the MECHANISTIC explanation for the §6nn 4% LOO
gap closure on hard. POOLED: 38.6% (68/176) of hard questions are
recoverable by ZERO of 6 helpers; 47.7% are nearly unrecoverable
(0-1 of 6); only 16.5% universally recoverable (6/6). PER-PRIMARY:
math 53.1% mutually unrecoverable, law 52.9%, medicine 34.3%,
physics 34.1%, biology 19.4% (with 41.9% universally recoverable).
Mean recovery / 6 matches §6hh per-primary W2C exactly. The §6cc
WHO-asymmetry on hard (67.69× cell-mean) is driven by primary-
level differences in RECOVERABILITY, not by helper effects.
For nearly half of hard questions, no helper choice can help —
this is the structural reason why orchestration fails on hard.
Twenty-one converging primary-effect tests now. Figure expanded
to 24×3 = 72 panels (added BP per-primary stacked recovery-bin
distribution, BQ pooled recovery distribution, BR twenty-one-test
triangulation card).

DONE THIS SESSION ALSO: #34 (subject decomposition of mutually-
unrecoverable hard) — §6qq breaks down §6pp's per-primary unrec
rate by MMLU subject. WITHIN MATH PRIMARY: high_school_mathematics
75% unrec (9/12), college_mathematics 75% (3/4), abstract_algebra
50% (3/6), elementary_mathematics 20% (2/10) — math primary's 17
unrec questions skew heavily toward HS/college math; the largest
subject (elementary_math, 22/50) is the most recoverable. WITHIN
LAW: professional_law 56% (18/32), jurisprudence 0% — law's 18
unrec are essentially all professional_law. WITHIN BIOLOGY:
high_school_biology 13.6% (3/22) — the most recoverable subject
globally. Reviewer I "subject-mix" partially-answered. Twenty-two
converging primary-effect tests now (§6qq flagged ⚠ in
triangulation card). Figure expanded to 25×3 = 75 panels (added
BS subject-level horizontal bar chart, BT within-primary
heterogeneity range plot, BU twenty-two-test triangulation).

DONE THIS SESSION ALSO: #35 (subject-stratified WHO ratio) —
§6rr replaces the §6m primary-identity ANOVA factor with subject-
identity in the same two-way variance decomposition. STRIKING:
subject/helper variance ratio = 21.73× (filtered n>=5) vs primary/
helper = 22.11× — essentially identical. SS_subject = 72.1% vs
SS_primary = 83.3%; SS_helper stays at 3.3% (vs 3.8%); SS_inter
rises from 12.9% to 24.6%. Cohen's f subject = 1.61 (huge),
helper f stays small (0.19-0.20) under both decompositions.
Within-primary subject row-mean spread (n-weighted): math 15.4 pp
(college_math -5.6 to elem_math +9.8), medicine 22.2, biology 0.5
(UNIQUELY HOMOGENEOUS), physics 21.7. The "row factor dominates
helper factor" finding is robust to switching row factor levels.
Reviewer I "subject-mix" concern now FULLY ANSWERED. Twenty-three
converging row-effect tests now. Figure expanded to 26×3 = 78
panels (added BV §6m vs §6rr SS comparison, BW filtered subject
row means, BX twenty-three-test triangulation).

DONE THIS SESSION ALSO: #36 (hard-subset subject-stratified WHO
ratio) — §6ss extends §6rr to the §6cc hard subset. STRIKING:
subject/helper hard ratio = 56.54× > primary/helper hard = 50.34×.
Subject-stratification STRENGTHENS the hard WHO finding (50.3× →
56.5×). SS_subject_hard = 80.2% (highest single-factor share in
audit); SS_helper_hard = 1.4% (down from 3.3% on full grid). Cohen
f subject_hard = 2.01 (huge); helper f stays at 0.12 (small).
Subject row means on hard (= W2C rate): math (4.2-38%), medicine
(28-57%), biology (57-67%), law (18.8%), physics (40-62%). Biology
homogeneous at 9.3 pp spread. college_mathematics 4.2% W2C: only
1 of 24 hard-cell recoveries succeed. high_school_biology 66.7% W2C:
largest robustly-recovered subject. Reviewer I FULLY ANSWERED at
both stratification levels. Twenty-four converging row-effect tests
now. Figure expanded to 27×3 = 81 panels (added BY four-way SS
comparison, BZ filtered hard W2C rates, CA twenty-four-test
triangulation card).

DONE THIS SESSION ALSO: #37 (per-subject hard W2C bootstrap CI) —
§6tt formal 95% CIs around §6ss point W2C rates via question-cluster
bootstrap (n=2000) per subject + 136 pairwise tests with BH-FDR +
Bonferroni-136. CLEANEST CI SEPARATION: hs_biology W2C [50.8, 81.8]
vs college_math [0, 12.5] — 39 pp non-overlap. 32 uncorrected, 20
BH-FDR, 0 Bonferroni-136 (precision floor 1/2001 ≈ 0.001 just above
α/136 = 0.0004). Top BH-FDR survivors: college_math vs hs_biology
-62.5 pp, hs_biology vs prof_law +47.9 pp. Cross-primary biology-
vs-math/law: 5/10 BH-FDR (3 hs_biology pairs all pass). Reviewer F
"where are CIs" answered at most-granular subject level. Twenty-five
converging tests. Figure expanded to 28×3 = 84 panels (CB CI forest
plot, CC pairwise heatmap, CD twenty-five-test card).

DONE THIS SESSION ALSO: §10 abstract directive 8th revision (post-
§6tt) — consolidates §6qq/§6rr/§6ss/§6tt subject-stratification
findings into the canonical paper-ready paragraph. New triangulation
table with 25 row-effect tests; new defensible one-paragraph abstract
sentence anchoring on the row-factor-invariance + difficulty-
amplification finding.

DONE THIS SESSION ALSO: #38 (easy-subset subject-stratified WHO
ratio) — §6uu mirrors §6ss on the easy subset. RESULT: easy primary/
helper = 1.31x; easy subject/helper = 1.64x. Both small; WHO-
asymmetry confirmed HARD-ONLY at both row factor levels. Helper
variance is sizeable on easy (22-24%, vs 1.4-1.7% hard, 3.3-3.8%
full). Cohen f easy: primary 0.63 (medium), subject 0.82 (large),
helper 0.53 (medium). Six-way WHO ratio family complete: full 22.1×
/ 21.7×; hard 50.3× / 56.5×; easy 1.31× / 1.64×. Subject-vs-primary
choice moves ratio <14%; difficulty regime moves it 30-40×. CLEANEST
DISAMBIGUATION between row-factor-choice (small) and difficulty-
regime (large). C2W rate easy: hs_biology 11.1% (lowest) → prof_law
33.3% (highest, single largest "abandons own correct answer" rate).
Twenty-six converging row-effect tests now. Figure expanded to 29×3
= 87 panels (CE six-way SS bar chart, CF log-scale ratio across
difficulty, CG twenty-six-test final triangulation).

DONE THIS SESSION ALSO: #39 (per-subject easy C2W bootstrap CI) —
§6vv mirrors §6tt on the easy regime. RESULT: 5 filtered subjects ×
10 pairwise tests. Per-subject 95% CIs on easy C2W: hs_biology
[3.3, 21.1], elementary_math [5.6, 22.2], college_biology [4.2,
29.2], prof_medicine [2.8, 44.4], prof_law [16.7, 52.8]. Pairwise:
2 uncorrected (hs_biology vs prof_law p=0.022; elementary_math vs
prof_law p=0.029), 0 BH-FDR, 0 Bonferroni-10. The §6uu C2W hierarchy
on easy is DIRECTION-CORRECT but NOT statistically robust at subject
granularity — confirms hard-regime is statistical backbone, easy-
regime is correlative. Twenty-seven row-effect tests now (§6vv ⚠).
Figure expanded to 30×3 = 90 panels.

DONE THIS SESSION ALSO: claim_evidence_map.md C9 row updated with
§6cc–§6vv evidence rows (8 new rows: difficulty stratification,
conditional rates, cell-level helper roles, mutual-unrecoverability,
subject-stratified WHO family, per-subject hard W2C CI, per-subject
easy C2W CI, 8th-revision defensible headline). C9 is now the
audit's authoritative paper-side claim.

DONE THIS SESSION ALSO: #40 (high-iteration §6tt bootstrap) — §6ww
re-runs §6tt at n_iter=50000 (precision floor 0.00004 < α/136 =
0.000368). STRIKING: 11 of 136 subject-pair tests survive
Bonferroni-136 at α=0.05 (vs 0 at n=2000). §6tt's 0/Bonferroni
was a precision-floor artifact. 11 survivors: 8 involve college_math
(W2C 4.2%), 3 involve hs_math; cross-primary headline contrasts
hs_biology vs college_math (-62.5 pp), hs_biology vs prof_law
(+47.9 pp), college_med vs prof_law (+31.2 pp); within-math survivor
college_math vs elementary_math (-34.2 pp) confirms §6qq subject-
mix heterogeneity at Bonferroni-136. §6qq triangulation flag
upgraded from ⚠ to ✓. §10 abstract directive 9th revision cites the
Bonferroni-136 survivors. Twenty-eight row-effect tests now. Figure
expanded to 31×3 = 93 panels (CK correction-survival bar chart 2k vs
50k, CL 11 Bonferroni-136 survivors forest plot, CM twenty-eight-
test final triangulation card).

ALL 40 audit follow-ups now complete.

The audit now has FIVE converging tests on the helper effect, all
showing no structure:

  Test                              Helper effect signal
  §6r ANOVA helper main             F = 0.96, p = 0.44 (none)
  §6t strict-shuffle (combined)     p = 0.001 (combined w/ primary)
  §6u within-row permutation        p = 0.65 (none)
  §6h col-mean vs self-solo         r = +0.51 (n=5; aggregation only)
  §6v per-q vs self-solo            rho = +0.064 (none)

And FOUR converging tests on the WHO-asymmetry primary effect:
  §6f within-cell bootstrap          CI [1.89, 8.39] excludes 1
  §6f question-clustered bootstrap   CI [2.20, 7.45] excludes 1
  §6r ANOVA primary main             F = 26.55, p < 1e-10
  §6u within-column permutation      p < 0.0001

Primary effect: rejected H0 in 4/4 tests with primary resolution.
Helper effect: not rejected in 4/4 tests with helper resolution
(only the §6t combined test gave p = 0.001, decomposed into
primary-driven by §6u).

**Audit deepening §6n/§6o/§6p/§6q** (2026-05-05): per-MMLU-subject
heterogeneity, Wilson 95% CI per cell (only 17/30 sig), helper col_std,
and subject-stratified WHO ratio. The 65.2%-vs-83.3% rows-of-total
shift on the subject grid means the "primary identity dominates"
framing partly subsumes within-primary subject heterogeneity:
SS_primary_marginal = 43.4% of total, SS_subject_within_primary =
21.8%, SS_helper = 3.3%, SS_residual (subject × helper interaction)
= 31.5%. Audit §6q has the full reattribution and the revised
abstract framing ("the primary's question-set composition dominates"
rather than "primary identity dominates").

### 6. Next priority — re-run verified pair-grid with new schema (NOT YET RUNNING)

The 1.7B verified pair-grid needs a re-run on A40 (or PACE) to
populate the new `qid` and `pre_a_full` fields emitted by the
patched runner. Without it:

- `recompute_pre_a_letters.py` cannot disambiguate true parsing
  failures from regex-misses → §6g letter-only rate-ratio numbers
  remain provisional.
- The 1.7B Full FT pair-grid (audit follow-up #5 scaffold) cannot
  be paired with a fresh LoRA grid that uses identical questions.
- The matched-solo-accuracy 1.7B FT comparison (the headline
  experiment that lets us claim "rank constraint causes" not just
  "rank constraint correlates with") still depends on a working A40
  with pulled FT checkpoints.

A40 access requires "GPU ACCESS GRANTED" message per CLAUDE.md.
Until then: paper-side work (claim_evidence_map.md updates) and
deeper audit analysis on the existing data.

## PI directive 2026-05-03

A40 authorized: `ssh root@69.30.85.238 -p 22192`.

Active phase: (a) Full FT specialists at matched solo accuracy, no
quantization shortcuts. Per-domain Full FT training -> matched-solo-accuracy
selection -> 5x5 Full FT pair-grid -> compare LoRA-vs-FullFT.

*Current state (2026-05-03 09:42 UTC)*:
- A40 bootstrapped: src + 5 LoRA adapters (1.3 GB) on `/workspace/` of
  `root@69.30.85.238:22192`. Dependencies installed.
- Medicine Full FT *running* (PID 1554, setsid'd, 1.21 it/s, ~1h45m ETA).
  Output: `/workspace/adapters_1p7b_full_ft/medicine/`, save_steps=1500,
  save_only_model=True so each checkpoint is ~3.4 GB.
- After medicine: math (gsm8k) -> biology (pubmedqa) -> law (casehold) ->
  physics (sciq), all single-rank Full FT, ~1h45m each, total ~9h.

After all 5 Full FT trained:
1. Run matched-solo-accuracy selection per domain: pick the FT checkpoint
   where solo accuracy on the pair-grid protocol (50 mixed-MMLU, 3 CoT
   rounds) matches the corresponding LoRA solo accuracy.
2. Run `run_verified_pair_grid.py` with FT checkpoints in place of LoRA
   adapters. Same 45 conditions, N=50, 3 rounds. Output:
   `results/verified_pair_grid_qwen3_1p7b_full_ft/matrix_results.json`.
3. Compare LoRA vs FullFT: row means, col means, asymmetry ratio,
   per-cell deltas.

## PI directive 2026-05-02

> *"Run (c) -> (a). Backup the data and pipelines to WD_BLACK after (c)
>  preliminary 5x5 LoRA pair-grid. Then I will switch to A40 to run (a)
>  the Full FT."*

LoRA roster of 5 specialists is finished. 5x5 LoRA pair-grid (c) DONE
2026-05-03 06:58 UTC. 45 conditions complete. WHO-asymmetry reproduces:
row spread 34 pp >> col spread 7.6 pp, ratio 4.47x (vs polluted-PACE
22.3x). See obj-040 + `figures/fig_verified_pair_grid_5x5.png`.

After (c) completes:
1. rsync `/workspace/adapters_1p7b_ood/` to `/media/dan/WD_BLACK/halulujah/`
2. rsync `results/specialist_verification/` and
   `results/verified_pair_grid_qwen3_1p7b/` to WD_BLACK
3. rsync the 1.7B-LoRA scripts to WD_BLACK
4. PI authorizes A40 48 GB switch
5. Run (a) Full FT with matched solo accuracy on A40
6. Run paired LoRA-vs-FullFT pair-grid on A40

## PI directive 2026-05-01

> *"Finish LoRA (without quantization compromises). And then I'll authorize A40."*

Plan: complete the LoRA specialist roster on Qwen3-1.7B (no 8-bit Adam or
quantization shortcuts), run the preliminary LoRA-only pair-grid collaboration
to confirm the WHO-asymmetry signal reproduces in the verified regime, *then*
switch to A40 48 GB for Full FT specialists at matched solo accuracy.

Specialist target before A40 transition: at least 4-5 verified Qwen3-1.7B LoRA
specialists, sufficient for a 4x4 or 5x5 pair-grid.

Currently have 3 (math, medicine, biology). Need 1-2 more — next: law on
CaseHOLD train, then physics on SciQ or ARC-Challenge.

## Recently Completed

- 2026-05-05: **obj-047 Audit follow-up #11** — `figures/audit_pair_swap_asymmetry.png` (figure-1 candidate). Mean |swap| 25.6 pp on deltas, max 46 pp (math↔biology). Headline pair: biology↔law +38 pp delta swap (biology primary +42 pp from law helper; swap roles → law gains +4 pp). Audit §6j rewritten under delta aggregator with the methodology caveat noted.
- 2026-05-05: **obj-046 Audit follow-up #10** — patched `collab_reasoning_scoped` to emit `raw_full`, runner to emit `pre_a_full`, wrote `src/scripts/recompute_pre_a_letters.py` with strict + permissive parsers. Smoke-tested on 12 patterns. Existing data has no pre_a_full yet → script falls back to baseline X-rate report (X rate 51.6%) and prompts re-run.
- 2026-05-05: **obj-045 Audit follow-up #9** — `src/scripts/clustered_bootstrap_who.py` exploits existing seed=42 question stability. Result: 95% CI **[2.20, 7.45]** (2000 iter), narrower than within-cell [1.89, 8.39]. Audit §6f prediction "clustering widens" empirically falsified. P(ratio > 1) = 100%, P(ratio > 2) = 99.1%. Defensible headline strengthens.
- 2026-05-05: **obj-044 Audit §6g–§6k deepening** — X-parsing pollution (51.6% X, 67.5% W2C is X-recovery, letter-only pooled rate ratio 0.92 not 0.70), helper-quality regression (Pearson +0.51), held vs flips (Spearman -0.92 W2C, +0.08 C2W), entropy diffuses (+0.45 bits not converges). Refreshed audit figure with 15 panels.
- 2026-05-05: Audit follow-up #7 — patched `src/scripts/run_4b_full_ft.py` to capture per_q (pre_a, pre_a_correct, c2w_per_started_correct, w2c_per_started_wrong). Bounded the existing §6a 4B FT rate ratio at **0.115–14.5** depending on unobserved pre_a accuracy; central estimate ~1.6. Audit §6a explicitly hedged.
- 2026-05-05: Audit follow-up #6 — recomputed WHO-asymmetry on restricted rosters. Direction survives all (3.05× to 6.27×); canonical headline 4.47× (full 5×5 + base helper, delta cells). Conservative envelope 3.05×–6.27×.
- 2026-05-05: Audit follow-up #5 — wrote `src/scripts/run_verified_pair_grid_ft.py` (FT pair-grid runner, mirrors LoRA grid). Needs A40 + manifest from #3.
- 2026-05-05: Audit follow-up #3 — wrote `src/scripts/select_matched_ft_checkpoint.py`. --dry-run verified; manifest_dry_run.json shows medicine has ckpts 4500/6000 (not 1500/3000 as planning.md said). Needs A40 to execute the actual scan.
- 2026-05-05: Audit follow-up #2 — locked canonical WHO aggregator; `src/scripts/compute_who_asymmetry.py` + `who_summary.json`. Headline 4.47×.
- 2026-05-05: Audit follow-up #1 — `paper/claim_evidence_map.md` C2/C5 revised under conditional-rate framing; old count-ratio claim preserved as C2.dep / C5.dep.
- 2026-05-05: Audit `tasks/audit-2026-05-05.md` + `figures/audit-2026-05-05.png` — discovered C2W:W2C count-ratio is base-rate confounded; rewrote `paper/claim_evidence_map.md` C2/C5 under conditional-rate framing.
- 2026-05-04: Full FT checkpoints pulled to WD_BLACK (`halulujah_2026-05-04_full_ft_checkpoints/`, biology+physics 4 ckpts each, math+medicine 2 ckpts each, **law NOT trained**).
- 2026-05-03: WD_BLACK backup at `halulujah_2026-05-03_pre_a40_handoff/` (1.3 GB)
- 2026-05-03: 5x5 LoRA pair-grid (45 conditions): WHO-asymmetry ratio 4.47x. obj-040.
- 2026-05-02: Pair-grid orchestrator `run_verified_pair_grid.py` written + launched
- 2026-05-03: A40 pod authorized + bootstrapped (91 scripts + 5 LoRA adapters synced). obj-041.
- 2026-05-01: Physics LoRA r=16 specialist on SciQ: 1/5 pass (college_physics +6). obj-039.
- 2026-05-01: Law LoRA r=16 on CaseHOLD: 1/4 pass, +24 CaseHOLD-test, MMLU law -25. obj-038.
- 2026-05-01: Math-Qwen3 LoRA r=16 specialist on GSM8K-train: 3/5 MCQ pass, GSM8K -5.5
- 2026-04-30: Biology HP sweep (Qwen3-1.7B + LoRA on PubMedQA-train): all ranks weakly pass, r=16 best
- 2026-04-30: Medicine HP sweep (Qwen3-1.7B + LoRA on MedQA-train): all ranks pass, r=64 best (3/7)
- 2026-04-29: CS specialist verification — failed (negative result, useful for paper)
- 2026-04-29: Math specialist verification — passed (off-the-shelf Qwen2.5-Math)
- 2026-04-29: Specialist dataset search plan written (`tasks/specialist_dataset_search.md`)
- 2026-04-29: Pivoted from polluted specialists to verification-gated pipeline
- 2026-04-28: MedQA OOD test → polluted-specialist diagnosis
- 2026-04-27: Empty-think-tag audit → archived all PACE-derived data. obj-042.
- 2026-04-27: Question-clustered bootstrap → 22.3x ratio [10.2, 48.0] (now suspended). obj-043.
- 2026-04-25: 4B 5x5 specialist-pair grid (now suspended pending verification)

## Active Pod Inventory

- /workspace usage: ~30 GB after 9.2 GB cleanup (mainly hf_cache 28 GB)
- /workspace/adapters_1p7b_ood/: empty, will fill with medicine_sweep
- WD_BLACK mirror: /media/dan/WD_BLACK/models — current
- Local backup: /home2/Documents/code/halulujah/archive/pre_RP_polluted_2026-04-27/

## Known Risks

- A single A4500 (20 GB) is the bottleneck. Cannot run two specialist trainings in
  parallel. Stay on sequential queue.
- /workspace has a per-volume quota; clean as we go.
- Medicine HP sweep is ~10.5 h: monitor first checkpoint output to catch OOM early.
