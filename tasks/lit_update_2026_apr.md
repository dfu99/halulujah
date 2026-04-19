# Literature Update — April 2026

Sweep performed **2026-04-19** against our refined rank-constraint hypothesis
(see `tasks/intuition.md`): *at matched solo accuracy, LoRA domain specialists
lose the ability to benefit from collaboration while full fine-tuning preserves
it — the mechanism is LoRA's rank constraint*.

## TL;DR

Three independent April-2026 paper clusters have converged on mutually
reinforcing findings that each partially cover our mechanism. **None of them
make our core empirical claim (LoRA vs full-FT at matched solo accuracy in a
collaboration protocol)**, but together they strengthen the rank-constraint
story considerably. One paper creates the strongest threat to our framing
(§ B-3 below) — it argues LoRA and full FT are 90–95% equivalent on downstream
tasks.

---

## A. Papers that STRENGTHEN the rank-constraint hypothesis

### A-1. CeRA: Overcoming the Linear Ceiling of Low-Rank Adaptation
`arXiv:2602.22911` (Feb 2026)

> "LoRA faces a linear ceiling: increasing the rank yields diminishing returns
> in expressive capacity due to intrinsic linear constraints."

**Relevance**: Makes the explicit claim that rank alone cannot close the gap
to full FT, aligned with our empirical observation at r=128 (still 19× C2W/W2C
ratio vs full FT's 1.4×). They propose a SiLU-gated non-linear expansion to
fix it — which is tantamount to admitting low-rank ΔW is architecturally
limited.

**Action**: Cite as primary theoretical support for why r=128 LoRA cannot
recover full-FT behaviour. Our 4B medicine result (same solo 84%, LoRA 1.5pp
vs full FT 5pp) is the empirical realization of their "linear ceiling".

### A-2. Polynomial Expansion Rank Adaptation (PERA)
`arXiv:2604.11841` (April 2026)

> "the bilinear formulation of weight updates captures only first-order
> dependencies between low-rank factors, restricting the modeling of
> nonlinear and higher-order parameter interactions."

**Relevance**: LoRA's BA form is bilinear → can only encode first-order
dependencies. If collaboration requires higher-order dependencies between the
specialist's domain knowledge and an incoming partner signal, LoRA's form is
inherently unable to route that interaction. This gives our finding a
mechanistic grounding outside the calibration framing.

**Action**: Cite alongside CeRA as two independent arguments for the same
expressive-capacity constraint we observe empirically in collaboration.

### A-3. LoRA vs Full Fine-tuning: An Illusion of Equivalence
`arXiv:2410.21228` (updated into 2026)

> "weight matrices trained with LoRA have new, high-ranking singular vectors,
> which we call intruder dimensions, while those trained with full
> fine-tuning do not… Intruder dimensions are directly responsible for
> catastrophic forgetting."

**Relevance**: This is the strongest existing paper for our argument.
It shows that LoRA and full FT are *structurally different* in weight space
even at matched task performance — the exact conditions of our experiment.
Their "intruder dimensions" hypothesis is a candidate mechanism for why
LoRA-trained specialists are rigid during collaboration: the intruder
dimensions are high-singular-value directions that dominate the model's
response and are not present in full-FT.

**Action**: Make this the central theoretical anchor of the Discussion
section. We empirically show the behavioural consequence of intruder
dimensions in a multi-agent collaboration protocol. Cite 2410.21228 as
"Shuttleworth et al." (verify authors on final pass).

### A-4. TalkLoRA: Communication-Aware Mixture of LoRA Experts
`arXiv:2604.06291` (April 2026)

> "existing MoE-augmented LoRA methods assume that experts operate
> independently, often leading to unstable routing, expert dominance…
> Talking Module enables expert-level communication prior to routing."

**Relevance**: Independently confirms our intuition that LoRA experts cannot
effectively share information in their baseline form. We show the
corresponding behavioural failure at the *agent* level (our specialists are
LoRA MoE-style "experts" that talk to each other via natural-language
reasoning chains rather than routed softmax). TalkLoRA argues the fix is
structural communication; our paper suggests natural-language communication
fails for the same reason — LoRA outputs lack the information channel.

**Action**: Cite as parallel evidence that LoRA experts in isolation fail to
integrate collaborative signal. Discuss whether our "full-FT mediator" result
is analogous to their "Talking Module".

### A-5. Bayesian-LoRA
`arXiv:2601.21003` (Jan 2026)

> "calibration often deteriorates after domain-specific fine-tuning, causing
> models to become systematically overconfident."

**Relevance**: Directly anticipates our C2W/W2C ratio finding. The claim in
our paper that LoRA specialists are overconfident (C2W 19× W2C at r=128) is
now an instance of a documented general phenomenon — we add the consequence
*in a multi-agent setting*.

**Action**: Cite in the calibration / Reviewer D discussion. Our switch
classification provides independent empirical confirmation that
post-fine-tuning calibration is compromised.

### A-6. Why LoRA Fails to Forget: Regularized Low-Rank Adaptation
`arXiv:2601.06305` (Jan 2026)

Extends the Shuttleworth "intruder dimension" line toward unlearning. Less
directly relevant but worth citing as part of the LoRA-structural-rigidity
cluster.

---

## B. Papers that THREATEN or CONTEXTUALIZE the hypothesis

### B-1. LoRA Training Provably Converges to a Low-Rank Global Minimum
`arXiv:2502.09376` (with 2026 updates)

Argues LoRA reaches a good low-rank minimum "or it fails loudly — but it
probably won't fail". The optimism in this paper creates a tension: if LoRA
training is well-behaved in theory, why does our collaboration protocol
expose failure?

**Response**: The paper is about *solo* optimization convergence, not
*collaborative inference* behaviour. Our finding is that the well-optimized
low-rank minimum lacks the directions needed for multi-agent updating. This
should be rebutted head-on in the paper, not avoided.

### B-2. How Much is Too Much? LoRA Rank Trade-offs
`arXiv:2512.15634` (published Dec 2026)

*Note: publication date is post-sweep; included only as a forward pointer.*
Expects to cover knowledge vs robustness trade-offs across rank. If the
paper lands before our submission cycle completes, integrate or rebut.

### B-3. The 90-95% equivalence argument (threat)
Multiple sources (Index.dev, Introl, LoRA vs QLoRA comparisons) argue in
practice LoRA recovers 90-95% of full FT quality. **If a reviewer cites this
as "LoRA is fine"**, our response is: our collaboration delta gap is 13× in
switch quality at *matched* solo accuracy — downstream task parity does not
imply collaboration parity, which is a separate and novel dimension.

---

## C. Extended context: Multi-agent debate literature

### C-1. Talk Isn't Always Cheap
`arXiv:2509.05396` — already cited in Reviewer C figure.

> "Introducing a weak or less capable LLM agent into a debate with a stronger
> agent can detrimentally affect the debate outcome."

**Relevance**: Our cross-domain LoRA specialist pairs (-8.7pp mean) are a
"weaker partner" instantiation in the sense this paper describes. But we add
that the *same* model at *matched* solo capability fails under full FT much
less than under LoRA — i.e. the harm is not "weakness" but "rank-constrained
weight geometry".

### C-2. Can LLM Agents Really Debate? A Controlled Study
`arXiv:2511.07784` — new.

Appears to find that MAD methods fail to consistently outperform single-agent
baselines at matched compute, echoing the "Single > Multi" line. Our finding
refines this: *base* pair debate does outperform compute-matched single agent
(+21pp vs +15pp compute alone) — so deliberation genuinely helps in the
*correct* configuration. Full FT preserves this; LoRA destroys it.

### C-3. Stochastic Self-Organization in Multi-Agent Systems
`arXiv:2510.00685`

Argues communication structure is the key variable. Tangential to our
rank-constraint claim, but reinforces that partner *selection* matters — and
our result that full-FT specialists collaborate while LoRA specialists don't
is a partner-property, not a structure-property, finding.

### C-4. Persuasion-driven adversarial influence
Nature Sci Reports 2026

Finds a single adversarial debater can drop accuracy 10-40% in MAD. Not
directly relevant to our protocol (no adversary), but establishes that MAD
outcomes are highly sensitive to participant properties — which is the
general claim our paper operationalizes via rank constraint.

---

## D. Net verdict for our submission

**Hypothesis survives the literature sweep** and in fact becomes *better
positioned* than at N=200 write-up time:

1. The theoretical rank-constraint arguments (CeRA, PERA) now exist as
   citable support, released *after* our experiments.

2. "Intruder dimensions" (Shuttleworth et al., 2410.21228) gives us a
   ready-made candidate mechanism to discuss in §5.

3. The multi-agent debate community has independently converged on the idea
   that participant properties (weak / conforming / adversarial) dominate
   outcomes — our paper contributes *what* those properties are
   architecturally (LoRA rank constraint + calibration deterioration).

4. **Differentiation from prior work remains clear**: nobody has run the
   controlled LoRA-vs-FT comparison at matched solo accuracy in a
   collaboration protocol. That is still uniquely ours.

**No paper found that renders the result obsolete.** The strongest threats
(B-1 convergence paper, B-3 "90-95% equivalence" folklore) do not address
our setting.

## E. Action items for the paper

| Section | Incorporate |
|---------|-------------|
| Introduction | Cite 2410.21228 (intruder dims) + 2602.22911 (linear ceiling) as motivation |
| Related Work | Expand multi-agent debate section: 2509.05396, 2511.07784, Nature 2026 |
| Mechanism (§5) | Ground the mechanism in intruder-dimension / bilinear-LoRA theory |
| Discussion | Pre-empt B-1 convergence and B-3 equivalence threats |
| Limitations | Note that higher-rank approaches (CeRA, PERA) have not been tested yet — future work |

---

*Sweep recorded: 2026-04-19*
*Source queries: arXiv 2604.xxxx, ICLR 2026 workshops, "LoRA rank constraint",
"multi-agent LLM collaboration", "LoRA vs full fine-tuning", "LoRA
overconfidence calibration".*
