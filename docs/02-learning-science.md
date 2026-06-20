# 02 — Learning Science (the rigor core)

This is the **why** behind every pedagogical decision. Each principle below maps to a concrete design rule and a tunable parameter. If a constant in `config.py` cannot be traced to a row here, it does not belong in the system.

Attributions are to the originating work at author/year granularity for credibility; treat them as pointers, not gospel — where the literature is contested (e.g., the exact optimal-difficulty target), we **estimate the parameter from the user's own data** rather than hardcode a number.

---

## P1 — Retrieval practice (the testing effect)

Actively recalling/producing a solution strengthens memory far more than re-reading it (Roediger & Karpicke, 2006). Generating before seeing the answer drives the *generation effect*.

- **Rule:** the unit of practice is *solving*, never reading. The tutor (`docs/04`) withholds the solution and forces a generation attempt before any hint; hints are a ladder, the full editorial is last resort.
- **Parameter:** `min_generation_attempts_before_hint = 1`; hint ladder depth.

## P2 — Desirable difficulty & the optimal-difficulty target

Conditions that introduce difficulty during practice slow immediate performance but improve long-term learning (Bjork, 1994). There is a sweet spot: too easy → no learning signal; too hard → failure + demotivation.

The information-theoretic anchor: for a Rasch item, **Fisher information about the learner's ability is maximized at P(correct)=0.5** (`docs/03`), so *for measuring* skill, ~50% items are optimal. For *learning*, the optimum is higher: Wilson et al. (2019, "the 85% rule") show that for gradient-based learners, training error ≈ 15% (accuracy ≈ 85%) maximizes the rate of improvement. Multi-step problem solving tolerates more failure than simple classification, and motivation matters, so the practical band is wider and lower.

- **Rule:** **Assess mode** targets first-attempt P(solve) ≈ 0.5 (max information). **Train mode** targets a band; default `[0.55, 0.80]`, center 0.7.
- **Critical:** the training target is a **prior, not a truth.** The system estimates the user's *personal* optimal difficulty by regressing observed learning gain (Δθ) against the difficulty gap (θ−b) and centering the band on the empirical maximizer (`docs/07`). Hardcoding 0.85 would be unrigorous.
- **Parameters:** `assess_target_p = 0.5`; `train_target_band = (0.55, 0.80)` (auto-tuned); `personal_optimum_estimator = on`.

## P3 — Spacing & the forgetting curve

Memory decays predictably (Ebbinghaus, 1885); practice distributed over time beats massed practice (Cepeda et al., 2006 meta-analysis), and reviewing at the point of near-forgetting maximizes efficiency.

- **Rule:** maintain a retention state per concept; schedule a review when predicted retrievability drops to a threshold. Use **FSRS** (difficulty/stability/retrievability) over fixed SM-2 intervals.
- **Parameters:** `target_retrievability = 0.90` (review trigger); FSRS weights (fit to the user's recall data, not the defaults, once enough reviews exist).

## P4 — Interleaving

Mixing problem types within a session (vs blocking one type) improves discrimination and transfer, at the cost of feeling harder (Rohrer & Taylor, 2007).

- **Rule:** the daily set **interleaves topics** — never serve N problems of one tag in a row. Shuffle across the gap-weighted topic set.
- **Parameter:** `max_consecutive_same_tag = 1`; daily-set topic spread.

## P5 — Mastery learning

Advance only after a competence criterion is met (Bloom, 1968; the "2-sigma" tutoring result). "Mastery" must be an explicit, measurable threshold — not "did a few problems."

- **Rule:** a topic is **mastered** when the posterior over θ_t is above the target band **with low variance** (confident, not lucky) **and** retention holds across ≥2 spaced reviews. Until then it stays in the active pool.
- **Parameters:** `mastery_threshold` (in rating units, relative to R); `mastery_posterior_sd_max`; `mastery_sustained_reviews = 2`.

## P6 — Zone of proximal development / prerequisite frontier

Learn at the frontier of current ability — just beyond what you can do unaided (Vygotsky). For CP this means respecting concept prerequisites (binary search → BS on answer → parallel BS; DP → tree DP → DP-on-broken-profile).

- **Rule:** maintain a prerequisite DAG of CP concepts; do not recommend a topic whose prerequisites are unmastered. Recommend along the DAG frontier.
- **Parameter:** the prereq DAG (`docs/03`); `frontier_only = on`.

## P7 — Metacognition & calibration

Learners (and models) must know what they don't know. A teaching system that *miscalibrates* difficulty teaches badly, and an over/under-confident learner studies the wrong things.

- **Rule:** the model's P(solve) must be calibrated (`docs/07`); surface confidence to the user (predicted solve prob + "why this problem"). Optionally elicit a pre-attempt confidence to detect over/under-confidence and adjust.
- **Parameter:** calibration is a **release gate**, not a metric to admire.

## P8 — Cold-start as adaptive diagnosis

Early on the model knows little; uninformed recommendations waste time. The principled fix is **Computerized Adaptive Testing**: select items to maximize information about θ, converging on an estimate in few items (Lord, 1980).

- **Rule:** new user / new topic → enter **Assess mode** (P8 = the trigger for the CAT objective in `docs/04`). Seed the prior from the user's current CF rating and observed history.
- **Parameter:** `assess_until_posterior_sd < τ` then switch to Train.

---

## Summary table

| # | Principle | Design rule | Parameter (default) |
|---|---|---|---|
| P1 | Retrieval practice | solve, don't read; generate before hints | `min_generation_attempts=1` |
| P2 | Desirable difficulty | assess@P=0.5, train@band; auto-tune to user | `train_band=(0.55,0.80)`, auto |
| P3 | Spacing | FSRS review at near-forgetting | `target_retrievability=0.90` |
| P4 | Interleaving | mix topics in a session | `max_consecutive_same_tag=1` |
| P5 | Mastery learning | confident posterior + sustained retention | `mastery_sustained_reviews=2` |
| P6 | ZPD / prereqs | recommend on the prereq-DAG frontier | `frontier_only=on` |
| P7 | Calibration | calibrated P(solve); a release gate | ECE gate (`docs/07`) |
| P8 | Adaptive diagnosis | cold-start → CAT info-maximization | `assess_until sd<τ` |

## The one rule above all

**Do not hardcode pedagogical truths.** Defaults here are priors. Where the user's own data can estimate a better value (optimal difficulty, FSRS weights, learning rates), the system estimates it and the eval suite checks that the estimate is stable and improves outcomes.
