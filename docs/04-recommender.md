# 04 — Recommender (the two-mode policy)

The recommender selects the next `K` problems. Its objective **switches by uncertainty**, which is the spine of the whole system.

```
if  posterior σ on goal-relevant topics is high   →   ASSESS  (maximize information)
else                                              →   TRAIN   (maximize learning gain)
```

`K` and the daily-set composition come from the time budget `H` (`docs/01`).

---

## Mode A — Assess (adaptive diagnosis / CAT)

**Objective:** reduce uncertainty about θ as fast as possible.

**Policy:** select the unseen problem maximizing Fisher information about the relevant θ_t — i.e. `b_p` closest to current μ_t (P(solve)≈0.5), restricted to topics whose σ_t exceeds the threshold. This is textbook Computerized Adaptive Testing (Lord, 1980), specialized to the Rasch model.

**When:** new user, new topic just opened on the prereq frontier, or periodic recalibration (e.g., after a long gap, or when predicted-vs-actual diverges).

**Exit:** switch a topic to Train once `σ_t < τ`.

---

## Mode B — Train (instructional policy)

**Objective:** maximize expected learning gain per unit time toward the goal.

### Topic selection

1. **Gap** per active topic: `gap_t = max(0, R_band(t) − μ_t)`.
2. **Weight** by goal relevance (rating-mode frequency or interview-mode importance) and by **review urgency** (FSRS: how far retrievability has decayed).
3. Restrict to the **prereq-DAG frontier** (`frontier_only`).
4. Sample topics proportional to `gap_t · weight_t`, then **interleave** (P4: `max_consecutive_same_tag=1`).

### Problem selection within a topic

Pick unseen problems whose predicted first-attempt P(solve) falls in the **desirable-difficulty band** (P2: default `(0.55,0.80)`, auto-tuned per user). Among candidates, rank by:
- problem quality (high `solvedCount`, contest > gym, well-rated),
- diversity from recent problems,
- information value (mild tie-break toward more-informative items so Train still sharpens the estimate).

### Daily-set composition (from the budget)

A blended set, not a monotone grind:
- ~**65%** target-band training problems (interleaved across weak topics),
- ~**20%** spaced reviews (FSRS-due solved problems — P3),
- ~**15%** stretch problems (P(solve)≈0.4) for growth and exploration.

### Anti-Goodhart

Optimizing the 0.7 success target alone degenerates into easy-variety grinding. The **coverage** (gap-weighting), **stretch**, and **review** terms are the guardrails. The eval suite (`docs/07`) explicitly checks that the policy doesn't collapse to low-difficulty churn.

---

## Explore / exploit & the learning curve

Both modes carry uncertainty, so selection uses **Thompson sampling / UCB** over the (topic × difficulty) space: sample θ from its posterior, act greedily on the sample. This gives principled exploration for free and unifies Assess (explore for information) and Train (exploit for gain) as two points on the same uncertainty-aware spectrum.

---

## Implementation progression (validate each against the last — `docs/07`)

1. **Greedy baseline (ship in M2):** the rules above, deterministic. Already useful.
2. **Contextual bandit (M5):** state = (μ, σ, retention, streak, goal); arms = topic×difficulty buckets; reward = realized learning gain (Δμ on the touched topic, from L2's learning-rate model) minus time cost. LinUCB / Thompson.
3. **Offline RL (M5+):** treat the *trajectory* to target as the objective (long-horizon credit, not myopic per-problem gain). Train offline on logged interactions (you can't freely online-train on one human). Evaluate by **offline policy evaluation** (replay / inverse-propensity) and **simulated skill trajectories** vs the baselines.

Reward design is the interesting research surface and rhymes with Crucible's RLVR work — including the failure mode where a learned policy *games* the reward (e.g., farming Δμ on noisy topics). Document that explicitly.

---

## Estimating the user's personal optimal difficulty (P2, the key rigor move)

Continuously fit `Δμ_t` (learning gain) as a function of the difficulty gap `(θ_eff − b_p)` at attempt time. The maximizer of that curve is the user's personal sweet spot; recenter `train_target_band` on it. The eval suite checks the estimate is stable and that recentering improves simulated gain. **This is what makes "teach me best" a measured claim rather than a slogan.**

## Acceptance (gates in `docs/07`)

- Mode switching behaves: high-σ topics route to Assess and σ drops quickly.
- Train policy keeps realized first-attempt success inside the (auto-tuned) band.
- Each tier beats the previous and beats random / fixed-curriculum / pure-difficulty-match on offline policy eval and simulated trajectories.
