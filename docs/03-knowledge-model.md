# 03 — Knowledge Model (the math)

Three layers. Ship L1 first; L2/L3 are upgrades validated against L1 in `docs/07`. The model exposes two functions everything else depends on:

```python
predict_solve(problem, state) -> (p: float, info: float)   # P(solve) and Fisher information
update(interaction) -> None                                # online posterior update
```

---

## L1 — Bayesian per-topic IRT (baseline, ship this)

### Latent skill

For each CF tag `t` (~37: dp, graphs, greedy, math, binary search, two pointers, dsu, trees, data structures, strings, bitmasks, number theory, combinatorics, geometry, flows, segment tree, …) maintain a **posterior** over skill θ_t, approximated as Gaussian `N(μ_t, σ_t²)`. σ_t is first-class — it drives Assess-vs-Train (`docs/04`).

### Outcome model (Rasch / 1PL)

CF gives ground-truth problem difficulty `b_p`, so we only estimate the learner:

```
P(solve p | θ) = σ( (θ_eff(p) − b_p) / s )
```

- `s` is a temperature (logistic scale); CF ratings are already ~Elo-spaced, so initialize `s` to the Elo scale and fit.
- `θ_eff(p)` aggregates the θ_t of the problem's tags. **Default: min** (a multi-tag problem is gated by your weakest required skill). Alternatives to evaluate: learned softmin, attention over tags. This credit-assignment is the messiest choice in the system — see Honesty below.

### Update (online Bayesian / Elo)

On each interaction, update the contributing μ_t toward the outcome and shrink σ_t by the information gained. Two acceptable implementations:
- **Closed-form Elo-style:** μ ← μ + K·(y − p), with K ∝ σ² (Kalman-like gain) so uncertain skills move faster. Reduce σ via the Fisher update below.
- **Online Laplace:** maintain (μ, σ²) by a one-step Gaussian update of the logistic likelihood.

Distinguish **in-contest solve** from **upsolve** and weight first-attempt outcomes most — conflating them inflates μ.

### Fisher information (powers Assess mode)

For the Rasch model the information an item carries about θ is

```
I(θ) = P(θ)·(1 − P(θ)) / s²     →  maximized when P = 0.5  (i.e. b_p ≈ θ_eff)
```

So **Assess mode** picks the unseen problem with `b_p` closest to current μ (max info ⇒ fastest σ reduction). **Train mode** deliberately picks easier (P in the desirable-difficulty band), trading information for learning gain. This tension *is* why the two modes exist.

### Cold-start prior

Seed each μ_t from the user's current CF rating (a global prior) and immediately refine per-topic from history. New topics start at the global prior with high σ.

---

## L2 — Performance Factors Analysis (learning-rate layer)

Rasch is static — it doesn't model that *practicing a skill improves it*. PFA (Pavlik et al., 2009) and the Additive Factors Model (Cen, Koedinger & Junker, 2006) add per-skill learning rates: the logit gains a term in the count of prior successes/failures on that skill.

```
logit P = β_skill + γ_skill·(#prior_correct) + ρ_skill·(#prior_incorrect) + (θ − b)
```

- Gives **per-skill learning rates** (γ, ρ) — directly useful: which topics you learn fast vs slow.
- Used in `docs/07` to fit **learning curves** and validate that practice on a skill actually reduces its error (power law of practice; Newell & Rosenbloom, 1981).

---

## L3 — Deep Knowledge Tracing (predictive upgrade)

Sequence model over the interaction stream → P(solve next). Start with **DKT** (LSTM; Piech et al., 2015); consider **SAKT** (self-attentive; Pandey & Karypis, 2019) for interpretability of which past items drive the prediction.

- **Input per step:** problem embedding, tag multi-hot, `b_p`, verdict, Δt since last (for recency/forgetting), in-contest flag.
- **Keep L1 as the interpretable backbone**; L3 is the predictive engine. The dashboard reads μ_t from L1; the recommender may use L3's P(solve) where it predicts better.
- **Honest expectation:** on a single user's (sparse) history, DKT may *not* beat L1/L2. Cold-start L3 from a public CF interaction dataset, and report the comparison either way — "IRT wins at n=1" is a legitimate result.

---

## Mastery (the P5 criterion, made precise)

A topic `t` is **mastered** iff:

```
μ_t ≥ R_band(t)            # confident skill at/above the goal band for that topic
AND  σ_t ≤ mastery_sd_max  # confident, not lucky
AND  retention(t) holds across ≥ mastery_sustained_reviews spaced reviews
```

Until all three hold, `t` stays in the active recommendation pool.

---

## Prerequisite DAG (P6)

A hand-seeded DAG of CP concepts with prerequisite edges, refined over time:

```
binary_search → bs_on_answer → parallel_binary_search
dp → dp_on_subsequences, tree_dp → dp_broken_profile, digit_dp
dsu → dsu_rollback, dsu_on_tree
graphs → shortest_paths → mcmf ; graphs → bridges/articulation
segment_tree → seg_tree_beats, persistent_segtree, li_chao
```

The recommender only opens a node once its parents are mastered (`frontier_only`). v3 idea: *learn* the DAG and the taxonomy from statement+editorial embeddings instead of trusting CF's coarse tags.

---

## Honesty / known modeling risks

- **Multi-tag credit assignment** (`θ_eff`) is the largest source of error. Don't over-claim per-topic precision; report it as a band and validate the aggregation choice in `docs/07`.
- **Tag noise**: CF tags are coarse and sometimes wrong. The learned-taxonomy idea (v3) is the principled fix.
- **Survivorship / selection**: you solve what you choose; the model sees a biased sample. Assess mode partly corrects this by deliberately sampling for information.
- **n=1 sparsity**: prefer the lower-variance model (L1/L2) until data justifies L3.
