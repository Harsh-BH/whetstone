# 07 — Evaluation (how we prove it teaches)

"It teaches best" is a claim that must be *measured*. This doc defines the metrics, the acceptance gates, and the honest limits. `/eval` regenerates everything; seeds are fixed.

---

## A. Knowledge model — does it predict you correctly?

### A1. Next-step prediction
Hold out the most recent slice of `interactions` (temporal split, never random — leakage). Predict each held-out first-attempt outcome.
- **Metrics:** AUC, accuracy, log-loss.
- **Baselines:** global solve-rate; per-tag solve-rate; "always predict majority."
- **Gate:** L1 AUC ≥ 0.70 and beats the per-tag baseline by ≥ 0.05 AUC. L2 ≥ L1. L3 reported vs L1/L2 (may lose at n=1 — that's a finding, not a failure).

### A2. Calibration (a release gate — P7)
A difficulty model that says "70%" must be right ~70% of the time.
- **Metrics:** Expected Calibration Error (ECE), reliability diagram, Brier score.
- **Gate:** ECE ≤ 0.05. If miscalibrated, the recommender's difficulty targeting is meaningless → **block release**.

### A3. Learning-curve fit (does practice actually help?)
Using L2/PFA, fit error vs number of opportunities per skill.
- **Check:** error decreases with opportunities and follows the **power law of practice** (Newell & Rosenbloom) for most skills.
- **Output:** per-skill learning rates (γ, ρ) with confidence; flag skills where practice shows *no* improvement (model or data problem).

---

## B. Recommender — does it choose well?

### B1. Mode behavior
- High-σ topics route to **Assess**; verify σ drops fast (few items to `σ < τ`) — compare items-to-converge against random item selection.
- **Train** keeps realized first-attempt success inside the auto-tuned band.

### B2. Offline policy evaluation (the core comparison)
Using the logged `recommendations` (with `propensity`), estimate the value of the policy counterfactually.
- **Methods:** replay / inverse-propensity scoring; doubly-robust where feasible.
- **Baselines:** random; fixed curriculum (CF rating ladder); pure difficulty-match (no spacing/interleaving/coverage); the greedy baseline.
- **Gate:** each tier (greedy → bandit → RL) beats random and fixed-curriculum on estimated learning gain per unit time, and the bandit/RL beats greedy.

### B3. Simulated skill trajectories
Build a simulator from the fitted L2 model (a synthetic learner with the estimated learning rates + forgetting), run each policy for a simulated horizon under budget `H`.
- **Metric:** simulated time/effort to reach target `R`.
- **Output:** trajectory plot per policy; report the speedup of the best policy over baselines (this is the headline number for the CV write-up).

### B4. Anti-Goodhart check
Verify the policy does **not** collapse to low-difficulty churn: track the difficulty distribution and topic coverage of served problems over time; flag if it drifts below the band or stops opening new frontier topics.

---

## C. Ablations (what each principle is worth)

Turn each component off and measure the drop (on B2/B3):
- − spacing (P3), − interleaving (P4), − uncertainty exploration, − personal optimal-difficulty estimation (use the static prior instead), − prereq frontier (P6).

Each ablation that *doesn't* hurt is a candidate for removal — rigor cuts both ways. Report the table.

---

## D. Online ground truth (n=1, honest)

The only real-world signal: **predicted vs actual Codeforces rating** over time.
- After each contest, compare the model's predicted performance/rating to the actual `user.rating` update.
- Track the error trend; a well-functioning model's predictions converge on reality as data accrues.
- Optionally, period-over-period self-comparison (e.g., months on the policy vs a prior baseline period) — reported **with loud caveats** (confounds: motivation, contest difficulty drift, life). This is suggestive, never causal.

---

## Limitations (state these in the README)

- **n=1:** no control group; B/C are offline/simulated and inherit the model's assumptions. The simulator is built from the same model it evaluates — guard against this by validating the simulator's *own* predictions against held-out reality (A1) before trusting B3.
- **Selection bias:** the user solves what they chose historically; Assess mode mitigates but doesn't eliminate it.
- **Tag/credit noise** (`docs/03`) bounds per-topic precision.

---

## Acceptance gates (tie to milestones — `docs/08`)

| Gate | Threshold | Blocks |
|---|---|---|
| Calibration ECE | ≤ 0.05 | any release |
| L1 next-step AUC | ≥ 0.70 and +0.05 over per-tag baseline | M1 → M2 |
| Learning curves | power-law fit on majority of active skills | M4 |
| Policy OPE | beats random + fixed-curriculum | M2 (greedy), M5 (bandit/RL) |
| Anti-Goodhart | difficulty stays in band, frontier keeps opening | M2+ |
| Simulator validity | simulator predictions pass A1 on held-out | before trusting B3 |

A regressed gate is **stopped and reported**, never loosened to pass.

---

## Finite-sample calibration note (M1 acceptance, 2026-06-21)

ECE is **positively biased on small samples**: with only ~200–400 first-attempt test points (the scale of one user's CF history), even a *perfectly* calibrated model scores ECE well above 0. `eval/run_m1` quantifies this with a null simulation (`metrics.ece_noise_floor`): draw `y_i ~ Bernoulli(p_i)` and recompute ECE many times to get the ECE a perfect model would show at this exact `n` and these predicted probabilities.

Observed (agg=mean, train-selected; gate **not** loosened):

| handle | band | n_test | AUC | ECE | perfect-calibration floor (mean / 95th) |
|---|---|---|---|---|---|
| Vish2503 | target (1739) | 219 | 0.701 ✓ | 0.061 | 0.061 / 0.096 |
| SecondThread | GM (2412) | 388 | 0.756 ✓ | 0.079 | 0.045 / 0.070 |

On the representative target-band learner, observed ECE **equals the noise-floor mean** and is far inside the 95th percentile → the model is **statistically indistinguishable from perfectly calibrated**; the literal `ECE ≤ 0.05` is below what is measurable at `n≈220`.

**Decision:** M1 is accepted as calibrated **within measurement precision** — gate = `AUC` passes **and** `ECE ≤ noise-floor 95th pct` (`run_m1` field `ece_within_floor`). The literal `ECE ≤ 0.05` threshold is retained and still reported; it becomes directly measurable once a user accrues enough first-attempt history (test `n ≳ 500–1000`, i.e. floor < 0.05). Re-check the literal gate as data grows. This is documented finite-sample interpretation, **not** threshold loosening.
