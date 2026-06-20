# 08 — Milestones

Each milestone has an **objective**, a **task checklist**, the **docs that govern it**, and an **acceptance gate** (from `docs/07`). Do them in order. Don't start a milestone without re-reading its governing docs.

---

## M0 — Spike (½ day)
**Objective:** prove CF data flows end to end.
**Governing:** `06`.
- [ ] `cf_client.py`: typed wrappers for `problemset.problems`, `user.status`, `user.rating`, `user.info` with rate-limit + backoff.
- [ ] Pull your full `user.status` + the catalog; sanity-check counts, tag coverage, rating coverage.
- [ ] `normalize.py`: collapse submissions → per-(user,problem) attempt episodes; flag in-contest vs upsolve.
- [ ] Load into a local Postgres via the `06` schema + Alembic migration.

**Gate:** episode counts and per-tag/per-rating distributions look correct against your CF profile.

---

## M1 — IRT baseline + skill radar (weekend)
**Objective:** an accurate, calibrated per-topic skill estimate you can see.
**Governing:** `03` (L1), `02` (P2/P8), `07` (A1/A2).
- [ ] `irt.py`: Rasch outcome model + Bayesian (μ,σ) online update + Fisher info; `θ_eff = min` over tags to start.
- [ ] Cold-start prior from your CF rating.
- [ ] `eval/`: temporal-split next-step AUC + calibration (ECE, reliability diagram).
- [ ] `web/`: skill radar (μ per tag, σ as band).

**Gate:** **ECE ≤ 0.05** and **AUC ≥ 0.70 (+0.05 over per-tag baseline).** Do not proceed if miscalibrated.

---

## M2 — Recommender v1 (greedy, two-mode) + daily set (few days)
**Objective:** a usable daily set you actually solve from.
**Governing:** `04`, `02` (P2/P4/P6), `07` (B1/B2/B4).
- [ ] `assess.py`: max-Fisher-info selection for high-σ topics.
- [ ] `train.py` + `compose.py`: gap-weighted, frontier-restricted, **interleaved** topic selection; desirable-difficulty band; 65/20/15 blend.
- [ ] `prereq_dag.py`: seed the concept DAG; `frontier_only`.
- [ ] Log every served problem with `mode`, `predicted_p`, `propensity` (needed for OPE).
- [ ] `web/`: "Today's set" with predicted P(solve) + why-this-problem.

**Gate:** mode switching behaves; realized success stays in band; greedy beats random + fixed-curriculum on OPE; anti-Goodhart check passes.

---

## M3 — Retention + full dashboard (few days)
**Objective:** it becomes a real daily tool.
**Governing:** `02` (P3/P5), `03` (mastery), `05`.
- [ ] `fsrs.py` + `scheduler.py`: retention state, due-review queue, review trigger at `target_retrievability=0.90`.
- [ ] Mastery criterion (confident posterior + sustained reviews); active-pool management.
- [ ] Dashboard: mastery-over-time, **predicted-vs-actual rating**, topic heatmap, contest-readiness score + "do these 5 → readiness X→Y".
- [ ] **Decision point:** keep Python monolith or carve a Go ingest/serving service (`05`).

**Gate:** review queue surfaces decaying topics; predicted rating tracks actual over ≥1 contest.

---

## M4 — PFA/AFM + DKT (predictive depth) (~1 week)
**Objective:** learning-rate modeling + a sequence model, rigorously compared.
**Governing:** `03` (L2/L3), `07` (A1/A3).
- [ ] `pfa.py`: per-skill learning rates; **learning-curve fits** (power-law check) in `eval/`.
- [ ] `dkt.py`: LSTM (or SAKT) over the interaction stream; cold-start from a public CF dataset, fine-tune on you.
- [ ] Report L1 vs L2 vs L3 next-step AUC honestly.

**Gate:** learning curves fit on the majority of active skills; model comparison table produced (L3 needn't win).

---

## M5 — Bandit / offline-RL recommender + A/B (1–2 weeks)
**Objective:** the research/CV centerpiece — a *learned* policy that provably beats the baseline.
**Governing:** `04` (progression), `07` (B2/B3), `02` (P2 personal optimum).
- [ ] `bandit.py`: contextual bandit (LinUCB/Thompson); reward = realized learning gain − time cost.
- [ ] Personal optimal-difficulty estimator (fit Δμ vs gap; recenter band).
- [ ] Offline RL on logged interactions for long-horizon trajectory optimization.
- [ ] `eval/ope.py` + `eval/sim.py`: OPE + simulated trajectories vs all baselines; **validate the simulator** (A1) first.

**Gate:** bandit/RL beats greedy + random + fixed-curriculum on OPE and simulated time-to-target; report the speedup.

---

## M6 — Claude tutor loop (few days)
**Objective:** retrieval-respecting help on failure.
**Governing:** `02` (P1).
- [ ] Hint ladder (concept → approach → key observation → near-solution), revealed only after a generation attempt.
- [ ] Concept diagnosis from the failed attempt → tag the missed concept → feed back as targeted weakness.
- [ ] Prototype as a Claude artifact first, then wire to the API.

**Gate:** failed-problem post-mortem produces a correct concept tag that updates the model.

---

## M7 — Polish, deploy, write-up (few days)
**Objective:** ship + a defensible results write-up.
- [ ] Deploy (compose; optional small cloud).
- [ ] README results: calibration, AUC table, learning curves, **policy speedup over baseline**, with the n=1 caveats.
- [ ] Demo (radar → daily set → solve → model update → contest-readiness).

**Gate:** every number in the write-up is regenerable via `/eval`.

---

## Suggested execution in Claude Code

- One milestone per branch/worktree; PR-sized commits scoped to the checklist.
- Use **plan mode** before M2, M4, M5 (multi-file, model-touching).
- After any `model/` or `recommender/` change: run `/eval`, paste the gate table into the PR. Red gate → stop and report.
- Keep `config.py` as the single source of pedagogical constants; annotate each with its `docs/02` rule.
