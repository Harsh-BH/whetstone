---
description: Run the full evaluation suite and print the metrics, calibration, learning curves, and the acceptance-gate table.
---

Run `eval/` end to end per `docs/07-evaluation.md` with fixed seeds, then print a report.

Produce:
1. **Knowledge model** — next-step AUC / accuracy / log-loss vs baselines (temporal split, never random); calibration (ECE, reliability diagram, Brier); per-skill learning-curve fits (power-law check) if PFA is built.
2. **Recommender** — mode-switching behavior; offline policy evaluation (replay/IPS) for every implemented policy vs random / fixed-curriculum / difficulty-match; simulated time-to-target trajectories; anti-Goodhart difficulty/coverage drift.
3. **Ablations** — drop spacing / interleaving / exploration / personal-optimum / frontier; report the deltas.
4. **Gate table** — the `docs/07` acceptance gates with PASS/FAIL.

Rules:
- If the simulator is used (B3), first verify it passes next-step prediction (A1) on held-out data; otherwise mark B3 untrustworthy.
- **Never loosen a gate to make it pass.** If a gate is red, print it red, summarize the likely cause, and stop. Do not edit thresholds in `docs/07` or `config.py` to go green.
- Paste this gate table into the PR for any change touching `model/` or `recommender/`.
