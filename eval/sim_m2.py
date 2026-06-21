"""M2 recommender gate (docs/07 B1/B4 + structural baseline comparison).

Honest scope: realized learning-gain OPE (docs/07 B2) needs a model where practice
improves *true* skill — IRT is a martingale (expected μ-change under its own
predictions is 0), so that requires L2/PFA (M4). At M2 we measure what is real now:
  B1 mode behavior — Assess drives σ below τ in FEWER items than random selection
       (Fisher-information gain; outcome-independent precision update).
  B4 anti-Goodhart — greedy stays in the desirable band and covers multiple frontier
       topics (does not collapse to easy churn).
  baselines — greedy concentrates more "pedagogical value" (gap-weighted, in-band, on
       the frontier) than random and the fixed CF-rating-ladder curriculum.
"""

import math
import random

from config import ASSESS_SIGMA_THRESHOLD, FRONTIER_MARGIN, PRIOR_SIGMA, TRAIN_TARGET_BAND
from model.irt import SkillModel, TagSkill
from recommender import prereq_dag
from recommender.candidates import Candidate


def _mu(model, t: str) -> float:
    sk = model.tags.get(t)
    return sk.mu if sk is not None else model.prior_mu


def _open_tags(model, r_band: float) -> set[str]:
    return prereq_dag.frontier(model, list(model.tags.keys()), r_band, FRONTIER_MARGIN)


def _in_band(p: float) -> bool:
    return TRAIN_TARGET_BAND[0] <= p <= TRAIN_TARGET_BAND[1]


def _cand_gap(model, c: Candidate, open_tags: set[str], r_band: float) -> float:
    gaps = [max(0.0, r_band - _mu(model, t)) for t in c.tags if t in open_tags]
    return max(gaps) if gaps else 0.0


# --- policies: (model, cands, k, rng, r_band) -> list[Candidate] ---


def greedy_select(model, cands, k, rng, r_band):
    open_tags = _open_tags(model, r_band)
    scored = []
    for c in cands:
        if not (set(c.tags) & open_tags):
            continue
        p, _ = model.predict_solve(c.b, c.tags)
        if not _in_band(p):
            continue
        scored.append((_cand_gap(model, c, open_tags, r_band), c))
    scored.sort(key=lambda x: -x[0])
    return [c for _, c in scored[:k]]


def random_select(model, cands, k, rng, r_band):
    return rng.sample(cands, min(k, len(cands)))


def fixed_curriculum_select(model, cands, k, rng, r_band):
    return sorted(cands, key=lambda c: c.b)[:k]  # ascending CF-rating ladder


def difficulty_match_select(model, cands, k, rng, r_band):
    inband = [c for c in cands if _in_band(model.predict_solve(c.b, c.tags)[0])]
    return inband[:k]  # band only, no coverage/gap-weighting


def pedagogical_value(model, selected, r_band) -> float:
    open_tags = _open_tags(model, r_band)
    total = 0.0
    for c in selected:
        p, _ = model.predict_solve(c.b, c.tags)
        if _in_band(p):
            total += _cand_gap(model, c, open_tags, r_band)
    return total


def coverage(selected) -> int:
    return len({t for c in selected for t in c.tags})


def in_band_fraction(model, selected) -> float:
    if not selected:
        return 0.0
    return sum(_in_band(model.predict_solve(c.b, c.tags)[0]) for c in selected) / len(selected)


# --- B1: Assess vs random items-to-converge on a single topic ---


def items_to_converge(
    true_theta: float, s: float, cand_bs: list[float], tau: float, max_items: int, mode: str, rng
) -> int:
    sk = TagSkill(mu=true_theta - 300, sigma=PRIOR_SIGMA)  # start uncertain, biased prior
    for n in range(1, max_items + 1):
        if mode == "assess":
            b = min(cand_bs, key=lambda b: abs(b - sk.mu))  # closest to μ -> max info
        else:
            b = rng.choice(cand_bs)
        p_belief = 1.0 / (1.0 + math.exp(-(sk.mu - b) / s))
        p_true = 1.0 / (1.0 + math.exp(-(true_theta - b) / s))
        y = 1 if rng.random() < p_true else 0
        info = p_belief * (1.0 - p_belief) / (s * s)
        sk.mu += sk.sigma * sk.sigma * (y - p_belief) / s
        sk.sigma = math.sqrt(1.0 / (1.0 / (sk.sigma * sk.sigma) + info))
        if sk.sigma < tau:
            return n
    return max_items


def run_gate(conn, user_id: str, horizon: int = 12) -> dict:
    from ingest.cf_client import CFClient
    from model.snapshot import fit_user
    from recommender.candidates import load_unseen

    client = CFClient()
    try:
        rating = client.user_info(user_id).rating
    except Exception:
        rating = None
    finally:
        client.close()
    r_band = float(rating) if rating else SkillModel().prior_mu
    model = fit_user(conn, user_id, rating)
    cands = load_unseen(conn, user_id)

    rng = random.Random(0)
    pol = {
        "greedy": greedy_select,
        "random": random_select,
        "fixed_curriculum": fixed_curriculum_select,
        "difficulty_match": difficulty_match_select,
    }
    metrics = {}
    for name, fn in pol.items():
        sel = fn(model, cands, horizon, random.Random(0), r_band)
        metrics[name] = {
            "value": pedagogical_value(model, sel, r_band),
            "mean_p": (
                (sum(model.predict_solve(c.b, c.tags)[0] for c in sel) / len(sel)) if sel else 0.0
            ),
            "in_band_frac": in_band_fraction(model, sel),
            "coverage": coverage(sel),
            "n": len(sel),
        }

    # Cap generous enough that Assess can actually reach σ<τ at the fitted s (larger s ->
    # smaller Fisher info per item -> more items needed); random should need many more.
    cap = 600
    cand_bs = [float(b) for b in range(800, 3001, 100)]
    assess_items = items_to_converge(
        r_band, model.s, cand_bs, ASSESS_SIGMA_THRESHOLD, cap, "assess", random.Random(1)
    )
    random_items = items_to_converge(
        r_band, model.s, cand_bs, ASSESS_SIGMA_THRESHOLD, cap, "random", random.Random(1)
    )

    g = metrics["greedy"]
    gate = {
        "beats_random": g["value"] > metrics["random"]["value"],
        "beats_fixed_curriculum": g["value"] > metrics["fixed_curriculum"]["value"],
        "realized_in_band": g["in_band_frac"] >= 0.8,  # B4: stays in the band
        "covers_multiple_topics": g["coverage"] >= 2,  # B4: not single-topic churn
        # B1: Assess must STRICTLY converge in fewer items AND actually reach τ (not cap).
        "assess_converges_faster": assess_items < random_items and assess_items < cap,
    }
    return {
        "r_band": r_band,
        "metrics": metrics,
        "assess_items": assess_items,
        "random_items": random_items,
        "gate": gate,
        "pass": all(gate.values()),
    }


def main(user_id: str = "Vish2503") -> None:
    import os

    from ingest import db

    user_id = os.environ.get("EVAL_HANDLE", user_id)
    conn = db.connect()
    try:
        res = run_gate(conn, user_id)
    finally:
        conn.close()
    print(f"\n=== M2 recommender gate — user={user_id} (R_band={res['r_band']:.0f}) ===")
    print(f"{'policy':<18}{'ped.value':>10}{'mean_P':>9}{'in_band':>9}{'coverage':>10}{'n':>4}")
    for name, m in res["metrics"].items():
        print(
            f"{name:<18}{m['value']:>10.0f}{m['mean_p']:>9.2f}{m['in_band_frac']:>9.2f}"
            f"{m['coverage']:>10}{m['n']:>4}"
        )
    print(
        f"\nAssess vs random items to σ<{ASSESS_SIGMA_THRESHOLD:.0f}: "
        f"{res['assess_items']} vs {res['random_items']}"
    )
    print("\n--- GATE (docs/07 B1/B4 + baselines; learning-gain OPE deferred to M4/M5) ---")
    for k, v in res["gate"].items():
        print(f"  {k:<28}: {'PASS' if v else 'FAIL'}")
    print("M2:", "PASS" if res["pass"] else "BLOCKED")


if __name__ == "__main__":
    main()
