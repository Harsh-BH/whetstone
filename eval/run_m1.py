"""M1 knowledge-model eval + acceptance gate (docs/07 A1/A2).

Temporal split, fit s on train, replay train, prequential test (predict then
update). Compare to global + per-tag baselines. Gate: AUC>=0.70, AUC>=per_tag+0.05,
ECE<=0.05. Deterministic; seeds fixed.
"""

from scipy.optimize import minimize_scalar

from eval import metrics
from eval.dataset import Record, load_records, temporal_split
from model.irt import SkillModel

# Bounds for the fitted logistic temperature s (CF-rating units). Wide enough that
# the optimum is interior, not clamped (a too-narrow grid was the M1 ECE bug).
S_BOUNDS = (50.0, 3000.0)


def _replay_train(recs: list[Record], s: float, prior_mu: float) -> SkillModel:
    m = SkillModel(s=s, prior_mu=prior_mu)
    for r in recs:
        m.update(r.b, r.tags, r.y)
    return m


def _prequential(model: SkillModel, recs: list[Record]) -> tuple[list[int], list[float]]:
    ys, ps = [], []
    for r in recs:
        p, _ = model.predict_solve(r.b, r.tags)
        ys.append(r.y)
        ps.append(p)
        model.update(r.b, r.tags, r.y)
    return ys, ps


def fit_s(train: list[Record], prior_mu: float, bounds=S_BOUNDS) -> float:
    """Fit the logistic temperature by minimizing prequential train log-loss (a
    proper scoring rule -> calibration). Continuous (scipy) so s isn't grid-clamped."""

    def loss(s: float) -> float:
        m = SkillModel(s=float(s), prior_mu=prior_mu)
        ys, ps = _prequential(m, train)
        if len(set(ys)) < 2:
            return 1e9
        return metrics.log_loss_(ys, ps)

    res = minimize_scalar(loss, bounds=bounds, method="bounded")
    return float(res.x)


def per_tag_baseline(train: list[Record], test: list[Record]) -> list[float]:
    rate: dict[str, list[int]] = {}
    glob = sum(r.y for r in train) / max(1, len(train))
    for r in train:
        for t in r.tags:
            rate.setdefault(t, []).append(r.y)
    means = {t: sum(v) / len(v) for t, v in rate.items()}
    out = []
    for r in test:
        vals = [means[t] for t in r.tags if t in means]
        out.append(sum(vals) / len(vals) if vals else glob)
    return out


def evaluate_records(recs: list[Record], cf_rating: float | None) -> dict:
    prior_mu = float(cf_rating) if cf_rating else SkillModel().prior_mu
    train, test = temporal_split(recs)
    s = fit_s(train, prior_mu)
    model = _replay_train(train, s, prior_mu)
    ys, ps = _prequential(model, test)
    bl = per_tag_baseline(train, test)
    auc = metrics.auc(ys, ps)
    per_tag_auc = metrics.auc(ys, bl)
    ece = metrics.ece(ys, ps)
    return {
        "n_train": len(train),
        "n_test": len(test),
        "s": s,
        "auc": auc,
        "accuracy": metrics.accuracy(ys, ps),
        "log_loss": metrics.log_loss_(ys, ps),
        "brier": metrics.brier(ys, ps),
        "ece": ece,
        "reliability": metrics.reliability(ys, ps),
        "per_tag_auc": per_tag_auc,
        "gate_auc": (auc >= 0.70) and (auc >= per_tag_auc + 0.05),
        "gate_ece": ece <= 0.05,
    }


def evaluate(conn, user_id: str, cf_rating: float | None = None) -> dict:
    return evaluate_records(load_records(conn, user_id), cf_rating)


def main(user_id: str = "SecondThread") -> None:
    from ingest import db
    from ingest.cf_client import CFClient

    client = CFClient()
    try:
        rating = client.user_info(user_id).rating
    except Exception:
        rating = None
    finally:
        client.close()
    conn = db.connect()
    try:
        res = evaluate(conn, user_id, cf_rating=rating)
    finally:
        conn.close()

    print(f"\n=== M1 knowledge-model eval — user={user_id} (cf_rating={rating}) ===")
    print(f"train/test: {res['n_train']}/{res['n_test']}   fit s={res['s']:.1f}")
    print(f"AUC        : {res['auc']:.3f}   (per-tag baseline {res['per_tag_auc']:.3f})")
    print(f"accuracy   : {res['accuracy']:.3f}")
    print(f"log-loss   : {res['log_loss']:.3f}   Brier {res['brier']:.3f}")
    print(f"ECE        : {res['ece']:.3f}")
    print("\nreliability (conf -> acc, n):")
    for bn in res["reliability"]:
        if bn["n"]:
            print(
                f"  [{bn['lo']:.1f}-{bn['hi']:.1f}] conf={bn['conf']:.2f} acc={bn['acc']:.2f} n={bn['n']}"
            )
    g_auc = "PASS" if res["gate_auc"] else "FAIL"
    g_ece = "PASS" if res["gate_ece"] else "FAIL"
    print("\n--- GATE ---")
    print(f"AUC>=0.70 & +0.05 over baseline : {g_auc}")
    print(f"ECE<=0.05                        : {g_ece}")
    print("M1 -> M2:", "PASS" if (res["gate_auc"] and res["gate_ece"]) else "BLOCKED")


if __name__ == "__main__":
    main()
