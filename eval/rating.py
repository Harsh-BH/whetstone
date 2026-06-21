"""Predicted-vs-actual Codeforces rating (docs/07 D) — the only real-world signal.

Ingests `user.rating` history and predicts a scalar rating from the per-tag θ-vector
(goal-weighted mean of μ_tag). The θ→rating map is rough by construction (multi-tag
credit, docs/03) — treat the comparison as suggestive, never causal (n=1).
"""

from recommender.train import goal_weights


def ingest_ratings(conn, client, handle: str) -> int:
    changes = client.user_rating(handle)
    rows = [(handle, ch.contest_id, ch.new_rating, ch.update_time) for ch in changes]
    if rows:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO ratings (user_id, contest_id, new_rating, update_time)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (user_id, contest_id) DO UPDATE SET
                    new_rating=EXCLUDED.new_rating, update_time=EXCLUDED.update_time
                """,
                rows,
            )
    return len(rows)


def predict_rating(conn, model, r_band: float) -> float:
    tags = list(model.tags.keys())
    if not tags:
        return model.prior_mu
    w = goal_weights(conn, tags, r_band)
    total = sum(w.values())
    if total <= 0:  # no catalog frequency signal -> plain mean of μ
        return sum(model.tags[t].mu for t in tags) / len(tags)
    return sum(model.tags[t].mu * w[t] for t in tags) / total


def latest_actual(conn, user_id: str) -> int | None:
    row = conn.execute(
        "SELECT new_rating FROM ratings WHERE user_id = %s ORDER BY update_time DESC LIMIT 1",
        (user_id,),
    ).fetchone()
    return row["new_rating"] if row else None


def predicted_vs_actual(conn, model, user_id: str, r_band: float, tol: float = 300.0) -> dict:
    actual = latest_actual(conn, user_id)
    predicted = predict_rating(conn, model, r_band)
    err = abs(predicted - actual) if actual is not None else None
    return {
        "predicted": predicted,
        "actual": actual,
        "abs_error": err,
        "tracks": err is not None and err <= tol,
    }
