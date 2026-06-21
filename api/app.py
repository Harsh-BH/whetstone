"""FastAPI service: skill radar data + the daily set (docs/05). Single-user.

GET /skills?user=     -> latest per-tag μ/σ snapshot (for the radar)
GET /daily-set?user=  -> today's interleaved two-mode set with why-this-problem

ponytail: /daily-set refits the model on demand (fast at one user's scale). Cache from
the topic_skill snapshot if it ever hurts. Each call logs a served set (for OPE).
"""

import copy
import random
import re

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from eval import rating as rating_mod
from ingest import db
from ingest.cf_client import CFClient
from model import mastery
from model.snapshot import fit_user
from recommender.compose import daily_set
from retention import scheduler

app = FastAPI(title="Whetstone")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _cf_url(pid: str) -> str:
    m = re.match(r"(\d+)(.+)", pid)
    return f"https://codeforces.com/problemset/problem/{m.group(1)}/{m.group(2)}" if m else ""


def _rating(user: str) -> float | None:
    client = CFClient()
    try:
        return client.user_info(user).rating
    except Exception:
        return None
    finally:
        client.close()


@app.get("/skills")
def skills(user: str = ""):
    user = user or settings.cf_handle
    conn = db.connect()
    try:
        rows = conn.execute(
            """
            SELECT tag, mu, sigma FROM topic_skill
            WHERE user_id = %s
              AND snapshot_at = (SELECT max(snapshot_at) FROM topic_skill WHERE user_id = %s)
            ORDER BY tag
            """,
            (user, user),
        ).fetchall()
    finally:
        conn.close()
    return [{"tag": r["tag"], "mu": r["mu"], "sigma": r["sigma"]} for r in rows]


@app.get("/daily-set")
def daily(user: str = "", h: float = 0.0):
    user = user or settings.cf_handle
    h = h or float(settings.weekly_hours)
    rating = _rating(user)
    r_band = float(rating) if rating else float(settings.target_rating)
    conn = db.connect()
    try:
        model = fit_user(conn, user, rating)
        items = daily_set(conn, model, user, r_band=r_band, h_hours=h, rng=random.Random())
    finally:
        conn.close()
    return [
        {
            "pid": it["pid"],
            "url": _cf_url(it["pid"]),
            "b": it["b"],
            "tags": it["tags"],
            "mode": it["mode"],
            "predicted_p": round(it["predicted_p"], 3),
            "why": it["why"],
        }
        for it in items
    ]


def _readiness_score(predicted: float, r_band: float) -> int:
    return round(100 * min(1.0, predicted / r_band))


@app.get("/reviews")
def reviews(user: str = ""):
    user = user or settings.cf_handle
    conn = db.connect()
    try:
        scheduler.save_reviews(conn, user, scheduler.cold_start_state(conn, user))
        conn.commit()
        due = scheduler.due_queue(conn, user)
    finally:
        conn.close()
    return [
        {
            "concept": d["concept"],
            "retrievability": round(d["retrievability"], 3),
            "due_at": d["due_at"].isoformat() if d["due_at"] else None,
        }
        for d in due
    ]


@app.get("/mastery")
def mastery_ep(user: str = ""):
    user = user or settings.cf_handle
    r_band = float(settings.target_rating)
    rating = _rating(user)
    conn = db.connect()
    try:
        model = fit_user(conn, user, rating)
        state = scheduler.cold_start_state(conn, user)
        ms = mastery.mastered_set(model, state, list(model.tags.keys()), r_band)
    finally:
        conn.close()
    return [
        {"tag": t, "mu": round(sk.mu), "sigma": round(sk.sigma), "mastered": t in ms}
        for t, sk in sorted(model.tags.items())
    ]


@app.get("/rating-history")
def rating_history(user: str = ""):
    user = user or settings.cf_handle
    rating = _rating(user)
    conn = db.connect()
    try:
        rows = conn.execute(
            "SELECT new_rating, update_time FROM ratings WHERE user_id=%s ORDER BY update_time",
            (user,),
        ).fetchall()
        model = fit_user(conn, user, rating)
        predicted = rating_mod.predict_rating(conn, model, float(settings.target_rating))
    finally:
        conn.close()
    return {
        "actual": [{"t": r["update_time"], "rating": r["new_rating"]} for r in rows],
        "predicted": round(predicted),
    }


@app.get("/readiness")
def readiness(user: str = "", h: float = 0.0):
    user = user or settings.cf_handle
    h = h or float(settings.weekly_hours)
    r_band = float(settings.target_rating)
    rating = _rating(user)
    conn = db.connect()
    try:
        model = fit_user(conn, user, rating)
        state = scheduler.cold_start_state(conn, user)
        current = rating_mod.predict_rating(conn, model, r_band)
        items = daily_set(conn, model, user, r_band, h, random.Random(), reviews=state)
        projected_model = copy.deepcopy(model)
        for it in items:  # project skill if the recommended set is solved
            projected_model.update(it["b"], it["tags"], 1)
        projected = rating_mod.predict_rating(conn, projected_model, r_band)
    finally:
        conn.close()
    return {
        "readiness": _readiness_score(current, r_band),
        "projected": _readiness_score(projected, r_band),
        "target": settings.target_rating,
        "do_these": [it["pid"] for it in items],
    }
