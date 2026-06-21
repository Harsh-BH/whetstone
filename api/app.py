"""FastAPI service: skill radar data + the daily set (docs/05). Single-user.

GET /skills?user=     -> latest per-tag μ/σ snapshot (for the radar)
GET /daily-set?user=  -> today's interleaved two-mode set with why-this-problem

ponytail: /daily-set refits the model on demand (fast at one user's scale). Cache from
the topic_skill snapshot if it ever hurts. Each call logs a served set (for OPE).
"""

import random
import re

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from ingest import db
from ingest.cf_client import CFClient
from model.snapshot import fit_user
from recommender.compose import daily_set

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
