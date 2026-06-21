"""M3 acceptance gate (docs/07/08): the review queue surfaces decaying topics, and the
model's predicted rating tracks the user's actual CF rating over ≥1 contest.
"""

from config import settings
from eval import rating as rating_mod
from model.irt import SkillModel
from model.snapshot import fit_user
from retention import scheduler


def run_gate(conn, user_id: str) -> dict:
    from ingest.cf_client import CFClient

    client = CFClient()
    try:
        actual_rating = client.user_info(user_id).rating
        rating_mod.ingest_ratings(conn, client, user_id)
    except Exception:
        actual_rating = None
    finally:
        client.close()
    conn.commit()

    r_band = float(settings.target_rating)  # the system's target R, used to weight prediction
    model = fit_user(conn, user_id, actual_rating)

    scheduler.save_reviews(conn, user_id, scheduler.cold_start_state(conn, user_id))
    conn.commit()
    due = scheduler.due_queue(conn, user_id)
    pva = rating_mod.predicted_vs_actual(conn, model, user_id, r_band)

    gate = {
        "review_queue_surfaces_decay": len(due) >= 1,
        "predicted_tracks_actual": pva["tracks"],
    }
    return {
        "due_count": len(due),
        "due_top": due[:5],
        "pva": pva,
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
    print(f"\n=== M3 gate — user={user_id} ===")
    pva = res["pva"]
    print(
        f"predicted rating: {pva['predicted']:.0f}   actual: {pva['actual']}   "
        f"abs error: {pva['abs_error']:.0f}"
        if pva["abs_error"] is not None
        else "no actual rating"
    )
    print(f"due-review queue: {res['due_count']} concepts decayed below target")
    for d in res["due_top"]:
        print(f"  {d['concept']:<22} R={d['retrievability']:.2f}")
    print("\n--- GATE (docs/07 B-retention + D) ---")
    for k, v in res["gate"].items():
        print(f"  {k:<30}: {'PASS' if v else 'FAIL'}")
    print("M3:", "PASS" if res["pass"] else "BLOCKED")


if __name__ == "__main__":
    main()
