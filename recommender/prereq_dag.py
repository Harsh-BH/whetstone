"""Hand-seeded prerequisite DAG over Codeforces tags (docs/02 P6, docs/03).

A topic is on the recommendation frontier only once its prerequisites are
"satisfied". M2 proxy for "satisfied": μ_prereq ≥ R_band − margin (real mastery
criterion replaces this in M3). Tags absent from PREREQS are roots (always open).
"""

# child tag -> list of prerequisite tags. Roots are omitted (empty prereqs).
PREREQS: dict[str, list[str]] = {
    # foundational skills built on the basics
    "binary search": ["sortings", "implementation"],
    "two pointers": ["sortings", "implementation"],
    "dp": ["implementation", "brute force"],
    "dfs and similar": ["implementation"],
    "strings": ["implementation"],
    "bitmasks": ["math", "implementation"],
    "number theory": ["math"],
    "combinatorics": ["math"],
    "data structures": ["implementation", "sortings"],
    "divide and conquer": ["sortings", "binary search"],
    "ternary search": ["binary search"],
    # mid
    "probabilities": ["math", "combinatorics"],
    "dsu": ["data structures"],
    "graphs": ["dfs and similar"],
    "hashing": ["strings", "number theory"],
    "games": ["dp"],
    "matrices": ["math", "dp"],
    "geometry": ["math"],
    "schedules": ["greedy", "sortings"],
    "meet-in-the-middle": ["bitmasks", "brute force"],
    "expression parsing": ["implementation", "strings"],
    "chinese remainder theorem": ["number theory"],
    "interactive": ["binary search"],
    # deeper
    "trees": ["graphs", "dfs and similar"],
    "shortest paths": ["graphs"],
    "2-sat": ["graphs", "dfs and similar"],
    "string suffix structures": ["strings", "hashing"],
    "fft": ["math", "divide and conquer"],
    "flows": ["graphs", "shortest paths"],
    "graph matchings": ["flows"],
}


def frontier(
    model, all_tags: list[str], r_band: float, margin: float, mastered: set[str] | None = None
) -> set[str]:
    """Tags whose every prerequisite is satisfied. Roots (no prereqs) are always open.

    M3: if `mastered` is given, a prereq is satisfied only when it is mastered (the real
    P6 criterion). M2 fallback (mastered=None): the μ ≥ r_band − margin proxy. Reads μ
    without mutating the model (prior for unseen tags)."""

    def mu(tag: str) -> float:
        sk = model.tags.get(tag)
        return sk.mu if sk is not None else model.prior_mu

    threshold = r_band - margin

    def satisfied(prereq: str) -> bool:
        if mastered is not None:
            return prereq in mastered
        return mu(prereq) >= threshold

    return {t for t in all_tags if all(satisfied(p) for p in PREREQS.get(t, []))}
