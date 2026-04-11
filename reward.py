"""
reward.py
=========
Dense reward shaping for the SQL Data Quality Agent.
Provides partial credit at every step, not just episode end.
"""

from tasks import clamp_score


DANGEROUS_KEYWORDS = [
    "DROP TABLE",
    "DROP DATABASE",
    "TRUNCATE",
    "DELETE FROM",  # only penalise if no WHERE clause follows
]


def _is_destructive_no_where(sql: str) -> bool:
    """Return True if a DELETE has no WHERE clause (mass delete)."""
    upper = sql.strip().upper()
    if upper.startswith("DELETE FROM") and "WHERE" not in upper:
        return True
    return False


def _has_drop(sql: str) -> bool:
    upper = sql.strip().upper()
    return "DROP TABLE" in upper or "DROP DATABASE" in upper


def compute_reward(
    prev_score: float,
    curr_score: float,
    sql: str,
    action_result: str,
    step: int,
    max_steps: int,
    success_threshold: float = 0.85,
) -> float:
    """
    Compute step reward.

    Components:
    - Progress bonus:  10x the quality score delta (main signal)
    - Milestone bonus: extra reward for large improvements
    - Error penalty:   small penalty for SQL errors
    - Destructive penalty: large penalty for mass DELETE or DROP
    - Efficiency bonus: small bonus for finishing early

    Returns a value strictly in (0, 1) — never exactly 0.0 or 1.0.
    """
    delta = curr_score - prev_score

    # --- Main progress signal ---
    progress = delta * 10.0  # scaled so 0.1 improvement -> +1.0 reward

    # --- Milestone bonus ---
    milestone = 0.0
    if delta >= 0.15:
        milestone = 0.3
    elif delta >= 0.08:
        milestone = 0.15
    elif delta >= 0.03:
        milestone = 0.05

    # --- Penalty for errors ---
    error_penalty = 0.0
    if action_result.startswith("error"):
        error_penalty = -0.05

    # --- Penalty for destructive SQL ---
    destructive_penalty = 0.0
    if _has_drop(sql):
        destructive_penalty = -1.0
    elif _is_destructive_no_where(sql):
        destructive_penalty = -0.5

    # --- Efficiency bonus: reward finishing early (uses task-specific threshold) ---
    efficiency = 0.0
    if curr_score >= success_threshold:
        remaining_steps = max_steps - step
        efficiency = 0.05 * remaining_steps  # small bonus per step saved

    total = progress + milestone + error_penalty + destructive_penalty + efficiency

    # Clamp to (0.01, 0.99) — safe margin from boundaries
    # clamp_score handles all edge cases (NaN, inf, out of range)
    return clamp_score(total)


def score_to_grade(score: float) -> str:
    """Human-readable letter grade for a quality score."""
    if score >= 0.95:
        return "A+"
    elif score >= 0.85:
        return "A"
    elif score >= 0.75:
        return "B"
    elif score >= 0.60:
        return "C"
    elif score >= 0.40:
        return "D"
    else:
        return "F"