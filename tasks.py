"""
tasks.py
========
Task registry for the SQL Data Quality Agent.

Each task defines:
  - task_id, name, difficulty, description
  - max_steps
  - db_generator function reference
  - grader: computes a quality score 0.0–1.0 from a live SQLite connection
"""

import math
import sqlite3
from typing import Dict, Any, Callable, Optional
from pydantic import BaseModel, field_validator, model_validator


# ---------------------------------------------------------------------------
# Universal clamping — THE single source of truth for score safety
# ---------------------------------------------------------------------------

_SCORE_LOW = 0.01
_SCORE_HIGH = 0.99


def _safe_float(v: Any) -> float:
    """Convert any value to a safe float. Returns 0.5 for any invalid input."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.5
    if math.isnan(f) or math.isinf(f):
        return 0.5
    return f


def clamp_score(score: Any, low: float = _SCORE_LOW, high: float = _SCORE_HIGH) -> float:
    """Clamp score strictly into (0, 1) exclusive — required by OpenEnv Phase 2 validator.

    Default range: [0.01, 0.99] — safe margin from both boundaries.
    Handles NaN, Inf, None, strings, and floating-point edge cases.
    
    CRITICAL: The validator checks that no score equals exactly 0.0 or 1.0.
    This function guarantees the output is always in [low, high].
    """
    f = _safe_float(score)
    # Clamp to specified range
    result = max(low, min(high, f))
    # Final paranoid guard against floating-point weirdness
    if result <= 0.0 or result >= 1.0:
        return 0.5
    return round(result, 4)


def clamp_ratio(ratio: Any) -> float:
    """Clamp a ratio value strictly into (0, 1) exclusive.
    Uses [0.001, 0.999] to preserve more precision for ratios.
    """
    f = _safe_float(ratio)
    if f <= 0.0:
        return 0.001
    if f >= 1.0:
        return 0.999
    result = max(0.001, min(0.999, f))
    if result <= 0.0 or result >= 1.0:
        return 0.5
    return round(result, 4)


class QualityReport(BaseModel):
    null_ratio: float = 0.01
    duplicate_ratio: float = 0.01
    type_error_ratio: float = 0.01
    constraint_violation_ratio: float = 0.01
    value_error_ratio: float = 0.01
    overall_score: float = 0.5
    details: Dict[str, Any] = {}

    @field_validator("overall_score", mode="before")
    @classmethod
    def _clamp_overall_score(cls, v: Any) -> float:
        """Clamp overall_score strictly into (0, 1) — OpenEnv Phase 2 requirement."""
        return clamp_score(v)

    @field_validator(
        "null_ratio",
        "duplicate_ratio",
        "type_error_ratio",
        "constraint_violation_ratio",
        "value_error_ratio",
        mode="before",
    )
    @classmethod
    def _validate_ratios(cls, v: Any) -> float:
        """Clamp ratio fields — keep in (0, 1) exclusive for safety."""
        return clamp_ratio(v)

    @model_validator(mode="after")
    def _final_check(self):
        """Absolute final guard: overall_score and all ratios strictly in (0, 1)."""
        # Guard overall_score
        s = self.overall_score
        if s is None or (isinstance(s, float) and (math.isnan(s) or math.isinf(s))) or s <= 0.0 or s >= 1.0:
            self.overall_score = 0.5
        # Guard all ratios
        for field_name in ["null_ratio", "duplicate_ratio", "type_error_ratio",
                           "constraint_violation_ratio", "value_error_ratio"]:
            val = getattr(self, field_name)
            if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))) or val <= 0.0 or val >= 1.0:
                setattr(self, field_name, 0.01)
        return self

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Override model_dump to ensure all numeric fields are clamped on serialization."""
        d = super().model_dump(**kwargs)
        # Final safety net before JSON serialization
        d["overall_score"] = clamp_score(d.get("overall_score", 0.5))
        for ratio_field in ["null_ratio", "duplicate_ratio", "type_error_ratio",
                            "constraint_violation_ratio", "value_error_ratio"]:
            d[ratio_field] = clamp_ratio(d.get(ratio_field, 0.01))
        return d


# ---------------------------------------------------------------------------
# Task 1 Grader — Null Patrol
# ---------------------------------------------------------------------------

def grade_null_patrol(conn: sqlite3.Connection) -> QualityReport:
    """Score based on how many email/phone nulls remain."""
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM customers")
    total = cur.fetchone()[0]
    if total == 0:
        return QualityReport(overall_score=0.5)

    cur.execute("SELECT COUNT(*) FROM customers WHERE email IS NULL")
    null_emails = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM customers WHERE phone IS NULL")
    null_phones = cur.fetchone()[0]

    total_nullable_fields = total * 2  # email + phone
    total_nulls = null_emails + null_phones
    null_ratio_raw = total_nulls / total_nullable_fields if total_nullable_fields > 0 else 0.0

    # Scale score: 0% nulls -> 0.85, 100% nulls -> 0.15 (safe range)
    quality = 1.0 - null_ratio_raw
    score = 0.15 + quality * 0.70  # Maps quality [0, 1] -> [0.15, 0.85]

    return QualityReport(
        null_ratio=null_ratio_raw,
        overall_score=score,
        details={
            "total_rows": total,
            "null_emails": null_emails,
            "null_phones": null_phones,
        },
    )


# ---------------------------------------------------------------------------
# Task 2 Grader — Duplicate Destroyer
# ---------------------------------------------------------------------------

def grade_duplicate_destroyer(conn: sqlite3.Connection) -> QualityReport:
    """Score based on how many duplicate order_ids remain."""
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM orders")
    total = cur.fetchone()[0]
    if total == 0:
        return QualityReport(overall_score=0.5)

    # Count rows that are NOT the earliest row for their order_id
    cur.execute("""
        SELECT COUNT(*) FROM orders
        WHERE row_id NOT IN (
            SELECT MIN(row_id) FROM orders GROUP BY order_id
        )
    """)
    duplicate_count = cur.fetchone()[0]

    duplicate_ratio_raw = duplicate_count / total if total > 0 else 0.0
    # Scale score: 0% dupes -> 0.85, 100% dupes -> 0.15 (safe range)
    quality = 1.0 - duplicate_ratio_raw
    score = 0.15 + quality * 0.70  # Maps quality [0, 1] -> [0.15, 0.85]

    return QualityReport(
        duplicate_ratio=duplicate_ratio_raw,
        overall_score=score,
        details={
            "total_rows": total,
            "duplicate_rows": duplicate_count,
        },
    )


# ---------------------------------------------------------------------------
# Task 3 Grader — Constraint Cascade
# ---------------------------------------------------------------------------

def grade_constraint_cascade(conn: sqlite3.Connection) -> QualityReport:
    """
    Composite score (weighted):
      30% — type errors (price not numeric, e.g. '$12.99')
      30% — FK violations (inventory rows referencing missing product_ids)
      20% — value errors (negative quantities)
      20% — category normalisation (mixed-case like 'electronics', 'BOOKS')
    """
    cur = conn.cursor()

    # --- Type errors in products.price ---
    cur.execute("SELECT COUNT(*) FROM products")
    total_products = cur.fetchone()[0] or 1

    cur.execute("SELECT price FROM products")
    prices = [row[0] for row in cur.fetchall()]
    type_errors = 0
    for p in prices:
        try:
            float(str(p).strip())
        except (ValueError, TypeError):
            type_errors += 1

    type_error_ratio_raw = type_errors / total_products if total_products > 0 else 0.0

    # --- FK violations in inventory ---
    cur.execute("SELECT COUNT(*) FROM inventory")
    total_inv = cur.fetchone()[0] or 1

    cur.execute("""
        SELECT COUNT(*) FROM inventory
        WHERE product_id NOT IN (SELECT product_id FROM products)
    """)
    fk_violations = cur.fetchone()[0]
    fk_ratio_raw = fk_violations / total_inv if total_inv > 0 else 0.0

    # --- Negative quantities ---
    cur.execute("SELECT COUNT(*) FROM inventory WHERE quantity < 0")
    neg_qty = cur.fetchone()[0]
    neg_ratio_raw = neg_qty / total_inv if total_inv > 0 else 0.0

    # --- Mixed-case category names ---
    VALID_CATEGORIES = {"Electronics", "Clothing", "Food", "Books", "Tools"}
    cur.execute("SELECT category FROM products")
    categories = [row[0] for row in cur.fetchall()]
    case_errors = sum(1 for c in categories if c not in VALID_CATEGORIES)
    case_error_ratio_raw = case_errors / total_products if total_products > 0 else 0.0

    # Weighted composite (4 dimensions) scaled to (0.15, 0.85)
    type_score  = 1.0 - type_error_ratio_raw
    fk_score    = 1.0 - fk_ratio_raw
    neg_score   = 1.0 - neg_ratio_raw
    case_score  = 1.0 - case_error_ratio_raw
    quality = (0.30 * type_score) + (0.30 * fk_score) + (0.20 * neg_score) + (0.20 * case_score)
    overall = 0.15 + quality * 0.70  # Maps quality [0, 1] -> [0.15, 0.85]

    return QualityReport(
        type_error_ratio=type_error_ratio_raw,
        constraint_violation_ratio=fk_ratio_raw,
        value_error_ratio=neg_ratio_raw,
        overall_score=overall,
        details={
            "total_products": total_products,
            "type_errors": type_errors,
            "total_inventory": total_inv,
            "fk_violations": fk_violations,
            "negative_quantity_rows": neg_qty,
            "category_case_errors": case_errors,
        },
    )


# ---------------------------------------------------------------------------
# Task Registry
# ---------------------------------------------------------------------------

class TaskDefinition(BaseModel):
    task_id: str
    name: str
    difficulty: str  # easy | medium | hard
    description: str
    max_steps: int
    success_threshold: float  # score needed to consider done

    class Config:
        arbitrary_types_allowed = True


TASK_REGISTRY: Dict[str, Dict[str, Any]] = {
    "null_patrol": {
        "meta": TaskDefinition(
            task_id="null_patrol",
            name="Null Patrol",
            difficulty="easy",
            description=(
                "A customers table has ~20% of email and phone fields set to NULL. "
                "Your job is to fill them with sensible placeholder values so the "
                "null ratio drops below 5%."
            ),
            max_steps=15,
            success_threshold=0.85,
        ),
        "grader": grade_null_patrol,
        "db_tables": ["customers"],
        "hints": [
            "Inspect NULLs: SELECT COUNT(*) FROM customers WHERE email IS NULL OR phone IS NULL",
            "UPDATE rows where email is NULL — replace with a placeholder string",
            "UPDATE rows where phone is NULL — replace with a placeholder string",
        ],
    },
    "duplicate_destroyer": {
        "meta": TaskDefinition(
            task_id="duplicate_destroyer",
            name="Duplicate Destroyer",
            difficulty="medium",
            description=(
                "An orders table has ~15% duplicate rows (same order_id with different "
                "inserted_at timestamps). Identify and remove the duplicates, keeping "
                "the earliest entry for each order_id."
            ),
            max_steps=20,
            success_threshold=0.85,
        ),
        "grader": grade_duplicate_destroyer,
        "db_tables": ["orders"],
        "hints": [
            "First inspect: SELECT order_id, COUNT(*) as cnt FROM orders GROUP BY order_id HAVING cnt > 1",
            "You need to keep the earliest row per order_id and delete the rest",
            "Hint: use MIN(row_id) grouped by order_id to identify which rows to keep",
        ],
    },
    "constraint_cascade": {
        "meta": TaskDefinition(
            task_id="constraint_cascade",
            name="Constraint Cascade",
            difficulty="hard",
            description=(
                "A products + inventory database has FOUR categories of issues: "
                "(1) ~20% of product prices are stored as strings like '$12.99' instead of numeric values, "
                "(2) ~15% of inventory rows reference non-existent product_ids (FK violations), "
                "(3) ~10% of inventory quantities are negative, "
                "(4) ~25% of product category names have inconsistent casing (e.g. 'electronics', 'BOOKS'). "
                "Fix all four categories to reach a composite quality score >= 0.80."
            ),
            max_steps=30,
            success_threshold=0.80,
        ),
        "grader": grade_constraint_cascade,
        "db_tables": ["products", "inventory"],
        "hints": [
            "Inspect issues: SELECT DISTINCT category FROM products; SELECT price FROM products WHERE price LIKE '$%'",
            "Check inventory: SELECT COUNT(*) FROM inventory WHERE product_id NOT IN (SELECT product_id FROM products)",
            "Valid categories are: Electronics, Clothing, Food, Books, Tools (title-case exactly)",
            "Quantities should be non-negative; prices should be castable to float",
        ],
    },
}


def get_task(task_id: str) -> Optional[Dict[str, Any]]:
    return TASK_REGISTRY.get(task_id)


def list_tasks():
    return [v["meta"].model_dump() for v in TASK_REGISTRY.values()]