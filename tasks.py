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

import sqlite3
from typing import Dict, Any, Callable, Optional
from pydantic import BaseModel, field_validator, model_validator


def clamp_score(score: float, low: float = 0.01, high: float = 0.99) -> float:
    """Clamp score to (0, 1) exclusive — required by OpenEnv Phase 2 validator.
    
    Use ONLY for overall_score and reward values.
    Do NOT use for ratio fields (null_ratio, duplicate_ratio, etc.) which can be 0.0-1.0.
    """
    try:
        score = float(score)
    except (TypeError, ValueError):
        score = 0.01
    clamped = max(low, min(high, score))
    # After rounding that may happen elsewhere, ensure we never hit exact 0 or 1
    if clamped <= 0.0:
        clamped = low
    if clamped >= 1.0:
        clamped = high
    return clamped


class QualityReport(BaseModel):
    null_ratio: float = 0.0
    duplicate_ratio: float = 0.0
    type_error_ratio: float = 0.0
    constraint_violation_ratio: float = 0.0
    value_error_ratio: float = 0.0
    overall_score: float = 0.01
    details: Dict[str, Any] = {}

    @field_validator("overall_score", mode="before")
    @classmethod
    def _clamp_overall_score(cls, v: float) -> float:
        """Clamp overall_score to (0, 1) exclusive — OpenEnv Phase 2 requirement."""
        try:
            v = float(v)
        except (TypeError, ValueError):
            v = 0.01
        clamped = max(0.01, min(0.99, v))
        # Double-check after potential float weirdness
        if clamped <= 0.0:
            clamped = 0.01
        if clamped >= 1.0:
            clamped = 0.99
        return clamped
    
    @field_validator(
        "null_ratio",
        "duplicate_ratio", 
        "type_error_ratio",
        "constraint_violation_ratio",
        "value_error_ratio",
        mode="before",
    )
    @classmethod
    def _validate_ratios(cls, v: float) -> float:
        """Clamp ratio fields to [0, 1] inclusive — ratios CAN be exactly 0 or 1."""
        try:
            v = float(v)
        except (TypeError, ValueError):
            v = 0.0
        # Ratios can be 0.0 or 1.0, just ensure in valid range
        return max(0.0, min(1.0, v))
    
    @model_validator(mode="after")
    def _final_check(self):
        """Final safety check: overall_score must NEVER be exactly 0.0 or 1.0."""
        if self.overall_score is not None:
            if self.overall_score <= 0.0:
                self.overall_score = 0.01
            elif self.overall_score >= 1.0:
                self.overall_score = 0.99
        return self


# ---------------------------------------------------------------------------
# Task 1 Grader — Null Patrol
# ---------------------------------------------------------------------------

def grade_null_patrol(conn: sqlite3.Connection) -> QualityReport:
    """Score based on how many email/phone nulls remain."""
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM customers")
    total = cur.fetchone()[0]
    if total == 0:
        return QualityReport(overall_score=0.01)

    cur.execute("SELECT COUNT(*) FROM customers WHERE email IS NULL")
    null_emails = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM customers WHERE phone IS NULL")
    null_phones = cur.fetchone()[0]

    total_nullable_fields = total * 2  # email + phone
    total_nulls = null_emails + null_phones
    null_ratio = total_nulls / total_nullable_fields if total_nullable_fields > 0 else 0.0

    # Scale score directly to (0.01, 0.99) to avoid exact 0 or 1
    score = 0.01 + (1.0 - null_ratio) * 0.98

    return QualityReport(
        null_ratio=round(null_ratio, 4),
        overall_score=clamp_score(round(score, 4)),
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
        return QualityReport(overall_score=0.01)

    # Count rows that are NOT the earliest row for their order_id
    cur.execute("""
        SELECT COUNT(*) FROM orders
        WHERE row_id NOT IN (
            SELECT MIN(row_id) FROM orders GROUP BY order_id
        )
    """)
    duplicate_count = cur.fetchone()[0]

    duplicate_ratio = duplicate_count / total if total > 0 else 0.0
    # Scale score directly to (0.01, 0.99) to avoid exact 0 or 1
    score = 0.01 + (1.0 - duplicate_ratio) * 0.98

    return QualityReport(
        duplicate_ratio=round(duplicate_ratio, 4),
        overall_score=clamp_score(round(score, 4)),
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

    type_error_ratio = type_errors / total_products if total_products > 0 else 0.0

    # --- FK violations in inventory ---
    cur.execute("SELECT COUNT(*) FROM inventory")
    total_inv = cur.fetchone()[0] or 1

    cur.execute("""
        SELECT COUNT(*) FROM inventory
        WHERE product_id NOT IN (SELECT product_id FROM products)
    """)
    fk_violations = cur.fetchone()[0]
    fk_ratio = fk_violations / total_inv if total_inv > 0 else 0.0

    # --- Negative quantities ---
    cur.execute("SELECT COUNT(*) FROM inventory WHERE quantity < 0")
    neg_qty = cur.fetchone()[0]
    neg_ratio = neg_qty / total_inv if total_inv > 0 else 0.0

    # --- Mixed-case category names ---
    VALID_CATEGORIES = {"Electronics", "Clothing", "Food", "Books", "Tools"}
    cur.execute("SELECT category FROM products")
    categories = [row[0] for row in cur.fetchall()]
    case_errors = sum(1 for c in categories if c not in VALID_CATEGORIES)
    case_error_ratio = case_errors / total_products if total_products > 0 else 0.0

    # Weighted composite (4 dimensions) scaled to (0.01, 0.99)
    type_score  = 1.0 - type_error_ratio
    fk_score    = 1.0 - fk_ratio
    neg_score   = 1.0 - neg_ratio
    case_score  = 1.0 - case_error_ratio
    overall_raw = (0.30 * type_score) + (0.30 * fk_score) + (0.20 * neg_score) + (0.20 * case_score)
    overall = 0.01 + overall_raw * 0.98  # Scale to (0.01, 0.99)

    return QualityReport(
        type_error_ratio=round(type_error_ratio, 4),
        constraint_violation_ratio=round(fk_ratio, 4),
        value_error_ratio=round(neg_ratio, 4),
        overall_score=clamp_score(round(overall, 4)),
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
            success_threshold=0.95,
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
            success_threshold=0.95,
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
                "Fix all four categories to reach a composite quality score >= 0.90."
            ),
            max_steps=30,
            success_threshold=0.90,
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