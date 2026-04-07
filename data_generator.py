"""
data_generator.py
=================
Generates synthetic "dirty" SQLite databases for each task.
Each generator returns a seeded, reproducible in-memory SQLite connection
with deliberate data quality issues baked in.
"""

import sqlite3
import random
import string
from typing import Optional


def _rand_email(rng: random.Random) -> str:
    name = "".join(rng.choices(string.ascii_lowercase, k=6))
    domain = rng.choice(["gmail.com", "yahoo.com", "outlook.com", "company.org"])
    return f"{name}@{domain}"


def _rand_phone(rng: random.Random) -> str:
    return f"{rng.randint(100,999)}-{rng.randint(100,999)}-{rng.randint(1000,9999)}"


def _rand_name(rng: random.Random) -> str:
    first = rng.choice(["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Henry"])
    last = rng.choice(["Smith", "Jones", "Williams", "Brown", "Davis", "Miller", "Wilson"])
    return f"{first} {last}"


# ---------------------------------------------------------------------------
# Task 1: Null Patrol — customers table with ~20% NULL emails/phones
# ---------------------------------------------------------------------------

def generate_null_patrol_db(seed: int = 42) -> sqlite3.Connection:
    """50-row customers table; ~20% email and phone values are NULL."""
    rng = random.Random(seed)
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE customers (
            id       INTEGER PRIMARY KEY,
            name     TEXT    NOT NULL,
            email    TEXT,
            phone    TEXT,
            city     TEXT,
            created  TEXT
        )
    """)

    cities = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad", "Pune", "Kolkata"]
    rows = []
    for i in range(1, 51):
        email = None if rng.random() < 0.20 else _rand_email(rng)
        phone = None if rng.random() < 0.20 else _rand_phone(rng)
        rows.append((
            i,
            _rand_name(rng),
            email,
            phone,
            rng.choice(cities),
            f"2024-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}",
        ))

    cur.executemany(
        "INSERT INTO customers VALUES (?, ?, ?, ?, ?, ?)", rows
    )
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Task 2: Duplicate Destroyer — orders table with ~15% duplicates
# ---------------------------------------------------------------------------

def generate_duplicate_destroyer_db(seed: int = 42) -> sqlite3.Connection:
    """200-row orders table; ~15% are duplicate order records (same order_id)."""
    rng = random.Random(seed)
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE orders (
            row_id      INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id    TEXT    NOT NULL,
            customer_id INTEGER NOT NULL,
            amount      REAL    NOT NULL,
            status      TEXT    NOT NULL,
            inserted_at TEXT    NOT NULL
        )
    """)

    statuses = ["pending", "shipped", "delivered", "cancelled"]
    base_orders = []
    for i in range(1, 171):  # 170 unique orders
        base_orders.append({
            "order_id": f"ORD-{i:04d}",
            "customer_id": rng.randint(1, 50),
            "amount": round(rng.uniform(10, 1000), 2),
            "status": rng.choice(statuses),
            "inserted_at": f"2024-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}T{rng.randint(0,23):02d}:{rng.randint(0,59):02d}:00",
        })

    all_rows = list(base_orders)
    # Add ~30 duplicates (same order_id, later inserted_at)
    for _ in range(30):
        orig = rng.choice(base_orders)
        dup = dict(orig)
        dup["inserted_at"] = f"2024-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}T{rng.randint(0,23):02d}:{rng.randint(0,59):02d}:00"
        all_rows.append(dup)

    rng.shuffle(all_rows)
    cur.executemany(
        "INSERT INTO orders (order_id, customer_id, amount, status, inserted_at) VALUES (:order_id, :customer_id, :amount, :status, :inserted_at)",
        all_rows,
    )
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Task 3: Constraint Cascade — products + inventory with 4 data quality issues
# ---------------------------------------------------------------------------

def generate_constraint_cascade_db(seed: int = 42) -> sqlite3.Connection:
    """
    products + inventory tables.
    Issues:
      - ~20% prices stored as strings like '$12.99' instead of numeric
      - ~15% inventory rows reference non-existent product_ids (FK violations)
      - ~10% inventory quantities are negative
      - ~25% product category names have wrong casing (e.g. 'electronics', 'BOOKS')
    """
    rng = random.Random(seed)
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE products (
            product_id   INTEGER PRIMARY KEY,
            name         TEXT    NOT NULL,
            category     TEXT    NOT NULL,
            price        TEXT    NOT NULL,   -- intentionally TEXT to store dirty data
            active       INTEGER NOT NULL DEFAULT 1
        )
    """)

    cur.execute("""
        CREATE TABLE inventory (
            inv_id      INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id  INTEGER NOT NULL,
            warehouse   TEXT    NOT NULL,
            quantity    INTEGER NOT NULL,
            updated_at  TEXT    NOT NULL
        )
    """)

    categories = ["Electronics", "Clothing", "Food", "Books", "Tools"]
    # Wrong-case variants for the 4th dirty dimension
    dirty_case = ["electronics", "CLOTHING", "food", "BOOKS", "tools",
                  "Electronics".upper(), "cLoThInG", "FOOD", "books", "TOOLS"]

    product_names = [
        "Laptop", "T-Shirt", "Rice Bag", "Python Book", "Hammer",
        "Headphones", "Jeans", "Coffee", "Novel", "Screwdriver",
        "Tablet", "Jacket", "Tea", "Textbook", "Wrench",
        "Monitor", "Shoes", "Sugar", "Comics", "Drill",
        "Keyboard", "Socks", "Pasta", "Magazine", "Pliers",
    ]

    # 25 products
    for i in range(1, 26):
        name = product_names[(i - 1) % len(product_names)]
        price_val = round(rng.uniform(5, 500), 2)
        # ~20% prices stored as dirty strings
        if rng.random() < 0.20:
            price = f"${price_val}"
        else:
            price = str(price_val)
        # ~25% categories with wrong casing
        if rng.random() < 0.25:
            category = rng.choice(dirty_case)
        else:
            category = rng.choice(categories)
        cur.execute(
            "INSERT INTO products VALUES (?, ?, ?, ?, 1)",
            (i, name, category, price),
        )

    warehouses = ["WH-North", "WH-South", "WH-East", "WH-West"]

    # 80 inventory rows
    for j in range(1, 81):
        # ~15% reference non-existent product_ids (26–35)
        if rng.random() < 0.15:
            product_id = rng.randint(26, 35)
        else:
            product_id = rng.randint(1, 25)

        # ~10% negative quantities
        qty = rng.randint(-50, -1) if rng.random() < 0.10 else rng.randint(0, 500)

        cur.execute(
            "INSERT INTO inventory (product_id, warehouse, quantity, updated_at) VALUES (?, ?, ?, ?)",
            (
                product_id,
                rng.choice(warehouses),
                qty,
                f"2024-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}",
            ),
        )

    conn.commit()
    return conn


TASK_DB_GENERATORS = {
    "null_patrol": generate_null_patrol_db,
    "duplicate_destroyer": generate_duplicate_destroyer_db,
    "constraint_cascade": generate_constraint_cascade_db,
}
