"""
Order lookup. This module is the integration point for the real order/CRM system.

Every lookup is scoped to the requesting customer (order_id AND user_id), so one
customer can never read another customer's order, and a "not found" answer is the
same whether the order does not exist or belongs to someone else.

Backed by a local SQLite table until a real system is connected. To load demo
data for your own email:  python orders.py seed you@example.com
"""

import os
import re
import sqlite3
import sys

DB_PATH = os.getenv("ORDERS_DB_PATH", "orders.db")

# An order id must contain a digit, so "order never arrived" does not yield "never".
_ID = r"((?=[A-Z0-9-]*\d)[A-Z0-9][A-Z0-9-]{3,19})"
_ORDER_PATTERNS = [
    re.compile(rf"\border\s*(?:id|number|no\.?|num)?\s*[:#]?\s*{_ID}\b", re.IGNORECASE),
    re.compile(rf"#\s*{_ID}\b", re.IGNORECASE),
    re.compile(r"\b(\d{5,12})\b"),
]


def init_orders_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS orders (
            order_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            status TEXT NOT NULL,
            eta TEXT,
            total REAL NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def extract_order_id(*texts: str):
    """Return the first order id found in the given texts (checked in order), or None."""
    for text in texts:
        for pattern in _ORDER_PATTERNS:
            match = pattern.search(text or "")
            if match:
                return match.group(1).upper()
    return None


def get_order(order_id: str, user_id: str):
    """Return the order as a dict, or None if it does not exist or is not this user's."""
    if not order_id or not user_id:
        return None
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT order_id, status, eta, total FROM orders WHERE upper(order_id) = upper(?) AND user_id = ?",
        (order_id.strip(), user_id.strip().lower()),
    ).fetchone()
    conn.close()
    if not row:
        return None
    return {"order_id": row[0], "status": row[1], "eta": row[2], "total": float(row[3])}


def _seed_demo(user_id: str):
    init_orders_db()
    conn = sqlite3.connect(DB_PATH)
    conn.executemany(
        "INSERT OR REPLACE INTO orders (order_id, user_id, status, eta, total) VALUES (?, ?, ?, ?, ?)",
        [
            ("12345", user_id.strip().lower(), "in transit", "in 2 days", 75.0),
            ("20002", user_id.strip().lower(), "delivered", None, 30.0),
        ],
    )
    conn.commit()
    conn.close()


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "seed":
        _seed_demo(sys.argv[2])
        print(f"Seeded demo orders 12345 and 20002 for {sys.argv[2]}")
    else:
        print("usage: python orders.py seed <email>")
