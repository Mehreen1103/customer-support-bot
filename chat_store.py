"""
Chat history storage. Separate from LangGraph's own checkpoint DB.
"""

import sqlite3
import uuid
from datetime import datetime, timedelta, timezone

DB_PATH = "chat_history.db"

# A claimed review that was never completed (e.g. the reviewer's app crashed) can be re-claimed after this.
REVIEW_CLAIM_TIMEOUT_SECONDS = 300


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            thread_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            category TEXT,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS threads (
            thread_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            title TEXT NOT NULL
        )
    """)
    # Drafts waiting for a human reviewer. Kept in the DB (not st.session_state) so a page
    # refresh or restart cannot lose them. status: pending | processing | approved | rejected | cancelled
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pending_reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            draft TEXT NOT NULL,
            refund_amount REAL NOT NULL DEFAULT 0,
            reason TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'pending',
            created_at TEXT NOT NULL,
            claimed_at TEXT,
            decided_by TEXT,
            decided_at TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_reviews_thread ON pending_reviews (thread_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_reviews_status ON pending_reviews (status)")
    conn.commit()
    conn.close()


def _thread_owner(conn, thread_id: str):
    """Return the owning user_id of a thread, or None if the thread does not exist yet."""
    row = conn.execute("SELECT user_id FROM threads WHERE thread_id = ?", (thread_id,)).fetchone()
    if row:
        return row[0]
    row = conn.execute("SELECT user_id FROM messages WHERE thread_id = ? LIMIT 1", (thread_id,)).fetchone()
    return row[0] if row else None


def user_can_access_thread(thread_id: str, user_id: str) -> bool:
    """True if the thread is new (unclaimed) or owned by this user."""
    conn = sqlite3.connect(DB_PATH)
    owner = _thread_owner(conn, thread_id)
    conn.close()
    return owner is None or owner == user_id


def set_thread_title(thread_id: str, user_id: str, title: str):
    conn = sqlite3.connect(DB_PATH)
    owner = _thread_owner(conn, thread_id)
    if owner is not None and owner != user_id:
        conn.close()
        raise PermissionError("Thread belongs to another user")
    conn.execute(
        "INSERT OR REPLACE INTO threads (thread_id, user_id, title) VALUES (?, ?, ?)",
        (thread_id, user_id, title),
    )
    conn.commit()
    conn.close()


def new_thread_id() -> str:
    return str(uuid.uuid4())


def save_message(user_id: str, thread_id: str, role: str, content: str, category: str = None):
    conn = sqlite3.connect(DB_PATH)
    owner = _thread_owner(conn, thread_id)
    if owner is not None and owner != user_id:
        conn.close()
        raise PermissionError("Thread belongs to another user")
    conn.execute(
        "INSERT INTO messages (user_id, thread_id, role, content, category, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (user_id, thread_id, role, content, category, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


def get_threads(user_id: str):
    """Return list of (thread_id, title, last_active) for a user, most recently active first."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        """
        SELECT m.thread_id,
               (SELECT title FROM threads t WHERE t.thread_id = m.thread_id) as saved_title,
               (SELECT content FROM messages m2 WHERE m2.thread_id = m.thread_id AND m2.role = 'user' ORDER BY m2.id ASC LIMIT 1) as first_msg,
               MAX(m.created_at) as last_active
        FROM messages m
        WHERE m.user_id = ?
        GROUP BY m.thread_id
        ORDER BY last_active DESC
        """,
        (user_id,),
    ).fetchall()
    conn.close()
    threads = []
    for thread_id, saved_title, first_msg, last_active in rows:
        if saved_title:
            title = saved_title
        elif first_msg:
            title = (first_msg[:30] + "...") if len(first_msg) > 30 else first_msg
        else:
            title = "New conversation"
        threads.append((thread_id, title, last_active))
    return threads


def delete_thread(thread_id: str, user_id: str):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM messages WHERE thread_id = ? AND user_id = ?", (thread_id, user_id))
    conn.execute("DELETE FROM threads WHERE thread_id = ? AND user_id = ?", (thread_id, user_id))
    # A deleted conversation must not receive a reply later.
    conn.execute(
        "UPDATE pending_reviews SET status = 'cancelled' "
        "WHERE thread_id = ? AND user_id = ? AND status IN ('pending', 'processing')",
        (thread_id, user_id),
    )
    conn.commit()
    conn.close()


def get_messages(thread_id: str, user_id: str):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT role, content, category, created_at FROM messages WHERE thread_id = ? AND user_id = ? ORDER BY id ASC",
        (thread_id, user_id),
    ).fetchall()
    conn.close()
    return [{"role": r[0], "content": r[1], "category": r[2], "created_at": r[3]} for r in rows]


# ---------- Human review queue ----------

def _now() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat()


def _review_row(row):
    keys = ("id", "thread_id", "user_id", "draft", "refund_amount", "reason", "status", "created_at")
    return dict(zip(keys, row))


_REVIEW_COLUMNS = "id, thread_id, user_id, draft, refund_amount, reason, status, created_at"


def create_pending_review(user_id: str, thread_id: str, draft: str, refund_amount: float = 0.0, reason: str = "") -> int:
    """Queue a draft for human review. Idempotent: a thread has at most one open review."""
    conn = sqlite3.connect(DB_PATH)
    owner = _thread_owner(conn, thread_id)
    if owner is not None and owner != user_id:
        conn.close()
        raise PermissionError("Thread belongs to another user")
    row = conn.execute(
        "SELECT id FROM pending_reviews WHERE thread_id = ? AND status IN ('pending', 'processing')",
        (thread_id,),
    ).fetchone()
    if row:
        conn.close()
        return row[0]
    cur = conn.execute(
        "INSERT INTO pending_reviews (thread_id, user_id, draft, refund_amount, reason, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (thread_id, user_id, draft, float(refund_amount or 0.0), reason or "", _now()),
    )
    conn.commit()
    review_id = cur.lastrowid
    conn.close()
    return review_id


def get_open_review(thread_id: str, user_id: str):
    """The customer-side view: is a reply to this user's thread still awaiting review?"""
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        f"SELECT {_REVIEW_COLUMNS} FROM pending_reviews "
        "WHERE thread_id = ? AND user_id = ? AND status IN ('pending', 'processing') ORDER BY id DESC LIMIT 1",
        (thread_id, user_id),
    ).fetchone()
    conn.close()
    return _review_row(row) if row else None


def list_open_reviews():
    """Reviewer-side queue, oldest first. For the reviewer console only; it is not scoped to a customer."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        f"SELECT {_REVIEW_COLUMNS} FROM pending_reviews WHERE status IN ('pending', 'processing') ORDER BY id ASC"
    ).fetchall()
    conn.close()
    return [_review_row(r) for r in rows]


def claim_review(review_id: int) -> bool:
    """Atomically take a review so two reviewers (or a double click) cannot both act on it."""
    cutoff = (datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(seconds=REVIEW_CLAIM_TIMEOUT_SECONDS)).isoformat()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute(
        "UPDATE pending_reviews SET status = 'processing', claimed_at = ? "
        "WHERE id = ? AND (status = 'pending' OR (status = 'processing' AND claimed_at < ?))",
        (_now(), review_id, cutoff),
    )
    conn.commit()
    claimed = cur.rowcount == 1
    conn.close()
    return claimed


def complete_review(review_id: int, decision: str, reviewer: str):
    status = "approved" if decision == "approve" else "rejected"
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "UPDATE pending_reviews SET status = ?, decided_by = ?, decided_at = ? WHERE id = ? AND status = 'processing'",
        (status, reviewer, _now(), review_id),
    )
    conn.commit()
    conn.close()


def release_review(review_id: int):
    """Put a claimed review back in the queue (the decision could not be processed)."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "UPDATE pending_reviews SET status = 'pending', claimed_at = NULL WHERE id = ? AND status = 'processing'",
        (review_id,),
    )
    conn.commit()
    conn.close()
