"""
Chat history storage. Separate from LangGraph's own checkpoint DB.
"""

import sqlite3
import uuid
from datetime import datetime

DB_PATH = "chat_history.db"


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
    conn.commit()
    conn.close()


def set_thread_title(thread_id: str, user_id: str, title: str):
    conn = sqlite3.connect(DB_PATH)
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


def delete_thread(thread_id: str):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM messages WHERE thread_id = ?", (thread_id,))
    conn.execute("DELETE FROM threads WHERE thread_id = ?", (thread_id,))
    conn.commit()
    conn.close()


def get_messages(thread_id: str):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT role, content, category, created_at FROM messages WHERE thread_id = ? ORDER BY id ASC",
        (thread_id,),
    ).fetchall()
    conn.close()
    return [{"role": r[0], "content": r[1], "category": r[2], "created_at": r[3]} for r in rows]