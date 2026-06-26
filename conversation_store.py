# conversation_store.py
import sqlite3
from pathlib import Path

DB_PATH = Path("conversation.db")

def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role      TEXT NOT NULL,
            content   TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_session ON messages(session_id)")
    conn.commit()
    return conn

def save_message(session_id: str, role: str, content: str):
    with _conn() as conn:
        conn.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content)
        )

def get_history(session_id: str, limit: int = 10) -> list[dict]:
    with _conn() as conn:
        rows = conn.execute("""
            SELECT role, content FROM messages
            WHERE session_id = ?
            ORDER BY created_at DESC LIMIT ?
        """, (session_id, limit)).fetchall()
    return [{"role": r, "content": c} for r, c in reversed(rows)]

def list_sessions() -> list[str]:
    with _conn() as conn:
        rows = conn.execute("""
            SELECT session_id FROM messages
            GROUP BY session_id
            ORDER BY MAX(created_at) DESC
        """).fetchall()
    return [r[0] for r in rows]

def delete_session(session_id: str):
    with _conn() as conn:
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))