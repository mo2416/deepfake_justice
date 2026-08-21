# auth.py — Authentication & Authorization for Deepfake Detection UI
# Storage: SQLite (stdlib) · Hashing: PBKDF2-HMAC-SHA256 + salt

import sqlite3
import hashlib
import secrets
from datetime import datetime

DB_PATH = "deepfake.db"


# ─────────────────────────────────────────────────────────────────────────────
# DB Initialisation
# ─────────────────────────────────────────────────────────────────────────────

def init_db() -> None:
    """Create the authentication and analysis tables."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT    UNIQUE NOT NULL,
            email         TEXT    UNIQUE NOT NULL,
            password_hash TEXT    NOT NULL,
            salt          TEXT    NOT NULL,
            role          TEXT    DEFAULT 'analyst',
            created_at    TEXT    NOT NULL,
            last_login    TEXT,
            is_active     INTEGER DEFAULT 1
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL,
            filename    TEXT    NOT NULL,
            file_type   TEXT    NOT NULL,
            verdict     TEXT    NOT NULL,
            score       REAL    NOT NULL,
            confidence  REAL    NOT NULL,
            sha256      TEXT    NOT NULL,
            model_name  TEXT    NOT NULL,
            elapsed_ms  REAL    NOT NULL,
            analyzed_at TEXT    NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    conn.commit()


    conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _hash_password(password: str, salt: str) -> str:
    key = hashlib.pbkdf2_hmac(
        "sha256", password.encode(), salt.encode(), 200_000
    )
    return key.hex()


def _create_user(conn: sqlite3.Connection, username: str, email: str,
                 password: str, role: str = "analyst") -> bool:
    salt = secrets.token_hex(32)
    ph = _hash_password(password, salt)
    try:
        conn.execute(
            "INSERT INTO users (username,email,password_hash,salt,role,created_at) "
            "VALUES (?,?,?,?,?,?)",
            (username, email, ph, salt, role, datetime.now().isoformat()),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Public auth API
# ─────────────────────────────────────────────────────────────────────────────

def authenticate(username: str, password: str) -> dict | None:
    """Verify credentials. Returns user dict or None."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT id,username,email,password_hash,salt,role,is_active "
        "FROM users WHERE username=?",
        (username,),
    )
    row = c.fetchone()
    if not row:
        conn.close()
        return None
    uid, uname, email, ph, salt, role, active = row
    if not active or _hash_password(password, salt) != ph:
        conn.close()
        return None
    conn.execute(
        "UPDATE users SET last_login=? WHERE id=?",
        (datetime.now().isoformat(), uid),
    )
    conn.commit()
    conn.close()
    return {"id": uid, "username": uname, "email": email, "role": role}


def register_user(username: str, email: str, password: str) -> tuple[bool, str]:
    """Register a new analyst account."""
    username = username.strip()
    email = email.strip().lower()
    if len(username) < 3:
        return False, "Username must be at least 3 characters."
    if len(password) < 8:
        return False, "Password must be at least 8 characters."
    if not any(c.isupper() for c in password):
        return False, "Password must contain at least one uppercase letter."
    if not any(c.isdigit() for c in password):
        return False, "Password must contain at least one digit."
    if "@" not in email or "." not in email:
        return False, "Invalid email address."
    conn = sqlite3.connect(DB_PATH)
    ok = _create_user(conn, username, email, password, "analyst")
    conn.close()
    return (True, "Account created! You can now sign in.") if ok \
        else (False, "Username or email already taken.")


# ─────────────────────────────────────────────────────────────────────────────
# User management (admin)
# ─────────────────────────────────────────────────────────────────────────────

def get_all_users() -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT id,username,email,role,created_at,last_login,is_active "
        "FROM users ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    keys = ["id", "username", "email", "role", "created_at", "last_login", "is_active"]
    return [dict(zip(keys, r)) for r in rows]


def toggle_user_status(user_id: int) -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE users SET is_active=1-is_active WHERE id=?", (user_id,))
    conn.commit()
    conn.close()


def change_user_role(user_id: int, role: str) -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE users SET role=? WHERE id=?", (role, user_id))
    conn.commit()
    conn.close()


def delete_user(user_id: int) -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM users WHERE id=? AND username!='admin'", (user_id,))
    conn.commit()
    conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# Analysis history
# ─────────────────────────────────────────────────────────────────────────────

def save_analysis(user_id: int, filename: str, file_type: str,
                  result: dict, sha256: str) -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO analyses "
        "(user_id,filename,file_type,verdict,score,confidence,sha256,model_name,elapsed_ms,analyzed_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)",
        (user_id, filename, file_type, result["verdict"], result["score"],
         result["confidence"], sha256, result["model_name"],
         result["elapsed_ms"], datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


def get_user_analyses(user_id: int, limit: int = 30) -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT filename,file_type,verdict,score,confidence,sha256,"
        "model_name,elapsed_ms,analyzed_at "
        "FROM analyses WHERE user_id=? ORDER BY analyzed_at DESC LIMIT ?",
        (user_id, limit),
    ).fetchall()
    conn.close()
    keys = ["filename","file_type","verdict","score","confidence",
            "sha256","model_name","elapsed_ms","analyzed_at"]
    return [dict(zip(keys, r)) for r in rows]


def get_all_analyses(limit: int = 100) -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT u.username,a.filename,a.file_type,a.verdict,a.score,"
        "a.confidence,a.sha256,a.model_name,a.elapsed_ms,a.analyzed_at "
        "FROM analyses a JOIN users u ON a.user_id=u.id "
        "ORDER BY a.analyzed_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    keys = ["username","filename","file_type","verdict","score",
            "confidence","sha256","model_name","elapsed_ms","analyzed_at"]
    return [dict(zip(keys, r)) for r in rows]


def get_stats() -> dict:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    total_users    = c.execute("SELECT COUNT(*) FROM users WHERE is_active=1").fetchone()[0]
    total_analyses = c.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]
    total_fakes    = c.execute("SELECT COUNT(*) FROM analyses WHERE verdict='FAKE'").fetchone()[0]
    total_real     = c.execute("SELECT COUNT(*) FROM analyses WHERE verdict='REAL'").fetchone()[0]
    conn.close()
    return {
        "total_users":    total_users,
        "total_analyses": total_analyses,
        "total_fakes":    total_fakes,
        "total_real":     total_real,
        "detection_rate": round(total_fakes / total_analyses * 100, 1)
                          if total_analyses else 0.0,
    }
