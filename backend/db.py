import json, sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "interview_twin.db"
SCHEMA = """
CREATE TABLE IF NOT EXISTS attempts(
  id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, uid TEXT NOT NULL,
  topic TEXT, category TEXT, difficulty TEXT, question TEXT, user_answer TEXT, reference TEXT,
  score INTEGER, label TEXT, feedback TEXT, missing TEXT, created_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS ix_att_session ON attempts(session_id);
CREATE TABLE IF NOT EXISTS users(
  id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL COLLATE NOCASE, password_hash TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS mock_sessions(
  id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, category TEXT, topic TEXT, difficulty TEXT, count INTEGER,
  timer_seconds INTEGER, uids TEXT NOT NULL, started_at TEXT NOT NULL, ended_at TEXT);
CREATE TABLE IF NOT EXISTS resumes(
  session_id TEXT PRIMARY KEY, filename TEXT, parsed TEXT, created_at TEXT);
"""


def conn(path=None):
    c = sqlite3.connect(path or DB_PATH)
    c.row_factory = sqlite3.Row
    c.executescript(SCHEMA)
    cols = {r["name"] for r in c.execute("PRAGMA table_info(attempts)")}   # light migration for databases made by older versions
    if "mock_id" not in cols:
        c.execute("ALTER TABLE attempts ADD COLUMN mock_id INTEGER")
    if "covered" not in cols:
        c.execute("ALTER TABLE attempts ADD COLUMN covered TEXT DEFAULT '[]'")
    return c


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save_attempt(c, **k):
    c.execute("""INSERT INTO attempts(session_id,uid,topic,category,difficulty,question,user_answer,reference,
                 score,label,feedback,missing,created_at,mock_id,covered) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
              (k["session_id"], k["uid"], k["topic"], k["category"], k["difficulty"], k["question"], k["user_answer"],
               k["reference"], k["score"], k["label"], json.dumps(k["feedback"]), json.dumps(k["missing"]), now(), k.get("mock_id"), json.dumps(k.get("covered", []))))
    c.commit()


def attempts(c, session_id, limit=500):
    return [dict(r) for r in c.execute("SELECT * FROM attempts WHERE session_id=? ORDER BY id DESC LIMIT ?", (session_id, limit))]


def save_resume(c, session_id, filename, parsed):
    c.execute("INSERT OR REPLACE INTO resumes VALUES(?,?,?,?)", (session_id, filename, json.dumps(parsed), now()))
    c.commit()


def get_resume(c, session_id):
    r = c.execute("SELECT * FROM resumes WHERE session_id=?", (session_id,)).fetchone()
    return json.loads(r["parsed"]) if r else None
