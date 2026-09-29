"""SQLite persistence shared by all tools."""
import sqlite3
import threading

from bot import config

_lock = threading.RLock()  # connect() runs under it via db()
_conn: sqlite3.Connection | None = None

SCHEMA = """
CREATE TABLE IF NOT EXISTS events(
  id INTEGER PRIMARY KEY, kind TEXT NOT NULL, title TEXT NOT NULL,
  start TEXT NOT NULL, end TEXT NOT NULL, location TEXT DEFAULT '', notes TEXT DEFAULT '',
  remind_minutes INTEGER DEFAULT 0, reminded INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS tasks(
  id INTEGER PRIMARY KEY, title TEXT NOT NULL, due TEXT DEFAULT '', priority INTEGER DEFAULT 2,
  done INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS notes(
  id INTEGER PRIMARY KEY, title TEXT NOT NULL, body TEXT NOT NULL, tags TEXT DEFAULT '');
CREATE TABLE IF NOT EXISTS mod_log(
  id INTEGER PRIMARY KEY, user TEXT NOT NULL, action TEXT NOT NULL, reason TEXT DEFAULT '',
  at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS lessons(
  id INTEGER PRIMARY KEY, trigger TEXT DEFAULT '', action TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS examples(
  id INTEGER PRIMARY KEY, phrase TEXT NOT NULL, tool TEXT NOT NULL, args TEXT DEFAULT '{}',
  source TEXT DEFAULT 'user', weight INTEGER DEFAULT 1, at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS blocklist(word TEXT PRIMARY KEY);
CREATE TABLE IF NOT EXISTS jobs(
  id INTEGER PRIMARY KEY, goal TEXT NOT NULL, run_at TEXT NOT NULL, done INTEGER DEFAULT 0,
  result TEXT DEFAULT '');
"""


def connect(path: str | None = None) -> sqlite3.Connection:
    global _conn
    with _lock:
        _conn = sqlite3.connect(path or str(config.db_path()), check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.executescript(SCHEMA)
        _conn.commit()
    return _conn


def db() -> sqlite3.Connection:
    return _conn if _conn is not None else connect()


def execute(sql: str, params: tuple = ()) -> sqlite3.Cursor:
    with _lock:
        cur = db().execute(sql, params)
        db().commit()
        return cur


def query(sql: str, params: tuple = ()) -> list[dict]:
    with _lock:
        return [dict(r) for r in db().execute(sql, params).fetchall()]


import contextlib


@contextlib.contextmanager
def scratch():
    """Temporarily use an empty in-memory database (e.g. for certification), then restore the real one."""
    global _conn
    with _lock:
        saved = _conn
        _conn = None
        connect(":memory:")
    try:
        yield
    finally:
        with _lock:
            _conn.close()
            _conn = saved
