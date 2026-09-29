"""In-chat moderation: word filter plus a warn/mute log per named user (no external platform)."""
import re

from bot import memory
from bot.registry import tool

DEFAULT_BLOCKLIST = frozenset({"fuck", "shit", "bitch", "asshole", "cunt"})
_WORD = re.compile(r"[^\W_]+(?:'[^\W_]+)?")


def blocklist() -> set:
    """Built-in words plus the ones added by the user (persisted in the database, so they survive restarts)."""
    return set(DEFAULT_BLOCKLIST) | {r["word"] for r in memory.query("SELECT word FROM blocklist")}


def check_text(text: str) -> list[str]:
    return sorted(set(_WORD.findall(text.lower())) & blocklist())


@tool("Check text against the profanity blocklist. Returns flagged words.")
def moderate_text(text: str) -> dict:
    hits = check_text(text)
    return {"flagged": bool(hits), "words": hits}


@tool("Add a word to the moderation blocklist.")
def add_blocked_word(word: str) -> dict:
    tokens = _WORD.findall(word.lower())
    if len(tokens) != 1:
        raise ValueError("give a single word to block")
    memory.execute("INSERT OR IGNORE INTO blocklist(word) VALUES(?)", (tokens[0],))
    return {"blocked": tokens[0], "blocklist_size": len(blocklist())}


@tool("Record a moderation action (warn, mute, unmute) against a user name, with a reason.")
def moderate_user(user: str, action: str, reason: str = "") -> dict:
    if action not in ("warn", "mute", "unmute"):
        raise ValueError("action must be warn, mute or unmute")
    memory.execute("INSERT INTO mod_log(user,action,reason) VALUES(?,?,?)", (user, action, reason))
    return {"recorded": True, "status": user_status(user)}


def user_status(user: str) -> dict:
    log = memory.query("SELECT action FROM mod_log WHERE user=? ORDER BY id", (user,))
    warns = sum(1 for r in log if r["action"] == "warn")
    muted = False
    for r in log:
        if r["action"] == "mute":
            muted = True
        elif r["action"] == "unmute":
            muted = False
    return {"warnings": warns, "muted": muted}


@tool("Get the moderation history and status of a user.")
def user_moderation_status(user: str) -> dict:
    return {"status": user_status(user), "log": memory.query("SELECT * FROM mod_log WHERE user=?", (user,))}
