"""In-chat moderation: word filter plus a warn/mute log per named user (no external platform)."""
import re

from bot import memory
from bot.registry import tool

DEFAULT_BLOCKLIST = {"fuck", "shit", "bitch", "asshole", "cunt"}
_blocklist: set[str] = set(DEFAULT_BLOCKLIST)


def check_text(text: str) -> list[str]:
    words = set(re.findall(r"[a-z']+", text.lower()))
    return sorted(words & _blocklist)


@tool("Check text against the profanity blocklist. Returns flagged words.")
def moderate_text(text: str) -> dict:
    hits = check_text(text)
    return {"flagged": bool(hits), "words": hits}


@tool("Add a word to the moderation blocklist.")
def add_blocked_word(word: str) -> dict:
    _blocklist.add(word.lower())
    return {"blocklist_size": len(_blocklist)}


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
