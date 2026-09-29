import os
from pathlib import Path

MODEL = os.environ.get("BOT_MODEL", "claude-sonnet-5-5")
MAX_TOKENS = 4096
MAX_ITERATIONS = 12          # tool-use loop cap per user turn
AUTONOMOUS_MAX_ITERATIONS = 30
CODE_TIMEOUT_S = 10


def home() -> Path:
    p = Path(os.environ.get("BOT_HOME", "~/.k3-bot")).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return p


def workspace() -> Path:
    p = home() / "workspace"
    p.mkdir(parents=True, exist_ok=True)
    return p


def db_path() -> Path:
    return home() / "bot.db"


SYSTEM_PROMPT = """You are K3, a multi-purpose assistant. You can chat, brainstorm, write and run \
code, analyze data, create designs (SVG/HTML), manage a calendar (classes, appointments, schedules), \
organize tasks and notes, moderate text, run utilities and play games. Use the provided tools \
whenever they make the answer more accurate or actually perform an action; never claim to have \
done something you did not do with a tool. Ask before destructive actions. Be concise."""
