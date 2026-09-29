"""Offline engine: understands instructions with no model or API.

Order: (1) taught rules ("learn") -> (2) built-in command grammar -> (3) help text.
Command text may also be an alias-expansion of a taught rule, e.g.
  learn "morning news" => research today's top tech news
"""
import json
import re
import uuid

from bot import memory
from bot.llm import Provider, Reply, ToolCall

HELP = ("I'm running offline (no model configured), so I understand these commands:\n"
        "  search <q> | research <q> | fetch <url> | crawl <url> [depth N] [max N] | scrape <url> <css selector>\n"
        "  calc <expr> | roll 2d6 | flip | now | add task <t> | tasks | note <title>: <body> | notes <q>\n"
        "  events | cancel event <id> | learn \"<trigger>\" => <command> | lessons | forget <id> | tools\n"
        "Set ANTHROPIC_API_KEY, or BOT_BASE_URL (Ollama/OpenAI-compatible), for free-form conversation.")

_LEARN = re.compile(r'^learn\s+["\'“](.+?)["\'”]\s*(?:=>|->|=)\s*(.+)$', re.I | re.S)


def parse(text: str, depth: int = 0):
    """Return (tool_name, args) | ('__text__', str) for a user instruction."""
    t = text.strip()
    low = t.lower()
    if m := _LEARN.match(t):
        return "learn_instruction", {"trigger": m.group(1), "action": m.group(2).strip()}
    if depth < 3:  # taught rules: longest matching trigger wins
        rules = sorted(memory.query("SELECT * FROM lessons WHERE trigger<>''"), key=lambda r: -len(r["trigger"]))
        for r in rules:
            if r["trigger"].lower() in low:
                return parse(r["action"], depth + 1)
    if low in ("lessons", "what have you learned"):
        return "list_lessons", {}
    if m := re.match(r"forget\s+(\d+)$", low):
        return "forget_lesson", {"id": int(m.group(1))}
    if low == "tools":
        return "__text__", "Tools: " + ", ".join(sorted(__import__("bot.registry", fromlist=["x"])._TOOLS))
    if m := re.match(r"(?:web\s+)?search\s+(?:for\s+)?(.+)", t, re.I):
        return "web_search", {"query": m.group(1)}
    if m := re.match(r"research\s+(.+)", t, re.I):
        return "research", {"question": m.group(1)}
    if m := re.match(r"(?:fetch|open|read)\s+(https?://\S+)", t, re.I):
        return "web_fetch", {"url": m.group(1)}
    if m := re.match(r"crawl\s+(https?://\S+)(.*)", t, re.I):
        rest = m.group(2)
        args = {"start_url": m.group(1)}
        if d := re.search(r"depth\s+(\d+)", rest, re.I):
            args["max_depth"] = int(d.group(1))
        if n := re.search(r"max\s+(\d+)", rest, re.I):
            args["max_pages"] = int(n.group(1))
        return "crawl", args
    if m := re.match(r"scrape\s+(https?://\S+)\s+(.+)", t, re.I):
        return "scrape", {"url": m.group(1), "selector": m.group(2).strip()}
    if m := re.match(r"(?:calc|calculate)\s+(.+)", t, re.I):
        return "calculate", {"expression": m.group(1)}
    if m := re.match(r"roll\s*(\d*d\d+)?$", low):
        return "roll_dice", {"spec": m.group(1) or "1d6"}
    if low in ("flip", "flip a coin", "flip coin"):
        return "flip_coin", {}
    if low in ("now", "time", "date"):
        return "now", {}
    if m := re.match(r"add task\s+(.+)", t, re.I):
        return "add_task", {"title": m.group(1)}
    if low == "tasks":
        return "list_tasks", {}
    if m := re.match(r"note\s+(.+?)\s*:\s*(.+)", t, re.I | re.S):
        return "add_note", {"title": m.group(1), "body": m.group(2)}
    if m := re.match(r"notes\s+(.+)", t, re.I):
        return "search_notes", {"query": m.group(1)}
    if low == "events":
        return "list_events", {}
    if m := re.match(r"cancel event\s+(\d+)$", low):
        return "cancel_event", {"id": int(m.group(1))}
    return "__text__", HELP


class OfflineProvider(Provider):
    name = "offline"

    def complete(self, system, messages, tools, max_tokens=4096) -> Reply:
        last = messages[-1]["content"]
        if not isinstance(last, str):  # tool results came back: just relay them
            return Reply("\n".join(str(b["content"]) for b in last))
        name, args = parse(last)
        if name == "__text__":
            return Reply(args)
        return Reply(tool_calls=[ToolCall(uuid.uuid4().hex[:8], name, args)])
