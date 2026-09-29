"""Offline engine: understands instructions with no model or API.

Order: (1) taught rules (`learn`) -> (2) exact command grammar -> (3) trained intent learner
(bot/learner.py) for free-form phrasing -> (4) help text.
"""
import re
import uuid

from bot import learner, memory, registry
from bot.llm import Provider, Reply, ToolCall

HELP = ("I'm running offline (no model configured), so I understand these commands:\n"
       "  search <q> | research <q> | fetch <url> | crawl <url> [depth N] [max N] | scrape <url> <css selector>\n"
       "  calc <expr> | roll 2d6 | flip | now | add task <t> | tasks | note <title>: <body> | notes <q>\n"
       "  events | cancel event <id> | learn \"<trigger>\" => <command> | lessons | forget <id> | tools\n"
       "  train | train status | train add \"<phrase>\" => <command> | train from <file> | train reset\n"
       "  good | wrong => <command>   (teach me from my last answer)\n"
       "I can also handle free-form phrasing once trained. For open conversation set ANTHROPIC_API_KEY or "
       "BOT_BASE_URL (Ollama/OpenAI-compatible).")

TRAIN_TOOLS = {"train_model", "training_status", "add_training_example", "train_from_file", "reset_training",
               "mark_wrong", "mark_good"}
_LEARN = re.compile(r'^learn\s+["\'“](.+?)["\'”]\s*(?:=>|->|=)\s*(.+)$', re.I | re.S)
_TRAIN_ADD = re.compile(r'^train add\s+["\'“](.+?)["\'”]\s*(?:=>|->|=)\s*(.+)$', re.I | re.S)


def parse_grammar(text: str):
    """Exact command grammar. Returns (tool, args) or None when the text is not a known command."""
    t = text.strip()
    low = t.lower()
    if m := _LEARN.match(t):
        return "learn_instruction", {"trigger": m.group(1), "action": m.group(2).strip()}
    if m := _TRAIN_ADD.match(t):
        return "add_training_example", {"phrase": m.group(1), "command": m.group(2).strip()}
    if low == "train":
        return "train_model", {}
    if low == "train status":
        return "training_status", {}
    if low == "train reset":
        return "reset_training", {}
    if m := re.match(r"train from\s+(\S+)$", t, re.I):
        return "train_from_file", {"path": m.group(1)}
    if m := re.match(r"wrong\s*(?:=>|->|:)\s*(.+)$", t, re.I | re.S):
        return "mark_wrong", {"command": m.group(1).strip()}
    if low in ("good", "correct", "that was right"):
        return "mark_good", {}
    if low in ("lessons", "what have you learned"):
        return "list_lessons", {}
    if m := re.match(r"forget\s+(\d+)$", low):
        return "forget_lesson", {"id": int(m.group(1))}
    if low == "tools":
        return "__text__", "Tools: " + ", ".join(sorted(registry._TOOLS))
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
    return None


def parse(text: str, depth: int = 0):
    """Return (tool_name, args) | ('__text__', str) for a user instruction."""
    low = text.strip().lower()
    if depth < 3 and not _LEARN.match(text.strip()):  # taught rules: longest matching trigger wins
        rules = sorted(memory.query("SELECT * FROM lessons WHERE trigger<>''"), key=lambda r: -len(r["trigger"]))
        for r in rules:
            if r["trigger"].lower() in low:
                return parse(r["action"], depth + 1)
    if (g := parse_grammar(text)) is not None:
        return g
    guess = learner.interpret(text)
    if guess is None:
        return "__text__", HELP
    tool, args = guess["tool"], guess["args"]
    if registry.is_destructive(tool):
        return "__text__", (f"That sounds like {tool}({args}), but {learner.NEVER_AUTORUN_HINT}. "
                            "Use the explicit command (e.g. `cancel event 3`) to confirm.")
    if guess["missing"]:
        return "__text__", f"I think you want {tool}, but I still need: {', '.join(guess['missing'])}."
    return tool, args


class OfflineProvider(Provider):
    name = "offline"

    def complete(self, system, messages, tools, max_tokens=4096) -> Reply:
        last = messages[-1]["content"]
        if not isinstance(last, str):  # tool results came back: just relay them
            return Reply("\n".join(str(b["content"]) for b in last))
        name, args = parse(last)
        if name == "__text__":
            return Reply(args)
        if name not in TRAIN_TOOLS:
            learner.LAST.clear()
            learner.LAST.update(phrase=last, tool=name, args=args)
        return Reply(tool_calls=[ToolCall(uuid.uuid4().hex[:8], name, args)])
