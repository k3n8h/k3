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
       "  good | wrong => <command> | that means <command>   (teach me from my last answer or an unrecognized phrase)\n"
       "I can also handle free-form phrasing once trained. For open conversation set ANTHROPIC_API_KEY or "
       "BOT_BASE_URL (Ollama/OpenAI-compatible).")

TRAIN_TOOLS = {"train_model", "training_status", "add_training_example", "train_from_file", "reset_training",
               "mark_wrong", "mark_good"}
UNKNOWN = "__unknown__"     # nothing understood the phrase: remember it so it can be taught
_POLITE_PREFIX = re.compile(r"^(?:(?:hey\s+)?k3[,\s]+)?(?:(?:(?:can|could|would|will)\s+you|please)\s+)+", re.I)
_LEARN = re.compile(r'^learn\s+["\'“](.+?)["\'”]\s*(?:=>|->|=)\s*(.+)$', re.I | re.S)
_TRAIN_ADD = re.compile(r'^train add\s+["\'“](.+?)["\'”]\s*(?:=>|->|=)\s*(.+)$', re.I | re.S)


def parse_grammar(text: str):
    """Exact command grammar. Returns (tool, args) or None when the text is not a known command."""
    t = _POLITE_PREFIX.sub("", text.strip())
    low = t.lower()
    if m := _LEARN.match(t):
        return "learn_instruction", {"trigger": m.group(1), "action": m.group(2).strip()}
    if m := _TRAIN_ADD.match(t):
        return "add_training_example", {"phrase": m.group(1), "command": m.group(2).strip()}
    if low in ("roles", "certify", "certification", "role report"):
        return "certify_roles", {}
    if low in ("capabilities", "features", "skills"):
        return "list_capabilities", {}
    if low == "train":
        return "train_model", {}
    if low == "train status":
        return "training_status", {}
    if low == "train reset":
        return "reset_training", {}
    if m := re.match(r"train from\s+(\S+)$", t, re.I):
        return "train_from_file", {"path": m.group(1)}
    if m := re.match(r"(?:wrong|that means|i meant)\s*(?:=>|->|:)\s*(.+)$", t, re.I | re.S):
        return "mark_wrong", {"command": m.group(1).strip()}
    if m := re.match(r"(?:that means|i meant)\s+(.+)$", t, re.I | re.S):   # only if the rest is a real command
        inner = parse_grammar(m.group(1))
        if inner and inner[0] not in ("__text__", "mark_wrong"):
            return "mark_wrong", {"command": m.group(1).strip()}
    if low in ("good", "correct", "that was right"):
        return "mark_good", {}
    if m := re.match(r"(?:run python|python)\s*:?\s+(.+)$", t, re.I | re.S):
        return "run_python", {"code": m.group(1)}
    if m := re.match(r"write file\s+(\S+?)\s*:\s*(.*)$", t, re.I | re.S):
        return "write_file", {"path": m.group(1), "content": m.group(2)}
    if m := re.match(r"(?:read|show) file\s+(\S+)$", t, re.I):
        return "read_file", {"path": m.group(1)}
    if low in ("files", "list files"):
        return "list_files", {}
    if m := re.match(r"describe\s+(\S+\.(?:csv|xlsx|xls))$", t, re.I):
        return "describe_data", {"path": m.group(1)}
    if m := re.match(r"query\s+(\S+\.(?:csv|xlsx|xls))\s+where\s+(.+)$", t, re.I):
        return "query_data", {"path": m.group(1), "filter": m.group(2).strip()}
    if m := re.match(r"schedule job\s+[\"'“](.+?)[\"'”]\s+at\s+(\S+)$", t, re.I):
        return "schedule_job", {"goal": m.group(1), "run_at": m.group(2)}
    if low in ("jobs", "list jobs"):
        return "list_jobs", {}
    if m := re.match(r"(?:complete|finish) task\s+(\d+)$", low):
        return "complete_task", {"id": int(m.group(1))}
    if m := re.match(r"delete note\s+(\d+)$", low):
        return "delete_note", {"id": int(m.group(1))}
    if low in ("lessons", "what have you learned"):
        return "list_lessons", {}
    if m := re.match(r"forget\s+(\d+)$", low):
        return "forget_lesson", {"id": int(m.group(1))}
    if low == "tools":
        return "__text__", "Tools: " + ", ".join(sorted(registry._TOOLS))
    if m := re.match(r"search\s+(?:my\s+)?notes?\s+(?:for\s+|about\s+|on\s+)?(.+)", t, re.I):
        return "search_notes", {"query": m.group(1)}
    if m := re.match(r"(?:web\s+)?search\s+(?:(?:the\s+)?(?:web|internet|net)\s+|online\s+)?(?:for\s+)?(.+)", t, re.I):
        return "web_search", {"query": m.group(1)}
    if m := re.match(r"(?:look\s+up|google|bing|duckduckgo)\s+(.+)", t, re.I):
        return "web_search", {"query": m.group(1)}
    if m := re.match(r"research\s+(.+)", t, re.I):
        return "research", {"question": m.group(1)}
    if m := re.match(r"(?:fetch|open|read)\s+(https?://\S+)", t, re.I):
        return "web_fetch", {"url": m.group(1)}
    if m := re.match(r"(?:crawl|spider|walk\s+through|explore|traverse|index)\s+(https?://\S+)(.*)", t, re.I):
        rest = m.group(2)
        args = {"start_url": m.group(1)}
        if d := re.search(r"depth\s+(\d+)|(\d+)\s*levels?", rest, re.I):
            args["max_depth"] = int(d.group(1) or d.group(2))
        if n := re.search(r"(?:max|up\s+to|limit)\s+(\d+)|(\d+)\s+pages", rest, re.I):
            args["max_pages"] = int(n.group(1) or n.group(2))
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
        return UNKNOWN, HELP
    tool, args = guess["tool"], guess["args"]
    if tool == "needs_model":
        return "__text__", ("That's an open-ended generation task (writing, coding, translating, brainstorming, "
                            "designing), which needs a language model. Set ANTHROPIC_API_KEY or BOT_BASE_URL "
                            "(Ollama/OpenAI-compatible) to enable it. Offline I can research the topic instead.")
    if registry.is_destructive(tool):
        return "__text__", (f"That sounds like {tool}({args}), but {learner.NEVER_AUTORUN_HINT}. "
                            "Use the explicit command (e.g. `cancel event 3`) to confirm.")
    if guess["missing"]:
        return "__text__", f"I think you want {tool}, but I still need: {', '.join(guess['missing'])}."
    return tool, args


class OfflineProvider(Provider):
    name = "offline"
    user_driven = True      # the user's own words choose every action; no model can be steered by web text

    def complete(self, system, messages, tools, max_tokens=4096) -> Reply:
        last = messages[-1]["content"]
        if not isinstance(last, str):  # tool results came back: just relay them
            return Reply("\n".join(str(b["content"]) for b in last))
        name, args = parse(last)
        if name == UNKNOWN:
            learner.remember(last, None, {})
            return Reply(args + "\n(Tell me what you meant with `wrong => <command>` or `that means <command>` "
                                "and I'll learn it.)")
        if name == "__text__":
            return Reply(args)
        if name not in TRAIN_TOOLS:
            learner.remember(last, name, args)
        return Reply(tool_calls=[ToolCall(uuid.uuid4().hex[:8], name, args)])
