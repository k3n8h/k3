"""On-device intent learner: phrase -> (tool, args). Pure Python, no API.

Classifier: multinomial Naive Bayes over binary word/bigram/char-trigram features (char trigrams give
typo tolerance). Slot filling: schema-driven extractors plus per-tool "trigger words" learned from the
training frames, which are stripped from the phrase edges to leave free-text arguments.
"""
import difflib
import json
import math
import re
import unicodedata
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from typing import Optional

from bot import config, memory, registry

URL_RE = re.compile(r"https?://[^\s\"'<>]+")
WORD_RE = re.compile(r"[^\W_]+(?:'[^\W_]+)?")
PSEUDO = {"chat", "needs_model"}   # intents that are not tools
DICE_RE = re.compile(r"\b\d*d\d+\b", re.I)
ALPHA = 0.5
MIN_KNOWN = 0.4       # share of a phrase's features that must have been seen in training
TEMPERATURE = 10.0
FAMILIARITY_POWER = 0.0
RUNNER_UP_MIN = 0.05
LOOKUP_FAMILY = {"research", "web_search"}   # read-only, overlapping intents: pool their probability
MIN_CONFIDENCE = 0.9
TRIGGER_DF = 0.0   # frames exclude slot values, so any frame word is command syntax
STOPS = {"please", "thanks", "thank", "you", "can", "could", "would", "hey", "k3", "me", "for", "the", "on", "about",
         "of", "to", "up", "some", "all", "every", "my", "a", "an", "and", "at", "from", "in", "into", "is", "it",
         "that", "i", "need", "want", "just", "then", "mind", "would", "like", "id", "d", "asap", "right", "now",
         "hey", "ok",
         # multilingual glue words
         "por", "favor", "gracias", "oye", "puedes", "quiero", "que", "una", "un", "el", "la", "los", "las", "de",
         "del", "para", "sobre", "mi", "mis", "s'il", "plait", "merci", "peux", "tu", "je", "voudrais", "dis",
         "le", "les", "des", "du", "sur", "une", "bitte", "danke", "kannst", "du", "ich", "mochte", "dass", "ein",
         "eine", "einen", "uber", "fur", "mir", "mich", "favor", "obrigado", "voce", "pode", "eu", "quero", "uma",
         "os", "as", "do", "da", "dos", "das", "sobre", "per", "grazie", "puoi", "vorrei", "ehi", "di", "il", "lo",
         "gli", "su", "mi", "ti", "com", "em", "en", "y", "e", "et", "und", "o", "ou", "oder", "a", "ao"}
NEVER_AUTORUN_HINT = "I won't run destructive actions from a guess"

LAST: dict = {}


def remember(phrase: str, tool: Optional[str], args: dict) -> None:
    """Record the last thing the bot did (tool=None: it understood nothing) for `good` / `wrong =>`."""
    LAST.clear()
    LAST.update(phrase=phrase, tool=tool, args=args)


   # most recent action taken for a user phrase, so `good` / `wrong` can learn from it


def fold(text: str) -> str:
    """Lowercase and strip accents so 'qué'/'que', 'münze'/'munze' share features."""
    return "".join(c for c in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(c))


def normalize(text: str) -> str:
    t = URL_RE.sub(" urltoken ", fold(text))
    t = DICE_RE.sub(" dicetoken ", t)
    return re.sub(r"\d+(?:\.\d+)?", " numtoken ", t)


def features(text: str) -> set:
    words = WORD_RE.findall(normalize(text))
    f = {"w:" + w for w in words}
    f |= {"x:" + name for name, tok in (("nonum", "numtoken"), ("nourl", "urltoken"), ("nodice", "dicetoken"))
          if tok not in words}          # absence of a value type is evidence too (calculate needs a number)
    f |= {f"b:{a}_{b}" for a, b in zip(words, words[1:])}
    for w in words:
        if len(w) >= 4 and not w.endswith("token"):
            p = f"#{w}#"
            f |= {"c:" + p[i:i + 3] for i in range(len(p) - 2)}
    return f


def _strip_values(phrase: str, args: dict) -> str:
    out = fold(phrase)
    for v in args.values():
        out = out.replace(fold(str(v)), " ")
    return out


class Model:
    def __init__(self):
        self.counts: dict[str, dict[str, int]] = {}
        self.totals: dict[str, int] = {}
        self.vocab: set = set()
        self.triggers: dict[str, list] = {}
        self.globals: list = []
        self.n_examples = 0

    def fit(self, examples: list[dict]) -> "Model":
        counts: dict = defaultdict(Counter)
        df: dict = defaultdict(Counter)
        n_tool: Counter = Counter()
        for ex in examples:
            feats = features(ex["phrase"])
            w = max(1, int(ex.get("weight", 1)))
            for f in feats:
                counts[ex["tool"]][f] += w
            frame = ex.get("frame") or _strip_values(ex["phrase"], ex.get("args") or {})
            n_tool[ex["tool"]] += 1
            for tok in set(WORD_RE.findall(fold(frame))):
                df[ex["tool"]][tok] += 1
        frame_df: Counter = Counter()
        value_df: Counter = Counter()
        for ex in examples:
            if ex["tool"] in PSEUDO:              # chit-chat / generative wording is content, not command syntax
                continue
            frame = ex.get("frame") or _strip_values(ex["phrase"], ex.get("args") or {})
            frame_df.update(set(WORD_RE.findall(fold(frame))))
            for v in (ex.get("args") or {}).values():
                value_df.update(set(WORD_RE.findall(fold(str(v)))))
        self.globals = sorted(w for w, k in frame_df.items() if k >= 2 and k > 2 * value_df[w])
        self.counts = {c: dict(v) for c, v in counts.items()}
        self.totals = {c: sum(v.values()) for c, v in self.counts.items()}
        self.vocab = {f for v in self.counts.values() for f in v}
        self.triggers = {c: sorted(t for t, k in df[c].items() if k / n_tool[c] > TRIGGER_DF) for c in n_tool}
        self.n_examples = len(examples)
        return self

    def rank(self, text: str) -> list[tuple[str, float]]:
        """All intents with calibrated probabilities, best first ([] if the phrase is unfamiliar)."""
        feats = features(text)
        known = [f for f in feats if f in self.vocab]
        if not known or len(known) / len(feats) < MIN_KNOWN:
            return []
        V = len(self.vocab)
        scores = {c: sum(math.log((self.counts[c].get(f, 0) + ALPHA) / (self.totals[c] + ALPHA * V)) for f in known)
                  for c in self.counts}
        z = {c: s / len(known) * TEMPERATURE for c, s in scores.items()}
        m = max(z.values())
        exp = {c: math.exp(v - m) for c, v in z.items()}
        tot = sum(exp.values())
        shrink = (len(known) / len(feats)) ** FAMILIARITY_POWER   # unfamiliar wording -> less sure
        return sorted(((c, v / tot * shrink) for c, v in exp.items()), key=lambda x: -x[1])

    def predict(self, text: str) -> tuple[Optional[str], float]:
        r = self.rank(text)
        return r[0] if r else (None, 0.0)

    def triggers_for(self, tool: str) -> list:
        return list(self.triggers.get(tool, [])) + self.globals

    def to_json(self) -> str:
        return json.dumps({"counts": self.counts, "triggers": self.triggers, "globals": self.globals,
                           "n": self.n_examples})

    @classmethod
    def from_json(cls, s: str) -> "Model":
        d = json.loads(s)
        m = cls()
        m.counts, m.triggers, m.n_examples = d["counts"], d["triggers"], d["n"]
        m.globals = d.get("globals", [])
        m.totals = {c: sum(v.values()) for c, v in m.counts.items()}
        m.vocab = {f for v in m.counts.values() for f in v}
        return m


# ---------------------------------------------------------------- slot extraction
_ALIASES = {"km": "km", "kilometer": "km", "kilometers": "km", "mi": "mi", "mile": "mi", "miles": "mi",
            "m": "m", "meter": "m", "meters": "m", "ft": "ft", "foot": "ft", "feet": "ft", "cm": "cm",
            "inch": "in", "inches": "in", "kg": "kg", "kilogram": "kg", "kilograms": "kg", "lb": "lb",
            "lbs": "lb", "pound": "lb", "pounds": "lb", "g": "g", "gram": "g", "grams": "g", "oz": "oz",
            "ounce": "oz", "ounces": "oz", "c": "c", "celsius": "c", "f": "f", "fahrenheit": "f",
            "kelvin": "k"}


def _edge(tok: str) -> str:
    return fold(tok).strip(".,!?:;'\"()¿¡«»")


_LEAD_PREPS = {"about", "on", "regarding", "sobre", "sur", "uber", "su", "acerca"}
_POLITE = STOPS


def _trim(toks: list, drop) -> str:
    def typo_of_command(tok: str) -> bool:      # "reserach", "scarpe": only the leading verb, same length
        e = _edge(tok)
        return len(e) >= 6 and any(len(d) == len(e) and e[0] == d[0] and sorted(e) == sorted(d) for d in drop)
    i, j = 0, len(toks)
    while i < j and (_edge(toks[i]) in drop or (i == 0 and typo_of_command(toks[i]))):
        i += 1
    while j > i and _edge(toks[j - 1]) in drop:
        j -= 1
    return " ".join(toks[i:j]).strip(" .,;:?!¿¡")


def free_text(text: str, triggers) -> Optional[str]:
    t = URL_RE.sub(" ", text)
    if q := re.search(r'"([^"]+)"|“([^”]+)”', t):
        return (q.group(1) or q.group(2)).strip()
    if m := re.search(r":(?:\s+|$)", t):          # "scan this for profanity: <content>"
        head, tail = t[:m.start()], t[m.end():]
        if tail.strip() and len(head.split()) <= 8:
            return _trim(tail.split(), _POLITE) or None
    drop = STOPS | set(triggers)
    out = _trim(t.split(), drop)
    toks = out.split()
    for k, tok in enumerate(toks[:6]):             # "... information on <topic>"
        if _edge(tok) in _LEAD_PREPS and k + 1 < len(toks):
            out = " ".join(toks[k + 1:])
            break
    return out or None


_OPS = [(r"to the power of|elevado a|hoch|puissance", "**"),
        (r"multiplied by|multiplicado por|multiplie par|multiplicado|moltiplicato per|times|vezes|fois|mal|por|per", "*"),
        (r"divided by|dividido por|divise par|geteilt durch|diviso per|diviso|over|entre", "/"),
        (r"plus|mas|mais|piu", "+"), (r"minus|menos|moins|meno", "-"), (r"\^", "**")]


def _expression(text: str) -> Optional[str]:
    t = fold(text)
    for pat, sym in _OPS:
        t = re.sub(rf"(?<![^\W\d_]){pat}(?![^\W\d_])", f" {sym} ", t)
    t = re.sub(r"(?<=\d)\s*x\s*(?=\d)", " * ", t)
    cands = [c.strip() for c in re.findall(r"[\d\.\(\)\s\+\-\*/%]{3,}", t)
             if re.search(r"\d", c) and re.search(r"[\+\-\*/%]", c)]
    return re.sub(r"\s+", " ", max(cands, key=len)) if cands else None


_WEEKDAYS = {d: i for i, d in enumerate(["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"])}


def parse_when(text: str, now: Optional[datetime] = None) -> dict:
    """Find a date, a time and a duration in English text. Returns {date, time, minutes, rest}."""
    now = now or datetime.now()
    t, cuts = text, []
    date = tm = minutes = None

    def take(m):
        cuts.append(m.span())

    if m := re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", t):
        date = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3))).date()
        take(m)
    elif m := re.search(r"\b(today|tomorrow)\b", t, re.I):
        date = (now + timedelta(days=1 if m.group(1).lower() == "tomorrow" else 0)).date()
        take(m)
    elif m := re.search(r"\b(?:(next|this)\s+)?(" + "|".join(_WEEKDAYS) + r")\b", t, re.I):
        ahead = (_WEEKDAYS[m.group(2).lower()] - now.weekday()) % 7
        if m.group(1) and m.group(1).lower() == "next" and ahead == 0:
            ahead = 7
        date = (now + timedelta(days=ahead)).date()
        take(m)
    rest = t
    for a, b in sorted(cuts, reverse=True):
        rest = rest[:a] + " " + rest[b:]
    cuts = []
    if m := re.search(r"\bfor\s+(\d+)\s*(hours?|hrs?|h|minutes?|mins?)\b", rest, re.I):
        minutes = int(m.group(1)) * (60 if m.group(2).lower().startswith("h") else 1)
        take(m)
    if m := re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", rest, re.I):
        h = int(m.group(1)) % 12 + (12 if m.group(3).lower() == "pm" else 0)
        tm = (h, int(m.group(2) or 0))
        take(m)
    elif m := re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", rest):
        tm = (int(m.group(1)), int(m.group(2)))
        take(m)
    elif m := re.search(r"\bnoon\b", rest, re.I):
        tm = (12, 0)
        take(m)
    for a, b in sorted(cuts, reverse=True):
        rest = rest[:a] + " " + rest[b:]
    if date is None and tm is not None:
        date = now.date() if datetime.combine(now.date(), datetime.min.time()).replace(hour=tm[0], minute=tm[1]) > now \
            else (now + timedelta(days=1)).date()
    return {"date": date, "time": tm, "minutes": minutes, "rest": rest}


def _extract_event(text: str, triggers, now: Optional[datetime] = None) -> tuple[dict, list]:
    w = parse_when(text, now)
    low = fold(text)
    kind = "class" if re.search(r"\b(class|lecture|lesson|course)\b", low) else \
        "appointment" if re.search(r"\b(appointment|appt)\b", low) else "event"
    args = {"kind": kind}
    if title := free_text(w["rest"], triggers):
        args["title"] = title
    if w["date"] and w["time"]:
        start = datetime.combine(w["date"], datetime.min.time()).replace(hour=w["time"][0], minute=w["time"][1])
        args["start"] = start.isoformat(timespec="minutes")
        args["end"] = (start + timedelta(minutes=w["minutes"] or 60)).isoformat(timespec="minutes")
    return args, [r for r in ("kind", "title", "start", "end") if r not in args]


def extract_args(tool: str, text: str, triggers) -> tuple[dict, list]:
    text = re.sub(r"\bk3\b", " ", text, flags=re.I)
    if tool == "add_event":
        return _extract_event(text, triggers)
    schema = registry._TOOLS[tool]["definition"]["input_schema"]
    props, required = schema["properties"], schema["required"]
    args: dict = {}
    url = URL_RE.search(text)
    num = re.search(r"\d+(?:\.\d+)?", URL_RE.sub(" ", text))
    for name, spec in props.items():
        v = None
        if name in ("url", "start_url"):
            v = url.group(0).rstrip(".,)") if url else None
        elif name == "expression":
            v = _expression(text)
        elif name == "spec":
            m = DICE_RE.search(text)
            v = m.group(0).lower() if m else None
        elif name == "date":
            d = parse_when(text)["date"]
            v = d.isoformat() if d else None
        elif name == "id":
            m = re.search(r"\b(\d+)\b", text)
            v = int(m.group(1)) if m else None
        elif name == "path":
            m = re.search(r"[\w./-]+\.(?:csv|xlsx|xls|json|md|txt|py|html|svg|png)\b", text, re.I)
            v = m.group(0) if m else None
        elif name == "max_depth":
            m = re.search(r"depth\s+(\d+)", text, re.I)
            v = int(m.group(1)) if m else None
        elif name == "max_pages":
            m = re.search(r"(?:max|limit)\s+(\d+)|(\d+)\s+pages", text, re.I)
            v = int(m.group(1) or m.group(2)) if m else None
        elif name == "value":
            v = float(num.group(0)) if num else None
        elif name in ("from_unit", "to_unit"):
            continue
        elif name in ("title", "body") and "body" in props and "title" in props:
            continue
        elif name in required and spec["type"] == "string":
            v = free_text(text, triggers)
        if v is not None:
            args[name] = v
    if "from_unit" in props:
        units = [_ALIASES[w] for w in WORD_RE.findall(text.lower()) if w in _ALIASES]
        if len(units) >= 2:
            a, b = (units[1], units[0]) if re.search(r"how many|how much", text, re.I) else units[:2]
            args["from_unit"], args["to_unit"] = a, b
    if "body" in props and "title" in props:
        if ":" in text:
            head, body = text.split(":", 1)
            title, body = free_text(head, triggers), _trim(body.split(), _POLITE)
        else:
            body = free_text(text, triggers)
            title = (body or "")[:30] or None
        if title:
            args["title"] = title
        if body:
            args["body"] = body
    return args, [r for r in required if r not in args]


# ---------------------------------------------------------------- persistence / training
_cache: tuple = (None, None, None)


def model_path():
    return config.home() / "model.json"


def add_example(phrase: str, tool: str, args: dict, source: str = "user", weight: int = 1) -> int:
    return memory.execute("INSERT INTO examples(phrase,tool,args,source,weight) VALUES(?,?,?,?,?)",
                          (phrase, tool, json.dumps(args), source, weight)).lastrowid


def stored_examples() -> list[dict]:
    return [{"phrase": r["phrase"], "tool": r["tool"], "args": json.loads(r["args"] or "{}"),
             "weight": r["weight"], "source": r["source"]} for r in memory.query("SELECT * FROM examples")]


def _valid(examples: list[dict]) -> list[dict]:
    registry.load_all()
    return [e for e in examples if e["tool"] in registry._TOOLS]


def _eval(model: Model, tests: list[dict]) -> dict:
    norm = lambda a: json.dumps({k: float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v
                                 for k, v in a.items()}, sort_keys=True)
    acted = wrong = 0
    groups: dict = {"tool": defaultdict(lambda: [0, 0, 0]), "lang": defaultdict(lambda: [0, 0, 0]),
                    "cap": defaultdict(lambda: [0, 0, 0])}
    for e in tests:
        r = resolve(model, e["phrase"])
        tool = r["tool"] if r else "chat"          # abstaining is the correct answer for chit-chat
        ok = tool == e["tool"]
        slot = ok and (tool in PSEUDO or (not r["missing"] and norm(r["args"]) == norm(e["args"])))
        acted += tool not in PSEUDO
        wrong += tool not in PSEUDO and not ok
        for kind, key in (("tool", e["tool"]), ("lang", e.get("lang", "?")), ("cap", e.get("cap", "?"))):
            g = groups[kind][key]
            g[0] += 1
            g[1] += ok
            g[2] += slot
    n = sum(v[0] for v in groups["tool"].values()) or 1
    rate = lambda d, i: {k: round(v[i] / v[0], 2) for k, v in sorted(d.items())}
    return {"intent_accuracy": round(sum(v[1] for v in groups["tool"].values()) / n, 3),
            "slot_accuracy": round(sum(v[2] for v in groups["tool"].values()) / n, 3),
            "per_tool": rate(groups["tool"], 1), "per_tool_slots": rate(groups["tool"], 2),
            "per_language": rate(groups["lang"], 1), "per_capability": rate(groups["cap"], 1),
            "wrong_action_rate": round(wrong / n, 3), "action_rate": round(acted / n, 3), "examples": n}


def fit_all(evaluate: bool = False) -> dict:
    """(Re)train from seed + stored examples and save. With evaluate, also measure on held-out phrasings."""
    from bot.training import seed
    global _cache
    user = _valid(stored_examples())
    report: dict = {"seed_examples": 0, "user_examples": len(user)}
    if evaluate:
        held = Model().fit(seed.generate(split="train", seed=1) + user)
        report["heldout"] = _eval(held, seed.generate(split="test", seed=2))
        report["seen_phrasing"] = _eval(held, seed.generate(split="train", seed=3))["intent_accuracy"]
    data = seed.generate(seed=0) + user
    report["seed_examples"] = len(data) - len(user)
    model = Model().fit(data)
    model_path().write_text(model.to_json())
    _cache = (None, None, None)
    return report


def get_model() -> Model:
    global _cache
    p = model_path()
    if not p.exists():
        fit_all()
    key = (str(p), p.stat().st_mtime_ns)
    if _cache[0] != key:
        _cache = (key, None, Model.from_json(p.read_text()))
    return _cache[2]


def resolve(model: Model, text: str) -> Optional[dict]:
    """Pick the intent + arguments. If the top intent lacks required slots, a runner-up whose slots
    are all present wins (e.g. 'what is calculus' is not arithmetic). None = abstain."""
    registry.load_all()
    ranked = model.rank(text)
    if not ranked:
        return None
    top_t, top_p = ranked[0]
    if top_t not in ("calculate", "chat") or (top_t == "chat" and top_p < 0.6):
        expr = _expression(text)                 # spoken arithmetic: "12 times 7", "cuanto es 12 por 7"
        if expr and re.fullmatch(r"[\d\.\(\)\s\+\-\*/%]+", expr) and \
                any(t == "calculate" and p > 1e-6 for t, p in ranked[:6]):
            return {"tool": "calculate", "args": {"expression": expr}, "confidence": round(top_p, 3), "missing": []}
    if top_t in LOOKUP_FAMILY and top_p < MIN_CONFIDENCE:   # search vs research is a near-tie by nature
        fam = sum(p for t, p in ranked if t in LOOKUP_FAMILY)
        if fam >= MIN_CONFIDENCE:
            top_p = fam
    if top_t == "chat" or top_p < MIN_CONFIDENCE:
        return None
    first = None
    for i, (t, p) in enumerate(ranked[:3]):
        if i and p < RUNNER_UP_MIN:
            break
        if t == "chat":
            continue
        if t == "needs_model":
            if i == 0:
                return {"tool": t, "args": {}, "confidence": round(p, 3), "missing": []}
            continue
        if t not in registry._TOOLS:
            continue
        args, missing = extract_args(t, text, model.triggers_for(t))
        cand = {"tool": t, "args": args, "confidence": round(p, 3), "missing": missing}
        if i == 0:
            first = cand
        if not missing:
            return cand
    return first


def interpret(text: str) -> Optional[dict]:
    """Best guess for a free-form phrase: {tool, args, confidence, missing} or None if unsure."""
    return resolve(get_model(), text)


def hints(text: str, k: int = 3) -> str:
    """Nearest user-taught examples (Jaccard on features) for few-shot prompting of a real LLM."""
    q = features(text)
    scored = []
    for e in stored_examples():
        f = features(e["phrase"])
        s = len(q & f) / (len(q | f) or 1)
        if s > 0.2:
            scored.append((s, e))
    scored.sort(key=lambda x: -x[0])
    if not scored:
        return ""
    return "\n\nHow this user has phrased similar requests before:\n" + "\n".join(
        f"- \"{e['phrase']}\" -> {e['tool']}({json.dumps(e['args'])})" for _, e in scored[:k])
