"""On-device intent learner: phrase -> (tool, args). Pure Python, no API.

Classifier: multinomial Naive Bayes over binary word/bigram/char-trigram features (char trigrams give
typo tolerance). Slot filling: schema-driven extractors plus per-tool "trigger words" learned from the
training frames, which are stripped from the phrase edges to leave free-text arguments.
"""
import difflib
import json
import math
import re
from collections import Counter, defaultdict
from typing import Optional

from bot import config, memory, registry

URL_RE = re.compile(r"https?://[^\s\"'<>]+")
WORD_RE = re.compile(r"[a-z0-9_']+")
DICE_RE = re.compile(r"\b\d*d\d+\b", re.I)
ALPHA = 0.5
MIN_KNOWN = 0.4       # share of a phrase's features that must have been seen in training
TEMPERATURE = 10.0
MIN_CONFIDENCE = 0.6
TRIGGER_DF = 0.04
STOPS = {"please", "thanks", "thank", "you", "can", "could", "would", "hey", "k3", "me", "for", "the", "on", "about",
         "of", "to", "up", "some", "all", "every", "my", "a", "an", "and", "at", "from", "in", "into", "is", "it",
         "that", "i", "need", "want", "just", "then"}
NEVER_AUTORUN_HINT = "I won't run destructive actions from a guess"

LAST: dict = {}   # most recent action taken for a user phrase, so `good` / `wrong` can learn from it


def normalize(text: str) -> str:
    t = URL_RE.sub(" urltoken ", text.lower())
    t = DICE_RE.sub(" dicetoken ", t)
    return re.sub(r"\d+(?:\.\d+)?", " numtoken ", t)


def features(text: str) -> set:
    words = WORD_RE.findall(normalize(text))
    f = {"w:" + w for w in words}
    f |= {f"b:{a}_{b}" for a, b in zip(words, words[1:])}
    for w in words:
        if len(w) >= 4 and not w.endswith("token"):
            p = f"#{w}#"
            f |= {"c:" + p[i:i + 3] for i in range(len(p) - 2)}
    return f


def _strip_values(phrase: str, args: dict) -> str:
    out = phrase.lower()
    for v in args.values():
        out = out.replace(str(v).lower(), " ")
    return out


class Model:
    def __init__(self):
        self.counts: dict[str, dict[str, int]] = {}
        self.totals: dict[str, int] = {}
        self.vocab: set = set()
        self.triggers: dict[str, list] = {}
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
            for tok in set(WORD_RE.findall(frame.lower())):
                df[ex["tool"]][tok] += 1
        self.counts = {c: dict(v) for c, v in counts.items()}
        self.totals = {c: sum(v.values()) for c, v in self.counts.items()}
        self.vocab = {f for v in self.counts.values() for f in v}
        self.triggers = {c: sorted(t for t, k in df[c].items() if k / n_tool[c] >= TRIGGER_DF) for c in n_tool}
        self.n_examples = len(examples)
        return self

    def predict(self, text: str) -> tuple[Optional[str], float]:
        feats = features(text)
        known = [f for f in feats if f in self.vocab]
        if not known or len(known) / len(feats) < MIN_KNOWN:
            return None, 0.0
        V = len(self.vocab)
        scores = {c: sum(math.log((self.counts[c].get(f, 0) + ALPHA) / (self.totals[c] + ALPHA * V)) for f in known)
                  for c in self.counts}
        z = {c: s / len(known) * TEMPERATURE for c, s in scores.items()}
        m = max(z.values())
        exp = {c: math.exp(v - m) for c, v in z.items()}
        tot = sum(exp.values())
        best = max(exp, key=exp.get)
        return best, exp[best] / tot

    def to_json(self) -> str:
        return json.dumps({"counts": self.counts, "triggers": self.triggers, "n": self.n_examples})

    @classmethod
    def from_json(cls, s: str) -> "Model":
        d = json.loads(s)
        m = cls()
        m.counts, m.triggers, m.n_examples = d["counts"], d["triggers"], d["n"]
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
    return tok.lower().strip(".,!?:;'\"()")


def free_text(text: str, triggers) -> Optional[str]:
    t = URL_RE.sub(" ", text)
    if q := re.search(r'"([^"]+)"|“([^”]+)”', t):
        return (q.group(1) or q.group(2)).strip()
    toks, drop = t.split(), STOPS | set(triggers)
    i, j = 0, len(toks)
    def dropped(tok: str) -> bool:
        e = _edge(tok)
        return e in drop or (len(e) >= 5 and bool(difflib.get_close_matches(e, drop, n=1, cutoff=0.8)))
    while i < j and dropped(toks[i]):
        i += 1
    while j > i and _edge(toks[j - 1]) in drop:
        j -= 1
    return " ".join(toks[i:j]).strip(" .?!") or None


def _expression(text: str) -> Optional[str]:
    t = text.lower()
    for a, b in [("to the power of", "**"), ("multiplied by", "*"), ("divided by", "/"), ("times", "*"),
                 ("plus", "+"), ("minus", "-"), ("over", "/"), ("^", "**")]:
        t = t.replace(a, f" {b} ")
    t = re.sub(r"(?<=\d)\s*x\s*(?=\d)", " * ", t)
    cands = [c.strip() for c in re.findall(r"[\d\.\(\)\s\+\-\*/%]{3,}", t)
             if re.search(r"\d", c) and re.search(r"[\+\-\*/%]", c)]
    return re.sub(r"\s+", " ", max(cands, key=len)) if cands else None


def extract_args(tool: str, text: str, triggers) -> tuple[dict, list]:
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
            title, body = free_text(head, triggers), body.strip()
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
    per: dict = defaultdict(lambda: [0, 0, 0])
    same = lambda a, b: json.dumps({k: float(v) if isinstance(v, (int, float)) else v for k, v in a.items()}, sort_keys=True) == \
        json.dumps({k: float(v) if isinstance(v, (int, float)) else v for k, v in b.items()}, sort_keys=True)
    for e in tests:
        tool, _ = model.predict(e["phrase"])
        tool = tool or "chat"          # abstaining is the correct answer for chit-chat
        per[e["tool"]][0] += 1
        if tool == e["tool"]:
            per[e["tool"]][1] += 1
            if tool == "chat":
                per[tool][2] += 1
                continue
            args, _ = extract_args(tool, e["phrase"], model.triggers.get(tool, []))
            per[e["tool"]][2] += same(args, e["args"])
    n = sum(v[0] for v in per.values()) or 1
    return {"intent_accuracy": round(sum(v[1] for v in per.values()) / n, 3),
            "slot_accuracy": round(sum(v[2] for v in per.values()) / n, 3),
            "per_tool": {t: round(v[1] / v[0], 2) for t, v in sorted(per.items())}}


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


def interpret(text: str) -> Optional[dict]:
    """Best guess for a free-form phrase: {tool, args, confidence, missing} or None if unsure."""
    model = get_model()
    tool, conf = model.predict(text)
    if tool is None or conf < MIN_CONFIDENCE or tool not in registry._TOOLS:
        return None
    args, missing = extract_args(tool, text, model.triggers.get(tool, []))
    return {"tool": tool, "args": args, "confidence": round(conf, 3), "missing": missing}


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
