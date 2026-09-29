"""On-device intent learner: phrase -> (tool, args). Pure Python, no API.

Classifier: multinomial Naive Bayes over binary word/bigram/char-trigram features (char trigrams give
typo tolerance). Slot filling: schema-driven extractors plus per-tool "trigger words" learned from the
training frames, which are stripped from the phrase edges to leave free-text arguments.
"""
import contextlib
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
TEMPERATURE = 12.0
FAMILIARITY_POWER = 0.0
MIN_DF = 3
RUNNER_UP_MIN = 0.05
# Read-only intents that overlap by nature; their probabilities are pooled before applying the threshold.
FAMILIES = [{"research", "web_search"}, {"web_fetch", "scrape"}]   # single-shot and read-only; never crawl
MIN_CONFIDENCE = 0.9          # for tools that change state (add/write/moderate/crawl...)
MIN_CONFIDENCE_APPEND = 0.8    # append-only, easily ignored: adding a task or a note
APPEND_ONLY = {"add_task", "add_note"}
MIN_CONFIDENCE_READONLY = 0.8  # a wrong guess on a read-only tool is harmless, so ask less of it
READ_ONLY = {"web_search", "research", "web_fetch", "scrape", "list_events", "find_free_slots", "list_tasks",
             "search_notes", "list_files", "read_file", "describe_data", "moderate_text", "user_moderation_status",
             "calculate", "now", "date_add", "convert_units", "make_poll", "roll_dice", "flip_coin", "pick_random",
             "scramble_word", "list_lessons", "list_jobs", "list_capabilities", "certify_roles", "training_status"}
TRIGGER_DF = 0.0   # frames exclude slot values, so any frame word is command syntax
STOPS = {"please", "thanks", "thank", "you", "can", "could", "would", "hey", "k3", "me", "for", "the", "on", "about",
         "of", "to", "up", "some", "all", "every", "my", "a", "an", "and", "at", "from", "in", "into", "is", "it",
         "that", "i", "need", "want", "just", "then", "mind", "would", "like", "id", "d", "asap", "right", "now",
         "hey", "ok", "before", "after", "ever", "already", "again", "yet", "still", "far", "ago", "pls", "so",
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


class Triggers(list):
    """Trigger words for a tool: `own` are words from that tool's command wording (safe to strip anywhere at the
    edges); the rest are pooled from other tools and only stripped from the front, so a content word that happens
    to be another tool's command word (e.g. 'review' in 'project review') survives at the end of a title."""
    own: set = frozenset()


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
            if ex.get("no_frame"):
                continue
            n_tool[ex["tool"]] += 1
            for tok in set(WORD_RE.findall(fold(frame))):
                df[ex["tool"]][tok] += 1
        frame_df: Counter = Counter()
        value_df: Counter = Counter()
        for ex in examples:
            if ex["tool"] in PSEUDO or ex.get("no_frame"):   # chit-chat / unframed wording is not command syntax
                continue
            frame = ex.get("frame") or _strip_values(ex["phrase"], ex.get("args") or {})
            frame_df.update(set(WORD_RE.findall(fold(frame))))
            for v in (ex.get("args") or {}).values():
                value_df.update(set(WORD_RE.findall(fold(str(v)))))
        self.globals = sorted(w for w, k in frame_df.items() if k >= 2 and k > 2 * value_df[w])
        self.counts = {c: dict(v) for c, v in counts.items()}
        if MIN_DF > 1:      # rare features are mostly one-off content words; they only add noise to confidence
            total: Counter = Counter()
            for v in self.counts.values():
                total.update(v)
            self.counts = {c: {f: n for f, n in v.items() if total[f] >= MIN_DF} for c, v in self.counts.items()}
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

    def triggers_for(self, tool: str) -> "Triggers":
        t = Triggers(list(self.triggers.get(tool, [])) + self.globals)
        t.own = set(self.triggers.get(tool, []))
        return t

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


def _trim(toks: list, drop, tail_drop=None) -> str:
    def typo_of_command(tok: str) -> bool:      # "reserach", "scarpe": only the leading verb, same length
        e = _edge(tok)
        return len(e) >= 6 and any(len(d) == len(e) and e[0] == d[0] and sorted(e) == sorted(d) for d in drop)
    i, j = 0, len(toks)
    while i < j and (_edge(toks[i]) in drop or (i == 0 and typo_of_command(toks[i]))):
        i += 1
    tail = drop if tail_drop is None else tail_drop
    while j > i and _edge(toks[j - 1]) in tail:
        j -= 1
    return " ".join(toks[i:j]).strip(" .,;:?!¿¡")


_POLITE_TAIL = {"please", "pls", "thanks", "thank", "asap", "now", "me"}


def _trim_tail(text: str) -> str:
    """Only strip trailing politeness ('please', 'thanks', 'for me', 'right now'); content words stay intact."""
    toks = text.split()
    while toks and (_edge(toks[-1]) in _POLITE_TAIL or (len(toks) > 1 and _edge(toks[-1]) == "for" )):
        toks.pop()
    while len(toks) > 1 and _edge(toks[-1]) in ("for", "right"):
        toks.pop()
    return " ".join(toks).strip(" .,;:?!¿¡")


def free_text(text: str, triggers) -> Optional[str]:
    t = URL_RE.sub(" ", text)
    if q := re.search(r'"([^"]+)"|“([^”]+)”', t):
        return (q.group(1) or q.group(2)).strip()
    if m := re.search(r":(?:\s+|$)", t):          # "scan this for profanity: <content>"
        head, tail = t[:m.start()], t[m.end():]
        if tail.strip() and len(head.split()) <= 8:
            return _trim_tail(tail) or None
    drop = STOPS | set(triggers)
    own = getattr(triggers, "own", None)
    out = _trim(t.split(), drop, STOPS | set(own) if own is not None else None)
    toks = out.split()
    for k, tok in enumerate(toks[:6]):             # "... information on <topic>"
        if _edge(tok) in _LEAD_PREPS and k + 1 < len(toks):
            out = " ".join(toks[k + 1:])
            break
    return out or None


_OPS = [(r"to the power of|elevado a|hoch|puissance|в степени", "**"),
        (r"multiplied by|умножить на|multiplicado por|multiplie par|multiplicado|moltiplicato per|times|vezes|fois|mal|por|per", "*"),
        (r"divided by|разделить на|dividido por|divise par|geteilt durch|diviso per|diviso|over|entre", "/"),
        (r"plus|mas|mais|piu|плюс", "+"), (r"minus|menos|moins|meno|минус", "-"), (r"\^", "**")]


def _expression(text: str) -> Optional[str]:
    t = re.sub(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}:\d{2}\b", " ", fold(text))   # dates/times are not arithmetic
    for pat, sym in _OPS:
        t = re.sub(rf"(?<![^\W\d_]){pat}(?![^\W\d_])", f" {sym} ", t)
    t = re.sub(r"(?<=\d)\s*x\s*(?=\d)", " * ", t)
    cands = [c.strip() for c in re.findall(r"[\d\.\(\)\s\+\-\*/%]{3,}", t)
             if re.search(r"[\d\)]\s*[\+\-\*/%]\s*[\d\(]", c)]         # needs operand op operand
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
    kind = "class" if re.search(r"\b(class|lecture|lesson|course|clase|classe|cours|kurs|unterricht|lezione|urok|занятие|урок)\b", low) else \
        "appointment" if re.search(r"\b(appointment|appt|cita|rendez-vous|termin|consulta|appuntamento|встреча)\b", low) else "event"
    args = {"kind": kind}
    if title := free_text(w["rest"], triggers):
        title = re.sub(r"^(?:class|lecture|lesson|course|appointment|appt|meeting|event)\s+", "", title, flags=re.I) or title
        args["title"] = title
    if w["date"] and w["time"]:
        start = datetime.combine(w["date"], datetime.min.time()).replace(hour=w["time"][0], minute=w["time"][1])
        args["start"] = start.isoformat(timespec="minutes")
        args["end"] = (start + timedelta(minutes=w["minutes"] or 60)).isoformat(timespec="minutes")
    return args, [r for r in ("kind", "title", "start", "end") if r not in args]


_DAY_UNITS = r"(days?|dias?|jours?|tage?|giorni|дн\w*|weeks?|semanas?|semaines?|wochen?|settimane|недел\w*)"
_BACKWARD = r"\b(before|ago|earlier|prior|minus|antes|avant|vor|meno|menos|moins|назад)\b"


_SELECTOR_NOISE = {"elements", "element", "headings", "heading", "tags", "tag", "items", "nodes", "listed", "links",
                   "entries", "there", "found"}
_PRONOUNS = {"her", "him", "them", "he", "she", "they", "it", "me", "us", "user", "member", "usuario", "usuário",
             "utilisateur", "l'utilisateur", "nutzer", "utente", "l'utente", "пользователя", "al", "den", "o"}


def _split_options(body: str) -> list:
    return [x.strip() for x in re.split(r",|\s(?:or|vs\.?|ou|oder|o|oppure|или)\s", body) if x.strip()]


def _extract_mod_user(text: str, triggers) -> tuple[dict, list]:
    f = fold(text)
    action = "unmute" if re.search(r"\bun-?mute|lift the mute|take the mute off|restore the chat|\brelease\b|speak again|talk again", f) else \
        "mute" if re.search(r"\b(?:mute|silence|gag|time[- ]?out|silencia\w*|silencie|sourdine|muet|sperre|silenzia\w*|заглуш\w*)", f) else \
        "warn" if re.search(r"\b(?:warn|warning|strike|advierte|advirta|avise|avertis|verwarne|avvisa|ammonisci|предупред\w*)", f) else None
    reason = None
    head = text
    if m := re.search(r"(?:\s(?:for|because|since|por|pour|wegen|per|за)\s|:\s*)(.+)$", text, re.I):
        reason, head = m.group(1).strip(" .!"), text[:m.start()]
    words = set(triggers) | {"user", "member", "the", "warn", "warning", "mute", "silence", "unmute", "issue", "give",
                             "please", "to"}
    left = _trim(head.split(), STOPS | words)
    args: dict = {}
    if action:
        args["action"] = action
    all_toks = [t.strip("'\"") for t in left.split()]
    toks = [t for t in all_toks if t.lower() not in _PRONOUNS]
    if len(toks) < len(all_toks) and all_toks:      # "mallory keeps spamming, warn her": 'her' points back at the first name
        first = [t for t in all_toks[:1] if t.lower() not in _PRONOUNS and fold(t) not in STOPS | words]
        toks = first or toks
        if first:
            args["user"] = first[0]
            toks = []
    if toks:
        args["user"] = toks[-1]
    if reason:
        args["reason"] = reason
    return args, [r for r in ("user", "action") if r not in args]


def _extract_poll(text: str, triggers) -> tuple[dict, list]:
    m = re.search(r"\bwith options\b|\boptions\b|\bcon opciones\b|\bavec les options\b|\bmit optionen\b|\bcom opções\b|"
                  r"\bcon opzioni\b|\bс вариантами\b|:", text, re.I)
    args: dict = {}
    if m:
        head, tail = text[:m.start()], text[m.end():]
        if "?" in tail and not re.search(r"\boptions\b", text, re.I):     # "poll: best season? summer or winter"
            qpart, tail = tail.split("?", 1)
            head = head + " " + qpart
        opts = _split_options(_trim_tail(tail))
        if len(opts) >= 2:
            args["options"] = opts
    else:
        head = text
    cut = re.search(r"\b(?:poll|vote|encuesta|votación|sondage|vote|umfrage|abstimmung|enquete|votação|sondaggio|votazione|опрос|голосование)\b"
                    r"(?:\s+(?:about|on|sobre|über|su|о))?", head, re.I)      # drop the verb phrase up to 'poll'
    q = free_text(head[cut.end():] if cut else head, triggers)
    if q:
        args["question"] = q
    return args, [r for r in ("question", "options") if r not in args]


def _extract_chart(text: str, triggers) -> tuple[dict, list]:
    f = fold(text)
    args: dict = {}
    if m := re.search(r"[\w./-]+\.(?:csv|xlsx|xls)\b", text, re.I):
        args["path"] = m.group(0)
    kind = "hist" if re.search(r"\b(?:histogram|hist)\b", f) else "line" if re.search(r"\bline\b", f) else \
        "bar" if re.search(r"\bbar\b", f) else None
    if kind:
        args["kind"] = kind
    xm, ym = re.search(r"\bx\s+(\w+)", f), re.search(r"\by\s+(\w+)", f)
    if xm and ym:
        args["x"], args["y"] = xm.group(1), ym.group(1)
    elif m := re.search(r"\bhist(?:ogram)?\s+of\s+(\w+)", f):
        args["x"] = m.group(1)
    elif m := re.search(r"\b(\w+)\s+(?:by|per|vs|versus|over|against)\s+(\w+)", f):
        args["y"], args["x"] = m.group(1), m.group(2)
    return args, [r for r in ("path", "x") if r not in args]


def _days(text: str) -> Optional[int]:
    """Signed day offset: sums every 'N days / N weeks' quantity; months/years are ambiguous -> None."""
    f = re.sub(r"\d{4}-\d{2}-\d{2}", " ", fold(text))
    if re.search(r"\d+\s*(?:months?|years?|meses|mes|mois|monate?|anos?|mesi|anni|месяц\w*|лет|год\w*)", f):
        return None
    qty = re.findall(rf"(\d+)\s*{_DAY_UNITS}\b", f)
    if qty:
        total = sum(int(n) * (7 if re.match(r"(?:week|seman|wochen|settiman|недел)", u) else 1) for n, u in qty)
    else:
        nums = {int(n) for n in re.findall(r"\b\d{1,4}\b", f)}
        if len(nums) != 1:
            return None
        total = nums.pop()
    return -total if re.search(_BACKWARD, f) else total


def extract_args(tool: str, text: str, triggers) -> tuple[dict, list]:
    text = re.sub(r"\bk3\b", " ", text, flags=re.I)
    if tool == "add_event":
        return _extract_event(text, triggers)
    special = {"moderate_user": _extract_mod_user, "make_poll": _extract_poll, "make_chart": _extract_chart}
    if tool in special:
        return special[tool](text, triggers)
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
        elif name == "days":
            v = _days(text)
        elif name == "options" and spec["type"] == "array":
            cue = re.search(r"(?:\bbetween\b|\bfrom\b|\bamong\b|\bof\b|:)\s+(.+)$", text, re.I)
            body = _trim_tail(cue.group(1)) if cue else (free_text(text, triggers) or "")
            parts = [x.strip() for x in re.split(r",|\s(?:or|vs\.?|ou|oder|o|oppure|или)\s", body) if x.strip()]
            v = parts if len(parts) >= 2 else None
        elif name == "selector":
            toks = (free_text(text, triggers) or "").split()
            while toks and _edge(toks[-1]) in _SELECTOR_NOISE:
                toks.pop()
            v = " ".join(toks) or None
        elif name == "word":
            cue = re.search(r"\b(?:word|term|palabra|mot|wort|palavra|parola|слово)\s+(\S+)", text, re.I)
            if cue:
                v = cue.group(1).strip("'\"?.,")
            else:                                     # first token that isn't a command/filler word
                skip = STOPS | set(triggers)
                v = next((t.strip("'\"?.,") for t in text.split() if _edge(t) not in skip), None)
        elif name == "user":
            toks = [t.removesuffix("'s").strip("'\"?.,") for t in (free_text(text, triggers) or "").split()]
            toks = [t for t in toks if t and t.lower() not in _PRONOUNS]
            v = toks[-1] if toks else None
        elif name == "id":
            plain = re.sub(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}:\d{2}\b", " ", text)   # dates/times hold no ids
            ids = {int(n) for n in re.findall(r"\b\d+\b", plain)}     # "tasks 3 and 4" is ambiguous: ask
            v = ids.pop() if len(ids) == 1 else None
        elif name == "path":
            m = re.search(r"[\w./-]+\.(?:csv|xlsx|xls|json|md|txt|py|html|svg|png)\b", text, re.I)
            v = m.group(0) if m else None
        elif name == "max_depth":
            m = re.search(r"(?:depth|profundidad|profondeur|tiefe|profundidade|profondità|глубин\w*)\s+(\d+)|(\d+)\s*levels?", text, re.I)
            v = int(m.group(1) or m.group(2)) if m else None
        elif name == "max_pages":
            m = re.search(r"(?:max|limit|hasta|jusqu'à|bis zu|até|fino a|до)\s+(\d+)|(\d+)\s+(?:pages|páginas|seiten|pagine|страниц\w*)", text, re.I)
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
            title, body = free_text(head, triggers), _trim_tail(body)
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
    acted = wrong = wrong_mut = 0
    groups: dict = {"tool": defaultdict(lambda: [0, 0, 0]), "lang": defaultdict(lambda: [0, 0, 0]),
                    "cap": defaultdict(lambda: [0, 0, 0])}
    for e in tests:
        r = resolve(model, e["phrase"])
        tool = r["tool"] if r else "chat"          # abstaining is the correct answer for chit-chat
        ok = tool == e["tool"]
        slot = ok and (tool in PSEUDO or (not r["missing"] and norm(r["args"]) == norm(e["args"])))
        acted += tool not in PSEUDO
        wrong += tool not in PSEUDO and not ok
        wrong_mut += tool not in PSEUDO and not ok and tool not in READ_ONLY and tool not in APPEND_ONLY
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
            "wrong_action_rate": round(wrong / n, 3), "wrong_state_change_rate": round(wrong_mut / n, 3), "action_rate": round(acted / n, 3), "examples": n}


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


_override: Optional[Model] = None


@contextlib.contextmanager
def using(model: Model):
    """Use `model` instead of the saved one for the duration (e.g. certifying the shipped seed model)."""
    global _override
    saved, _override = _override, model
    try:
        yield model
    finally:
        _override = saved


def get_model() -> Model:
    global _cache
    if _override is not None:
        return _override
    p = model_path()
    if not p.exists():
        fit_all()
    key = (str(p), p.stat().st_mtime_ns)
    if _cache[0] != key:
        _cache = (key, None, Model.from_json(p.read_text()))
    return _cache[2]


_PARTICIPLE = re.compile(r"\b(?:muted|silenced|warned|silenciado|silenciada|silenziato|silenziata|stummgeschaltet|verwarnt|"
                         r"заглушен\w*|en sourdine|avertido|avvisato)\b")


def _retarget_status_question(text: str, cand: dict) -> dict:
    """'is bob muted' / 'está silenciado bob' ask about a state; only the imperative verb mutes."""
    if cand["tool"] == "moderate_user" and _PARTICIPLE.search(fold(text)):
        args, missing = extract_args("user_moderation_status", text, get_model().triggers_for("user_moderation_status"))
        return {"tool": "user_moderation_status", "args": args, "confidence": cand["confidence"], "missing": missing}
    return cand


_CRAWL_CUES = re.compile(r"crawl|spider|explor|index|walk|map|follow|depth|levels?|pages?|site|sitemap|rastre|recorr|profundidad|"
                         r"p[aá]ginas|parcour|profondeur|durchsuch|erkund|tiefe|seiten|percorr|scansion|esplor|pagine|обойд|исследуй|глубин|страниц")


def _has_command_evidence(model: "Model", text: str, tool: str) -> bool:
    """A state-changing action needs at least one word from that tool's own command wording (or a typo of one);
    a statistical guess built only from character n-grams ('blah blah' -> add_blocked_word) is not enough."""
    all_own = set(model.triggers.get(tool, []))
    own = all_own - STOPS
    toks = WORD_RE.findall(fold(text))
    if any(t in own for t in toks):
        return True
    if len({t for t in toks if t in all_own}) >= 2:      # "i need to ...": filler words, but a real command frame
        return True
    if tool == "add_event" and re.search(r"\d{4}-\d{2}-\d{2}", text):
        return True
    return any(len(t) >= 5 and difflib.get_close_matches(t, own, n=1, cutoff=0.85) for t in toks)


def _guard_crawl(text: str, cand: dict) -> dict:
    """A bare URL request without any 'walk the site' wording is a page read, not a multi-page crawl."""
    if cand["tool"] == "crawl" and not _CRAWL_CUES.search(fold(URL_RE.sub(" ", text))):
        return {"tool": "web_fetch", "args": {"url": cand["args"].get("start_url", "")}, "confidence": cand["confidence"],
                "missing": [] if cand["args"].get("start_url") else ["url"]}
    return cand


def _url_fallback(text: str, top_t: str) -> Optional[dict]:
    """A message that carries a URL but was not confidently claimed by anything else (and isn't chit-chat about
    something state-changing) is treated as 'read this page': read-only, so a wrong guess is harmless."""
    m = URL_RE.search(text)
    if m and top_t not in ("chat",) and top_t not in STATE_CHANGING_HINT:
        return {"tool": "web_fetch", "args": {"url": m.group(0).rstrip(".,)")}, "confidence": 0.5, "missing": []}
    return None


STATE_CHANGING_HINT = {"add_task", "add_note", "add_event", "cancel_event", "moderate_user", "add_blocked_word",
                       "schedule_job", "write_file", "complete_task", "learn_instruction"}


def resolve(model: Model, text: str) -> Optional[dict]:
    """Pick the intent + arguments. If the top intent lacks required slots, a runner-up whose slots
    are all present wins (e.g. 'what is calculus' is not arithmetic). None = abstain."""
    registry.load_all()
    ranked = model.rank(text)
    if not ranked:
        return None
    top_t, top_p = ranked[0]
    if top_t != "chat" or top_p < 0.6:
        expr = _expression(text)                 # spoken arithmetic: "12 times 7", "cuanto es 12 por 7"
        if expr and re.fullmatch(r"[\d\.\(\)\s\+\-\*/%]+", expr) and \
                any(t == "calculate" and p > 1e-6 for t, p in ranked[:6]):
            return {"tool": "calculate", "args": {"expression": expr}, "confidence": round(top_p, 3), "missing": []}
    need = MIN_CONFIDENCE_READONLY if top_t in READ_ONLY else MIN_CONFIDENCE_APPEND if top_t in APPEND_ONLY \
        else MIN_CONFIDENCE
    if top_p < need:
        for fam_set in FAMILIES:
            if top_t in fam_set:
                fam = sum(p for t, p in ranked if t in fam_set)
                if fam >= need:
                    top_p = fam
    if top_t == "chat" or top_p < need:
        return _url_fallback(text, top_t)
    if top_t not in READ_ONLY and top_t not in PSEUDO and not _has_command_evidence(model, text, top_t):
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
        cand = _guard_crawl(text, _retarget_status_question(text, {"tool": t, "args": args, "confidence": round(p, 3),
                                                                   "missing": missing}))
        if i == 0:
            first = cand
        if not missing:
            return cand
    if first and first["missing"] and first["tool"] in ("scrape",) and URL_RE.search(text):
        return {"tool": "web_fetch", "args": {"url": URL_RE.search(text).group(0).rstrip(".,)")},
                "confidence": first["confidence"], "missing": []}
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
