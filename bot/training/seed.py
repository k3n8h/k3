"""Synthetic training data: templates x paraphrases x slot fillers, with light noise.

Each tool has several phrasing templates. `split="train"` drops every 4th template so
`split="test"` measures generalization to phrasings the model never saw.
"""
import random
import re

TOPICS = ["solar panels", "the french revolution", "python asyncio", "electric cars", "climate change",
          "quantum computing", "sourdough bread", "the roman empire", "machine learning", "coffee prices",
          "renewable energy", "space telescopes"]
URLS = ["https://example.com", "https://news.ycombinator.com", "https://docs.python.org/3/",
        "https://en.wikipedia.org/wiki/Python", "https://blog.example.org/posts"]
SELECTORS = ["h1", "p.x", "a.link", "div.price", "ul li", "h2", "span.title", "table tr"]
EXPRS = [("12 times 7", "12 * 7"), ("15 + 27", "15 + 27"), ("100 divided by 8", "100 / 8"),
         ("(3+4)*5", "(3+4)*5"), ("9 minus 4", "9 - 4"), ("2 plus 2 times 3", "2 + 2 * 3"),
         ("18 * 3", "18 * 3"), ("250 / 5", "250 / 5")]
DICE = ["2d6", "d20", "3d8", "1d10", "d6", "4d4"]
TITLES = ["buy milk", "email the team", "finish the report", "call mom", "book flights", "water the plants",
          "renew passport", "prepare slides"]
BODIES = ["eggs, bread, butter", "ideas for the launch", "password hint is the dog's name",
          "meeting moved to friday", "read chapter three"]
PATHS = ["sales.csv", "data/users.csv", "report.xlsx", "results/q3.csv"]
UNITS = [("km", "km"), ("kilometers", "km"), ("mi", "mi"), ("miles", "mi"), ("m", "m"), ("meters", "m"),
         ("ft", "ft"), ("feet", "ft"), ("kg", "kg"), ("kilograms", "kg"), ("lb", "lb"), ("pounds", "lb"),
         ("g", "g"), ("grams", "g"), ("oz", "oz"), ("ounces", "oz"), ("c", "c"), ("celsius", "c"),
         ("f", "f"), ("fahrenheit", "f")]
GROUPS = [["km", "mi", "m", "ft"], ["kg", "lb", "g", "oz"], ["c", "f"]]
CHAT = ["tell me a joke", "how are you today", "hello", "hi there", "who are you", "what is the meaning of life",
        "write me a poem", "thanks a lot", "good morning", "you are awesome", "what's your favorite color",
        "i'm feeling sad", "explain how photosynthesis works", "give me advice about my career",
        "what do you think about pineapple on pizza", "sing me a song", "goodbye", "can you help me",
        "how does the internet work", "why is the sky blue", "translate hello to spanish",
        "write a story about a dragon", "what should i have for dinner", "tell me something interesting",
        "who won the world cup", "how tall is mount everest", "are you a robot", "let's chat",
        "what can you do", "ok", "nevermind", "that's funny", "blah blah", "hmm", "lol", "cool thanks",
        "what's the weather like", "play some music", "order me a pizza", "call my mother"]
PREFIXES = ["", "", "", "please ", "hey k3, ", "can you ", "could you ", "k3 "]
SUFFIXES = ["", "", "", " please", " thanks", " for me"]


def _fields(rng: random.Random) -> dict:
    et, es = rng.choice(EXPRS)
    group = rng.choice(GROUPS)
    a, b = rng.sample(group, 2)
    surface = lambda canon: rng.choice([s for s, c in UNITS if c == canon])
    return dict(topic=rng.choice(TOPICS), url=rng.choice(URLS), sel=rng.choice(SELECTORS), expr=et, expr_sym=es,
                dice=rng.choice(DICE), title=rng.choice(TITLES), body=rng.choice(BODIES),
                path=rng.choice(PATHS), id=rng.randint(1, 30), d=rng.randint(1, 3), n=rng.choice([5, 10, 20, 30]),
                value=rng.choice([5, 10, 12.5, 100, 3, 72, 20]), fu=surface(a), fu_c=a, tu=surface(b), tu_c=b)


# tool -> (templates, {arg: (placeholder, transform|None)}); an arg is included only if its
# placeholder appears in the template.
SPECS: dict[str, tuple[list[str], dict]] = {
    "web_search": (["search {topic}", "search for {topic}", "web search {topic}", "google {topic}",
                    "look up {topic}", "find {topic} online", "search the web for {topic}",
                    "find me links about {topic}", "what does the web say about {topic}", "browse for {topic}", "show me search results for {topic}", "who has written about {topic}"], {"query": ("topic", None)}),
    "research": (["research {topic}", "research {topic} for me", "do some research on {topic}",
                  "dig up information on {topic}", "investigate {topic}", "find out everything about {topic}",
                  "i need a research summary of {topic}", "research and cite sources on {topic}",
                  "write up research on {topic}", "give me a sourced report on {topic}", "what's the latest on {topic}", "compile a briefing about {topic}", "gather info on {topic}", "study {topic} and summarize with links"],
                 {"question": ("topic", None)}),
    "web_fetch": (["fetch {url}", "open {url}", "read {url}", "go to {url}", "get the page at {url}",
                   "show me {url}", "what does {url} say", "load {url}", "download {url}", "visit {url}", "pull up {url}", "summarize the page {url}"], {"url": ("url", None)}),
    "scrape": (["scrape {url} {sel}", "scrape {sel} from {url}", "extract {sel} from {url}",
                "grab all {sel} on {url}", "pull {sel} elements from {url}", "get every {sel} at {url}", "parse {url} and return the {sel}", "fetch all the {sel} from {url}", "list the {sel} found on {url}", "scrape {url} for {sel}"],
               {"url": ("url", None), "selector": ("sel", None)}),
    "crawl": (["crawl {url}", "crawl {url} depth {d}", "crawl {url} max {n}", "spider {url}",
               "crawl the site {url}", "map out the pages of {url}", "follow links on {url}",
               "index the whole site {url}", "crawl {url} up to {n} pages", "crawl {url} going {d} levels deep", "explore every page linked from {url}", "walk the site {url}"],
              {"start_url": ("url", None), "max_depth": ("d", None), "max_pages": ("n", None)}),
    "calculate": (["calc {expr}", "what is {expr}", "what's {expr}", "how much is {expr}", "compute {expr}",
                   "work out {expr}", "{expr} = ?", "figure out {expr}", "evaluate {expr}", "solve {expr}", "what does {expr} equal", "{expr}"],
                  {"expression": ("expr", lambda f: f["expr_sym"])}),
    "roll_dice": (["roll {dice}", "roll a {dice}", "throw {dice}", "roll some dice", "roll the dice",
                   "give me a dice roll {dice}", "let's roll {dice}", "roll two dice", "roll {dice} for me", "i want to roll {dice}", "throw the dice", "roll a die"], {"spec": ("dice", None)}),
    "flip_coin": (["flip a coin", "flip coin", "toss a coin", "heads or tails", "coin flip", "flip a coin for me", "flip a quick coin", "give me heads or tails", "let a coin decide", "do a coin toss"], {}),
    "now": (["what time is it", "what's the time", "current date and time", "what day is it today",
             "what's today's date", "tell me the time", "what is the date", "what's the current time", "tell me today's date", "what is the time now", "what day of the week is it"], {}),
    "add_task": (["add task {title}", "add a task to {title}", "remind me to {title}", "todo: {title}",
                  "i need to {title}", "put {title} on my to-do list", "new task {title}", "create a task {title}", "add {title} to my tasks", "make a todo to {title}", "don't let me forget to {title}", "task: {title}"],
                 {"title": ("title", None)}),
    "list_tasks": (["tasks", "show my tasks", "what's on my to-do list", "list my tasks", "what do i need to do",
                    "show todo list", "my pending tasks", "what's left to do", "show pending todos", "what tasks are open", "read me my to-do list"], {}),
    "add_note": (["note {title}: {body}", "take a note {title}: {body}", "save a note titled {title}: {body}",
                  "jot down {title}: {body}", "write down {title}: {body}", "note to self {title}: {body}", "make a note {title}: {body}", "remember {title}: {body}", "note: {title}: {body}", "log a note called {title}: {body}"],
                 {"title": ("title", None), "body": ("body", None)}),
    "search_notes": (["notes {topic}", "search my notes for {topic}", "find notes about {topic}",
                      "what did i write about {topic}", "look through my notes for {topic}",
                      "any notes on {topic}", "search notes {topic}", "look for {topic} in my notes", "find my note about {topic}", "did i save anything on {topic}"], {"query": ("topic", None)}),
    "list_events": (["events", "show my schedule", "what's on my calendar", "list my appointments",
                     "what do i have coming up", "show my calendar", "upcoming events",
                     "what's my schedule this week", "what appointments do i have", "what's coming up on my calendar", "do i have any classes today", "show upcoming classes and meetings"], {}),
    "cancel_event": (["cancel event {id}", "delete event {id}", "remove appointment {id}",
                      "cancel appointment {id}", "drop event number {id}", "cancel meeting {id}", "get rid of event {id}", "cancel the appointment numbered {id}", "delete meeting number {id}", "remove event {id}"],
                     {"id": ("id", None)}),
    "list_lessons": (["lessons", "what have you learned", "show what you've learned", "list your lessons",
                      "what did i teach you", "what do you remember", "show everything you've been taught", "what rules have i taught you", "list learned instructions", "what are your lessons"], {}),
    "convert_units": (["convert {value} {fu} to {tu}", "how many {tu} in {value} {fu}", "{value} {fu} in {tu}",
                       "change {value} {fu} into {tu}", "what is {value} {fu} in {tu}", "{value} {fu} to {tu}", "convert {value} {fu} into {tu}", "express {value} {fu} in {tu}", "{value} {fu} equals how many {tu}", "how many {tu} is {value} {fu}"],
                      {"value": ("value", lambda f: float(f["value"])), "from_unit": ("fu", lambda f: f["fu_c"]),
                       "to_unit": ("tu", lambda f: f["tu_c"])}),
    "scramble_word": (["scramble a word", "word scramble", "play a word game", "give me an anagram",
                       "unscramble game", "word puzzle", "let's play word scramble", "give me a scrambled word", "i want an anagram puzzle", "play unscramble"], {}),
    "list_files": (["list files", "show my files", "what files are in the workspace", "ls",
                    "show workspace files", "list the workspace", "what files do i have", "show me the files", "list everything in my workspace", "dir"], {}),
    "describe_data": (["describe {path}", "summarize the data in {path}", "analyze {path}", "what's in {path}",
                       "give me stats for {path}", "profile {path}", "take a look at {path}", "show a summary of {path}", "how many rows are in {path}", "inspect {path}", "analyze the file {path}"],
                      {"path": ("path", None)}),
}


def _noise(template: str, rng: random.Random) -> str:
    """Case/typo noise on the template words only (never on slot values)."""
    parts = re.split(r"(\{[^}]*\})", template)
    out = []
    for p in parts:
        if p.startswith("{") or len(p) < 5 or rng.random() > 0.15:
            out.append(p)
            continue
        i = rng.randrange(1, len(p) - 2)
        if p[i].isalpha() and p[i + 1].isalpha():
            p = p[:i] + p[i + 1] + p[i] + p[i + 2:]  # swap adjacent letters
        out.append(p)
    return "".join(out)


def generate(per_tool: int = 40, seed: int = 0, split: str = "all") -> list[dict]:
    """Return examples: {phrase, tool, args, frame, weight}. frame = phrase with slot values removed."""
    rng = random.Random(seed)
    out = []
    for tool, (templates, argmap) in SPECS.items():
        idx = [i for i in range(len(templates))
               if split == "all" or (i % 4 == 3) == (split == "test")]
        for k in range(per_tool):
            t = templates[idx[k % len(idx)]]
            f = _fields(rng)
            noisy = _noise(t, rng)
            pre, suf = rng.choice(PREFIXES), rng.choice(SUFFIXES)
            phrase = pre + noisy.format(**f) + suf
            frame = pre + re.sub(r"\{[^}]*\}", " ", noisy) + suf
            args = {a: (fn(f) if fn else f[ph]) for a, (ph, fn) in argmap.items() if "{" + ph + "}" in t}
            out.append({"phrase": phrase, "tool": tool, "args": args, "frame": frame, "weight": 1})
    idx = [i for i in range(len(CHAT)) if split == "all" or (i % 4 == 3) == (split == "test")]
    for k in range(per_tool):
        c = CHAT[idx[k % len(idx)]]
        out.append({"phrase": rng.choice(PREFIXES) + c, "tool": "chat", "args": {}, "frame": c, "weight": 1})
    return out
