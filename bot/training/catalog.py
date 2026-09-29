"""The bot's organized knowledge of itself: capabilities, instruction styles, languages, task types,
subjects and skills. This is the single source for the `capabilities` command, docs/CAPABILITIES.md,
the training-data generator's topic pool, and the per-capability training report.

Every registered tool must appear in exactly the capability that owns it (tests enforce this).
`needs_model` marks capabilities that only work fully with an LLM provider: offline, the bot recognizes
the request and says so instead of pretending.
"""

CAPABILITIES = [
    dict(id="research", name="Research & web", offline=True,
         tools=["web_search", "research", "web_fetch", "scrape", "crawl"],
         summary="Search the web, read pages, scrape by CSS selector, crawl sites politely, write sourced research files.",
         examples=["research solar panels", "scrape https://x.com h1", "crawl https://a.com depth 2"]),
    dict(id="schedule", name="Schedules, classes & appointments", offline=True,
         tools=["add_event", "list_events", "find_free_slots", "cancel_event", "schedule_job", "list_jobs"],
         summary="Book classes/appointments/events with conflict checks and reminders; find free time; schedule unattended jobs.",
         examples=["book a dentist appointment on 2030-03-05 at 14:00", "what's on my calendar", "when am i free on 2030-03-05"]),
    dict(id="organize", name="Organizing (tasks & notes)", offline=True,
         tools=["add_task", "list_tasks", "complete_task", "add_note", "search_notes", "delete_note"],
         summary="To-do list with priorities/dues and searchable notes.",
         examples=["add task buy milk", "note ideas: launch plan", "search my notes for budget"]),
    dict(id="coding", name="Coding & files", offline=True,
         tools=["write_file", "read_file", "list_files", "run_python"],
         summary="Read/write workspace files and run Python with a timeout. (Writing new code needs a model.)",
         examples=["show file notes.md", "list files"]),
    dict(id="data", name="Data analysis", offline=True,
         tools=["describe_data", "query_data", "make_chart"],
         summary="Summarize, filter/group and chart CSV/XLSX files.",
         examples=["describe sales.csv", "analyze the file report.xlsx"]),
    dict(id="design", name="Designs & creating", offline=True,
         tools=["save_svg", "save_html", "save_brainstorm"],
         summary="Persist SVG/HTML designs and brainstorm lists. (Authoring them needs a model.)",
         examples=[]),
    dict(id="moderation", name="Moderation", offline=True,
         tools=["moderate_text", "add_blocked_word", "moderate_user", "user_moderation_status"],
         summary="Profanity filter, warn/mute log (in this chat only).",
         examples=["is this offensive: what the hell"]),
    dict(id="utilities", name="Utilities", offline=True,
         tools=["calculate", "now", "date_add", "convert_units", "make_poll"],
         summary="Arithmetic, date/time, unit conversion, polls.",
         examples=["what's 12 times 7", "convert 5 km to miles", "what time is it"]),
    dict(id="fun", name="Fun & games", offline=True,
         tools=["roll_dice", "flip_coin", "pick_random", "scramble_word"],
         summary="Dice, coin, random picks, word scramble.",
         examples=["roll 2d6", "heads or tails"]),
    dict(id="teaching", name="Learning & training", offline=True,
         tools=["learn_instruction", "list_lessons", "forget_lesson", "train_model", "training_status",
                "add_training_example", "train_from_file", "mark_wrong", "mark_good", "reset_training",
                "list_capabilities", "certify_roles"],
         summary="Teach shortcuts and phrasings, correct mistakes, retrain the on-device intent learner.",
         examples=["train add \"gimme a d20\" => roll d20", "wrong => research solar panels"]),
    dict(id="generative", name="Writing, coding, design, translation, brainstorming", offline=False, tools=[],
         summary="Open-ended generation (essays, code, poems, emails, translations, brainstorms, designs). "
                 "Recognized offline, produced only when a model provider is connected.",
         examples=["write me an essay about volcanoes", "translate hello to french"]),
    dict(id="chat", name="Conversation", offline=False, tools=[],
         summary="Free-form chat; offline the bot abstains rather than guessing.", examples=["hello"]),
]

INSTRUCTION_STYLES = {
    "imperative": "research solar panels",
    "polite request": "could you please research solar panels",
    "question": "what is 12 times 7",
    "telegraphic": "solar panels research",
    "desire / indirect": "i'd like to know about solar panels",
    "with typos": "reserach solar pnaels",
    "code-switched": "busca solar panels",
    "taught shortcut": 'learn "morning news" => research top tech news',
    "correction": "wrong => research solar panels",
}

# support: trained = synthetic + tested; model-only = works via an LLM provider, not the offline learner.
LANGUAGES = {
    "en": dict(name="English", script="Latin", support="trained"),
    "es": dict(name="Spanish", script="Latin", support="trained"),
    "fr": dict(name="French", script="Latin", support="trained"),
    "de": dict(name="German", script="Latin", support="trained"),
    "pt": dict(name="Portuguese", script="Latin", support="trained"),
    "it": dict(name="Italian", script="Latin", support="trained"),
    "ru": dict(name="Russian", script="Cyrillic", support="trained"),
    "*": dict(name="Any other language", script="any", support="model-only (or teach it: `train add` / `train from`)"),
}

TASK_TYPES = ["lookup", "create", "modify", "schedule", "analyze", "automate", "organize", "converse", "play", "teach"]

SKILLS = {
    "research": ["web_search", "research", "web_fetch", "scrape", "crawl"],
    "time management": ["add_event", "find_free_slots", "list_events", "schedule_job", "add_task"],
    "note-taking": ["add_note", "search_notes"],
    "data literacy": ["describe_data", "query_data", "make_chart"],
    "quantitative reasoning": ["calculate", "convert_units", "date_add"],
    "programming support": ["write_file", "read_file", "run_python"],
    "content safety": ["moderate_text", "moderate_user"],
    "play": ["roll_dice", "flip_coin", "pick_random", "scramble_word"],
    "self-improvement": ["learn_instruction", "train_model", "mark_wrong", "mark_good"],
}

# Subjects double as the topic pool for research/search/notes training phrases.
SUBJECTS = {
    "Natural sciences": ["photosynthesis", "black holes", "plate tectonics", "the periodic table", "dna replication",
                         "quantum mechanics", "climate change", "the water cycle", "evolution", "ocean currents"],
    "Mathematics": ["linear algebra", "prime numbers", "the pythagorean theorem", "probability", "calculus",
                    "graph theory", "statistics", "fibonacci numbers"],
    "Technology & computing": ["python asyncio", "machine learning", "quantum computing", "blockchain", "solar panels",
                               "electric cars", "cybersecurity", "rust ownership", "sql joins", "http caching"],
    "History": ["the french revolution", "the roman empire", "the industrial revolution", "the cold war",
                "ancient egypt", "the silk road", "world war two", "the renaissance"],
    "Geography & travel": ["the amazon rainforest", "mount everest", "the sahara desert", "visa rules for japan",
                           "the great barrier reef", "iceland volcanoes", "the nile river"],
    "Health & fitness": ["sleep hygiene", "intermittent fasting", "marathon training", "vitamin d", "meditation",
                         "cholesterol", "first aid basics"],
    "Business & finance": ["compound interest", "index funds", "startup fundraising", "supply chains", "coffee prices",
                           "inflation", "content marketing", "cash flow forecasting"],
    "Arts, literature & languages": ["impressionism", "shakespeare sonnets", "jazz history", "spanish verbs",
                                     "japanese kanji", "film noir", "poetry meter", "french cuisine"],
    "Society & law": ["copyright law", "renewable energy policy", "urban planning", "voting systems", "human rights",
                      "tenant rights"],
    "Everyday life": ["sourdough bread", "houseplant care", "budget meal prep", "bike maintenance", "space telescopes",
                      "home composting", "learning guitar", "public speaking"],
}


def all_topics() -> list[str]:
    return [t for ts in SUBJECTS.values() for t in ts]


def capability_of(tool: str) -> str:
    for c in CAPABILITIES:
        if tool in c["tools"]:
            return c["id"]
    return {"chat": "chat", "needs_model": "generative"}.get(tool, "unknown")


def render_markdown() -> str:
    L = ["# K3 capabilities", "", "_Generated from `bot/training/catalog.py` (`python -m bot.train --docs`)._", "",
         "## Capabilities", ""]
    for c in CAPABILITIES:
        L.append(f"### {c['name']}  ({'works offline' if c['offline'] else 'needs a model provider'})")
        L.append(c["summary"])
        if c["tools"]:
            L.append("Tools: " + ", ".join(f"`{t}`" for t in c["tools"]))
        if c["examples"]:
            L.append("Try: " + "; ".join(f"`{e}`" for e in c["examples"]))
        L.append("")
    L += ["## Instruction styles the learner is trained on", ""]
    L += [f"- **{k}**: `{v}`" for k, v in INSTRUCTION_STYLES.items()]
    L += ["", "## Languages", ""]
    L += [f"- {v['name']} (`{k}`, {v['script']}): {v['support']}" for k, v in LANGUAGES.items()]
    L += ["", "## Task types", "", ", ".join(TASK_TYPES), "", "## Skills", ""]
    L += [f"- **{k}**: " + ", ".join(f"`{t}`" for t in v) for k, v in SKILLS.items()]
    L += ["", "## Subject areas (topic pool for training)", ""]
    L += [f"- **{k}**: " + ", ".join(v) for k, v in SUBJECTS.items()]
    return "\n".join(L) + "\n"


def render_summary(area: str = "") -> str:
    rows = [c for c in CAPABILITIES if not area or area.lower() in (c["id"], c["name"].lower())
            or area.lower() in c["name"].lower()]
    return "\n".join(f"- {c['name']} [{'offline' if c['offline'] else 'needs model'}]: {c['summary']}"
                     for c in rows) or f"No capability matches '{area}'."
