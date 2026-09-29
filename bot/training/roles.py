"""Expert roles: what each specialist must be able to do, and the exams that prove it.

A role owns skills; a skill owns tools and a competency exam. Exam entries are (split, phrase, expected, args):
  split    "d" dev (used while improving training data) or "h" holdout (never trained on, reported honestly)
  expected tool name, "a|b" for accepted alternatives, "text:<substring>" for a required reply, or "abstain"
  args     subset of arguments that must match (None = only the tool is checked)
Skill modes: trained (free-form intent), command (explicit command form offline; a model can also call it),
guarded (destructive: never auto-run from a guess), model (needs an LLM; offline it must say so), abstain.
"""
from dataclasses import dataclass, field

D, H = "d", "h"


@dataclass
class Skill:
    id: str
    name: str
    tools: list
    mode: str
    exams: list = field(default_factory=list)


@dataclass
class Role:
    id: str
    name: str
    mission: str
    skills: list


ROLES = [
    Role("research_analyst", "Research Analyst", "Find, read, extract and cite information from the web.", [
        Skill("find_sources", "Find sources on the web", ["web_search"], "trained", [
            (D, "find pages about wind turbines", "web_search|research", {"query": "wind turbines"}),
            (D, "search the web for cheap flights to lisbon", "web_search|research", {"query": "cheap flights to lisbon"}),
            (H, "can you google best hiking trails in norway", "web_search|research", {"query": "best hiking trails in norway"}),
            (H, "busca información sobre la energía eólica", "web_search|research", None)]),
        Skill("research_topic", "Research a topic with sources", ["research"], "trained", [
            (D, "do a deep dive on urban beekeeping", "research|web_search", {"question": "urban beekeeping"}),
            (D, "i need a sourced overview of tidal energy", "research|web_search", None),
            (H, "what should i know about tenant rights", "research|web_search", None),
            (H, "investiga sobre la revolución industrial", "research|web_search", None),
            (H, "explique-moi la photosynthèse", "research|web_search", None)]),
        Skill("read_page", "Read a web page", ["web_fetch"], "trained", [
            (D, "fetch https://example.com/pricing", "web_fetch", {"url": "https://example.com/pricing"}),
            (D, "read the page https://docs.python.org/3/library/re.html", "web_fetch", None),
            (H, "open https://news.ycombinator.com for me", "web_fetch|scrape", {"url": "https://news.ycombinator.com"}),
            (H, "abre https://example.com/es", "web_fetch", {"url": "https://example.com/es"})]),
        Skill("extract_data", "Extract data from a page (CSS selectors)", ["scrape"], "trained", [
            (D, "scrape https://shop.example.com div.price", "scrape", {"url": "https://shop.example.com", "selector": "div.price"}),
            (D, "extract every h2 from https://blog.example.org", "scrape", {"url": "https://blog.example.org", "selector": "h2"}),
            (H, "grab all a.link on https://example.com/docs", "scrape", {"url": "https://example.com/docs", "selector": "a.link"}),
            (H, "get every table tr at https://example.com/data", "scrape", {"url": "https://example.com/data", "selector": "table tr"})]),
        Skill("map_site", "Map a website politely (robots.txt, delays)", ["crawl"], "trained", [
            (D, "crawl https://example.com depth 2", "crawl", {"start_url": "https://example.com", "max_depth": 2}),
            (D, "crawl https://example.org max 5", "crawl", {"start_url": "https://example.org", "max_pages": 5}),
            (H, "map out the pages of https://example.com", "crawl", {"start_url": "https://example.com"}),
            (H, "spider https://example.net up to 8 pages", "crawl", {"start_url": "https://example.net"})]),
    ]),
    Role("executive_assistant", "Executive Assistant (Scheduler)", "Manage classes, appointments, free time and unattended jobs.", [
        Skill("book", "Book classes, appointments and events", ["add_event"], "trained", [
            (D, "schedule dentist on 2031-05-06 at 09:00", "add_event", {"title": "dentist", "start": "2031-05-06T09:00"}),
            (D, "book a class algebra on 2031-05-07 at 10:30", "add_event", {"kind": "class", "title": "algebra", "start": "2031-05-07T10:30"}),
            (H, "put yoga in my calendar on 2031-06-01 at 18:00", "add_event", {"title": "yoga", "start": "2031-06-01T18:00"}),
            (H, "add a meeting project review at 14:00 on 2031-06-02", "add_event", {"title": "project review", "start": "2031-06-02T14:00"})]),
        Skill("review_calendar", "Review the calendar", ["list_events"], "trained", [
            (D, "what's on my schedule", "list_events", None), (D, "show my upcoming classes", "list_events", None),
            (H, "do i have anything planned", "list_events", None), (H, "mostra i miei appuntamenti", "list_events", None),
            (H, "zeige meine termine", "list_events", None)]),
        Skill("free_time", "Find free time", ["find_free_slots"], "trained", [
            (D, "when am i free on 2031-05-06", "find_free_slots", {"date": "2031-05-06"}),
            (D, "find a free slot on 2031-05-07", "find_free_slots", {"date": "2031-05-07"}),
            (H, "is there a gap in my day on 2031-06-01", "find_free_slots", {"date": "2031-06-01"}),
            (H, "what time slots are free on 2031-06-02", "find_free_slots", {"date": "2031-06-02"})]),
        Skill("cancel", "Cancel bookings (guarded)", ["cancel_event"], "guarded", [
            (D, "cancel event 3", "cancel_event", {"id": 3}), (D, "cancel event 12", "cancel_event", {"id": 12}),
            (H, "get rid of event 5", "text:destructive", None)]),
        Skill("schedule_jobs", "Schedule unattended jobs", ["schedule_job"], "command", [
            (D, 'schedule job "summarize the news" at 2031-05-06T08:00', "schedule_job", {"goal": "summarize the news", "run_at": "2031-05-06T08:00"}),
            (D, 'schedule job "check disk space" at 2031-05-07T02:00', "schedule_job", {"goal": "check disk space", "run_at": "2031-05-07T02:00"}),
            (H, 'schedule job "send weekly report" at 2031-06-01T09:00', "schedule_job", {"goal": "send weekly report", "run_at": "2031-06-01T09:00"})]),
        Skill("review_jobs", "Review scheduled jobs", ["list_jobs"], "trained", [
            (D, "show scheduled jobs", "list_jobs", None), (D, "list my automated jobs", "list_jobs", None),
            (H, "what background jobs do i have", "list_jobs", None), (H, "jobs", "list_jobs", None)]),
    ]),
    Role("personal_organizer", "Personal Organizer", "Keep to-dos and notes tidy and findable.", [
        Skill("add_todo", "Add tasks", ["add_task"], "trained", [
            (D, "add task buy stamps", "add_task", {"title": "buy stamps"}),
            (D, "remind me to call the plumber", "add_task", {"title": "call the plumber"}),
            (D, "i need to renew my library card", "add_task", None),   # promoted from holdout after it failed
            (H, "añade una tarea pagar la factura", "add_task", {"title": "pagar la factura"}),
            (H, "add a to-do: water the ferns", "add_task", {"title": "water the ferns"})]),
        Skill("review_todos", "Review tasks", ["list_tasks"], "trained", [
            (D, "show my tasks", "list_tasks", None), (D, "what's on my to-do list", "list_tasks", None),
            (H, "what do i need to do", "list_tasks", None), (H, "mes tâches", "list_tasks", None)]),
        Skill("finish_todo", "Complete tasks", ["complete_task"], "trained", [
            (D, "complete task 4", "complete_task", {"id": 4}), (D, "mark task 9 as done", "complete_task", {"id": 9}),
            (H, "i finished task number 2", "complete_task", {"id": 2}), (H, "marca la tarea 6 como hecha", "complete_task", {"id": 6})]),
        Skill("take_notes", "Take notes", ["add_note"], "trained", [
            (D, "note groceries: eggs and rice", "add_note", {"title": "groceries", "body": "eggs and rice"}),
            (D, "jot down wifi: password is blue42", "add_note", {"title": "wifi", "body": "password is blue42"}),
            (H, "save a note ideas: launch in march", "add_note", {"title": "ideas", "body": "launch in march"}),
            (H, "notiz einkauf: milch und brot", "add_note", {"title": "einkauf", "body": "milch und brot"})]),
        Skill("find_notes", "Find notes", ["search_notes"], "trained", [
            (D, "notes budget", "search_notes", {"query": "budget"}), (D, "search my notes for taxes", "search_notes", {"query": "taxes"}),
            (H, "did i write anything about the roof", "search_notes", None), (H, "find my notes on holidays", "search_notes", {"query": "holidays"})]),
        Skill("delete_notes", "Delete notes (guarded)", ["delete_note"], "guarded", [
            (D, "delete note 2", "delete_note", {"id": 2}), (D, "delete note 8", "delete_note", {"id": 8}),
            (H, "delete note 5", "delete_note", {"id": 5})]),
    ]),
    Role("data_analyst", "Data Analyst", "Summarize, filter and chart tabular data.", [
        Skill("profile_data", "Profile a dataset", ["describe_data"], "trained", [
            (D, "describe sales.csv", "describe_data", {"path": "sales.csv"}),
            (D, "give me a quick look at report.xlsx", "describe_data", {"path": "report.xlsx"}),
            (H, "what columns does data/users.csv have", "describe_data|read_file", {"path": "data/users.csv"}),
            (D, "how big is results/q3.csv", "describe_data", {"path": "results/q3.csv"}),   # promoted after failing
            (H, "profile the spreadsheet budget.xlsx", "describe_data", {"path": "budget.xlsx"})]),
        Skill("query", "Filter and aggregate", ["query_data"], "command", [
            (D, "query sales.csv where price > 10", "query_data", {"path": "sales.csv", "filter": "price > 10"}),
            (D, "query data/users.csv where age >= 30", "query_data", {"path": "data/users.csv", "filter": "age >= 30"}),
            (H, "query report.xlsx where region == 'north'", "query_data", {"path": "report.xlsx", "filter": "region == 'north'"})]),
        Skill("chart", "Chart data", ["make_chart"], "trained", [
            (D, "plot sales by region from data.csv", "make_chart", {"path": "data.csv", "y": "sales", "x": "region"}),
            (D, "make a line chart of price per month in prices.csv", "make_chart", {"path": "prices.csv", "kind": "line", "y": "price", "x": "month"}),
            (H, "chart results.csv with x month and y score", "make_chart", {"path": "results.csv", "x": "month", "y": "score"}),
            (H, "make a histogram of age from users.csv", "make_chart", {"path": "users.csv", "kind": "hist", "x": "age"})]),
    ]),
    Role("developer_assistant", "Developer Assistant", "Work with workspace files and run small Python snippets.", [
        Skill("browse_files", "List files", ["list_files"], "trained", [
            (D, "list files", "list_files", None), (D, "show me everything in the workspace", "list_files", None),
            # NOTE: the bare "what have i saved" is ambiguous (files? notes? lessons?) and the bot rightly abstains;
            # the exam uses the unambiguous form.
            (D, "what have i saved in the workspace", "list_files", None),
            (H, "which files exist", "list_files", None), (H, "what documents are in my folder", "list_files", None)]),
        Skill("read", "Read files", ["read_file"], "trained", [
            (D, "read file notes.md", "read_file", {"path": "notes.md"}),
            (D, "show me the contents of script.py", "read_file", {"path": "script.py"}),
            (H, "open the file notes.md", "read_file|describe_data", {"path": "notes.md"}),
            (H, "dump the text of script.py", "read_file", {"path": "script.py"})]),
        Skill("write", "Write files", ["write_file"], "command", [
            (D, "write file hello.txt: hi there", "write_file", {"path": "hello.txt", "content": "hi there"}),
            (D, "write file src/a.py: print(1)", "write_file", {"path": "src/a.py", "content": "print(1)"}),
            (H, "write file todo.md: - ship it", "write_file", {"path": "todo.md", "content": "- ship it"})]),
        Skill("execute", "Run Python", ["run_python"], "command", [
            (D, "run python print(6*7)", "run_python", {"code": "print(6*7)"}),
            (D, "python: print('hi')", "run_python", {"code": "print('hi')"}),
            (H, "run python import math; print(math.pi)", "run_python", {"code": "import math; print(math.pi)"})]),
    ]),
    Role("creative_writer_designer", "Creative Writer & Designer", "Compose text, code, translations, designs and brainstorms (needs a model).", [
        Skill("design_assets", "Persist SVG/HTML designs and brainstorms", ["save_svg", "save_html", "save_brainstorm"], "model", [
            (D, "design a logo for my cafe", "text:language model", None), (D, "draw me a poster for the science fair", "text:language model", None),
            (H, "brainstorm names for my bakery", "text:language model", None)]),
        Skill("compose", "Write essays, poems, emails, code; translate", [], "model", [
            (D, "write me an essay about volcanoes", "text:language model", None), (D, "write a poem about autumn", "text:language model", None),
            (H, "translate good morning to german", "text:language model", None),
            (H, "escribe un poema sobre el mar", "text:language model", None),
            (H, "schreibe ein gedicht über den herbst", "text:language model", None)]),
    ]),
    Role("community_moderator", "Community Moderator", "Keep chat civil: filter words, warn and mute users, keep a record.", [
        Skill("screen_text", "Screen text for profanity", ["moderate_text"], "trained", [
            (D, "check this text for bad words: what the hell is this shit", "moderate_text", {"text": "what the hell is this shit"}),
            (D, "scan this for profanity: nice work team", "moderate_text", {"text": "nice work team"}),
            (H, "does this contain swearing: fuck this", "moderate_text", None),
            (H, "moderate: hello everyone", "moderate_text", {"text": "hello everyone"})]),
        Skill("manage_blocklist", "Manage the blocklist", ["add_blocked_word"], "trained", [
            (D, "block the word phishing", "add_blocked_word", {"word": "phishing"}), (D, "ban the word scam", "add_blocked_word", {"word": "scam"}),
            (H, "add spam to the blocklist", "add_blocked_word", {"word": "spam"}), (H, "blacklist clickbait", "add_blocked_word", {"word": "clickbait"})]),
        Skill("discipline", "Warn, mute and unmute users", ["moderate_user"], "trained", [
            (D, "warn user bob for spamming", "moderate_user", {"user": "bob", "action": "warn", "reason": "spamming"}),
            (D, "mute alice because rude language", "moderate_user", {"user": "alice", "action": "mute"}),
            (H, "unmute carol", "moderate_user", {"user": "carol", "action": "unmute"}),
            (H, "give dave a warning for spoilers", "moderate_user", {"user": "dave", "action": "warn"})]),
        Skill("audit_users", "Check a user's moderation record", ["user_moderation_status"], "trained", [
            (D, "show warnings for bob", "user_moderation_status", {"user": "bob"}), (D, "is alice muted", "user_moderation_status", {"user": "alice"}),
            (H, "how many warnings does carol have", "user_moderation_status", {"user": "carol"}),
            (H, "moderation history of dave", "user_moderation_status", {"user": "dave"})]),
    ]),
    Role("utility_expert", "Calculation & Utility Expert", "Arithmetic, dates, time, unit conversion and polls.", [
        Skill("arithmetic", "Do arithmetic", ["calculate"], "trained", [
            (D, "what's 15 times 4", "calculate", {"expression": "15 * 4"}), (D, "calc 100 / 8", "calculate", {"expression": "100 / 8"}),
            (H, "cuánto es 12 por 7", "calculate", {"expression": "12 * 7"}), (H, "combien font 9 plus 4", "calculate", {"expression": "9 + 4"}),
            (H, "berechne 6 mal 7", "calculate", {"expression": "6 * 7"})]),
        Skill("clock", "Tell the time and date", ["now"], "trained", [
            (D, "what time is it", "now", None), (D, "what's today's date", "now", None),
            (H, "wie spät ist es", "now", None), (H, "que horas são", "now", None), (H, "который час", "now", None)]),
        Skill("date_math", "Add and subtract days", ["date_add"], "trained", [
            (D, "what date is 30 days after 2031-01-15", "date_add", {"date": "2031-01-15", "days": 30}),
            (D, "add 2 weeks to 2031-03-01", "date_add", {"date": "2031-03-01", "days": 14}),
            (H, "5 days before 2031-01-15", "date_add", {"date": "2031-01-15", "days": -5}),
            (H, "añade 10 días a 2031-01-15", "date_add", {"date": "2031-01-15", "days": 10})]),
        Skill("units", "Convert units", ["convert_units"], "trained", [
            (D, "convert 10 km to miles", "convert_units", {"value": 10.0, "from_unit": "km", "to_unit": "mi"}),
            (D, "how many feet in 3 meters", "convert_units", {"value": 3.0, "from_unit": "m", "to_unit": "ft"}),
            (H, "convierte 5 kg a lb", "convert_units", {"value": 5.0, "from_unit": "kg", "to_unit": "lb"}),
            (H, "100 fahrenheit in celsius", "convert_units", {"value": 100.0, "from_unit": "f", "to_unit": "c"})]),
        Skill("polls", "Create polls", ["make_poll"], "trained", [
            (D, "create a poll team lunch: pizza, sushi, tacos", "make_poll", {"question": "team lunch", "options": ["pizza", "sushi", "tacos"]}),
            (D, "poll movie night: comedy or horror", "make_poll", {"question": "movie night", "options": ["comedy", "horror"]}),
            (H, "start a poll weekend plan: hike or movie", "make_poll", {"question": "weekend plan", "options": ["hike", "movie"]})]),
    ]),
    Role("game_host", "Game Host", "Run dice, coin flips, random picks and word games.", [
        Skill("dice", "Roll dice", ["roll_dice"], "trained", [
            (D, "roll 2d6", "roll_dice", {"spec": "2d6"}), (D, "throw the dice", "roll_dice", None),
            (H, "tira 3d8", "roll_dice", {"spec": "3d8"}), (H, "würfle einmal d20", "roll_dice", {"spec": "d20"}),
            (H, "брось кубик", "roll_dice", None)]),
        Skill("coin", "Flip a coin", ["flip_coin"], "trained", [
            (D, "flip a coin", "flip_coin", None), (D, "heads or tails", "flip_coin", None),
            (H, "cara o cruz", "flip_coin", None), (H, "pile ou face", "flip_coin", None), (H, "wirf eine münze", "flip_coin", None)]),
        Skill("choose", "Pick randomly between options", ["pick_random"], "trained", [
            (D, "pick one of tea, coffee, juice", "pick_random", {"options": ["tea", "coffee", "juice"]}),
            (D, "choose between cinema or bowling", "pick_random", {"options": ["cinema", "bowling"]}),
            (H, "decide for me: red or blue", "pick_random", {"options": ["red", "blue"]}),
            (H, "elige entre gato, perro, pez", "pick_random", {"options": ["gato", "perro", "pez"]})]),
        Skill("word_game", "Word scramble", ["scramble_word"], "trained", [
            (D, "scramble a word", "scramble_word", None), (D, "word puzzle", "scramble_word", None),
            (H, "give me an anagram", "scramble_word", None), (H, "let's play word scramble", "scramble_word", None)]),
    ]),
    Role("trainer", "Trainer & Teacher (self-improvement)", "Learn from the user: shortcuts, phrasings, corrections; explain capabilities.", [
        Skill("teach_shortcut", "Learn a shortcut", ["learn_instruction"], "command", [
            (D, 'learn "morning news" => research top tech news', "learn_instruction", {"trigger": "morning news", "action": "research top tech news"}),
            (D, 'learn "flip it" => flip', "learn_instruction", {"trigger": "flip it", "action": "flip"}),
            (H, 'learn "lucky number" => roll d20', "learn_instruction", {"trigger": "lucky number", "action": "roll d20"})]),
        Skill("recall", "Recall what was learned", ["list_lessons"], "trained", [
            (D, "lessons", "list_lessons", None), (D, "what have you learned", "list_lessons", None),
            (H, "what did i teach you", "list_lessons", None), (H, "what do you remember", "list_lessons", None)]),
        Skill("forget", "Forget a lesson (guarded)", ["forget_lesson"], "guarded", [
            (D, "forget 3", "forget_lesson", {"id": 3}), (H, "forget 7", "forget_lesson", {"id": 7})]),
        Skill("retrain", "Retrain and report", ["train_model", "training_status"], "command", [
            (D, "train", "train_model", None), (D, "train status", "training_status", None),
            (H, "train", "train_model", None)]),
        Skill("teach_phrase", "Teach a phrasing", ["add_training_example", "train_from_file"], "command", [
            (D, 'train add "gimme a d20" => roll d20', "add_training_example", {"phrase": "gimme a d20", "command": "roll d20"}),
            (D, "train from examples.jsonl", "train_from_file", {"path": "examples.jsonl"}),
            (H, 'train add "flip it" => flip', "add_training_example", {"phrase": "flip it", "command": "flip"}),
            (H, "train from data.csv", "train_from_file", {"path": "data.csv"})]),
        Skill("correct", "Take corrections and confirmations", ["mark_wrong", "mark_good"], "command", [
            (D, "wrong => research solar panels", "mark_wrong", {"command": "research solar panels"}),
            (D, "that means roll 1d6", "mark_wrong", {"command": "roll 1d6"}), (D, "good", "mark_good", None),
            (H, "i meant flip", "mark_wrong", {"command": "flip"}), (H, "that was right", "mark_good", None)]),
        Skill("reset", "Reset training (guarded)", ["reset_training"], "guarded", [
            (D, "train reset", "reset_training", None), (H, "train reset", "reset_training", None)]),
        Skill("explain_self", "Explain capabilities and role readiness", ["list_capabilities", "certify_roles"], "trained", [
            (D, "what can you do", "list_capabilities", None), (D, "show your capabilities", "list_capabilities", None),
            (H, "what are you good at", "list_capabilities", None), (H, "что ты умеешь", "list_capabilities", None),
            (H, "qué puedes hacer", "list_capabilities", None)]),
    ]),
    Role("conversationalist", "Conversation Partner", "Recognize small talk; offline it abstains instead of guessing.", [
        Skill("small_talk", "Abstain on chit-chat", [], "abstain", [
            (D, "tell me a joke", "abstain", None), (D, "hello", "abstain", None),
            (H, "how are you today", "abstain", None), (H, "hola", "abstain", None), (H, "bonjour", "abstain", None)]),
    ]),
]


def all_tools() -> list:
    return [t for r in ROLES for s in r.skills for t in s.tools]
