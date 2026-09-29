"""Synthetic training data: templates x paraphrases x slot fillers x languages, with light noise.

`split="train"` drops every 4th template of each tool/language so `split="test"` measures generalization
to phrasings never seen in training. Examples carry `lang` and `cap` (capability) tags for reporting.
"""
import random
import re

from bot.training import catalog, i18n

TOPICS = catalog.all_topics()
URLS = ["https://example.com", "https://news.ycombinator.com", "https://docs.python.org/3/",
        "https://en.wikipedia.org/wiki/Python", "https://blog.example.org/posts", "https://www.bbc.com/news",
        "https://arxiv.org/abs/1706.03762"]
SELECTORS = ["h1", "p.x", "a.link", "div.price", "ul li", "h2", "span.title", "table tr"]
DICE = ["2d6", "d20", "3d8", "1d10", "d6", "4d4"]
_ACTS = ["buy", "call", "email", "send", "finish", "book", "pay", "renew", "prepare", "fix", "clean", "write", "review",
         "schedule", "return", "pick up", "order", "cancel", "update", "print"]
_OBJS = ["the invoice", "milk", "mom", "the report", "flights", "the rent", "my passport", "the slides", "the bike",
         "the kitchen", "a summary", "the contract", "the parcel", "groceries", "the dentist", "tax forms", "the landlord"]
TITLES = [f"{a} {o}" for a in _ACTS for o in _OBJS][::4] + ["buy milk", "email the team", "finish the report", "call mom", "book flights", "water the plants",
          "renew passport", "prepare slides", "study for the exam", "pay the rent"]
EVENT_TITLES = ["chemistry", "spanish class", "board meeting", "gym session", "book club", "coffee with sam", "tax review",
                "design workshop", "language exchange", "guitar practice", "vet visit", "sprint planning", "dentist", "math", "team sync", "yoga", "piano practice", "physics", "haircut", "project review"]
BODIES = ["eggs, bread, butter", "ideas for the launch", "password hint is the dog's name",
          "meeting moved to friday", "read chapter three"]
PATHS = ["sales.csv", "data/users.csv", "report.xlsx", "results/q3.csv", "notes.md", "script.py"]
UNITS = [("km", "km"), ("kilometers", "km"), ("mi", "mi"), ("miles", "mi"), ("m", "m"), ("meters", "m"),
         ("ft", "ft"), ("feet", "ft"), ("kg", "kg"), ("kilograms", "kg"), ("lb", "lb"), ("pounds", "lb"),
         ("g", "g"), ("grams", "g"), ("oz", "oz"), ("ounces", "oz"), ("c", "c"), ("celsius", "c"),
         ("f", "f"), ("fahrenheit", "f")]
GROUPS = [["km", "mi", "m", "ft"], ["kg", "lb", "g", "oz"], ["c", "f"]]
DATES = ["2030-03-05", "2030-11-21", "2031-01-09", "2030-07-14"]
TIMES = ["09:00", "14:00", "10:30", "16:15", "08:45"]
KINDS = [("class", "class"), ("lecture", "class"), ("appointment", "appointment"), ("meeting", "event"),
         ("event", "event")]
OPTIONS = [("pizza, sushi, tacos", ["pizza", "sushi", "tacos"]), ("red or blue", ["red", "blue"]),
           ("tea, coffee or juice", ["tea", "coffee", "juice"]), ("cinema or bowling", ["cinema", "bowling"]),
           ("rock, paper, scissors", ["rock", "paper", "scissors"])]
USERS = ["bob", "alice", "dave42", "mallory", "carol", "erin", "frank", "grace", "henry", "ivan", "judy", "kevin",
         "laura", "mike", "nina", "oscar", "peggy", "quinn", "rob", "sara"]
REASONS = ["spamming links", "rude language", "off topic posts", "posting spoilers"]
MOD_VERBS = [("lift the mute on", "unmute"), ("restore the chat privileges of", "unmute"), ("release", "unmute"),
             ("take the mute off", "unmute"), ("gag", "mute"), ("time out", "mute"), ("timeout", "mute"), ("warn", "warn"), ("give a warning to", "warn"), ("issue a warning to", "warn"), ("mute", "mute"),
             ("silence", "mute"), ("unmute", "unmute")]
BLOCK_WORDS = ["spoiler", "crypto", "lottery", "casino", "giveaway", "phishing", "scam", "spam", "clickbait", "viagra",
               "gambling", "airdrop", "jackpot", "payday", "porn", "bitcoin", "pyramid", "coupon"]
POLL_QS = ["office snacks", "best season", "game night", "team offsite", "lunch place", "movie night", "logo color",
           "meeting day", "favorite fruit", "best movie", "team lunch spot", "weekend plan", "project name"]
CHART_COLS = ["revenue", "team", "cost", "quarter", "year", "units", "rating", "price", "sales", "region", "month", "age", "score"]
CHART_KINDS = [("bar", "bar"), ("line", "line"), ("histogram", "hist")]
OFFENSIVE = ["what the hell is this shit", "you are an asshole", "this is bullshit", "fuck this", "great post thanks",
             "nice work team", "see you all tomorrow", "you are all idiots", "this thread is useless", "welcome everyone"]
CHAT = {
    "en": ["tell me a joke", "how are you today", "hello", "hi there", "who are you", "what is the meaning of life",
           "thanks a lot", "good morning", "you are awesome", "what's your favorite color", "i'm feeling sad",
           "give me advice about my career", "what do you think about pineapple on pizza", "sing me a song",
           "goodbye", "let's chat", "ok", "nevermind", "that's funny", "blah blah", "hmm",
           "lol", "cool thanks", "what's the weather like", "play some music", "order me a pizza",
           "call my mother", "are you a robot", "good night", "i love you", "you're wrong", "wow", "really",
           "how was your day", "what are you doing", "i'm bored", "that's interesting", "you're funny", "not really",
           "maybe later", "see you tomorrow", "nice to meet you", "i don't know", "sounds good", "no thanks",
           "yes please", "what's up", "long time no see", "you are so smart"],
    **{l: t["chat"] for l, t in i18n.LANG_TEMPLATES.items()},
}
PREFIXES = ["", "", "", "please ", "hey k3, ", "can you ", "could you ", "k3 ", "i'd like you to ", "i want you to ",
            "would you mind "]
SUFFIXES = ["", "", "", " please", " thanks", " for me", " asap", " right now"]


def _expr(rng: random.Random, lang: str) -> tuple[str, str]:
    if rng.random() < 0.25:
        return rng.choice([("(3+4)*5", "(3+4)*5"), ("18 * 3", "18 * 3"), ("250 / 5", "250 / 5"),
                           ("15 + 27", "15 + 27")])
    sym, words = rng.choice(i18n.OPWORDS.get(lang, i18n.OPWORDS["en"]))
    a, b = rng.randint(2, 99), rng.randint(2, 99)
    return f"{a} {rng.choice(words)} {b}", f"{a} {sym} {b}"


def _fields(rng: random.Random, lang: str = "en") -> dict:
    et, es = _expr(rng, lang)
    group = rng.choice(GROUPS)
    a, b = rng.sample(group, 2)
    surface = lambda canon: rng.choice([s for s, c in UNITS if c == canon])
    kw, kind = rng.choice(KINDS)
    mv = rng.choice(i18n.MOD_VERBS.get(lang) or MOD_VERBS)
    loc = i18n.FILLERS.get(lang)
    local = {k: rng.choice(v) for k, v in loc.items()} if loc and rng.random() < 0.75 else {}
    return dict(topic=rng.choice(TOPICS), url=rng.choice(URLS), sel=rng.choice(SELECTORS), expr=et, expr_sym=es,
                dice=rng.choice(DICE), title=rng.choice(TITLES), body=rng.choice(BODIES),
                path=rng.choice(PATHS), id=rng.randint(1, 30), d=rng.randint(1, 3), n=rng.choice([5, 10, 20, 30]),
                value=rng.choice([5, 10, 12.5, 100, 3, 72, 20]), fu=surface(a), fu_c=a, tu=surface(b), tu_c=b,
                date=rng.choice(DATES), time=rng.choice(TIMES), kind=kw, kind_c=kind,
                days=rng.choice([3, 7, 10, 14, 30, 45, 90]), user=rng.choice(USERS), reason=rng.choice(REASONS),
                verb=mv[0], verb_c=mv[1], word=rng.choice(BLOCK_WORDS), q=rng.choice(POLL_QS),
                cx=(cc := rng.sample(CHART_COLS, 2))[0], cy=cc[1], ckind=(ck := rng.choice(CHART_KINDS))[0], ckind_c=ck[1], opts=(o := rng.choice(OPTIONS))[0], opts_l=o[1],
                etitle=rng.choice(EVENT_TITLES), off=rng.choice(OFFENSIVE), mins=rng.choice([30, 45, 60, 90]),
                ) | local


def _end(f: dict) -> str:
    from datetime import datetime, timedelta
    return (datetime.fromisoformat(f"{f['date']}T{f['time']}") + timedelta(minutes=60)).isoformat(timespec="minutes")


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



EXTRA_EN = {
    "research": ["explain {topic}", "what is {topic}", "teach me about {topic}", "tell me about {topic}",
                 "give me an overview of {topic}", "how does {topic} work"],
    "certify_roles": ["which experts are ready", "show role readiness", "how well trained are your roles",
                      "certify your roles", "are you ready for each role", "role certification report",
                      "check which specialists are trained", "give me the readiness of your experts"],
    "list_capabilities": ["what can you do", "show your capabilities", "what are your skills", "list your features",
                          "what are you able to do", "help me see what you can do"],
    "needs_model": ["draw me a poster for the school fair", "create a flyer for my event", "make a banner for my shop",
                    "sketch a floor plan of my flat", "design a website for my shop", "design a business card",
                    "write me an essay about {topic}", "write a python function that sorts a list",
                    "translate hello to french", "brainstorm ideas for {topic}", "design a logo for my cafe",
                    "write a poem about {topic}", "help me debug my code", "draft an email to my boss",
                    "compose a story about {topic}", "create a business plan for a bakery", "write a cover letter",
                    "summarize this article for me", "write me a song about {topic}", "come up with names for my dog"],
    "add_event": ["schedule {etitle} on {date} at {time}", "book a {kind} {etitle} on {date} at {time}",
                  "add a {kind} {etitle} at {time} on {date}", "put {etitle} in my calendar on {date} at {time}",
                  "set up a {kind}: {etitle}, {date} {time}", "i have a {kind} {etitle} on {date} at {time}",
                  "create {kind} {etitle} for {date} {time}", "block {date} at {time} for {etitle}"],
    "find_free_slots": ["when am i free on {date}", "find a free slot on {date}", "any free time on {date}",
                        "what's open on {date}", "show free slots for {date}", "am i free on {date}"],
    "moderate_text": ["is this offensive: {off}", "check this text for bad words: {off}", "moderate: {off}",
                      "scan this for profanity: {off}", "does this contain swearing: {off}", "filter this: {off}"],
    "list_jobs": ["show scheduled jobs", "what jobs are queued", "list my automated jobs", "any pending autonomous jobs",
                  "show job results", "what background jobs do i have", "are there scheduled tasks running later",
                  "list the unattended jobs"],
    "add_blocked_word": ["block the word {word}", "add {word} to the blocklist", "ban the word {word}",
                         "filter out the word {word}", "blacklist {word}", "add {word} to the banned words",
                         "don't allow the word {word}", "start filtering {word}"],
    "moderate_user": ["{verb} user {user}", "{verb} {user} for {reason}", "{verb} {user} because {reason}",
                      "please {verb} {user}", "{verb} {user}: {reason}", "{verb} the user {user} for {reason}"],
    "user_moderation_status": ["what is {user}'s moderation status", "show warnings for {user}",
                               "how many warnings does {user} have", "moderation history of {user}",
                               "is {user} muted", "check {user}'s moderation record", "show the mod log for {user}",
                               "what has {user} been warned about"],
    "make_poll": ["create a poll {q}: {opts}", "make a poll about {q} with options {opts}", "poll {q}: {opts}",
                  "start a poll {q}: {opts}", "set up a vote on {q}: {opts}", "run a poll on {q} with options {opts}"],
    "make_chart": ["plot {cy} by {cx} from {path}", "make a {ckind} chart of {cy} per {cx} in {path}",
                   "chart {path} with x {cx} and y {cy}", "draw a {ckind} chart from {path} with x {cx} and y {cy}",
                   "graph {cy} against {cx} using {path}", "visualize {cy} over {cx} in {path}",
                   "show a histogram of {cx} in {path}", "make a histogram of {cx} from {path}"],
    "complete_task": ["mark task {id} as done", "complete task {id}", "finish task {id}", "task {id} is done",
                      "check off task {id}", "tick off task {id}", "i finished task number {id}", "done with task {id}"],
    "date_add": ["what date is {days} days after {date}", "add {days} days to {date}", "{days} days from {date}",
                 "what is {days} days after {date}", "count {days} days forward from {date}",
                 "which day is {days} days past {date}"],
    "pick_random": ["pick one of {opts}", "choose between {opts}", "randomly choose from {opts}",
                    "decide for me: {opts}", "help me choose: {opts}", "select one at random from {opts}"],
    "read_file": ["show file {path}", "read the file {path}", "cat {path}", "display the contents of {path}",
                  "view the text of {path}", "print out {path}"],
}
EXTRA_ARGS = {
    "add_event": {"kind": ("etitle", lambda f: f["kind_c"]),
                  "title": ("etitle", lambda f: f["etitle"]), "start": ("etitle", lambda f: f"{f['date']}T{f['time']}"),
                  "end": ("etitle", _end)},
    "find_free_slots": {"date": ("date", None)},
    "moderate_text": {"text": ("off", None)},
    "read_file": {"path": ("path", None)},
    "complete_task": {"id": ("id", None)},
    "list_jobs": {}, "certify_roles": {}, "add_blocked_word": {"word": ("word", None)},
    "moderate_user": {"user": ("user", None), "action": ("verb", lambda f: f["verb_c"]), "reason": ("reason", None)},
    "user_moderation_status": {"user": ("user", None)},
    "make_poll": {"question": ("q", None), "options": ("opts", lambda f: f["opts_l"])},
    "make_chart": {"path": ("path", None), "x": ("cx", None), "y": ("cy", None), "kind": ("ckind", lambda f: f["ckind_c"])},
    "date_add": {"date": ("date", None), "days": ("days", None)},
    "pick_random": {"options": ("opts", lambda f: f["opts_l"])},
    "research": {"question": ("topic", None)},
    "list_capabilities": {}, "needs_model": {},
}
for _t, _extra in EXTRA_EN.items():
    if _t in SPECS:
        SPECS[_t][0].extend(_extra)
    else:
        SPECS[_t] = (list(_extra), EXTRA_ARGS[_t])


# More phrasing variety per tool (breadth of wording is what lets the learner handle unseen requests).
MORE_EN = {
    "web_search": ["look for {topic} on the internet", "i want to find articles about {topic}", "find pages about {topic}",
                   "run a web search for {topic}", "search online for {topic}", "get me some links on {topic}",
                   "bing {topic}", "any websites about {topic}"],
    "research": ["research the topic of {topic}", "put together a report on {topic}", "i need to learn about {topic}",
                 "read up on {topic} for me", "look into {topic} thoroughly", "collect sources about {topic}",
                 "what do experts say about {topic}", "prepare a literature review of {topic}", "brief me on {topic}",
                 "find reliable information about {topic}"],
    "web_fetch": ["get me the text of {url}", "can you read the page {url}", "pull the content from {url}",
                  "visit the site {url} and tell me what it says", "grab {url}", "browse to {url}",
                  "summarize the page {url}", "look at {url}", "check out {url}", "what is on {url}"],
    "scrape": ["scrape all the {sel} from {url}", "collect every {sel} on {url}", "harvest {sel} elements from {url}"],
    "describe_data": ["give me an overview of the dataset {path}", "how many rows and columns are in {path}",
                      "show the columns of {path}", "what does the data in {path} look like",
                      "summary statistics for {path}", "explore the table {path}"],
    "read_file": ["show me the contents of {path}", "open {path} and print it", "type out {path}", "read {path}",
                  "let me see the file {path}", "dump the text of {path}"],
    "find_free_slots": ["what does my schedule look like on {date}", "do i have any openings on {date}",
                        "find me an open time on {date}", "which times are available on {date}",
                        "is there a gap in my day on {date}", "when can i fit a meeting on {date}"],
    "list_capabilities": ["what are you capable of", "tell me what you can help with", "show me your features",
                          "what commands do you support", "what tools do you have", "give me a list of things you do",
                          "how can you help me", "show help"],
    "flip_coin": ["toss a coin", "let's flip a coin", "heads or tails?", "flip it", "coin toss please",
                  "decide with a coin"],
    "list_files": ["show me everything in the workspace", "what documents do i have", "list my files",
                   "which files exist", "display the file list", "what's in my folder"],
    "crawl": ["crawl {url}", "spider the website {url} up to {n} pages", "scan the whole site at {url}",
              "follow the links from {url} {d} levels", "index {url} recursively"],
    "list_events": ["what's next on my agenda", "do i have meetings today", "show my upcoming appointments",
                    "what's scheduled this week", "read out my calendar", "what classes and events are coming"],
    "search_notes": ["look through my notes about {topic}", "which notes mention {topic}", "find what i noted on {topic}",
                     "retrieve my notes on {topic}", "notes about {topic}"],
    "add_task": ["put {title} on my todo list", "i should {title}", "remind me to {title} later", "add {title} to my todo",
                 "queue up a task: {title}", "todo {title}"],
    "now": ["what's the time right now", "tell me the current date", "what is today", "time please", "what date is it"],
    "needs_model": ["write a blog post about {topic}", "generate a python script that renames files",
                    "translate this paragraph into german", "invent a story for my kids", "write a haiku",
                    "help me brainstorm a startup idea", "draw me a logo", "proofread my essay",
                    "rewrite this sentence to sound formal", "write unit tests for my function",
                    "explain this code line by line", "make me a workout plan"],
}
for _t, _extra in MORE_EN.items():
    SPECS[_t][0].extend(_extra)


# Second variety pass, targeted at the tools that scored lowest on held-out wordings.
MORE_EN2 = {
    "add_task": ["i have to {title}", "i should probably {title}", "gotta {title}", "make sure i {title}",
                 "add an item: {title}"],
    "describe_data": ["what size is {path}", "how large is {path}", "how many rows does {path} have",
                      "what's the structure of {path}", "check the shape of {path}"],
    "list_files": ["what have i saved so far", "show me what i've saved", "what did i save in the workspace",
                   "what's in my workspace", "what do i have stored", "show what files i've created",
                   "which documents have i created", "list what's saved in the workspace"],
    "find_free_slots": ["are there any free hours on {date}", "what time slots are free on {date}", "check my availability on {date}",
                        "when is my schedule empty on {date}", "show open times for {date}", "what's my availability for {date}",
                        "can you find a gap on {date}", "find a time on {date} when nothing is booked"],
    "list_files": ["list all files", "show the files in my workspace", "what have i saved", "ls workspace",
                   "give me a directory listing", "which files are there", "files please", "show me my documents folder"],
    "list_capabilities": ["what features do you offer", "how do you work", "what can i ask you", "what are you good at",
                          "tell me about yourself and what you do", "capabilities", "what services do you provide",
                          "what kinds of tasks can you handle"],
    "web_fetch": ["fetch the page {url}", "show me what {url} contains", "load the webpage {url}", "read me {url}",
                  "bring up {url}", "extract the text from {url}", "get the article at {url}", "download and read {url}"],
    "describe_data": ["describe the spreadsheet {path}", "show me stats about {path}", "what columns does {path} have",
                      "profile the dataset {path}", "give me a quick look at {path}", "analyze {path} for me",
                      "how big is {path}", "give me the shape and types of {path}"],
    "list_events": ["what's on my schedule", "show me my upcoming classes", "do i have anything planned", "what's on for today",
                    "list my meetings", "what's booked this week", "tell me my appointments", "show my agenda"],
    "search_notes": ["find my notes on {topic}", "search notes for {topic}", "what notes do i have about {topic}",
                     "look up {topic} in my notes", "did i write anything about {topic}", "pull up my notes about {topic}",
                     "check my notes for {topic}", "show notes mentioning {topic}"],
    "add_event": ["schedule a {kind} for {etitle} on {date} at {time}", "reserve {date} at {time} for {etitle}",
                  "add {etitle} to my calendar on {date} at {time}", "i need a {kind} called {etitle} on {date} at {time}",
                  "please book {etitle} for {date} at {time}", "make a calendar entry {etitle} {date} {time}",
                  "plan a {kind}: {etitle} on {date} at {time}", "{date} at {time}: {etitle}"],
    "add_task": ["add a to-do: {title}", "please add {title} to my task list", "note that i have to {title}",
                 "create a reminder to {title}", "add to my todos: {title}", "i need to remember to {title}",
                 "put a task for {title}", "new todo item {title}"],
    "web_search": ["find web pages about {topic}", "find articles on {topic}", "find sites about {topic}",
                   "get me web pages on {topic}", "look for websites covering {topic}",
                   "can you google {topic}", "search the internet for {topic}", "find websites on {topic}",
                   "lookup {topic} online", "what can i find about {topic} on the web", "search {topic} for me",
                   "show me web results for {topic}", "search up {topic}"],
    "research": ["can you research {topic}", "do a deep dive on {topic}", "give me a summary of {topic} with sources",
                 "what should i know about {topic}", "help me understand {topic}", "write a fact sheet on {topic}",
                 "find and read sources about {topic}", "study up on {topic}"],
    "scrape": ["get all {sel} elements from {url}", "extract every {sel} from the page {url}", "scrape the {sel} tags at {url}"],
}
for _t, _extra in MORE_EN2.items():
    SPECS[_t][0].extend(_extra)


# Compositional paraphrases: verb phrase x object phrase cross products give broad, sensible wording variety
# (far more than hand-listing) without copying any exam phrase.
def _cross(verbs, objs, sep=" "):
    return [f"{v}{sep}{o}" for v in verbs for o in objs]


COMPOSED = {
    "moderate_text": _cross(["check", "scan", "test", "look", "detect", "screen"],
                            ["for bad language: {off}", "for profanity: {off}", "for swearing: {off}",
                             "for offensive words: {off}", "for rude words: {off}"])
                     + _cross(["is this", "does this look", "tell me whether this is", "check if this is"],
                            ["rude: {off}", "offensive: {off}", "abusive: {off}", "toxic: {off}", "inappropriate: {off}"])
                     + _cross(["filter", "review", "screen", "vet", "moderate"],
                              ["this comment: {off}", "this message: {off}", "this post: {off}", "this reply: {off}"]),
    "web_search": _cross(["find", "look for", "hunt down", "search for", "dig up", "locate", "look up", "seek out", "get me"],
                         ["links about {topic}", "pages on {topic}", "websites about {topic}", "articles regarding {topic}",
                          "results for {topic}", "sites covering {topic}", "reviews of {topic}", "info online about {topic}"]),
    "research": _cross(["write up", "give me an overview of", "brief me on", "summarize", "get me up to speed on",
                        "explain", "dig into", "report on", "investigate"],
                       ["{topic}", "{topic} with sources", "the history of {topic}", "everything about {topic}"]),
    "list_events": _cross(["show me", "tell me", "list", "read out", "check", "give me"],
                          ["my agenda", "my schedule", "my calendar", "my meetings", "my appointments",
                           "my upcoming events", "my classes this week", "what i have planned"]),
    "list_tasks": _cross(["show me", "tell me", "list", "read out", "check", "give me"],
                         ["my tasks", "my to-do list", "my checklist", "my pending chores", "what's outstanding",
                          "what i still need to do", "my open items"]),
    "add_task": _cross(["add", "put", "note", "queue up", "log", "jot down"],
                       ["a task to {title}", "a reminder to {title}", "{title} to my to-do list",
                        "{title} on my checklist", "an item: {title}"])
                + ["todo: {title}", "to do: {title}", "task: {title}", "remember to {title} tomorrow", "{title} by friday",
                   "don't forget to {title} today", "i must {title}", "we need to {title}", "reminder: {title}"],
    "add_note": _cross(["remember this", "save this", "note this down", "write this down", "keep this", "store this"],
                       ["as {title}: {body}", "under {title}: {body}", "with the title {title}: {body}"]),
    "list_files": _cross(["show", "list", "display", "what's in", "browse"],
                         ["my files", "the workspace", "my folder", "my documents", "everything i've saved"]),
    "read_file": ["show what is written in {path}", "what does {path} say", "what's written in {path}",
                  "read out what's in {path}"] + _cross(["let me look at", "show me", "print", "display", "read out", "open up", "let me read"],
                        ["{path}", "the file {path}", "the contents of {path}"]),
    "list_lessons": _cross(["show", "list", "tell me", "read back"],
                           ["the shortcuts you've stored", "what i've taught you", "the phrases you learned",
                            "everything you remember", "your learned rules"]),
    "add_blocked_word": _cross(["censor", "block", "ban", "filter", "blacklist", "disallow", "forbid"],
                               ["the word {word}", "the term {word}", "{word} in chat", "{word} from messages"]),
    "user_moderation_status": _cross(["show", "pull", "check", "give me", "what's"],
                                     ["the moderation file on {user}", "{user}'s record", "{user}'s warnings",
                                      "the mod history for {user}", "{user}'s moderation status"])
                              + ["has {user} been warned before", "was {user} ever muted", "any strikes against {user}"],
    "describe_data": _cross(["give me the lowdown on", "summarise", "summarize", "tell me about", "give me stats on",
                             "break down", "give me an overview of", "analyse", "analyze"],
                            ["{path}", "the file {path}", "the numbers in {path}", "the data inside {path}"]),
    "find_free_slots": ["do i have time for a meeting on {date}", "is there room for an appointment on {date}",
                        "any space for a call on {date}", "do i have room for something on {date}", "any availability on {date}", "is {date} open anywhere",
                        "check whether {date} has open hours", "can i fit something in on {date}", "free time on {date}",
                        "am i booked solid on {date}", "how does {date} look for openings"],
    "certify_roles": ["how ready are the experts", "are your specialists trained", "role readiness", "are your roles certified",
                      "which of your roles need more training"],
}
for _t, _extra in COMPOSED.items():
    SPECS[_t][0].extend(_extra)


def _lang_specs() -> dict:
    """{lang: {tool: (templates, argmap)}} reusing the English arg maps for translated tools."""
    out = {"en": SPECS}
    for lang, tmap in i18n.LANG_TEMPLATES.items():
        out[lang] = {t: (tpl, SPECS[t][1]) for t, tpl in tmap.items() if t in SPECS}
    return out


def _noise(template: str, rng: random.Random) -> str:
    """Typo noise on template words only (never on slot values)."""
    parts = re.split(r"(\{[^}]*\})", template)
    out = []
    for p in parts:
        if p.startswith("{") or len(p) < 5 or rng.random() > 0.12:
            out.append(p)
            continue
        i = rng.randrange(1, len(p) - 2)
        if p[i].isalpha() and p[i + 1].isalpha():
            p = p[:i] + p[i + 1] + p[i] + p[i + 2:]  # swap adjacent letters
        out.append(p)
    return "".join(out)


def _pick(n: int, split: str) -> list[int]:
    return [i for i in range(n) if split == "all" or (i % 4 == 3) == (split == "test")]


def curriculum() -> list[dict]:
    """Training examples taken from each role's DEV exam phrasings (never the holdout ones). Each role's skills
    are practiced on real-sounding requests in addition to the generated templates."""
    from bot.training import roles
    out = []
    for role in roles.ROLES:
        for sk in role.skills:
            for split, phrase, expected, args in sk.exams:
                if split != roles.D:
                    continue
                if expected == "abstain":
                    tool = "chat"
                elif expected.startswith("text:"):
                    tool = "needs_model" if "language model" in expected else None
                else:
                    tool = expected.split("|")[0]
                if tool is None or tool in ("learn_instruction", "add_training_example", "mark_wrong", "mark_good") \
                        or (tool not in ("chat", "needs_model") and sk.mode in ("command", "guarded")):
                    continue                       # explicit-command skills are handled by the exact grammar
                frame = phrase.lower()
                for v in (args or {}).values():
                    for piece in (v if isinstance(v, list) else [v]):
                        frame = frame.replace(str(piece).lower(), " ")
                # without the argument values we can't tell command words from content words, so such examples
                # must not contribute "trigger words" (they'd strip the topic itself from the answer)
                out.append({"phrase": phrase, "tool": tool, "args": args or {}, "frame": frame, "weight": 2,
                            "no_frame": args is None and tool not in ("chat", "needs_model"),
                            "lang": "en", "cap": catalog.capability_of(tool)})
    return out


def generate(per_tool: int = 40, seed: int = 0, split: str = "all", per_lang: int = 16,
             langs: tuple = ()) -> list[dict]:
    """Examples: {phrase, tool, args, frame, weight, lang, cap}. frame = phrase with slot values removed."""
    rng = random.Random(seed)
    out = []
    for lang, specs in _lang_specs().items():
        if langs and lang not in langs:
            continue
        n_each = per_tool if lang == "en" else per_lang
        pre = PREFIXES if lang == "en" else i18n.PREFIXES[lang]
        suf = SUFFIXES if lang == "en" else i18n.SUFFIXES[lang]
        for tool, (templates, argmap) in specs.items():
            idx = _pick(len(templates), split)
            if not idx:                 # fewer than 4 templates: nothing to hold out for the test split
                continue
            for k in range(max(n_each, min(len(idx), 120))):
                t = templates[idx[k % len(idx)]]
                f = _fields(rng, lang)
                noisy = _noise(t, rng)
                p, s = rng.choice(pre), rng.choice(suf)
                phrase = p + noisy.format(**f) + s
                # verb/kind placeholders are command wording (not slot values), so they stay in the frame
                keep = re.sub(r"\{(verb|kind|ckind)\}", lambda m: str(f[m.group(1)]), noisy)
                frame = p + re.sub(r"\{[^}]*\}", " ", keep) + s
                args = {a: (fn(f) if fn else f[ph]) for a, (ph, fn) in argmap.items() if "{" + ph + "}" in t}
                if tool == "add_event":
                    args["kind"] = f["kind_c"] if "{kind}" in t else "event"
                if tool == "make_chart" and "histogram" in t:
                    args["kind"] = "hist"
                out.append({"phrase": phrase, "tool": tool, "args": args, "frame": frame, "weight": 1,
                            "lang": lang, "cap": catalog.capability_of(tool)})
        chat = CHAT[lang]
        idx = _pick(len(chat), split)
        for k in range(n_each):
            c = chat[idx[k % len(idx)]]
            p = rng.choice(pre)
            out.append({"phrase": p + c, "tool": "chat", "args": {}, "frame": c, "weight": 1,
                        "lang": lang, "cap": "chat"})
    if split == "all" and not langs:
        out += curriculum()
    return out
