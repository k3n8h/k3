"""End-to-end job scenarios per role: real tools, real state, isolated scratch DB + workspace.

Exams check that a request is *understood*; scenarios check that the role can actually *do the job*: multi-step
workflows where later steps depend on the state earlier ones created. Every scenario runs through the same offline
Agent a user talks to (with destructive confirmation denied), so refusals are part of what is verified.
"""
import contextlib
import json
import os
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from bot import config, llm, memory
from bot.agent import Agent


@contextlib.contextmanager
def env(**changes):
    """Set (value) or unset (None) environment variables, restoring every prior value even on error."""
    saved = {k: os.environ.get(k) for k in changes}
    try:
        for k, v in changes.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class Session:
    """A user chatting with the offline bot; `say` returns the reply text."""

    def __init__(self):
        self.agent = Agent(provider=llm.get_provider("offline"), confirm=lambda n, a: False)

    def say(self, text: str) -> str:
        return self.agent.run(text)

    def data(self, text: str):
        out = self.say(text)
        try:
            return json.loads(out)
        except (ValueError, TypeError):
            return out


def _need(cond, msg):
    if not cond:
        raise AssertionError(msg)


# ---------------------------------------------------------------- scenarios (each raises AssertionError on failure)
def research_analyst():
    class H(BaseHTTPRequestHandler):
        PAGES = {"/": '<title>Home</title><main><p class=x>alpha</p><a href="/a">A</a><a href="/hidden">H</a></main>',
                 "/a": "<title>A</title><main><p class=x>beta</p></main>", "/hidden": "<title>no</title>secret",
                 "/robots.txt": "User-agent: *\nDisallow: /hidden\n"}

        def do_GET(self):
            body = self.PAGES.get(self.path)
            self.send_response(200 if body else 404)
            self.send_header("Content-Type", "text/plain" if self.path.endswith("robots.txt") else "text/html")
            self.end_headers()
            self.wfile.write((body or "").encode())

        def log_message(self, *a):
            pass

    from bot import web
    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    saved_robots = dict(web._robots)
    web._robots.clear()
    try:
        base, s = f"http://127.0.0.1:{srv.server_port}", Session()
        with env(BOT_ALLOW_PRIVATE_NETS="1", BOT_CRAWL_DELAY="0", BOT_TRUST_PROXY_ENV=None):
            page = s.data(f"fetch {base}/")
            _need(isinstance(page, dict) and page["title"] == "Home" and "alpha" in page["text"], "read a page")
            _need(s.data(f"scrape {base}/ p.x") == ["alpha"], "scrape by CSS selector")
            crawl = s.data(f"crawl {base}/ depth 1")
            urls = {p["url"]: p for p in crawl["results"]}
            _need(f"{base}/a" in urls and "skipped" in urls.get(f"{base}/hidden", {}),
                  "crawl follows links and honors robots.txt")
            _need(s.data(f"abre {base}/a")["title"] == "A", "reads pages when asked in Spanish")
        with env(BOT_ALLOW_PRIVATE_NETS=None, BOT_TRUST_PROXY_ENV=None):     # guard back on: private hosts refused
            blocked = s.say("fetch http://10.0.0.5/").lower()
            _need("error" in blocked and "non-public" in blocked, f"refuses private addresses: {blocked[:80]}")
    finally:
        srv.shutdown()
        srv.server_close()
        web._robots.clear()
        web._robots.update(saved_robots)


def executive_assistant():
    s = Session()
    _need(s.data("schedule dentist on 2031-05-06 at 09:00")["created"] is True, "book an appointment")
    clash = s.data("schedule yoga on 2031-05-06 at 09:30")
    _need(clash["created"] is False and clash["conflicts"][0]["title"] == "dentist", "detect a double booking")
    rows = memory.query("SELECT title FROM events")
    _need([r["title"] for r in rows] == ["dentist"], "only the first booking exists")
    slots = s.data("when am i free on 2031-05-06")
    _need(slots and slots[0]["start"] == "2031-05-06T10:00", f"the booked 09:00 hour is not offered as free: {slots}")
    _need("declined" in s.say("cancel event 1"), "cancelling needs confirmation")
    _need(len(memory.query("SELECT * FROM events")) == 1, "guarded cancel did not run")
    _need(s.data("planifie yoga le 2031-06-01 à 10:00")["created"] is True, "books from a French request")
    job = s.data('schedule job "summarize the news" at 2031-05-07T08:00')
    _need(isinstance(job, dict) and isinstance(job["id"], int), f"schedule a job: {job}")
    _need([j["goal"] for j in s.data("jobs")] == ["summarize the news"], "list scheduled jobs")


def personal_organizer():
    s = Session()
    s.say("add task buy stamps")
    s.say("add task call the plumber")
    _need("buy stamps" in s.say("tasks") and "call the plumber" in s.say("tasks"), "tasks are listed")
    s.say("complete task 1")
    listing = s.say("tasks")
    _need("buy stamps" not in listing and "call the plumber" in listing, "completed tasks leave the open list")
    added = s.data("añade una tarea pagar la factura")
    _need(isinstance(added, dict) and isinstance(added["id"], int), f"adds a task from Spanish: {added}")
    _need("pagar la factura" in [t["title"] for t in s.data("mis tareas")], "lists it back in Spanish")
    s.say("note groceries: eggs and rice")
    _need("groceries" in s.say("notes eggs"), "notes are searchable by body")
    _need("declined" in s.say("delete note 1"), "deleting a note needs confirmation")
    _need(len(memory.query("SELECT * FROM notes")) == 1, "guarded delete did not run")


def data_analyst():
    s = Session()
    s.say("write file sales.csv: region,price\nnorth,10\nnorth,30\nsouth,5")
    desc = s.data("describe sales.csv")
    _need(desc["shape"] == [3, 2], "profile a dataset")
    q = s.data("query sales.csv where price > 8")
    _need(sorted(r["price"] for r in q) == [10, 30], "filter rows")
    _need("error" in s.say("query sales.csv where __import__('os').system('id') == 1").lower(), "unsafe filters are refused")
    out = s.say("plot price by region from sales.csv")
    _need("saved" in out and os.path.exists(os.path.join(config.workspace(), "chart.png")), "chart file is created")


def developer_assistant():
    s = Session()
    _need("wrote" in s.say("write file hello.txt: hi there"), "write a file")
    _need("hi there" in s.say("read file hello.txt"), "read it back")
    _need("hello.txt" in s.say("files"), "list files")
    res = s.data("run python print(6*7)")
    _need(res["exit_code"] == 0 and res["stdout"].strip() == "42", "run python")
    _need("error" in s.say("read file ../secret.txt").lower(), "cannot escape the workspace")


def creative_writer_designer():
    s = Session()
    for ask in ("write me an essay about volcanoes", "design a logo for my cafe", "escribe un poema sobre el mar"):
        _need("language model" in s.say(ask), f"honest about needing a model: {ask}")


def community_moderator():
    s = Session()
    _need(s.data("moderate: hello everyone")["flagged"] is False, "clean text passes")
    _need(s.data("moderate: this is a phishing link")["flagged"] is False, "not blocked before it is added")
    s.say("block the word phishing")
    hit = s.data("moderate: this is a phishing link")
    _need(hit["flagged"] is True and hit["words"] == ["phishing"], "newly blocked words are caught")
    s.say("advierte a alice por spam")
    _need(s.data("cuántas advertencias tiene alice")["status"]["warnings"] == 1, "moderates in Spanish")
    s.say("warn user bob for spamming")
    s.say("mute bob because rude language")
    status = s.data("show warnings for bob")
    _need(status["status"] == {"warnings": 1, "muted": True}, f"record reflects warn+mute: {status}")
    s.say("unmute bob")
    _need(s.data("show warnings for bob")["status"]["muted"] is False, "unmute clears the mute")


def utility_expert():
    s = Session()
    _need(s.data("what's 15 times 4") == 60, "arithmetic")
    _need(s.data("combien font 9 plus 4") == 13, "arithmetic in French")
    _need(s.data("what date is 30 days after 2030-01-15")["date"] == "2030-02-14", "date math")
    _need(abs(s.data("convert 10 km to miles") - 6.2137) < 0.01, "unit conversion")
    _need("pizza" in s.say("create a poll team lunch: pizza, sushi, tacos"), "poll ballot")
    _need("error" in s.say("calc __import__('os')").lower(), "calculator refuses code")


def game_host():
    s = Session()
    rolls = s.data("roll 3d6")
    _need(len(rolls["rolls"]) == 3 and 3 <= rolls["total"] <= 18, "dice in range")
    _need(s.data("flip a coin") in ("heads", "tails"), "coin")
    _need(s.data("wirf eine münze") in ("heads", "tails"), "coin in German")
    _need(s.data("pick one of tea, coffee, juice") in ("tea", "coffee", "juice"), "random pick from the given options")
    word = s.data("scramble a word")
    _need(sorted(word["scrambled"]) == sorted(word["answer"]), "scramble is a real anagram")


def trainer():
    s = Session()
    s.say('learn "lucky number" => roll 1d2')
    _need('"total"' in s.say("give me my lucky number please"), "taught shortcut is applied")
    _need("lucky number" in s.say("lessons"), "lessons are recalled")
    s.say("yeet a cube")
    out = s.say("that means roll 1d6")
    _need("learned" in out, "corrections are learned")
    from bot import learner
    _need(learner.interpret("yeet a cube")["tool"] == "roll_dice", "the correction changes later behavior")
    _need("declined" in s.say("forget 1"), "forgetting needs confirmation")
    _need("Research" in s.say("what can you do"), "explains its capabilities")


def conversationalist():
    s = Session()
    for chat in ("tell me a joke", "hello", "hola", "bonjour"):
        _need("offline" in s.say(chat), f"small talk abstains instead of acting: {chat}")
    _need(not memory.query("SELECT * FROM tasks") and not memory.query("SELECT * FROM notes"), "small talk changed nothing")


SCENARIOS = {fn.__name__: fn for fn in (research_analyst, executive_assistant, personal_organizer, data_analyst,
                                        developer_assistant, creative_writer_designer, community_moderator,
                                        utility_expert, game_host, trainer, conversationalist)}


def run(role_id: str) -> tuple[bool, str]:
    """Run one role's scenario in a scratch DB and a scratch BOT_HOME (both restored afterwards)."""
    from bot import learner
    fn = SCENARIOS.get(role_id)
    if fn is None:
        return False, "no scenario defined"
    old_home = os.environ.get("BOT_HOME")
    learner.LAST.clear()
    pinned, learner._override = learner._override, None      # scenarios train (and retrain) their own model in the temp home
    learner._cache = (None, None, None)
    with tempfile.TemporaryDirectory() as tmp, memory.scratch():
        os.environ["BOT_HOME"] = tmp
        try:
            fn()
            return True, "ok"
        except AssertionError as e:
            return False, str(e)
        except Exception as e:  # a crash is a failure with a reason, not a certification-run crash
            return False, f"{type(e).__name__}: {e}"
        finally:
            learner._override, learner._cache = pinned, (None, None, None)
            if old_home is None:
                os.environ.pop("BOT_HOME", None)
            else:
                os.environ["BOT_HOME"] = old_home
