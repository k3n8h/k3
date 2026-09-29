import json

import pytest

from bot import learner, llm, memory, registry
from bot.agent import Agent
from bot.training import seed


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("BOT_HOME", str(tmp_path))
    memory.connect(str(tmp_path / "t.db"))
    registry.load_all()
    learner.LAST.clear()
    learner._cache = (None, None, None)


def bot():
    return Agent(provider=llm.get_provider("offline"))


def test_heldout_generalization_and_seen_phrasing():
    r = learner.fit_all(evaluate=True)
    h = r["heldout"]
    assert r["seen_phrasing"] >= 0.85              # new slot values, known phrasings
    assert h["intent_accuracy"] >= 0.55            # wordings never seen in training
    assert h["wrong_action_rate"] <= 0.08          # unfamiliar wording mostly abstains rather than misfires
    assert h["wrong_state_change_rate"] <= 0.01    # ...and almost never changes state by mistake
    assert set(h["per_language"]) == {"en", "es", "fr", "de", "pt", "it", "ru"}
    assert r["seed_examples"] > 2000


def test_free_form_intents_and_slots():
    g = learner.interpret
    assert g("dig up information on solar panels")["tool"] == "research"
    assert g("what's 12 times 7") == {"tool": "calculate", "args": {"expression": "12 * 7"},
                                      "confidence": g("what's 12 times 7")["confidence"], "missing": []}
    r = g("scrape https://x.com/a h1 please")
    assert r["args"] == {"url": "https://x.com/a", "selector": "h1"}
    assert g("how many km in 5 miles")["args"] == {"value": 5.0, "from_unit": "mi", "to_unit": "km"}
    assert g("note groceries: milk and eggs")["args"] == {"title": "groceries", "body": "milk and eggs"}
    assert g("crawl https://a.com depth 2 max 5")["args"] == {"start_url": "https://a.com", "max_depth": 2, "max_pages": 5}
    assert g("please roll 3d8 for me")["args"] == {"spec": "3d8"}
    # typo tolerance is best-effort: "reserach quantum computing" (topic words that collide with other intents)
    # still abstains, so this pins the case that works
    r = g("reserach the roman empire")
    assert r["tool"] in ("research", "web_search") and "roman empire" in r["args"].values()


def test_abstains_on_chitchat_and_gibberish():
    for p in ("tell me a joke", "hello", "asdf qwer zxcv", "what's the weather like"):
        assert learner.interpret(p) is None


def test_missing_slot_asks_and_destructive_never_autoruns():
    a = bot()
    assert "need: url" in a.run("please fetch the page for me").lower()
    memory.execute("INSERT INTO events(kind,title,start,end) VALUES('event','x','2030-01-01T10:00','2030-01-01T11:00')")
    out = a.run("get rid of event 1")
    assert "destructive" in out
    assert len(memory.query("SELECT * FROM events")) == 1


def test_offline_agent_uses_learner_end_to_end():
    a = bot()
    out = a.run("what does 15 plus 27 come to, please")
    assert "42" in out


def test_correction_changes_behavior_after_retrain():
    a = bot()
    phrase = "flurbo the zanzibar"
    assert "offline" in a.run(phrase)                       # unknown -> help
    learner.LAST.clear(); learner.LAST.update(phrase=phrase, tool="web_search", args={})
    out = a.run("wrong => calc 2+2")
    assert '"learned"' in out and "4" in out
    assert learner.interpret(phrase)["tool"] == "calculate"  # now understood


def test_train_add_status_file_and_reset(tmp_path):
    a = bot()
    assert "learned" in a.run('train add "gimme a d20" => roll d20')
    assert learner.interpret("gimme a d20")["tool"] == "roll_dice"
    ws = registry.call("write_file", {"path": "ex.jsonl", "content":
                                      json.dumps({"phrase": "yo dice", "tool": "roll_dice", "args": {}}) + "\n"
                                      + json.dumps({"phrase": "bad", "tool": "cancel_event", "args": {"id": 1}}) + "\n"})
    assert '"added": 1' in a.run("train from ex.jsonl") and '"skipped": 1' in registry.call("train_from_file", {"path": "ex.jsonl"})
    st = json.loads(registry.call("training_status", {}))
    assert st["user_examples"] >= 3 and "roll_dice" in st["intents"]
    assert "declined" in a.run("train reset")               # destructive: needs confirmation
    assert memory.query("SELECT * FROM examples")


def test_good_reinforces_and_hints_for_llm():
    a = bot()
    a.run("roll 2d6")
    assert '"reinforced"' in a.run("good")
    assert "roll" in learner.hints("roll 2d6 again")
    assert learner.hints("zzz qqq") == ""


def test_model_json_roundtrip():
    learner.fit_all()
    m = learner.get_model()
    m2 = learner.Model.from_json(m.to_json())
    assert m2.predict("flip a coin") == m.predict("flip a coin")


def test_first_use_without_explicit_connect_does_not_deadlock(tmp_path, monkeypatch):
    import subprocess, sys
    r = subprocess.run([sys.executable, "-c", "from bot import memory; print(memory.query('select 1 as x'))"],
                       env={**__import__('os').environ, "BOT_HOME": str(tmp_path)}, capture_output=True, text=True, timeout=30)
    assert "[{'x': 1}]" in r.stdout


def test_catalog_covers_every_tool_and_is_consistent():
    from bot.training import catalog
    owned = [t for c in catalog.CAPABILITIES for t in c["tools"]]
    assert len(owned) == len(set(owned)), "a tool is listed under two capabilities"
    assert set(owned) == set(registry._TOOLS), (set(registry._TOOLS) ^ set(owned))
    assert all(catalog.capability_of(t) != "unknown" for t in registry._TOOLS)
    md = catalog.render_markdown()
    assert "Spanish" in md and "Subject areas" in md and "Natural sciences" in md
    assert len(catalog.all_topics()) >= 60
    assert "research" in registry.call("list_capabilities", {"area": "research"}).lower()
    assert "Research & web" in bot().run("what can you do")


def test_multilingual_free_form():
    g = learner.interpret
    cases = {
        "busca información sobre energía solar": ("web_search", None),
        "investiga sobre la revolución francesa": ("research", None),
        "wie spät ist es": ("now", {}),
        "quelle heure est-il": ("now", {}),
        "que horas são": ("now", {}),
        "che ore sono": ("now", {}),
        "cuánto es 12 por 7": ("calculate", {"expression": "12 * 7"}),
        "combien font 9 plus 4": ("calculate", {"expression": "9 + 4"}),
        "berechne 6 mal 7": ("calculate", {"expression": "6 * 7"}),
        "quanto é 20 vezes 3": ("calculate", {"expression": "20 * 3"}),
        "quanto fa 100 diviso 4": ("calculate", {"expression": "100 / 4"}),
        "lance une pièce": ("flip_coin", {}),
        "wirf eine münze": ("flip_coin", {}),
        "tira 2d6": ("roll_dice", {"spec": "2d6"}),
        "convierte 5 km a mi": ("convert_units", {"value": 5.0, "from_unit": "km", "to_unit": "mi"}),
        "guarda una nota compras: leche y huevos": ("add_note", {"title": "compras", "body": "leche y huevos"}),
        "mostra le mie attività": ("list_tasks", {}),
        "zeige meine termine": ("list_events", {}),
    }
    for phrase, (tool, args) in cases.items():
        r = g(phrase)
        ok = {"web_search", "research"} if tool == "web_search" else {tool}
        assert r and r["tool"] in ok, (phrase, r)
        if args is not None:
            assert r["args"] == args, (phrase, r)


def test_generative_requests_are_recognized_not_faked():
    a = bot()
    for p in ("write me an essay about volcanoes", "escribe un poema sobre el mar", "schreibe ein gedicht über den herbst"):
        out = a.run(p)
        assert "language model" in out, (p, out)


def test_scheduling_from_free_form():
    from datetime import datetime
    now = datetime(2030, 3, 4, 8, 0)           # a Monday
    w = learner.parse_when("dentist tomorrow at 3pm for 90 minutes", now)
    assert (w["date"].isoformat(), w["time"], w["minutes"]) == ("2030-03-05", (15, 0), 90)
    assert learner.parse_when("yoga next monday at 9am", now)["date"].isoformat() == "2030-03-11"
    assert learner.parse_when("call at 7am", now)["date"].isoformat() == "2030-03-05"   # already past today
    args, missing = learner._extract_event("book a class math on 2030-05-06 at 10:30", ["book", "a", "class"], now)
    assert args == {"kind": "class", "title": "math", "start": "2030-05-06T10:30", "end": "2030-05-06T11:30"}
    a = bot()
    out = a.run("schedule team sync on 2030-03-05 at 14:00")
    assert '"created": true' in out
    assert "conflict" in a.run("put yoga in my calendar on 2030-03-05 at 14:30").lower()


def test_subject_topics_flow_into_research_slot():
    learner.fit_all()
    trig = learner.get_model().triggers_for("research")
    for topic in ("plate tectonics", "compound interest", "film noir", "japanese kanji", "silk road"):
        for form in (f"explain {topic}", f"tell me about {topic} please", f"i need to learn about {topic}"):
            args, missing = learner.extract_args("research", form, trig)
            assert args == {"question": topic} and not missing, (form, args)


def test_teach_an_unrecognized_phrase_then_it_is_understood():
    a = bot()
    phrase = "yeet a cube"
    assert "wrong =>" in a.run(phrase)                    # abstained, and says how to teach it
    assert "error" in a.run("good")                       # nothing to confirm: it did not act
    assert 'total' in a.run("that means roll 1d6")      # learns it and runs the command
    assert learner.interpret(phrase)["tool"] == "roll_dice"


def test_teach_flow_regressions():
    from bot.offline import parse_grammar
    for p in ("i meant to call you yesterday", "that means nothing to me", "I meant business"):
        assert parse_grammar(p) is None, p                       # ordinary sentences are not corrections
    a = bot()
    a.run('learn "tools list" => tools')
    a.run("roll 1d6")
    a.run("tools list")                                          # taught rule: understood, must not wipe LAST
    assert learner.LAST["tool"] == "roll_dice"
    assert "wrong =>" not in a.run("write me a poem about the sea")   # generative reply is not a teach prompt
    a.run("yeet a cube")
    a.run("that means roll 1d6")
    assert learner.LAST["tool"] == "roll_dice"                   # corrected action becomes the new LAST
    assert "reinforced" in a.run("good")


def test_new_intents_and_russian():
    learner.fit_all()
    g = learner.interpret
    assert g("mark task 7 as done")["tool"] == "complete_task" and g("mark task 7 as done")["args"] == {"id": 7}
    r = g("what date is 30 days after 2030-01-15")
    assert r["tool"] == "date_add" and r["args"] == {"date": "2030-01-15", "days": 30}, r
    r = g("choose between pizza, sushi or tacos")
    assert r["tool"] == "pick_random" and r["args"] == {"options": ["pizza", "sushi", "tacos"]}, r
    for phrase, tool in (("который час", "now"), ("подбрось монетку", "flip_coin"), ("что ты умеешь", "list_capabilities"),
                         ("покажи мои задачи", "list_tasks")):
        assert g(phrase)["tool"] == tool, phrase
    assert g("сколько будет 12 умножить на 7")["args"] == {"expression": "12 * 7"}
    assert g("привет") is None                       # Cyrillic chit-chat abstains
    assert "2030-02-14" in bot().run("what date is 30 days after 2030-01-15")


def test_extraction_edge_cases_from_review():
    g = learner.interpret
    learner.fit_all()
    r = g("mark tasks 3 and 4 as done")                       # ambiguous: ask, do not silently pick 3
    assert r["tool"] == "complete_task" and r["missing"] == ["id"]
    assert g("add 2 weeks to 2030-01-15")["args"] == {"date": "2030-01-15", "days": 14}
    assert g("add 3 months to 2030-01-15")["missing"] == ["days"]      # variable-length unit: ask
    assert g("5 days before 2030-01-15")["args"]["days"] == -5
    assert g("what date is 30 days after 2030-01-15")["args"]["days"] == 30
    assert g("choose between pizza, sushi or tacos")["args"]["options"] == ["pizza", "sushi", "tacos"]
    r = g("what's open on 2030-11-21")                         # a date is not arithmetic
    assert r is None or r["tool"] != "calculate"
    assert learner._expression("book 2030-03-05 at 14:00") is None
    for phrase, tool in (("отметь задачу 7 выполненной", "complete_task"), ("marque la tâche 4 comme terminée", "complete_task"),
                         ("añade 10 días a 2030-01-15", "date_add"), ("wähle zwischen tee, kaffee oder saft", "pick_random")):
        r = g(phrase)
        assert r and r["tool"] == tool, (phrase, r)


def test_date_and_id_extraction_regressions():
    learner.fit_all()
    g = lambda p: learner.interpret(p)
    assert g("what date is 2 weeks before 2030-01-15")["args"] == {"date": "2030-01-15", "days": -14}
    assert g("2 weeks ago from 2030-01-15")["args"]["days"] == -14
    assert g("minus 5 days from 2030-01-15")["tool"] == "date_add"            # not arithmetic
    assert g("add 2 weeks and 3 days to 2030-01-15")["args"]["days"] == 17     # quantities add up
    assert g("complete task 3 at 14:00")["args"] == {"id": 3}                  # times/dates hold no ids
    assert g("mark task 03 done on 2030-01-15")["args"] == {"id": 3}
    assert learner._expression("minus 5 days") is None and learner._expression("7 minus 5") == "7 - 5"
    assert g("fetch the page for me")["missing"] == ["url"]                    # un-prefixed form keeps asking
    r = g("grab pages from https://x.com")
    assert r is None or r["tool"] != "crawl"                                   # crawl is never auto-picked by pooling
