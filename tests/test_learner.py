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
    assert r["seen_phrasing"] >= 0.95            # new slot values, known phrasings
    assert r["heldout"]["intent_accuracy"] >= 0.7  # phrasings never seen in training
    assert r["seed_examples"] > 500


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
    assert g("reserach quantum computing")["args"] == {"question": "quantum computing"}  # typo tolerant


def test_abstains_on_chitchat_and_gibberish():
    for p in ("tell me a joke", "hello", "asdf qwer zxcv", "what's the weather like"):
        assert learner.interpret(p) is None


def test_missing_slot_asks_and_destructive_never_autoruns():
    a = bot()
    assert "need: url" in a.run("please pull up the page for me").lower()
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
