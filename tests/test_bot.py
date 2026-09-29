from datetime import datetime
from types import SimpleNamespace

import pytest

from bot import memory, registry, scheduler


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("BOT_HOME", str(tmp_path))
    memory.connect(str(tmp_path / "t.db"))
    registry.load_all()


def test_schema_and_dispatch():
    defs = {d["name"]: d for d in registry.definitions()}
    assert {"add_event", "run_python", "describe_data", "roll_dice", "schedule_job"} <= set(defs)
    assert defs["add_event"]["input_schema"]["required"] == ["kind", "title", "start", "end"]
    assert defs["add_event"]["input_schema"]["properties"]["remind_minutes"]["type"] == "integer"
    assert registry.call("nope", {}).startswith("error")
    assert registry.call("calculate", {"expression": "2+3*4"}) == "14"
    assert registry.call("calculate", {"expression": "__import__('os')"}).startswith("error")


def test_calendar_conflicts_and_free_slots():
    a = registry.call("add_event", dict(kind="class", title="Math", start="2030-01-01T10:00", end="2030-01-01T11:00"))
    assert '"created": true' in a
    b = registry.call("add_event", dict(kind="appointment", title="Dentist", start="2030-01-01T10:30", end="2030-01-01T11:30"))
    assert '"created": false' in b
    slots = registry.call("find_free_slots", {"date": "2030-01-01", "minutes": 60})
    assert "09:00" in slots and "11:00" in slots
    assert registry.call("cancel_event", {"id": 1}) == '{"deleted": 1}'


def test_reminders():
    registry.call("add_event", dict(kind="event", title="Call", start="2030-01-01T10:00",
                                    end="2030-01-01T10:30", remind_minutes=15))
    sent = []
    scheduler._notify = sent.append
    scheduler.tick(datetime(2030, 1, 1, 9, 50))
    scheduler.tick(datetime(2030, 1, 1, 9, 55))
    assert len(sent) == 1


def test_code_sandbox(tmp_path):
    out = registry.call("run_python", {"code": "print(1+1)"})
    assert '"stdout": "2\\n"' in out
    assert "timed out" in registry.call("run_python", {"code": "while True: pass", "timeout": 1})
    assert registry.call("read_file", {"path": "../etc/passwd"}).startswith("error")


def test_data(tmp_path):
    registry.call("write_file", {"path": "d.csv", "content": "a,b\nx,1\nx,3\ny,5\n"})
    out = registry.call("query_data", {"path": "d.csv", "group_by": "a", "column": "b", "agg": "sum"})
    assert '"b": 4' in out and '"b": 5' in out
    assert "saved" in registry.call("make_chart", {"path": "d.csv", "x": "a", "y": "b"})


def test_moderation():
    assert '"flagged": true' in registry.call("moderate_text", {"text": "well shit"})
    registry.call("moderate_user", {"user": "bob", "action": "warn"})
    registry.call("moderate_user", {"user": "bob", "action": "mute"})
    assert '"muted": true' in registry.call("user_moderation_status", {"user": "bob"})


def test_utilities_and_fun():
    assert float(registry.call("convert_units", {"value": 100, "from_unit": "c", "to_unit": "f"})) == 212
    assert '"total"' in registry.call("roll_dice", {"spec": "2d6"})


class FakeProvider:
    def __init__(self, replies):
        self.replies = iter(replies)

    def complete(self, system, messages, tools, max_tokens=4096):
        return next(self.replies)


def test_agent_loop_and_destructive_denied():
    from bot.agent import Agent
    from bot.llm import Reply, ToolCall
    agent = Agent(provider=FakeProvider([Reply(tool_calls=[ToolCall("1", "add_task", {"title": "x"})]), Reply("done")]))
    assert agent.run("add a task") == "done"
    assert memory.query("SELECT title FROM tasks") == [{"title": "x"}]

    memory.execute("INSERT INTO notes(title,body) VALUES('a','b')")
    agent = Agent(provider=FakeProvider([Reply(tool_calls=[ToolCall("2", "delete_note", {"id": 1})]), Reply("ok")]))
    agent.run("delete it")
    assert len(memory.query("SELECT * FROM notes")) == 1  # denied by default


# ---- agent robustness
def test_provider_failure_is_reported_and_history_stays_consistent():
    from bot.agent import Agent
    from bot.llm import Reply, ToolCall

    class Flaky:
        def __init__(self):
            self.calls = 0

        def complete(self, system, messages, tools, max_tokens=4096):
            self.calls += 1
            if self.calls == 2:                      # second call (after a tool result) fails mid-turn
                raise ConnectionError("network down")
            return Reply(tool_calls=[ToolCall("1", "flip_coin", {})]) if self.calls == 1 else Reply("ok")

    a = Agent(provider=Flaky())
    out = a.run("flip a coin")
    assert "model call failed" in out and "ConnectionError" in out
    assert a.messages == []                          # the half-finished turn was rolled back entirely
    assert a.run("again") == "ok"                    # and the next turn works


def test_history_is_bounded_and_never_splits_tool_pairs():
    from bot.agent import Agent
    from bot.llm import Reply, ToolCall

    class Tooly:
        def __init__(self):
            self.n = 0

        def complete(self, system, messages, tools, max_tokens=4096):
            self.n += 1
            last = messages[-1]["content"]
            return Reply(tool_calls=[ToolCall(str(self.n), "flip_coin", {})]) if isinstance(last, str) else Reply("done")

    a = Agent(provider=Tooly())
    for i in range(80):
        a.run(f"turn {i}")
    assert len(a.messages) <= Agent.MAX_HISTORY + 4
    assert a.messages[0]["role"] == "user" and isinstance(a.messages[0]["content"], str)
    pending = set()
    for m in a.messages:                             # every tool_result has its tool_use still present
        if m["role"] == "assistant":
            pending |= {b["id"] for b in m["content"] if b["type"] == "tool_use"}
        elif not isinstance(m["content"], str):
            assert {b["tool_use_id"] for b in m["content"]} <= pending


def test_web_chat_serializes_concurrent_turns(monkeypatch):
    import threading
    import time
    from fastapi.testclient import TestClient
    from bot.interfaces import web
    active, overlap = [0], [False]

    class Slow:
        def complete(self, system, messages, tools, max_tokens=4096):
            from bot.llm import Reply
            active[0] += 1
            overlap[0] |= active[0] > 1
            time.sleep(0.05)
            active[0] -= 1
            return Reply("hi")

    from bot.agent import Agent
    monkeypatch.setattr(web, "_agent", Agent(provider=Slow()))
    c = TestClient(web.app)
    threads = [threading.Thread(target=lambda: c.post("/chat", json={"message": "x"})) for _ in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not overlap[0]
    assert len([m for m in web._agent.messages if m["role"] == "user"]) == 6     # no lost or corrupted turns
