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


def _resp(stop, *blocks):
    return SimpleNamespace(stop_reason=stop, content=list(blocks))


class FakeClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.messages = SimpleNamespace(create=lambda **kw: next(self.responses))


def test_agent_loop_and_destructive_denied():
    from bot.agent import Agent
    tu = SimpleNamespace(type="tool_use", id="1", name="add_task", input={"title": "x"})
    txt = SimpleNamespace(type="text", text="done")
    agent = Agent(client=FakeClient([_resp("tool_use", tu), _resp("end_turn", txt)]))
    assert agent.run("add a task") == "done"
    assert memory.query("SELECT title FROM tasks") == [{"title": "x"}]

    memory.execute("INSERT INTO notes(title,body) VALUES('a','b')")
    td = SimpleNamespace(type="tool_use", id="2", name="delete_note", input={"id": 1})
    agent = Agent(client=FakeClient([_resp("tool_use", td), _resp("end_turn", txt)]))
    agent.run("delete it")
    assert len(memory.query("SELECT * FROM notes")) == 1  # denied by default
