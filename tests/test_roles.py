import pytest

from bot import learner, memory, registry
from bot.training import certify, roles, seed


@pytest.fixture(scope="module")
def report():
    return certify.certify()


def test_every_tool_belongs_to_exactly_one_skill_and_role_tools_exist():
    registry.load_all()
    tools = roles.all_tools()
    assert len(tools) == len(set(tools)), "a tool is claimed by two skills"
    assert set(tools) == set(registry._TOOLS), set(tools) ^ set(registry._TOOLS)


def test_every_skill_has_dev_and_holdout_exams():
    for r in roles.ROLES:
        for s in r.skills:
            splits = {e[0] for e in s.exams}
            assert splits == {roles.D, roles.H}, (r.id, s.id, splits)
            assert s.mode in ("trained", "command", "guarded", "model", "abstain")


def test_holdout_phrases_are_never_training_data():
    train_phrases = {e["phrase"].lower() for e in seed.curriculum()}
    for r in roles.ROLES:
        for s in r.skills:
            for split, phrase, *_ in s.exams:
                if split == roles.H:
                    assert phrase.lower() not in train_phrases, phrase


def test_no_skill_has_structural_problems(report):
    for r in report["roles"]:
        for s in r["skills"]:
            assert not s["problems"], (r["id"], s["id"], s["problems"])


def test_trained_skills_cover_every_trained_language_or_say_which(report):
    for r in report["roles"]:
        for s in r["skills"]:
            if s["mode"] == "trained":
                assert s["languages"] and "en" in s["languages"], (r["id"], s["id"])


def test_role_exam_results_meet_the_bar(report):
    dev = [s["dev"] for r in report["roles"] for s in r["skills"]]
    hold = [s["holdout"] for r in report["roles"] for s in r["skills"]]
    assert sum(a for a, b in dev) / sum(b for a, b in dev) >= 0.9
    assert sum(a for a, b in hold) / sum(b for a, b in hold) >= 0.85        # unseen phrasings, all roles together
    weak = [(r["id"], s["id"]) for r in report["roles"] for s in r["skills"] if s["status"] != "certified"]
    assert not weak, weak
    assert all(r["certified"] for r in report["roles"])
    stress = [s["stress"] for r in report["roles"] for s in r["skills"]]
    assert sum(a for a, b in stress) / sum(b for a, b in stress) >= 0.88     # independent, differently-worded exams


def test_certification_does_not_touch_the_users_database_or_model(tmp_path, monkeypatch):
    monkeypatch.setenv("BOT_HOME", str(tmp_path))
    memory.connect(str(tmp_path / "t.db"))
    memory.execute("INSERT INTO lessons(trigger,action) VALUES('hi','now')")
    learner.fit_all()
    before = (tmp_path / "model.json").read_text()
    certify.certify("game_host")
    assert memory.query("SELECT trigger FROM lessons") == [{"trigger": "hi"}]      # real DB restored
    assert (tmp_path / "model.json").read_text() == before                          # saved model untouched


def test_role_report_is_available_from_the_bot_and_renders(report):
    from bot import llm
    from bot.agent import Agent
    out = Agent(provider=llm.get_provider("offline")).run("roles")
    assert "Research Analyst" in out and "Game Host" in out
    md = certify.render_markdown(report)
    assert "Role certification" in md and "| Skill | Mode |" in md
    assert "which experts are ready" in seed.SPECS["certify_roles"][0]


def test_stress_exams_are_independent_of_training_data():
    train = {e["phrase"].lower() for e in seed.curriculum()}
    assert {sk.id for r in roles.ROLES for sk in r.skills if sk.mode == "trained"} <= set(roles.STRESS) | {"explain_self"}
    for skill_id, items in roles.STRESS.items():
        for phrase, *_ in items:
            assert phrase.lower() not in train, phrase
        assert len(items) >= 3, skill_id


def test_every_trained_skill_covers_all_seven_languages(report):
    thin = {s["id"] for r in report["roles"] for s in r["skills"] if s["mode"] == "trained" and len(s["languages"]) < 7}
    assert not thin, thin


def test_state_changing_guesses_need_command_evidence_and_bare_urls_are_not_crawls():
    learner.fit_all()
    assert learner.interpret("blah blah") is None                        # n-gram luck is not evidence for add_blocked_word
    assert learner.interpret("hello there my friend") is None
    r = learner.interpret("fetch https://example.com/x")
    assert r["tool"] == "web_fetch"
    assert learner.interpret("crawl https://example.com and go 2 levels deep")["tool"] == "crawl"
    r = learner.interpret("está silenciado mallory")                    # a status question, not the imperative 'silencia'
    assert r["tool"] == "user_moderation_status"
    assert learner.interpret("silencia a mallory")["tool"] == "moderate_user"
    assert learner.interpret("mallory keeps spamming, warn her: links")["args"]["user"] == "mallory"


# ---- job scenarios: they must pass on the real system and FAIL when a role's tool is broken (negative controls)
def test_every_role_has_a_passing_job_scenario(report):
    from bot.training import scenarios
    assert set(scenarios.SCENARIOS) == {r.id for r in roles.ROLES}
    for r in report["roles"]:
        assert r["scenario_ok"], (r["id"], r["scenario_detail"])


@pytest.mark.parametrize("role_id, breakage", [
    ("executive_assistant", ("attr", "bot.tools.calendar", "find_conflicts", lambda *a, **k: [])),
    ("personal_organizer", ("tool", "complete_task", lambda id: {"updated": 0})),
    ("community_moderator", ("attr", "bot.tools.moderation", "check_text", lambda text: [])),
    ("utility_expert", ("tool", "calculate", lambda expression: 0)),
    ("game_host", ("tool", "flip_coin", lambda: "edge")),
    ("trainer", ("tool", "learn_instruction", lambda trigger, action: {"id": 0})),
    ("research_analyst", ("attr", "bot.web", "robots_allowed", lambda url: True)),
    ("developer_assistant", ("tool", "run_python", lambda code, timeout=10: {"exit_code": 1, "stdout": "", "stderr": ""})),
    ("data_analyst", ("tool", "describe_data", lambda path: {"shape": [0, 0]})),
    ("conversationalist", ("tool", "add_task", None)),
])
def test_scenarios_fail_when_the_roles_tools_are_broken(role_id, breakage, monkeypatch):
    import importlib
    from bot.training import scenarios
    kind = breakage[0]
    if kind == "attr":
        monkeypatch.setattr(importlib.import_module(breakage[1]), breakage[2], breakage[3])
    elif role_id == "conversationalist":
        # small talk must never touch state: simulate an over-eager bot that turns any text into a task
        from bot import offline
        monkeypatch.setattr(offline, "parse", lambda text, depth=0: ("add_task", {"title": text}))
    else:
        monkeypatch.setitem(registry._TOOLS[breakage[1]], "fn", breakage[2])
    ok, detail = scenarios.run(role_id)
    assert not ok, f"{role_id} scenario still passed with a broken tool"
    assert detail and detail != "ok"


# ---- isolation: certification must never leak into (or depend on) the caller's process state
def test_moderation_scenario_does_not_leak_into_the_real_blocklist(tmp_path, monkeypatch):
    from bot.tools import moderation
    from bot.training import scenarios
    monkeypatch.setenv("BOT_HOME", str(tmp_path))
    memory.connect(str(tmp_path / "real.db"))
    assert scenarios.run("community_moderator") == (True, "ok")
    assert "phishing" not in moderation.blocklist()                    # scratch DB, nothing leaked
    assert scenarios.run("community_moderator") == (True, "ok")        # repeatable: 'not blocked before added' holds again


def test_blocklist_is_persistent_and_validates_input(tmp_path, monkeypatch):
    from bot.tools import moderation
    monkeypatch.setenv("BOT_HOME", str(tmp_path))
    memory.connect(str(tmp_path / "a.db"))
    registry.call("add_blocked_word", {"word": "Phishing"})
    memory.connect(str(tmp_path / "a.db"))                              # "restart": new connection, same file
    assert "phishing" in moderation.blocklist() and moderation.check_text("a phishing link") == ["phishing"]
    assert registry.call("add_blocked_word", {"word": "two words"}).startswith("error")
    assert registry.call("add_blocked_word", {"word": ""}).startswith("error")
    assert moderation.check_text("ÑANDÚ shit") == ["shit"]              # non-ASCII text no longer confuses the filter


def test_research_scenario_restores_environment_and_robots_cache(monkeypatch):
    import os
    from bot import web
    from bot.training import scenarios
    monkeypatch.setenv("BOT_CRAWL_DELAY", "2.5")
    monkeypatch.delenv("BOT_ALLOW_PRIVATE_NETS", raising=False)
    web._robots["https://real.example:443"] = None                      # a cached real-site entry must survive
    try:
        assert scenarios.run("research_analyst") == (True, "ok")
        assert os.environ["BOT_CRAWL_DELAY"] == "2.5"
        assert "BOT_ALLOW_PRIVATE_NETS" not in os.environ
        assert "https://real.example:443" in web._robots
        assert len(web._robots) == 1                                    # the scenario's own server entry is gone
    finally:
        web._robots.pop("https://real.example:443", None)


def test_env_helper_restores_on_error():
    import os
    from bot.training.scenarios import env
    os.environ["K3_T1"] = "a"
    with pytest.raises(RuntimeError):
        with env(K3_T1="b", K3_T2="c"):
            assert os.environ["K3_T1"] == "b"
            raise RuntimeError
    assert os.environ["K3_T1"] == "a" and "K3_T2" not in os.environ
    del os.environ["K3_T1"]


def test_certify_roles_tool_runs_isolated_and_leaves_the_session_alone(tmp_path, monkeypatch):
    monkeypatch.setenv("BOT_HOME", str(tmp_path))
    memory.connect(str(tmp_path / "live.db"))
    memory.execute("INSERT INTO tasks(title) VALUES('keep me')")
    learner.LAST.update(phrase="x", tool="now", args={})
    conn_before = memory._conn
    out = registry.call("certify_roles", {"role": "game_host"})
    assert "Game Host: CERTIFIED" in out, out
    assert memory._conn is conn_before                                  # the live connection was never swapped
    assert memory.query("SELECT title FROM tasks") == [{"title": "keep me"}]
    assert learner.LAST == {"phrase": "x", "tool": "now", "args": {}}
    assert registry.call("certify_roles", {"role": "no_such_role"}).count("CERTIFIED") == 0
