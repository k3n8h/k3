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


def test_trained_skills_cover_all_seven_languages_except_the_documented_two(report):
    thin = {s["id"] for r in report["roles"] for s in r["skills"] if s["mode"] == "trained" and len(s["languages"]) < 7}
    assert thin <= {"chart", "explain_self"}, thin          # chart needs column names; certify_roles is English-only


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
