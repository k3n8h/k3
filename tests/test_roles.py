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
    assert len(weak) <= 2, weak                                           # at most a couple of documented gaps
    certified_roles = [r["id"] for r in report["roles"] if r["certified"]]
    assert len(certified_roles) >= 9


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
