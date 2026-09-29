"""Training tools: teach, retrain and inspect the on-device intent learner."""
import csv
import json

from bot import learner, memory, registry
from bot.registry import tool
from bot.tools.code import safe_path


def _command_to_call(command: str) -> tuple[str, dict]:
    from bot.offline import parse_grammar
    parsed = parse_grammar(command)
    if not parsed or parsed[0] == "__text__":
        raise ValueError("I couldn't read that as an exact command (e.g. `research solar panels`)")
    if registry.is_destructive(parsed[0]):
        raise ValueError("won't learn destructive commands from feedback")
    return parsed


@tool("Train (or retrain) the local intent learner from seed data plus everything taught so far; "
      "returns held-out accuracy on phrasings it has not seen.")
def train_model() -> dict:
    return learner.fit_all(evaluate=True)


@tool("Show how many examples the learner has and whether a model exists.")
def training_status() -> dict:
    ex = learner.stored_examples()
    by_source: dict = {}
    for e in ex:
        by_source[e["source"]] = by_source.get(e["source"], 0) + 1
    m = learner.get_model()
    return {"model_examples": m.n_examples, "intents": sorted(t for t in m.counts if t != "chat"),
            "user_examples": len(ex), "by_source": by_source}


@tool("Teach the learner that `phrase` means `command` (an exact command like 'research solar panels'), then retrain.")
def add_training_example(phrase: str, command: str) -> dict:
    name, args = _command_to_call(command)
    learner.add_example(phrase, name, args, source="user", weight=2)
    learner.fit_all()
    return {"learned": {"phrase": phrase, "tool": name, "args": args}}


@tool("Train from a workspace file: JSONL lines {\"phrase\",\"tool\",\"args\"} or CSV columns phrase,tool,args(json).")
def train_from_file(path: str) -> dict:
    p = safe_path(path)
    rows = []
    if p.suffix.lower() == ".csv":
        with p.open(newline="") as fh:
            rows = [{"phrase": r["phrase"], "tool": r["tool"], "args": json.loads(r.get("args") or "{}")}
                    for r in csv.DictReader(fh)]
    else:
        rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    added = skipped = 0
    for r in rows:
        if r.get("tool") in registry._TOOLS and not registry.is_destructive(r["tool"]) and r.get("phrase"):
            learner.add_example(r["phrase"], r["tool"], r.get("args") or {}, source="file", weight=2)
            added += 1
        else:
            skipped += 1
    learner.fit_all()
    return {"added": added, "skipped": skipped}


@tool("The last answer was wrong: `command` is what the user actually meant (exact command). "
      "Learns the correction, retrains and runs the corrected command.")
def mark_wrong(command: str) -> dict:
    if not learner.LAST:
        raise ValueError("nothing to correct yet")
    name, args = _command_to_call(command)
    learner.add_example(learner.LAST["phrase"], name, args, source="correction", weight=3)
    learner.fit_all()
    return {"learned": {"phrase": learner.LAST["phrase"], "tool": name, "args": args},
            "result": registry.call(name, args)}


@tool("The last answer was right: reinforce it.")
def mark_good() -> dict:
    if not learner.LAST:
        raise ValueError("nothing to confirm yet")
    l = learner.LAST
    learner.add_example(l["phrase"], l["tool"], l["args"], source="confirm", weight=2)
    learner.fit_all()
    return {"reinforced": l["phrase"]}


@tool("Forget all taught examples and the trained model (back to seed data only).", destructive=True)
def reset_training() -> dict:
    n = memory.execute("DELETE FROM examples").rowcount
    if learner.model_path().exists():
        learner.model_path().unlink()
    return {"deleted_examples": n}
