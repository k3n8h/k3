"""Headless training and reporting.

  python -m bot.train [examples.jsonl|csv]   train (optionally importing your examples first)
  python -m bot.train --report               also write docs/TRAINING_REPORT.md
  python -m bot.train --docs                 (re)write docs/CAPABILITIES.md from the catalog
"""
import json
import sys
from pathlib import Path

from bot import learner, registry
from bot.training import catalog

DOCS = Path(__file__).resolve().parent.parent / "docs"


def write_docs() -> Path:
    DOCS.mkdir(exist_ok=True)
    p = DOCS / "CAPABILITIES.md"
    p.write_text(catalog.render_markdown())
    return p


def render_report(r: dict) -> str:
    h = r["heldout"]
    tbl = lambda title, d: [f"### {title}", "", "| | intent accuracy |", "|---|---|"] + \
        [f"| {k} | {v:.0%} |" for k, v in d.items()] + [""]
    L = ["# Training report", "",
         f"Trained on {r['seed_examples']} generated + {r['user_examples']} taught examples.", "",
         "Measured on phrasings **held out from training** (every 4th template of each tool and language is "
         "never seen), so these numbers reflect generalization to new wording, not memorization.", "",
         f"- Familiar phrasings (new slot values): **{r['seen_phrasing']:.0%}** intent accuracy",
         f"- Unseen phrasings: **{h['intent_accuracy']:.0%}** intent, **{h['slot_accuracy']:.0%}** intent + arguments",
         f"- Unseen phrasings that triggered a *wrong* action: **{h['wrong_action_rate']:.1%}** "
         f"(the rest were either right or safely abstained with help text)", ""]
    L += tbl("By language", h["per_language"]) + tbl("By capability", h["per_capability"]) + \
        tbl("By tool / intent", h["per_tool"])
    L += ["_Misses are mostly abstentions: an unfamiliar wording falls through to help rather than guessing. "
          "Fix one with `wrong => <command>` or `train add \"<phrase>\" => <command>`._", ""]
    return "\n".join(L)


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    registry.load_all()
    if "--docs" in sys.argv:
        print("wrote", write_docs())
        return
    if args:
        print(registry.call("train_from_file", {"path": args[0]}))
    r = learner.fit_all(evaluate=True)
    if "--report" in sys.argv:
        DOCS.mkdir(exist_ok=True)
        (DOCS / "TRAINING_REPORT.md").write_text(render_report(r))
        write_docs()
        print("wrote docs/TRAINING_REPORT.md and docs/CAPABILITIES.md")
    print(json.dumps({k: v for k, v in r.items() if k != "heldout"}, indent=2))
    h = r["heldout"]
    print({k: h[k] for k in ("intent_accuracy", "slot_accuracy", "wrong_action_rate")})


if __name__ == "__main__":
    main()
