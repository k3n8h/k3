"""Role certification: does each expert have every skill its role requires, and can it prove it?

For each skill: (1) its tools exist and carry the right safety flags, (2) the intent is trained in which
languages, (3) the exam passes on dev phrasings (used while improving) and on holdout phrasings (never trained on).
Runs against the shipped seed model in a scratch database, so a user's own lessons/examples can't skew the result.
"""
from bot import learner, memory, registry
from bot.training import roles as R
from bot.training import seed

DEV_PASS = 0.75       # a skill is certified when dev >= 75% ...
HOLDOUT_PASS = 0.60   # ... and holdout >= 60%
STRESS_PASS = 0.60    # ... and the independent stress exam (different vocabulary/structure) >= 60%
TRAINED_LANGS = ["en", "es", "fr", "de", "pt", "it", "ru"]


def _same(want, got) -> bool:
    if isinstance(want, (int, float)) and not isinstance(want, bool):
        return isinstance(got, (int, float)) and float(want) == float(got)
    if isinstance(want, str):
        return isinstance(got, str) and want.strip().lower() == got.strip().lower()
    return want == got


def run_exam(phrase: str, expected: str, args) -> tuple[bool, str]:
    from bot import offline
    name, got = offline.parse(phrase)
    if expected == "abstain":
        return name == offline.UNKNOWN, f"{name}"
    if expected.startswith("text:"):
        return name in ("__text__", offline.UNKNOWN) and expected[5:] in str(got).lower(), f"{name}: {str(got)[:60]}"
    if name not in expected.split("|"):
        return False, f"{name}"
    if args:
        for k, v in args.items():
            if k not in got or not _same(v, got[k]):
                return False, f"{name} {got}"
    return True, name


def language_coverage(tool: str, examples: list) -> list:
    counts: dict = {}
    for e in examples:
        if e["tool"] == tool:
            counts[e["lang"]] = counts.get(e["lang"], 0) + 1
    return [l for l in TRAINED_LANGS if counts.get(l, 0) >= 5]


def certify(role_id: str = "") -> dict:
    registry.load_all()
    examples = seed.generate()
    model = learner.Model().fit(examples)
    out = {"roles": [], "trained_examples": len(examples)}
    with memory.scratch(), learner.using(model):
        for role in R.ROLES:
            if role_id and role.id != role_id:
                continue
            rr = {"id": role.id, "name": role.name, "mission": role.mission, "skills": []}
            for sk in role.skills:
                problems = []
                for t in sk.tools:
                    if t not in registry._TOOLS:
                        problems.append(f"tool {t} does not exist")
                    elif sk.mode == "guarded" and not registry.is_destructive(t):
                        problems.append(f"{t} is guarded but not flagged destructive")
                    elif sk.mode == "command" and t in ("run_python", "write_file", "schedule_job", "learn_instruction") \
                            and not registry._TOOLS[t].get("sensitive"):
                        problems.append(f"{t} runs code/persists state but is not flagged sensitive")
                langs = []
                if sk.mode == "trained":
                    per = [language_coverage(t, examples) for t in sk.tools]
                    langs = sorted(set.intersection(*map(set, per)), key=TRAINED_LANGS.index) if per else []
                    if not langs:
                        problems.append("no training examples")
                res = {"d": [0, 0], "h": [0, 0]}
                failures = []
                for split, phrase, expected, args in sk.exams:
                    ok, detail = run_exam(phrase, expected, args)
                    res[split][0] += ok
                    res[split][1] += 1
                    if not ok:
                        failures.append((split, phrase, expected, detail))
                stress = [0, 0]
                for phrase, expected, args in R.STRESS.get(sk.id, []):
                    ok, detail = run_exam(phrase, expected, args)
                    stress[0] += ok
                    stress[1] += 1
                    if not ok:
                        failures.append(("s", phrase, expected, detail))
                stress_rate = stress[0] / stress[1] if stress[1] else 1.0
                dev = res["d"][0] / res["d"][1] if res["d"][1] else 0
                hold = res["h"][0] / res["h"][1] if res["h"][1] else 0
                status = "certified" if (not problems and dev >= DEV_PASS and hold >= HOLDOUT_PASS
                                         and stress_rate >= STRESS_PASS) else "needs work"
                rr["skills"].append({"id": sk.id, "name": sk.name, "mode": sk.mode, "tools": sk.tools, "languages": langs,
                                     "dev": res["d"], "holdout": res["h"], "stress": stress, "problems": problems,
                                     "failures": failures, "status": status})
            n_ok = sum(s["status"] == "certified" for s in rr["skills"])
            rr["certified"] = n_ok == len(rr["skills"])
            rr["summary"] = f"{n_ok}/{len(rr['skills'])} skills certified"
            out["roles"].append(rr)
    return out


def render_markdown(rep: dict) -> str:
    L = ["# Role certification", "",
         "Each expert role lists the skills it needs. A skill is **certified** when its tools exist with the right safety "
         f"flags, it is trained, and its exams pass: dev phrasings >= {DEV_PASS:.0%} (part of the training curriculum), "
         f"holdout phrasings >= {HOLDOUT_PASS:.0%} (never trained on) and an independent **stress** exam >= {STRESS_PASS:.0%} "
         "(deliberately different vocabulary and structure: indirect requests, typos, other languages; never trained on, "
         "but failures were used to fix general extraction bugs and to write broader compositional paraphrases, so treat it "
         "as an upper-ish estimate). Exams run through the same offline path a "
         "user hits (exact grammar, then the learner), against the shipped seed model.", "",
         "Modes: **trained** free-form intent; **command** explicit command form offline (a model can also call the tool); "
         "**guarded** destructive, never auto-run from a guess; **model** needs an LLM; **abstain** small talk.", "",
         f"_Generated with `python -m bot.train --certify`; {rep['trained_examples']} generated training examples._", ""]
    for r in rep["roles"]:
        L += [f"## {r['name']}  -  {'CERTIFIED' if r['certified'] else 'needs work'} ({r['summary']})", "", r["mission"], "",
              "| Skill | Mode | Tools | Langs | Dev | Holdout | Stress | Status |", "|---|---|---|---|---|---|---|---|"]
        for s in r["skills"]:
            langs = ",".join(s["languages"]) if s["languages"] else "-"
            L.append(f"| {s['name']} | {s['mode']} | {', '.join(s['tools']) or '-'} | {langs} | {s['dev'][0]}/{s['dev'][1]} | "
                     f"{s['holdout'][0]}/{s['holdout'][1]} | {s['stress'][0]}/{s['stress'][1]} | {s['status']} |")
        bad = [(s["name"], s["problems"], s["failures"]) for s in r["skills"] if s["problems"] or s["failures"]]
        if bad:
            L += ["", "Open items:"]
            for name, problems, failures in bad:
                for p in problems:
                    L.append(f"- **{name}**: {p}")
                for split, phrase, expected, detail in failures:
                    L.append(f"- **{name}** [{ {'d': 'dev', 'h': 'holdout', 's': 'stress'}[split] }] `{phrase}` expected `{expected}`, got `{detail}`")
        L.append("")
    return "\n".join(L)


def summary(rep: dict) -> str:
    return "\n".join(f"- {r['name']}: {'CERTIFIED' if r['certified'] else 'needs work'} ({r['summary']})" for r in rep["roles"])
