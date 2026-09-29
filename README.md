# K3 — multi-purpose assistant

A Python assistant (terminal + web chat) with a provider-agnostic tool-use loop. It works with
Claude, any OpenAI-compatible endpoint (OpenAI, Ollama, vLLM, LM Studio), or **fully offline with no model**.

| Area | Tools |
|---|---|
| Schedules, classes, appointments | `add_event` (conflict check), `list_events`, `find_free_slots`, `cancel_event`, reminders |
| Organizing | tasks and notes |
| Coding | workspace files, `run_python` (timeout) |
| Data analysis | `describe_data`, `query_data`, `make_chart` (CSV/XLSX) |
| Designs / creating | `save_svg`, `save_html`, `save_brainstorm` |
| Moderation | word filter, warn/mute log (in-chat only) |
| Utilities / fun | calculator, dates, units, polls, dice, coin, word scramble |
| Research / web | `web_search` (no key), `research`, `web_fetch`, `scrape` (CSS selectors), `crawl` (robots.txt, delay, SSRF guard) |
| Learning | `learn_instruction`, `list_lessons`, `forget_lesson`: taught rules persist and are applied by every provider |
| Autonomous work | `schedule_job`; unattended goals run with a larger step budget, destructive tools auto-denied |

## Run
```
pip install -e .[dev]
# pick a backend (or nothing for offline mode):
export ANTHROPIC_API_KEY=...            # pip install -e .[anthropic]
# export BOT_BASE_URL=http://localhost:11434/v1 BOT_MODEL=llama3.1   # Ollama / OpenAI-compatible
# BOT_PROVIDER=anthropic|openai|offline forces a choice; default auto-detects
python -m bot.interfaces.cli          # terminal
uvicorn bot.interfaces.web:app        # web UI on :8000
pytest
```
Data lives in `~/.k3-bot` (override with `BOT_HOME`); model via `BOT_MODEL`.

Notes: `run_python` is a timeout/cwd-restricted subprocess, not a security sandbox. The web UI is
single-user/local and denies destructive tools (no interactive confirmation).

Offline mode understands a command grammar (`search`, `research`, `fetch`, `crawl`, `scrape`, `calc`, `add task`, ...)
and taught shortcuts: `learn "morning news" => research top tech news`. Type any unknown text to see the help.
Crawling honors robots.txt and `BOT_CRAWL_DELAY` (default 1s); fetches to private/loopback addresses are blocked.

## Training (on-device, no API)
Offline mode is backed by a small trained intent learner (`bot/learner.py`): a Naive Bayes classifier over
word/bigram/char-trigram features plus schema-driven slot extraction, with an abstain class so chit-chat and
gibberish fall through to help instead of triggering a tool. It is trained from three sources: synthetic
examples (`bot/training/seed.py`), examples you supply, and your live corrections.

```
train                              # retrain; prints held-out accuracy on phrasings it hasn't seen
train add "gimme a d20" => roll d20
train from examples.jsonl          # workspace file: {"phrase","tool","args"} lines or phrase,tool,args CSV
wrong => research solar panels     # correct the last answer (or teach one I did not understand: "that means <command>")
good                               # reinforce the last answer
train status | train reset
python -m bot.train [file]         # headless
```
The learner never auto-runs destructive tools, asks for missing arguments instead of guessing, and with a real
LLM configured, your taught examples are added to the prompt as few-shot hints. This is a small classifier, not
an LLM: it generalizes to new phrasings only as far as its examples cover.

## Capabilities, languages and training coverage
Everything the bot can do is organized in `bot/training/catalog.py` (capabilities, instruction styles, languages,
task types, skills, subject areas) and rendered to [`docs/CAPABILITIES.md`](docs/CAPABILITIES.md); ask the bot
`capabilities` or `what can you do`. Tests fail if a tool is missing from the catalog.

- **Languages trained:** English, Spanish, French, German, Portuguese, Italian (Latin script, accents folded) and Russian (Cyrillic).
  Any other language works through an LLM provider, or teach it with `train add` / `train from`.
- **Subjects:** ~80 topics across 10 subject areas seed the research/search/notes slots.
- **Recognized but not faked:** writing, coding, translating, brainstorming and design requests are classified
  as `needs_model`; offline the bot says it needs a model instead of pretending.
- **Scheduling from free text:** "book a class math on 2030-05-06 at 10:30", "dentist tomorrow at 3pm for 90 minutes".
- **Measured quality:** [`docs/TRAINING_REPORT.md`](docs/TRAINING_REPORT.md) (`python -m bot.train --report`), scored on
  wordings held out from training, broken down by language, capability and tool.

## Security model
- **Destructive tools** (delete/cancel/reset) always ask for confirmation; the web UI and unattended jobs deny them.
- The moderation blocklist is stored in the database (survives restarts); the web chat serializes turns; a failed model
  call is reported and rolled back; conversation history is bounded.
- **Sensitive tools** (`run_python`, `write_file`, `learn_instruction`, `schedule_job`, training imports) ask when a
  *model* chose the call, because text it read on the web could have steered it (prompt injection). In offline mode
  they run directly since your own words are the only input.
- The system prompt tells models that web/file/tool text is untrusted data.
- `query_data` filters reject function calls, attributes and `@variables`; aggregations are whitelisted.
- The web UI is single-user and only accepts `localhost`/`127.0.0.1` Host headers (extend with `BOT_ALLOWED_HOSTS`).
- Fetching blocks private/loopback/link-local addresses, re-checks every redirect hop, and connects to the exact IP that
  passed the check (TLS still validates the original hostname; each vetted address is tried in turn), so DNS rebinding
  cannot swap it. Proxy environment variables are ignored by default because a proxy would route by name and defeat the
  pin; on a host that *must* egress through a proxy set `BOT_TRUST_PROXY_ENV=1` (the address check still runs, but the
  proxy does its own DNS, so it is best-effort there). `SSL_CERT_FILE` / `REQUESTS_CA_BUNDLE` are honored either way.
- `run_python` is a timeout/cwd-restricted subprocess, **not** a sandbox: run the bot in a container if you connect a model.

## Expert roles and certification
The bot is organized as 11 expert roles (Research Analyst, Executive Assistant, Personal Organizer, Data Analyst,
Developer Assistant, Creative Writer & Designer, Community Moderator, Calculation & Utility Expert, Game Host,
Trainer & Teacher, Conversation Partner), each with the skills its job needs (`bot/training/roles.py`). Every skill
has a competency exam split into **dev** phrasings (part of the training curriculum) and **holdout** phrasings
(never trained on). `python -m bot.train --certify` (or just say `roles`) runs them against the shipped model in a
scratch database and writes [`docs/ROLE_CERTIFICATION.md`](docs/ROLE_CERTIFICATION.md) with per-skill results,
language coverage and open items. Tests fail if a tool has no owning skill, a skill lacks exams, or results drop.

Each role also runs an end-to-end **job scenario** (`bot/training/scenarios.py`; run in an isolated subprocess when you ask the bot for `roles`): a multi-step workflow with the real
tools in a scratch database and workspace (book, detect a double booking, refuse a cancel; add/complete tasks; write a
CSV then describe, filter and chart it; crawl a local site honoring robots.txt; warn, mute and audit a user, some steps
in other languages). Tests include negative controls that break each role's tool and require its scenario to fail.

Every skill also faces an independent **stress exam** (differently worded, indirect, typo'd, other languages; never
trained on). Confidence is risk-tiered: state-changing tools need 90% confidence *and* at least one word from that
tool's own command wording (so n-gram luck can't add a blocklist word from "blah blah"), append-only tools (tasks,
notes) and read-only ones need 80%; destructive tools are never auto-run from a guess. A bare URL is read, not
crawled, unless the wording asks to walk the site. Skills that need a model say so instead of pretending.
