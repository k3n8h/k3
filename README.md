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
