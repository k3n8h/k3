# K3 — multi-purpose Claude assistant

A Python assistant (terminal + web chat) built on the Claude API tool-use loop.

| Area | Tools |
|---|---|
| Schedules, classes, appointments | `add_event` (conflict check), `list_events`, `find_free_slots`, `cancel_event`, reminders |
| Organizing | tasks and notes |
| Coding | workspace files, `run_python` (timeout) |
| Data analysis | `describe_data`, `query_data`, `make_chart` (CSV/XLSX) |
| Designs / creating | `save_svg`, `save_html`, `save_brainstorm` |
| Moderation | word filter, warn/mute log (in-chat only) |
| Utilities / fun | calculator, dates, units, polls, dice, coin, word scramble |
| Autonomous work | `schedule_job`; unattended goals run with a larger step budget, destructive tools auto-denied |

## Run
```
pip install -e .[dev]
export ANTHROPIC_API_KEY=...
python -m bot.interfaces.cli          # terminal
uvicorn bot.interfaces.web:app        # web UI on :8000
pytest
```
Data lives in `~/.k3-bot` (override with `BOT_HOME`); model via `BOT_MODEL`.

Notes: `run_python` is a timeout/cwd-restricted subprocess, not a security sandbox. The web UI is
single-user/local and denies destructive tools (no interactive confirmation).
