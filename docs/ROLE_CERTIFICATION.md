# Role certification

Each expert role lists the skills it needs. A skill is **certified** when its tools exist with the right safety flags, it is trained, and its exams pass: dev phrasings >= 75% (part of the training curriculum), holdout phrasings >= 60% (never trained on) and an independent **stress** exam >= 60% (deliberately different vocabulary and structure: indirect requests, typos, other languages; never trained on, but failures were used to fix general extraction bugs and to write broader compositional paraphrases, so treat it as an upper-ish estimate). Exams run through the same offline path a user hits (exact grammar, then the learner), against the shipped seed model.

Modes: **trained** free-form intent; **command** explicit command form offline (a model can also call the tool); **guarded** destructive, never auto-run from a guess; **model** needs an LLM; **abstain** small talk.

_Generated with `python -m bot.train --certify`; 5026 generated training examples._

## Research Analyst  -  CERTIFIED (5/5 skills certified)

Find, read, extract and cite information from the web.

| Skill | Mode | Tools | Langs | Dev | Holdout | Stress | Status |
|---|---|---|---|---|---|---|---|
| Find sources on the web | trained | web_search | en,es,fr,de,pt,it,ru | 2/2 | 4/4 | 4/4 | certified |
| Research a topic with sources | trained | research | en,es,fr,de,pt,it,ru | 2/2 | 3/3 | 3/4 | certified |
| Read a web page | trained | web_fetch | en,es,fr,de,pt,it,ru | 2/2 | 2/2 | 3/4 | certified |
| Extract data from a page (CSS selectors) | trained | scrape | en,es,fr,de,pt,it,ru | 2/2 | 2/2 | 3/3 | certified |
| Map a website politely (robots.txt, delays) | trained | crawl | en,es,fr,de,pt,it,ru | 2/2 | 2/2 | 3/3 | certified |

Open items:
- **Research a topic with sources** [stress] `tell me what the literature says about intermittent fasting` expected `research|web_search`, got `__unknown__`
- **Read a web page** [stress] `bring me the contents of https://example.net/post/12` expected `web_fetch`, got `__text__`

## Executive Assistant (Scheduler)  -  CERTIFIED (6/6 skills certified)

Manage classes, appointments, free time and unattended jobs.

| Skill | Mode | Tools | Langs | Dev | Holdout | Stress | Status |
|---|---|---|---|---|---|---|---|
| Book classes, appointments and events | trained | add_event | en,es,fr,de,pt,it,ru | 2/2 | 2/2 | 4/4 | certified |
| Review the calendar | trained | list_events | en,es,fr,de,pt,it,ru | 2/2 | 3/3 | 3/4 | certified |
| Find free time | trained | find_free_slots | en,es,fr,de,pt,it,ru | 2/2 | 2/2 | 3/3 | certified |
| Cancel bookings (guarded) | guarded | cancel_event | - | 2/2 | 1/1 | 0/0 | certified |
| Schedule unattended jobs | command | schedule_job | - | 2/2 | 1/1 | 0/0 | certified |
| Review scheduled jobs | trained | list_jobs | en,es,fr,de,pt,it,ru | 2/2 | 2/2 | 3/3 | certified |

Open items:
- **Review the calendar** [stress] `qu'est-ce que j'ai au programme` expected `list_events`, got `__unknown__`

## Personal Organizer  -  CERTIFIED (6/6 skills certified)

Keep to-dos and notes tidy and findable.

| Skill | Mode | Tools | Langs | Dev | Holdout | Stress | Status |
|---|---|---|---|---|---|---|---|
| Add tasks | trained | add_task | en,es,fr,de,pt,it,ru | 3/3 | 2/2 | 3/4 | certified |
| Review tasks | trained | list_tasks | en,es,fr,de,pt,it,ru | 2/2 | 2/2 | 3/3 | certified |
| Complete tasks | trained | complete_task | en,es,fr,de,pt,it,ru | 2/2 | 2/2 | 3/3 | certified |
| Take notes | trained | add_note | en,es,fr,de,pt,it,ru | 2/2 | 2/2 | 3/3 | certified |
| Find notes | trained | search_notes | en,es,fr,de,pt,it,ru | 2/2 | 2/2 | 3/3 | certified |
| Delete notes (guarded) | guarded | delete_note | - | 2/2 | 1/1 | 0/0 | certified |

Open items:
- **Add tasks** [stress] `hay que sacar la basura` expected `add_task`, got `__unknown__`

## Data Analyst  -  CERTIFIED (3/3 skills certified)

Summarize, filter and chart tabular data.

| Skill | Mode | Tools | Langs | Dev | Holdout | Stress | Status |
|---|---|---|---|---|---|---|---|
| Profile a dataset | trained | describe_data | en,es,fr,de,pt,it,ru | 3/3 | 2/2 | 3/3 | certified |
| Filter and aggregate | command | query_data | - | 2/2 | 1/1 | 0/0 | certified |
| Chart data | trained | make_chart | en | 2/2 | 2/2 | 3/3 | certified |

## Developer Assistant  -  CERTIFIED (4/4 skills certified)

Work with workspace files and run small Python snippets.

| Skill | Mode | Tools | Langs | Dev | Holdout | Stress | Status |
|---|---|---|---|---|---|---|---|
| List files | trained | list_files | en,es,fr,de,pt,it,ru | 3/3 | 2/2 | 3/3 | certified |
| Read files | trained | read_file | en,es,fr,de,pt,it,ru | 2/2 | 2/2 | 2/3 | certified |
| Write files | command | write_file | - | 2/2 | 1/1 | 0/0 | certified |
| Run Python | command | run_python | - | 2/2 | 1/1 | 0/0 | certified |

Open items:
- **Read files** [stress] `show what is written in todo.txt` expected `read_file`, got `list_tasks`

## Creative Writer & Designer  -  CERTIFIED (2/2 skills certified)

Compose text, code, translations, designs and brainstorms (needs a model).

| Skill | Mode | Tools | Langs | Dev | Holdout | Stress | Status |
|---|---|---|---|---|---|---|---|
| Persist SVG/HTML designs and brainstorms | model | save_svg, save_html, save_brainstorm | - | 2/2 | 1/1 | 0/0 | certified |
| Write essays, poems, emails, code; translate | model | - | - | 2/2 | 2/3 | 0/0 | certified |

Open items:
- **Write essays, poems, emails, code; translate** [holdout] `translate good morning to german` expected `text:language model`, got `__unknown__: I'm running offline (no model configured), so I understand t`

## Community Moderator  -  CERTIFIED (4/4 skills certified)

Keep chat civil: filter words, warn and mute users, keep a record.

| Skill | Mode | Tools | Langs | Dev | Holdout | Stress | Status |
|---|---|---|---|---|---|---|---|
| Screen text for profanity | trained | moderate_text | en,es,fr,de,pt,it,ru | 2/2 | 2/2 | 2/3 | certified |
| Manage the blocklist | trained | add_blocked_word | en,es,fr,de,pt,it,ru | 2/2 | 2/2 | 3/3 | certified |
| Warn, mute and unmute users | trained | moderate_user | en,es,fr,de,pt,it,ru | 2/2 | 2/2 | 3/3 | certified |
| Check a user's moderation record | trained | user_moderation_status | en,es,fr,de,pt,it,ru | 2/2 | 2/2 | 3/3 | certified |

Open items:
- **Screen text for profanity** [stress] `filter this comment: great post thanks` expected `moderate_text`, got `moderate_text {'text': 'great post'}`

## Calculation & Utility Expert  -  CERTIFIED (5/5 skills certified)

Arithmetic, dates, time, unit conversion and polls.

| Skill | Mode | Tools | Langs | Dev | Holdout | Stress | Status |
|---|---|---|---|---|---|---|---|
| Do arithmetic | trained | calculate | en,es,fr,de,pt,it,ru | 2/2 | 3/3 | 4/4 | certified |
| Tell the time and date | trained | now | en,es,fr,de,pt,it,ru | 2/2 | 3/3 | 3/3 | certified |
| Add and subtract days | trained | date_add | en,es,fr,de,pt,it,ru | 2/2 | 2/2 | 3/3 | certified |
| Convert units | trained | convert_units | en,es,fr,de,pt,it,ru | 2/2 | 2/2 | 3/3 | certified |
| Create polls | trained | make_poll | en,es,fr,de,pt,it,ru | 2/2 | 1/1 | 3/3 | certified |

## Game Host  -  CERTIFIED (4/4 skills certified)

Run dice, coin flips, random picks and word games.

| Skill | Mode | Tools | Langs | Dev | Holdout | Stress | Status |
|---|---|---|---|---|---|---|---|
| Roll dice | trained | roll_dice | en,es,fr,de,pt,it,ru | 2/2 | 3/3 | 3/3 | certified |
| Flip a coin | trained | flip_coin | en,es,fr,de,pt,it,ru | 2/2 | 3/3 | 3/3 | certified |
| Pick randomly between options | trained | pick_random | en,es,fr,de,pt,it,ru | 2/2 | 2/2 | 3/3 | certified |
| Word scramble | trained | scramble_word | en,es,fr,de,pt,it,ru | 2/2 | 2/2 | 3/3 | certified |

## Trainer & Teacher (self-improvement)  -  CERTIFIED (8/8 skills certified)

Learn from the user: shortcuts, phrasings, corrections; explain capabilities.

| Skill | Mode | Tools | Langs | Dev | Holdout | Stress | Status |
|---|---|---|---|---|---|---|---|
| Learn a shortcut | command | learn_instruction | - | 2/2 | 1/1 | 0/0 | certified |
| Recall what was learned | trained | list_lessons | en,es,fr,de,pt,it,ru | 2/2 | 2/2 | 2/3 | certified |
| Forget a lesson (guarded) | guarded | forget_lesson | - | 1/1 | 1/1 | 0/0 | certified |
| Retrain and report | command | train_model, training_status | - | 2/2 | 1/1 | 0/0 | certified |
| Teach a phrasing | command | add_training_example, train_from_file | - | 2/2 | 2/2 | 0/0 | certified |
| Take corrections and confirmations | command | mark_wrong, mark_good | - | 3/3 | 2/2 | 0/0 | certified |
| Reset training (guarded) | guarded | reset_training | - | 1/1 | 1/1 | 0/0 | certified |
| Explain capabilities and role readiness | trained | list_capabilities, certify_roles | en | 2/2 | 3/3 | 3/3 | certified |

Open items:
- **Recall what was learned** [stress] `list the things i told you to remember` expected `list_lessons`, got `__unknown__`

## Conversation Partner  -  CERTIFIED (1/1 skills certified)

Recognize small talk; offline it abstains instead of guessing.

| Skill | Mode | Tools | Langs | Dev | Holdout | Stress | Status |
|---|---|---|---|---|---|---|---|
| Abstain on chit-chat | abstain | - | - | 2/2 | 3/3 | 0/0 | certified |
