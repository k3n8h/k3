# K3 capabilities

_Generated from `bot/training/catalog.py` (`python -m bot.train --docs`)._

## Capabilities

### Research & web  (works offline)
Search the web, read pages, scrape by CSS selector, crawl sites politely, write sourced research files.
Tools: `web_search`, `research`, `web_fetch`, `scrape`, `crawl`
Try: `research solar panels`; `scrape https://x.com h1`; `crawl https://a.com depth 2`

### Schedules, classes & appointments  (works offline)
Book classes/appointments/events with conflict checks and reminders; find free time; schedule unattended jobs.
Tools: `add_event`, `list_events`, `find_free_slots`, `cancel_event`, `schedule_job`, `list_jobs`
Try: `book a dentist appointment on 2030-03-05 at 14:00`; `what's on my calendar`; `when am i free on 2030-03-05`

### Organizing (tasks & notes)  (works offline)
To-do list with priorities/dues and searchable notes.
Tools: `add_task`, `list_tasks`, `complete_task`, `add_note`, `search_notes`, `delete_note`
Try: `add task buy milk`; `note ideas: launch plan`; `search my notes for budget`

### Coding & files  (works offline)
Read/write workspace files and run Python with a timeout. (Writing new code needs a model.)
Tools: `write_file`, `read_file`, `list_files`, `run_python`
Try: `show file notes.md`; `list files`

### Data analysis  (works offline)
Summarize, filter/group and chart CSV/XLSX files.
Tools: `describe_data`, `query_data`, `make_chart`
Try: `describe sales.csv`; `analyze the file report.xlsx`

### Designs & creating  (works offline)
Persist SVG/HTML designs and brainstorm lists. (Authoring them needs a model.)
Tools: `save_svg`, `save_html`, `save_brainstorm`

### Moderation  (works offline)
Profanity filter, warn/mute log (in this chat only).
Tools: `moderate_text`, `add_blocked_word`, `moderate_user`, `user_moderation_status`
Try: `is this offensive: what the hell`

### Utilities  (works offline)
Arithmetic, date/time, unit conversion, polls.
Tools: `calculate`, `now`, `date_add`, `convert_units`, `make_poll`
Try: `what's 12 times 7`; `convert 5 km to miles`; `what time is it`

### Fun & games  (works offline)
Dice, coin, random picks, word scramble.
Tools: `roll_dice`, `flip_coin`, `pick_random`, `scramble_word`
Try: `roll 2d6`; `heads or tails`

### Learning & training  (works offline)
Teach shortcuts and phrasings, correct mistakes, retrain the on-device intent learner.
Tools: `learn_instruction`, `list_lessons`, `forget_lesson`, `train_model`, `training_status`, `add_training_example`, `train_from_file`, `mark_wrong`, `mark_good`, `reset_training`, `list_capabilities`
Try: `train add "gimme a d20" => roll d20`; `wrong => research solar panels`

### Writing, coding, design, translation, brainstorming  (needs a model provider)
Open-ended generation (essays, code, poems, emails, translations, brainstorms, designs). Recognized offline, produced only when a model provider is connected.
Try: `write me an essay about volcanoes`; `translate hello to french`

### Conversation  (needs a model provider)
Free-form chat; offline the bot abstains rather than guessing.
Try: `hello`

## Instruction styles the learner is trained on

- **imperative**: `research solar panels`
- **polite request**: `could you please research solar panels`
- **question**: `what is 12 times 7`
- **telegraphic**: `solar panels research`
- **desire / indirect**: `i'd like to know about solar panels`
- **with typos**: `reserach solar pnaels`
- **code-switched**: `busca solar panels`
- **taught shortcut**: `learn "morning news" => research top tech news`
- **correction**: `wrong => research solar panels`

## Languages

- English (`en`, Latin): trained
- Spanish (`es`, Latin): trained
- French (`fr`, Latin): trained
- German (`de`, Latin): trained
- Portuguese (`pt`, Latin): trained
- Italian (`it`, Latin): trained
- Russian (`ru`, Cyrillic): trained
- Any other language (`*`, any): model-only (or teach it: `train add` / `train from`)

## Task types

lookup, create, modify, schedule, analyze, automate, organize, converse, play, teach

## Skills

- **research**: `web_search`, `research`, `web_fetch`, `scrape`, `crawl`
- **time management**: `add_event`, `find_free_slots`, `list_events`, `schedule_job`, `add_task`
- **note-taking**: `add_note`, `search_notes`
- **data literacy**: `describe_data`, `query_data`, `make_chart`
- **quantitative reasoning**: `calculate`, `convert_units`, `date_add`
- **programming support**: `write_file`, `read_file`, `run_python`
- **content safety**: `moderate_text`, `moderate_user`
- **play**: `roll_dice`, `flip_coin`, `pick_random`, `scramble_word`
- **self-improvement**: `learn_instruction`, `train_model`, `mark_wrong`, `mark_good`

## Subject areas (topic pool for training)

- **Natural sciences**: photosynthesis, black holes, plate tectonics, the periodic table, dna replication, quantum mechanics, climate change, the water cycle, evolution, ocean currents
- **Mathematics**: linear algebra, prime numbers, the pythagorean theorem, probability, calculus, graph theory, statistics, fibonacci numbers
- **Technology & computing**: python asyncio, machine learning, quantum computing, blockchain, solar panels, electric cars, cybersecurity, rust ownership, sql joins, http caching
- **History**: the french revolution, the roman empire, the industrial revolution, the cold war, ancient egypt, the silk road, world war two, the renaissance
- **Geography & travel**: the amazon rainforest, mount everest, the sahara desert, visa rules for japan, the great barrier reef, iceland volcanoes, the nile river
- **Health & fitness**: sleep hygiene, intermittent fasting, marathon training, vitamin d, meditation, cholesterol, first aid basics
- **Business & finance**: compound interest, index funds, startup fundraising, supply chains, coffee prices, inflation, content marketing, cash flow forecasting
- **Arts, literature & languages**: impressionism, shakespeare sonnets, jazz history, spanish verbs, japanese kanji, film noir, poetry meter, french cuisine
- **Society & law**: copyright law, renewable energy policy, urban planning, voting systems, human rights, tenant rights
- **Everyday life**: sourdough bread, houseplant care, budget meal prep, bike maintenance, space telescopes, home composting, learning guitar, public speaking
