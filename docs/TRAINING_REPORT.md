# Training report

Trained on 5248 generated + 0 taught examples.

Measured on phrasings **held out from training** (every 4th template of each tool and language is never seen), so these numbers reflect generalization to new wording, not memorization.

- Familiar phrasings (new slot values): **95%** intent accuracy
- Unseen phrasings: **76%** intent, **64%** intent + arguments
- Unseen phrasings that triggered a *wrong* action: **5.5%** (the rest were either right or safely abstained with help text)

### By language

| | intent accuracy |
|---|---|
| de | 80% |
| en | 81% |
| es | 79% |
| fr | 64% |
| it | 79% |
| pt | 80% |
| ru | 60% |

### By capability

| | intent accuracy |
|---|---|
| chat | 68% |
| coding | 68% |
| data | 83% |
| fun | 88% |
| generative | 39% |
| moderation | 78% |
| organize | 71% |
| research | 79% |
| schedule | 71% |
| teaching | 60% |
| utilities | 90% |

### By tool / intent

| | intent accuracy |
|---|---|
| add_blocked_word | 79% |
| add_event | 90% |
| add_note | 82% |
| add_task | 60% |
| calculate | 99% |
| cancel_event | 100% |
| certify_roles | 88% |
| chat | 68% |
| complete_task | 81% |
| convert_units | 82% |
| crawl | 85% |
| date_add | 100% |
| describe_data | 75% |
| find_free_slots | 79% |
| flip_coin | 93% |
| list_capabilities | 21% |
| list_events | 35% |
| list_files | 50% |
| list_jobs | 86% |
| list_lessons | 100% |
| list_tasks | 62% |
| make_chart | 85% |
| make_poll | 98% |
| moderate_text | 100% |
| moderate_user | 83% |
| needs_model | 39% |
| now | 71% |
| pick_random | 94% |
| read_file | 85% |
| research | 67% |
| roll_dice | 79% |
| scramble_word | 72% |
| scrape | 90% |
| search_notes | 62% |
| user_moderation_status | 65% |
| web_fetch | 87% |
| web_search | 79% |

_Misses are mostly abstentions: an unfamiliar wording falls through to help rather than guessing. Fix one with `wrong => <command>` or `train add "<phrase>" => <command>`._
