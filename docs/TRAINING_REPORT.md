# Training report

Trained on 5026 generated + 0 taught examples.

Measured on phrasings **held out from training** (every 4th template of each tool and language is never seen), so these numbers reflect generalization to new wording, not memorization.

- Familiar phrasings (new slot values): **95%** intent accuracy
- Unseen phrasings: **74%** intent, **63%** intent + arguments
- Unseen phrasings that triggered a *wrong* action: **5.2%** (the rest were either right or safely abstained with help text)

### By language

| | intent accuracy |
|---|---|
| de | 76% |
| en | 81% |
| es | 77% |
| fr | 59% |
| it | 77% |
| pt | 74% |
| ru | 58% |

### By capability

| | intent accuracy |
|---|---|
| chat | 75% |
| coding | 69% |
| data | 90% |
| fun | 88% |
| generative | 41% |
| moderation | 78% |
| organize | 69% |
| research | 68% |
| schedule | 69% |
| teaching | 52% |
| utilities | 88% |

### By tool / intent

| | intent accuracy |
|---|---|
| add_blocked_word | 78% |
| add_event | 90% |
| add_note | 79% |
| add_task | 57% |
| calculate | 96% |
| cancel_event | 100% |
| certify_roles | 100% |
| chat | 75% |
| complete_task | 76% |
| convert_units | 81% |
| crawl | 85% |
| date_add | 97% |
| describe_data | 80% |
| find_free_slots | 82% |
| flip_coin | 89% |
| list_capabilities | 24% |
| list_events | 32% |
| list_files | 55% |
| list_jobs | 78% |
| list_lessons | 100% |
| list_tasks | 65% |
| make_chart | 100% |
| make_poll | 99% |
| moderate_text | 100% |
| moderate_user | 90% |
| needs_model | 41% |
| now | 67% |
| pick_random | 94% |
| read_file | 82% |
| research | 49% |
| roll_dice | 85% |
| scramble_word | 72% |
| scrape | 90% |
| search_notes | 57% |
| user_moderation_status | 59% |
| web_fetch | 84% |
| web_search | 61% |

_Misses are mostly abstentions: an unfamiliar wording falls through to help rather than guessing. Fix one with `wrong => <command>` or `train add "<phrase>" => <command>`._
