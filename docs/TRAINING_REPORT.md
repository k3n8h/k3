# Training report

Trained on 3283 generated + 0 taught examples.

Measured on phrasings **held out from training** (every 4th template of each tool and language is never seen), so these numbers reflect generalization to new wording, not memorization.

- Familiar phrasings (new slot values): **94%** intent accuracy
- Unseen phrasings: **68%** intent, **57%** intent + arguments
- Unseen phrasings that triggered a *wrong* action: **6.2%** (the rest were either right or safely abstained with help text)

### By language

| | intent accuracy |
|---|---|
| de | 62% |
| en | 75% |
| es | 65% |
| fr | 59% |
| it | 62% |
| pt | 73% |
| ru | 52% |

### By capability

| | intent accuracy |
|---|---|
| chat | 63% |
| coding | 57% |
| data | 72% |
| fun | 91% |
| generative | 43% |
| moderation | 68% |
| organize | 64% |
| research | 49% |
| schedule | 52% |
| teaching | 57% |
| utilities | 90% |

### By tool / intent

| | intent accuracy |
|---|---|
| add_blocked_word | 50% |
| add_event | 97% |
| add_note | 86% |
| add_task | 43% |
| calculate | 99% |
| cancel_event | 100% |
| certify_roles | 100% |
| chat | 63% |
| complete_task | 62% |
| convert_units | 86% |
| crawl | 90% |
| date_add | 100% |
| describe_data | 45% |
| find_free_slots | 57% |
| flip_coin | 90% |
| list_capabilities | 34% |
| list_events | 21% |
| list_files | 33% |
| list_jobs | 60% |
| list_lessons | 93% |
| list_tasks | 61% |
| make_chart | 100% |
| make_poll | 100% |
| moderate_text | 100% |
| moderate_user | 72% |
| needs_model | 43% |
| now | 74% |
| pick_random | 100% |
| read_file | 82% |
| research | 39% |
| roll_dice | 85% |
| scramble_word | 88% |
| scrape | 78% |
| search_notes | 72% |
| user_moderation_status | 47% |
| web_fetch | 32% |
| web_search | 57% |

_Misses are mostly abstentions: an unfamiliar wording falls through to help rather than guessing. Fix one with `wrong => <command>` or `train add "<phrase>" => <command>`._
