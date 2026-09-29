# Training report

Trained on 2928 generated + 0 taught examples.

Measured on phrasings **held out from training** (every 4th template of each tool and language is never seen), so these numbers reflect generalization to new wording, not memorization.

- Familiar phrasings (new slot values): **93%** intent accuracy
- Unseen phrasings: **65%** intent, **53%** intent + arguments
- Unseen phrasings that triggered a *wrong* action: **3.4%** (the rest were either right or safely abstained with help text)

### By language

| | intent accuracy |
|---|---|
| de | 59% |
| en | 71% |
| es | 62% |
| fr | 59% |
| it | 64% |
| pt | 68% |
| ru | 52% |

### By capability

| | intent accuracy |
|---|---|
| chat | 73% |
| coding | 44% |
| data | 45% |
| fun | 88% |
| generative | 54% |
| moderation | 93% |
| organize | 64% |
| research | 48% |
| schedule | 50% |
| teaching | 38% |
| utilities | 80% |

### By tool / intent

| | intent accuracy |
|---|---|
| add_event | 100% |
| add_note | 92% |
| add_task | 39% |
| calculate | 84% |
| cancel_event | 100% |
| chat | 73% |
| complete_task | 76% |
| convert_units | 79% |
| crawl | 93% |
| date_add | 99% |
| describe_data | 45% |
| find_free_slots | 60% |
| flip_coin | 85% |
| list_capabilities | 22% |
| list_events | 18% |
| list_files | 17% |
| list_lessons | 90% |
| list_tasks | 51% |
| moderate_text | 93% |
| needs_model | 54% |
| now | 60% |
| pick_random | 100% |
| read_file | 70% |
| research | 35% |
| roll_dice | 85% |
| scramble_word | 72% |
| scrape | 90% |
| search_notes | 53% |
| web_fetch | 26% |
| web_search | 57% |

_Misses are mostly abstentions: an unfamiliar wording falls through to help rather than guessing. Fix one with `wrong => <command>` or `train add "<phrase>" => <command>`._
