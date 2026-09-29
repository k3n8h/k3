# Training report

Trained on 2640 generated + 0 taught examples.

Measured on phrasings **held out from training** (every 4th template of each tool and language is never seen), so these numbers reflect generalization to new wording, not memorization.

- Familiar phrasings (new slot values): **90%** intent accuracy
- Unseen phrasings: **62%** intent, **53%** intent + arguments
- Unseen phrasings that triggered a *wrong* action: **5.2%** (the rest were either right or safely abstained with help text)

### By language

| | intent accuracy |
|---|---|
| de | 66% |
| en | 69% |
| es | 60% |
| fr | 51% |
| it | 59% |
| pt | 66% |
| ru | 41% |

### By capability

| | intent accuracy |
|---|---|
| chat | 88% |
| coding | 53% |
| data | 50% |
| fun | 81% |
| generative | 55% |
| moderation | 93% |
| organize | 63% |
| research | 47% |
| schedule | 46% |
| teaching | 40% |
| utilities | 77% |

### By tool / intent

| | intent accuracy |
|---|---|
| add_event | 78% |
| add_note | 83% |
| add_task | 39% |
| calculate | 85% |
| cancel_event | 100% |
| chat | 88% |
| complete_task | 95% |
| convert_units | 79% |
| crawl | 93% |
| date_add | 47% |
| describe_data | 50% |
| find_free_slots | 28% |
| flip_coin | 81% |
| list_capabilities | 25% |
| list_events | 26% |
| list_files | 15% |
| list_lessons | 90% |
| list_tasks | 58% |
| moderate_text | 93% |
| needs_model | 55% |
| now | 75% |
| pick_random | 95% |
| read_file | 90% |
| research | 32% |
| roll_dice | 80% |
| scramble_word | 72% |
| scrape | 80% |
| search_notes | 57% |
| web_fetch | 27% |
| web_search | 60% |

_Misses are mostly abstentions: an unfamiliar wording falls through to help rather than guessing. Fix one with `wrong => <command>` or `train add "<phrase>" => <command>`._
