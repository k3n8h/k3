# Training report

Trained on 2280 generated + 0 taught examples.

Measured on phrasings **held out from training** (every 4th template of each tool and language is never seen), so these numbers reflect generalization to new wording, not memorization.

- Familiar phrasings (new slot values): **91%** intent accuracy
- Unseen phrasings: **66%** intent, **55%** intent + arguments
- Unseen phrasings that triggered a *wrong* action: **4.3%** (the rest were either right or safely abstained with help text)

### By language

| | intent accuracy |
|---|---|
| de | 73% |
| en | 68% |
| es | 67% |
| fr | 52% |
| it | 58% |
| pt | 70% |

### By capability

| | intent accuracy |
|---|---|
| chat | 93% |
| coding | 56% |
| data | 53% |
| fun | 84% |
| generative | 52% |
| moderation | 100% |
| organize | 70% |
| research | 48% |
| schedule | 55% |
| teaching | 51% |
| utilities | 78% |

### By tool / intent

| | intent accuracy |
|---|---|
| add_event | 90% |
| add_note | 92% |
| add_task | 51% |
| calculate | 82% |
| cancel_event | 100% |
| chat | 93% |
| convert_units | 72% |
| crawl | 100% |
| describe_data | 53% |
| find_free_slots | 25% |
| flip_coin | 80% |
| list_capabilities | 35% |
| list_events | 38% |
| list_files | 33% |
| list_lessons | 100% |
| list_tasks | 71% |
| moderate_text | 100% |
| needs_model | 52% |
| now | 81% |
| read_file | 80% |
| research | 37% |
| roll_dice | 95% |
| scramble_word | 62% |
| scrape | 65% |
| search_notes | 65% |
| web_fetch | 28% |
| web_search | 56% |

_Misses are mostly abstentions: an unfamiliar wording falls through to help rather than guessing. Fix one with `wrong => <command>` or `train add "<phrase>" => <command>`._
