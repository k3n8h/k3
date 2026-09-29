# Training report

Trained on 2280 generated + 0 taught examples.

Measured on phrasings **held out from training** (every 4th template of each tool and language is never seen), so these numbers reflect generalization to new wording, not memorization.

- Familiar phrasings (new slot values): **92%** intent accuracy
- Unseen phrasings: **63%** intent, **53%** intent + arguments
- Unseen phrasings that triggered a *wrong* action: **3.7%** (the rest were either right or safely abstained with help text)

### By language

| | intent accuracy |
|---|---|
| de | 72% |
| en | 62% |
| es | 65% |
| fr | 52% |
| it | 59% |
| pt | 74% |

### By capability

| | intent accuracy |
|---|---|
| chat | 93% |
| coding | 42% |
| data | 33% |
| fun | 87% |
| generative | 57% |
| moderation | 100% |
| organize | 69% |
| research | 47% |
| schedule | 43% |
| teaching | 41% |
| utilities | 79% |

### By tool / intent

| | intent accuracy |
|---|---|
| add_event | 47% |
| add_note | 91% |
| add_task | 57% |
| calculate | 84% |
| cancel_event | 100% |
| chat | 93% |
| convert_units | 76% |
| crawl | 93% |
| describe_data | 33% |
| find_free_slots | 10% |
| flip_coin | 84% |
| list_capabilities | 21% |
| list_events | 33% |
| list_files | 17% |
| list_lessons | 100% |
| list_tasks | 67% |
| moderate_text | 100% |
| needs_model | 57% |
| now | 78% |
| read_file | 68% |
| research | 42% |
| roll_dice | 92% |
| scramble_word | 80% |
| scrape | 60% |
| search_notes | 42% |
| web_fetch | 29% |
| web_search | 48% |

_Misses are mostly abstentions: an unfamiliar wording falls through to help rather than guessing. Fix one with `wrong => <command>` or `train add "<phrase>" => <command>`._
