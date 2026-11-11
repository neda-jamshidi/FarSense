# FarSense data

| File | Scenarios | Items |
|---|---:|---:|
| `farsense_train.jsonl.gz` | 4,000 | 28,000 |
| `farsense_test.jsonl.gz` | 210 | 1,470 |
| `examples.jsonl` | 1 | 7 (one per task, uncompressed for browsing) |

Each scenario contributes seven items: two True/False statements, one multiple-choice
question, one cloze sentence, one cause→effect pair, one counterfactual question and one
"why" question. Train and test share no scenario or source article. The split is the one
used in the paper and is identical to the
[`Kamyar-zeinalipour/CS_Farsi`](https://huggingface.co/datasets/Kamyar-zeinalipour/CS_Farsi)
dataset on the Hugging Face Hub.

## Loading

```python
import pandas as pd

test = pd.read_json("data/farsense_test.jsonl.gz", lines=True)
```

```python
from datasets import load_dataset

ds = load_dataset(
    "json",
    data_files={"train": "data/farsense_train.jsonl.gz", "test": "data/farsense_test.jsonl.gz"},
)
```

## Fields

| Field | Description |
|---|---|
| `id` | `fs<scenario>_<task>`, e.g. `fs0001_cloze`. The seven items of a scenario share the prefix. |
| `split` | `train` or `test` |
| `task` | `true_false_1`, `true_false_2`, `multiple_choice`, `cloze`, `cause_effect`, `counterfactual`, `explanation` |
| `article_title` | Title of the Farsi Wikipedia article the scenario was derived from |
| `source_paragraph` | The Wikipedia passage (101–499 words, median 200) |
| `source_word_count` | Word count of `source_paragraph` |
| `prompt` | The full model input: task instructions (English), the scenario and the question (Farsi), and the required JSON answer format |
| `target` | Gold answer as the one-field JSON the prompt asks for, e.g. `{"multiple_choice_question_answer_predict": "B"}`. This is the string used for fine-tuning and scoring. |
| `answer_key` | Name of the field inside `target` (without `_predict`) |
| `answer` | The gold value on its own (`True`/`False`, `A`–`D`, or Farsi text) |

`true_false_1` is the statement written to be true and `true_false_2` the one written to
be false (T/F-True and T/F-False in the paper). The scenario itself appears in `prompt`
after `Context (Farsi):`.

## Label distribution

| Task | Train | Test |
|---|---|---|
| multiple_choice | A 23 · B 3,816 · C 151 · D 10 | A 1 · B 198 · C 11 |
| true_false_1 | True 3,861 · False 139 | True 205 · False 5 |
| true_false_2 | False 3,878 · True 122 | False 205 · True 5 |

## Notes

- **Persian option letters.** In 132 multiple-choice items the options are labelled
  الف/ب/ج/د instead of A–D, and the gold answer in `target` uses the Persian letter.
  `answer` maps them to A–D (الف→A, ب→B, ج→C, د→D); `target` is kept as used in the paper.
- **True/False #2 key name.** The prompt template for `true_false_2` asks the model for the
  key `true_false_1_answer_predict`, while its `target` uses `true_false_2_answer_predict`.
  Scoring reads the value regardless of the key, so this does not affect results.
- **Repeated items.** 12 training articles (in 5 groups) received identical generated
  items, so 84 training rows repeat another row's prompt. The test split has none.

## Source and license

Passages come from Farsi Wikipedia; scenarios and tasks were generated with
Gemini-2.5-Flash and validated in two passes by native speakers (Section 3 of the paper).
Because the data is derived from Wikipedia text, it is released under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). The code in this
repository is under the Apache License 2.0.
