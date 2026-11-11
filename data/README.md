# FarSense dataset

`FarSense.csv.zip` contains the authors' supplied `FarSense.csv` without modifying the underlying CSV.

Extract it locally:

```bash
python -m zipfile -e data/FarSense.csv.zip data/
```

The extracted file is ~168 MB, larger than GitHub's 100 MB per-file limit. The compressed archive is much smaller, and avoids committing the oversized CSV.

## Columns

- `Paragraph_Word_Count`: source paragraph word count
- `Content_Paragraph`: contextual source paragraph
- `Article_Title`: source article title
- `task`: task label
- `prompt`: instruction/question (may contain embedded JSON templates)
- `assistant`: reference answer encoded as a JSON object (the field name ends in `_predict` even though the **CSV contains references**, not model predictions)

The multiple-choice references contain both Latin and Persian answer-letter variants. The new evaluator handles these without changing the CSV.

There are 29,470 rows: 4,210 for each of seven task labels (`multiple_choice`, `true_false_1`, `true_false_2`, `cloze`, `explanation`, `counterfactual`, `cause_effect`). This count was checked against the supplied archive, not against a separately released evaluation split.

**Important:** This file has no explicit `train`/`test` split or stable ID column. Evaluation requires a known, held-out subset and predictions aligned to its rows. Never present evaluation on training examples as test-set performance.
