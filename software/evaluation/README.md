# FarSense evaluation

## Recommended: lightweight evaluator

Use `evaluate.py` with Python 3.9+ and no extra packages:

```bash
python software/evaluation/evaluate.py --gold data/FarSense.csv --validate
python software/evaluation/evaluate.py --gold data/FarSense.csv --predictions my_predictions.csv --pred-column prediction --output-dir results
```

The script reads each reference from the supplied CSV's `assistant` JSON column using the exact key for its `task`. For model predictions, it accepts the same JSON format or bare strings/labels. It also accepts `true_false_1_answer_predict` in a **true_false_2 model answer**, because the supplied true_false_2 prompts contain a mistaken JSON key example. Gold references must use their correct key. Multiple-choice values are normalized across Latin letters `A/B/C/D` and Persian equivalents `الف/ب/ج/د`; both forms are present in the supplied dataset.

Prediction CSV requirements:

- A generated-only text column (`generated_text`, `prediction`, or `response` by default; `--pred-column` overrides).
- Either exactly one prediction per gold CSV row, preserving **identical order**, or a `row_index` column denoting zero-based positions in the chosen gold CSV. Do not match solely on title or task.
- For the uploaded 29,470-row dataset, **no original held-out split is provided**; you must establish the correct test subset yourself. Model inference using a different Hugging Face test split cannot be compared to the full CSV by assumed row position.

The new script computes task-group accuracy, macro precision/recall/F1 for classification, plus exact match and whitespace-token F1 for free text. These are practical sanity checks, *not* the source publication's exact evaluation protocol. Whitespace-token F1 is not morphologically aware for Persian.

## Historic `eval3_original.py`

The **unchanged original** scoring script is preserved for provenance. It expects a directory of CSV files with paired columns like `answer` and `answer_predict` and requires the user to choose task types with CLI flags. For example, if a file contains columns named `label`, `label_predict`, run:

```bash
pip install -r requirements-legacy-evaluation.txt
python software/evaluation/eval3_original.py path/to/paired_csv_dir/ --classification label --output results_legacy
```

For text scoring, it calculates BLEU-4, ROUGE-L, BERTScore and a model-embedding cosine similarity (using network-downloaded weights); for multi-label scoring, macro/micro F1 and Jaccard. **It does not parse the released CSV's `assistant` JSON by itself.** This script also loads pretrained transformers at import and calls an **English** DeBERTa checkpoint for BERTScore; carefully validate applicability to Persian and paper-method consistency before making a reproduction claim. The historic script was not executed end-to-end here.

The original source also includes older naming and CLI defaults; they were not altered to avoid accidentally changing experimental behavior.
