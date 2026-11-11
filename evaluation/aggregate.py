"""Compute the paper's summary scores (Section 4.2; Tables 6 and 9) from aggregate_metrics.csv.

  Classification = mean accuracy over T/F-True, T/F-False, MCQ, Cloze
  Free-form      = mean over Cause->Effect, Counterfactual, Explanation, Cloze-text of the
                   mean of (BLEU-4, ROUGE-L F1, BERTScore F1, cosine)
  Overall        = (Classification + Free-form) / 2

Usage:
    python evaluation/aggregate.py results/scores/aggregate_metrics.csv
    python evaluation/aggregate.py paper_results/finetuned_models.csv   # paper numbers
"""
import argparse

import pandas as pd

CLASSIFICATION = {
    "T/F-True": "true_false_1_answer_accuracy",
    "T/F-False": "true_false_2_answer_accuracy",
    "MCQ": "multiple_choice_answer_accuracy",
    "Cloze": "cloze_answer_accuracy",
}
FREE_FORM = {
    "Cause-Effect": "cause_effect_answer",
    "Counterfactual": "counterfactual_answer",
    "Explanation": "explanation_answer",
    "Cloze-text": "cloze_answer_text",
}
TEXT_METRICS = ["bleu4", "rougeL_f1", "bertscore_f1", "cosine"]


def summarize(agg: pd.DataFrame) -> pd.DataFrame:
    name_col = "model" if "model" in agg.columns else "file"
    out = pd.DataFrame({name_col: agg[name_col]})
    for name, col in CLASSIFICATION.items():
        out[name] = agg[col]
    for name, base in FREE_FORM.items():
        out[name] = agg[[f"{base}_{m}_mean" for m in TEXT_METRICS]].mean(axis=1)
    out["Classification"] = out[list(CLASSIFICATION)].mean(axis=1)
    out["Free-form"] = out[list(FREE_FORM)].mean(axis=1)
    out["Overall"] = (out["Classification"] + out["Free-form"]) / 2
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("metrics", nargs="?", default="results/scores/aggregate_metrics.csv")
    p.add_argument("-o", "--output", default=None, help="Optional CSV for the summary table.")
    args = p.parse_args()

    summary = summarize(pd.read_csv(args.metrics))
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(summary.round(3).to_string(index=False))
    if args.output:
        summary.to_csv(args.output, index=False)
        print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
