# -*- coding: utf-8 -*-
"""Score CSV files that hold `<base>` (gold) / `<base>_predict` (model) column pairs.

Run on the output of evaluation/prepare_for_eval.py:
    python evaluation/evaluate.py results/ready -o results/scores

Without column flags the FarSense columns are used:
  * classification (accuracy, macro P/R/F1): both True/False items, multiple choice, cloze
  * text (BLEU-4, ROUGE-L F1 with the hazm tokenizer, BERTScore F1, cosine similarity of
    mean-pooled HooshvareLab/bert-fa-base-uncased embeddings): cause->effect,
    counterfactual, explanation, cloze

For every input file this writes `<name>_scored.csv` (per-row metrics) and appends one row
to `aggregate_metrics.csv` in the output folder.
"""
from __future__ import annotations

import argparse
import pathlib
import re
from functools import lru_cache
from typing import List, Optional

import numpy as np
import pandas as pd
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm.auto import tqdm

FARSENSE_CLASSIFICATION = [
    "cloze_answer",
    "true_false_1_answer",
    "true_false_2_answer",
    "multiple_choice_answer",
]
FARSENSE_TEXT = [
    "cause_effect_answer",
    "counterfactual_answer",
    "explanation_answer",
    "cloze_answer_text",
]

EMBEDDING_MODEL = "HooshvareLab/bert-fa-base-uncased"
BERTSCORE_MODEL = "microsoft/deberta-xlarge-mnli"

_SMOOTH_FN = SmoothingFunction().method4
_SPLIT_RE = re.compile(r"[;,/| ]+")


@lru_cache(maxsize=1)
def _hazm_tokenizer():
    import hazm

    return hazm.WordTokenizer()


class HazmTokenizer:
    def tokenize(self, text):
        return _hazm_tokenizer().tokenize(text)


@lru_cache(maxsize=1)
def _rouge():
    return rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False, tokenizer=HazmTokenizer())


@lru_cache(maxsize=1)
def _embedder():
    from transformers import AutoModel, AutoTokenizer

    return AutoTokenizer.from_pretrained(EMBEDDING_MODEL), AutoModel.from_pretrained(EMBEDDING_MODEL).eval()


def _clean_label(x):
    """Normalise single-label entries: trim, case-fold, unify booleans."""
    if pd.isna(x):
        return ""
    x = str(x).strip()
    lx = x.lower()
    if lx in {"true", "yes", "1"}:
        return "true"
    if lx in {"false", "no", "0"}:
        return "false"
    return lx


def _to_label_set(cell) -> set[str]:
    if pd.isna(cell):
        return set()
    return {tok.strip() for tok in _SPLIT_RE.split(str(cell)) if tok.strip()}


def _row_bleu(ref: str, hyp: str) -> float:
    return sentence_bleu(
        [ref.split()], hyp.split(), smoothing_function=_SMOOTH_FN, weights=(0.25, 0.25, 0.25, 0.25)
    )


def _row_rouge_l_f1(ref: str, hyp: str) -> float:
    return _rouge().score(ref, hyp)["rougeL"].fmeasure


def _row_cosine(ref: str, hyp: str) -> float:
    import torch
    import torch.nn.functional as F

    tokenizer, model = _embedder()

    def embed(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            return model(**inputs).last_hidden_state.mean(dim=1).squeeze()

    return float(F.cosine_similarity(embed(ref), embed(hyp), dim=0))


def _bertscore_f1(refs: List[str], hyps: List[str]) -> List[float]:
    # Settings as in the paper (footnote 3). BERTScore is pairwise, so batching gives the
    # same per-row values as scoring one pair at a time.
    from bert_score import score as bert_score

    _, _, f1 = bert_score(hyps, refs, model_type=BERTSCORE_MODEL, lang="en", rescale_with_baseline=True)
    return [float(x) for x in f1]


def evaluate_file(
    csv_path: pathlib.Path,
    *,
    classification_cols: List[str],
    multilabel_cols: List[str],
    text_cols: List[str],
    output_dir: pathlib.Path,
    scored_suffix: str = "_scored",
):
    df = pd.read_csv(csv_path, dtype=str)
    per_file = {"file": csv_path.name}

    for base in classification_cols:
        pred = f"{base}_predict"
        if not {base, pred}.issubset(df.columns):
            print(f"[WARN] {csv_path.name}: missing '{base}' / '{pred}'")
            continue
        rows = df[base].notna()
        y_true = df.loc[rows, base].map(_clean_label)
        y_pred = df.loc[rows, pred].map(_clean_label)
        df.loc[rows, f"{base}_match"] = (y_true == y_pred).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
        per_file.update(
            {
                f"{base}_n": int(rows.sum()),
                f"{base}_accuracy": accuracy_score(y_true, y_pred),
                f"{base}_precision": prec,
                f"{base}_recall": rec,
                f"{base}_f1": f1,
            }
        )

    for base in multilabel_cols:
        pred = f"{base}_predict"
        if not {base, pred}.issubset(df.columns):
            print(f"[WARN] {csv_path.name}: missing '{base}' / '{pred}'")
            continue
        rows = df[base].notna()
        y_true_sets = df.loc[rows, base].map(_to_label_set)
        y_pred_sets = df.loc[rows, pred].map(_to_label_set)
        subset_match = (y_true_sets == y_pred_sets).astype(int)
        jaccards = [len(t & p) / len(t | p) if (t | p) else 1.0 for t, p in zip(y_true_sets, y_pred_sets)]
        df.loc[rows, f"{base}_subset_match"] = subset_match
        df.loc[rows, f"{base}_jaccard"] = jaccards

        mlb = MultiLabelBinarizer()
        y_true_bin = mlb.fit_transform(y_true_sets)
        y_pred_bin = mlb.transform(y_pred_sets)
        micro = precision_recall_fscore_support(y_true_bin, y_pred_bin, average="micro", zero_division=0)
        macro = precision_recall_fscore_support(y_true_bin, y_pred_bin, average="macro", zero_division=0)
        per_file.update(
            {
                f"{base}_subset_accuracy_mean": subset_match.mean(),
                f"{base}_micro_precision": micro[0],
                f"{base}_micro_recall": micro[1],
                f"{base}_micro_f1": micro[2],
                f"{base}_macro_precision": macro[0],
                f"{base}_macro_recall": macro[1],
                f"{base}_macro_f1": macro[2],
                f"{base}_jaccard_mean": float(np.mean(jaccards)),
            }
        )

    for base in text_cols:
        pred = f"{base}_predict"
        if not {base, pred}.issubset(df.columns):
            print(f"[WARN] {csv_path.name}: missing '{base}' / '{pred}'")
            continue
        rows = df[base].notna()
        refs = df.loc[rows, base].tolist()
        hyps = df.loc[rows, pred].fillna("").tolist()
        bleu, rouge, cosine = [], [], []
        for ref, hyp in tqdm(zip(refs, hyps), total=len(refs), desc=f"{csv_path.name}:{base}", leave=False):
            bleu.append(_row_bleu(ref, hyp))
            rouge.append(_row_rouge_l_f1(ref, hyp))
            cosine.append(_row_cosine(ref, hyp))
        bert = _bertscore_f1(refs, hyps)

        df.loc[rows, f"{base}_bleu4"] = bleu
        df.loc[rows, f"{base}_rougeL_f1"] = rouge
        df.loc[rows, f"{base}_bertscore_f1"] = bert
        df.loc[rows, f"{base}_cosine"] = cosine
        per_file.update(
            {
                f"{base}_bleu4_mean": float(np.mean(bleu)),
                f"{base}_rougeL_f1_mean": float(np.mean(rouge)),
                f"{base}_bertscore_f1_mean": float(np.mean(bert)),
                f"{base}_cosine_mean": float(np.mean(cosine)),
            }
        )

    scored_path = output_dir / f"{csv_path.stem}{scored_suffix}{csv_path.suffix}"
    df.to_csv(scored_path, index=False)
    print(f"Saved {scored_path}")
    return per_file


def evaluate_directory(
    dir_path,
    *,
    classification_cols: Optional[List[str]] = None,
    multilabel_cols: Optional[List[str]] = None,
    text_cols: Optional[List[str]] = None,
    aggregate_name: str = "aggregate_metrics.csv",
    output_dir="results/scores",
):
    dir_path = pathlib.Path(dir_path)
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for csv_path in sorted(dir_path.glob("*.csv")):
        if csv_path.name.endswith("_scored.csv") or csv_path.name == aggregate_name:
            continue
        rows.append(
            evaluate_file(
                csv_path,
                classification_cols=classification_cols or [],
                multilabel_cols=multilabel_cols or [],
                text_cols=text_cols or [],
                output_dir=output_dir,
            )
        )

    if not rows:
        print("No unscored CSV files found.")
        return
    agg = pd.DataFrame(rows)
    agg_path = output_dir / aggregate_name
    if agg_path.exists():
        agg = pd.concat([pd.read_csv(agg_path), agg], ignore_index=True)
        agg = agg.drop_duplicates(subset="file", keep="last")
    agg.to_csv(agg_path, index=False)
    print(f"Wrote summary to {agg_path}")


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("directory", help="Folder containing CSV files.")
    p.add_argument("-o", "--output", default="results/scores", help="Output folder (default: results/scores).")
    p.add_argument("--classification", nargs="*", default=None, help="Single-label column base names.")
    p.add_argument("--multilabel", nargs="*", default=None, help="Multi-label column base names.")
    p.add_argument("--text", nargs="*", default=None, help="Free-text column base names.")
    return p.parse_args()


def main():
    args = _parse_args()
    no_flags = args.classification is None and args.multilabel is None and args.text is None
    evaluate_directory(
        args.directory,
        classification_cols=FARSENSE_CLASSIFICATION if no_flags else args.classification,
        multilabel_cols=args.multilabel,
        text_cols=FARSENSE_TEXT if no_flags else args.text,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
