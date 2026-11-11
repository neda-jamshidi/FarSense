# -*- coding: utf-8 -*-
"""
evaluate_directory.py
=====================
Utility for scoring CSV files that contain pairs of columns in the form

    <base_name>          # gold / reference / ground-truth
    <base_name>_predict  # model prediction

Three task types are supported
------------------------------
1. **Single-label classification**
   * exact string match per row → 0/1 column `<base_name>_match`
   * macro accuracy / precision / recall / F1 appended to per-file metrics

2. **Multi-label classification** (each cell holds *sets* of labels, e.g. "A;C")
   * per-row subset accuracy, Jaccard, set sizes
   * micro / macro precision / recall / F1 + mean Jaccard in per-file metrics

3. **Text / generation metrics**
   * sentence-level BLEU-4, ROUGE-L F1, BERTScore F1, MiniLM cosine similarity

Per-row metrics are written back into a new CSV `<original>_scored.csv`.
Per-file aggregates are appended to / created in an `aggregate_metrics.csv`
inside an *output directory* (default: **reuls**).
"""

from __future__ import annotations

import argparse
import pathlib
import re
from typing import List, Optional

import numpy as np
import pandas as pd
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util as st_util
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm.auto import tqdm
from bert_score import score as bert_score
import hazm
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

# ───────────────────────────────────────────────────────────────────────────────
# Models & scorers – loaded once at import time
# ───────────────────────────────────────────────────────────────────────────────

global_hazm_tokenizer = hazm.WordTokenizer()


class HazmTokenizer:
    def tokenize(self, text):
        return global_hazm_tokenizer.tokenize(text)


tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/bert-fa-base-uncased")
model = AutoModel.from_pretrained("HooshvareLab/bert-fa-base-uncased")

def get_embedding(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze()


_SMOOTH_FN = SmoothingFunction().method4
_ROUGE = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False, tokenizer=HazmTokenizer())
_SPLIT_RE = re.compile(r"[;,/| ]+")  # delimiters for multi-label sets

# ───────────────────────────────────────────────────────────────────────────────
# Helper functions
# ───────────────────────────────────────────────────────────────────────────────


def _clean_label(x: str):
    """Normalise single-label entries: trim, case-fold, unify booleans."""
    if pd.isna(x):  # catches NaN and None
        return ""
    x = str(x).strip()
    lx = x.lower()
    if lx in {"true", "yes", "1"}:
        return "true"
    if lx in {"false", "no", "0"}:
        return "false"
    return lx  # letters / words → lowercase exact


def _to_label_set(cell: str) -> set[str]:
    """Split a cell into a *set* of labels for multi-label tasks."""
    if pd.isna(cell):
        return set()
    return {tok.strip() for tok in _SPLIT_RE.split(str(cell)) if tok.strip()}


# ───────────────────────────────────────────────────────────────────────────────
# Text-metric helpers (BLEU, ROUGE-L, cosine, BERTScore)
# ───────────────────────────────────────────────────────────────────────────────


def _row_bleu(ref: str, hyp: str) -> float:
    return sentence_bleu(
        [ref.split()], hyp.split(), smoothing_function=_SMOOTH_FN, weights=(0.25, 0.25, 0.25, 0.25)
    )


def _row_rouge_l_f1(ref: str, hyp: str) -> float:
    return _ROUGE.score(ref, hyp)["rougeL"].fmeasure


def _row_cosine(ref: str, hyp: str) -> float:
    ref_emb = get_embedding(ref)
    hyp_emb = get_embedding(hyp)
    return float(F.cosine_similarity(ref_emb, hyp_emb, dim=0))


def _row_bert_f1(ref: str, hyp: str) -> float:
    _, _, f1_arr = bert_score(
        [hyp], [ref], model_type="microsoft/deberta-xlarge-mnli", lang="en", rescale_with_baseline=True
    )
    return float(f1_arr[0])


def _text_metrics(ref: str, hyp: str):
    bleu = _row_bleu(ref, hyp)
    rouge = _row_rouge_l_f1(ref, hyp)
    cosine = _row_cosine(ref, hyp)
    bert_f1 = _row_bert_f1(ref, hyp)
    return bleu, rouge, bert_f1, cosine


# ───────────────────────────────────────────────────────────────────────────────
# Core evaluation of a single CSV file
# ───────────────────────────────────────────────────────────────────────────────


def evaluate_file(
    csv_path: pathlib.Path,
    *,
    classification_cols: List[str],
    multilabel_cols: List[str],
    text_cols: List[str],
    scored_suffix: str = "_scored",
    output_dir: Optional[pathlib.Path] = None,
):
    """
    Read *csv_path*, add metric columns, write <stem>_scored.csv into *output_dir*
    (or alongside the input if *output_dir* is None), return per-file metric dict.
    """

    df = pd.read_csv(csv_path, dtype=str)  # read everything as string
    per_file = {"file": csv_path.name}

    # ─── Single-label classification ────────────────────────────────────────
    for base in classification_cols:
        pred = f"{base}_predict"
        if {base, pred}.issubset(df.columns):
            y_true = df[base].map(_clean_label)
            y_pred = df[pred].map(_clean_label)

            match_col = f"{base}_match"
            df[match_col] = (y_true == y_pred).astype(int)

            acc = accuracy_score(y_true, y_pred)
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average="macro", zero_division=0
            )
            per_file.update(
                {
                    f"{base}_accuracy": acc,
                    f"{base}_precision": prec,
                    f"{base}_recall": rec,
                    f"{base}_f1": f1,
                }
            )
        else:
            print(f"[WARN] {csv_path.name}: missing '{base}' / '{pred}'")

    # ─── Multi-label classification ─────────────────────────────────────────
    for base in multilabel_cols:
        pred = f"{base}_predict"
        if {base, pred}.issubset(df.columns):
            y_true_sets = df[base].map(_to_label_set)
            y_pred_sets = df[pred].map(_to_label_set)

            subset_match = (y_true_sets == y_pred_sets).astype(int)
            df[f"{base}_subset_match"] = subset_match

            jaccards = [
                len(t & p) / len(t | p) if (t | p) else 1.0 for t, p in zip(y_true_sets, y_pred_sets)
            ]
            df[f"{base}_jaccard"] = jaccards

            # diagnostic counts (optional)
            df[f"{base}_num_true"] = y_true_sets.map(len)
            df[f"{base}_num_pred"] = y_pred_sets.map(len)
            df[f"{base}_num_overlap"] = [len(t & p) for t, p in zip(y_true_sets, y_pred_sets)]

            # binarise for global metrics
            mlb = MultiLabelBinarizer()
            y_true_bin = mlb.fit_transform(y_true_sets)
            y_pred_bin = mlb.transform(y_pred_sets)

            micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
                y_true_bin, y_pred_bin, average="micro", zero_division=0
            )
            macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
                y_true_bin, y_pred_bin, average="macro", zero_division=0
            )

            per_file.update(
                {
                    f"{base}_subset_accuracy_mean": subset_match.mean(),
                    f"{base}_micro_precision": micro_p,
                    f"{base}_micro_recall": micro_r,
                    f"{base}_micro_f1": micro_f1,
                    f"{base}_macro_precision": macro_p,
                    f"{base}_macro_recall": macro_r,
                    f"{base}_macro_f1": macro_f1,
                    f"{base}_jaccard_mean": float(np.mean(jaccards)),
                }
            )
        else:
            print(f"[WARN] {csv_path.name}: missing '{base}' / '{pred}'")

    # ─── Text / generation metrics ──────────────────────────────────────────
    for base in text_cols:
        pred = f"{base}_predict"
        if {base, pred}.issubset(df.columns):
            bleu_vals, rouge_vals, bert_vals, cos_vals = [], [], [], []
            for ref, hyp in tqdm(
                zip(df[base].fillna(""), df[pred].fillna("")),
                total=len(df),
                desc=f"Scoring {csv_path.name}:{base}",
                leave=False,
            ):
                bleu, rouge, bert_f1, cosine = _text_metrics(ref, hyp)
                bleu_vals.append(bleu)
                rouge_vals.append(rouge)
                bert_vals.append(bert_f1)
                cos_vals.append(cosine)

            df[f"{base}_bleu4"] = bleu_vals
            df[f"{base}_rougeL_f1"] = rouge_vals
            df[f"{base}_bertscore_f1"] = bert_vals
            df[f"{base}_cosine"] = cos_vals

            per_file.update(
                {
                    f"{base}_bleu4_mean": float(np.mean(bleu_vals)),
                    f"{base}_rougeL_f1_mean": float(np.mean(rouge_vals)),
                    f"{base}_bertscore_f1_mean": float(np.mean(bert_vals)),
                    f"{base}_cosine_mean": float(np.mean(cos_vals)),
                }
            )
        else:
            print(f"[WARN] {csv_path.name}: missing '{base}' / '{pred}'")

    # ─── write scored CSV ───────────────────────────────────────────────────
    if output_dir is None:
        scored_path = csv_path.with_name(f"{csv_path.stem}{scored_suffix}{csv_path.suffix}")
    else:
        scored_path = output_dir / f"{csv_path.stem}{scored_suffix}{csv_path.suffix}"

    df.to_csv(scored_path, index=False)
    print(f"✔ Saved {scored_path.relative_to(scored_path.parent.parent) if output_dir else scored_path.name}")

    return per_file


# ───────────────────────────────────────────────────────────────────────────────
# Directory-level driver
# ───────────────────────────────────────────────────────────────────────────────


def evaluate_directory(
    dir_path: str | pathlib.Path,
    *,
    classification_cols: Optional[List[str]] = None,
    multilabel_cols: Optional[List[str]] = None,
    text_cols: Optional[List[str]] = None,
    aggregate_name: str = "aggregate_metrics.csv",
    output_dir: str | pathlib.Path = "reuls",
):
    classification_cols = classification_cols or []
    multilabel_cols = multilabel_cols or []
    text_cols = text_cols or []

    dir_path = pathlib.Path(dir_path)
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    for csv_path in dir_path.glob("*.csv"):
        if csv_path.name.endswith("_scored.csv") or csv_path.name == aggregate_name:
            continue  # skip already scored & summary file
        rows.append(
            evaluate_file(
                csv_path,
                classification_cols=classification_cols,
                multilabel_cols=multilabel_cols,
                text_cols=text_cols,
                output_dir=output_dir,
            )
        )

    if rows:
        agg_df_new = pd.DataFrame(rows)
        agg_path = output_dir / aggregate_name
        if agg_path.exists():
            agg_df_old = pd.read_csv(agg_path)
            agg_df_new = pd.concat([agg_df_old, agg_df_new], ignore_index=True)
        agg_df_new.to_csv(agg_path, index=False)
        print(f"★ Wrote summary → {agg_path.relative_to(agg_path.parent.parent)}")
    else:
        print("No unscored CSV files found.")


# ───────────────────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────────────────


def _parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate CSV files that contain <col>/<col>_predict pairs."
    )
    p.add_argument("directory", help="Folder containing CSV files.")
    p.add_argument(
        "-o",
        "--output",
        default="results",
        help="Folder that will receive *_scored.csv and aggregate_metrics.csv (default: reuls).",
    )
    p.add_argument(
        "--classification",
        nargs="*",
        default=[],
        help="Base names of *single-label* classification columns.",
    )
    p.add_argument(
        "--multilabel",
        nargs="*",
        default=[],
        help="Base names of *multi-label* columns (labels separated by ; , / | space).",
    )
    p.add_argument(
        "--text",
        nargs="*",
        default=[],
        help="Base names of *text/generation* columns.",
    )
    return p.parse_args()


def main():
    args = _parse_args()
    evaluate_directory(
        args.directory,
        classification_cols=args.classification,
        multilabel_cols=args.multilabel,
        text_cols=args.text,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
