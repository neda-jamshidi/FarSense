#!/usr/bin/env python3
"""Validate FarSense CSV references and score aligned model predictions.

No external Python libraries required. Classification: accuracy and macro
precision/recall/F1. Generation: exact match and whitespace-token F1.
These are simple diagnostics, NOT the original publication's full metrics.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import re
import sys

TASK_KEYS = {
    "multiple_choice": "multiple_choice_question_answer_predict",
    "true_false_1": "true_false_1_answer_predict",
    "true_false_2": "true_false_2_answer_predict",
    "cloze": "cloze_answer_predict",
    "explanation": "explanation_answer_predict",
    "counterfactual": "counterfactual_answer_predict",
    "cause_effect": "cause_effect_effect_predict",
}
CLASS_LABELS = {
    "multiple_choice": ("A", "B", "C", "D"),
    "true_false_1": ("true", "false"),
    "true_false_2": ("true", "false"),
}
PRED_COLUMNS = ("generated_text", "prediction", "response", "output")
DECODER = json.JSONDecoder()


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"Missing CSV header in {path}")
        return list(reader)


def json_objects(text):
    # Last matching JSON object is used if the model includes commentary.
    for match in re.finditer(r"\{", text):
        try:
            obj, _ = DECODER.raw_decode(text[match.start():])
        except ValueError:
            continue
        if isinstance(obj, dict):
            yield obj


def extract_answer(raw, task, reference=False):
    if raw is None or not str(raw).strip():
        return None
    text = str(raw).strip()
    keys = [TASK_KEYS[task]]
    if not reference and task == "true_false_2":
        # The supplied true_false_2 prompt incorrectly instructs this key.
        keys.append("true_false_1_answer_predict")
    objects = list(json_objects(text))
    matching = [obj[key] for obj in objects for key in keys if key in obj]
    if matching:
        return matching[-1]
    if reference or objects:
        return None
    # Bare outputs also supported; JSON object with a wrong key is not.
    return re.sub(r"^```(?:json|text)?\s*|\s*```$", "", text, flags=re.I).strip() or None


def normalise(value, task):
    if value is None:
        return None
    if task.startswith("true_false"):
        if isinstance(value, bool):
            return "true" if value else "false"
        return {"true":"true", "false":"false", "yes":"true", "no":"false",
                "1":"true", "0":"false"}.get(str(value).strip().casefold())
    if task == "multiple_choice":
        ans = str(value).strip().upper()
        # The released gold CSV mixes Latin A/B/C/D and Persian option letters.
        # Do not edit the underlying references; normalise only at scoring time.
        ans = {"الف":"A", "ب":"B", "ج":"C", "د":"D"}.get(ans, ans)
        return ans if ans in CLASS_LABELS[task] else None
    if not isinstance(value, str):
        return None
    return " ".join(value.split()) or None


def token_f1(ref, pred):
    r, p = ref.split(), pred.split()
    overlap = sum((Counter(r) & Counter(p)).values())
    return 2 * overlap / (len(r) + len(p)) if r or p else 1.0


def macro_prf(scored, task):
    ps, rs, f1s = [], [], []
    observed = {row["gold"] for row in scored} | {row["prediction"] for row in scored}
    labels = [label for label in CLASS_LABELS[task] if label in observed]
    for label in labels:
        tp = sum(r["gold"] == label and r["prediction"] == label for r in scored)
        fp = sum(r["gold"] != label and r["prediction"] == label for r in scored)
        fn = sum(r["gold"] == label and r["prediction"] != label for r in scored)
        p = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * recall / (p + recall) if p + recall else 0.0
        ps.append(p); rs.append(recall); f1s.append(f1)
    return (sum(ps)/len(ps), sum(rs)/len(rs), sum(f1s)/len(f1s))


def validate(gold):
    errors = []
    task_counts = Counter()
    for i, row in enumerate(gold):
        task = row.get("task", "")
        task_counts[task] += 1
        if task not in TASK_KEYS:
            errors.append((i, "unknown task: " + task))
        elif normalise(extract_answer(row.get("assistant", ""), task, reference=True), task) is None:
            errors.append((i, "invalid assistant reference for task " + task))
    return task_counts, errors


def aligned_predictions(path, column, n):
    rows = read_csv(path)
    if not rows:
        raise ValueError("Prediction CSV is empty")
    columns = rows[0].keys()
    column = column or next((c for c in PRED_COLUMNS if c in columns), None)
    if column not in columns:
        raise ValueError("Prediction column missing; use --pred-column. Available: " + ", ".join(columns))
    has_ids = "row_index" in columns
    if not has_ids and len(rows) != n:
        raise ValueError(f"Predictions have {len(rows)} rows, gold has {n}. "
                         "Use a row_index column (zero-based position in gold CSV) for subsets.")
    if column == "output":
        print("WARNING: The original inference 'output' contains the prompt as well as the answer. "
              "Generated-only predictions are recommended.", file=sys.stderr)
    output = {}
    for i, row in enumerate(rows):
        try:
            idx = int(row["row_index"]) if has_ids else i
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid row_index on line {i+2}") from e
        if idx < 0 or idx >= n or idx in output:
            raise ValueError(f"Duplicate or out-of-range row_index: {idx}")
        response = row.get(column, "") or ""
        if column == "output" and row.get("prompt") and row["prompt"] in response:
            response = response.rsplit(row["prompt"], 1)[-1]
        output[idx] = response
    return output, column


def evaluate(gold, predicted):
    by_task = defaultdict(list)
    all_rows = []
    for i, raw in sorted(predicted.items()):
        task = gold[i]["task"]
        ref = normalise(extract_answer(gold[i]["assistant"], task, reference=True), task)
        pred = normalise(extract_answer(raw, task), task)
        row = {"row_index":i, "task":task, "gold":ref, "prediction":pred,
               "parsed":int(pred is not None), "exact_match":int(pred is not None and pred == ref),
               "token_f1":(token_f1(ref, pred) if pred is not None else 0.0) if task not in CLASS_LABELS else ""}
        by_task[task].append(row)
        all_rows.append(row)
    summary = []
    for task, subset in sorted(by_task.items()):
        n = len(subset)
        item = {"task":task, "n":n, "parse_rate":sum(r["parsed"] for r in subset)/n,
                "exact_match_accuracy":sum(r["exact_match"] for r in subset)/n}
        if task in CLASS_LABELS:
            p, r, f1 = macro_prf(subset, task)
            item.update(macro_precision=p, macro_recall=r, macro_f1=f1)
        else:
            item["mean_whitespace_token_f1"] = sum(r["token_f1"] for r in subset)/n
        summary.append(item)
    return all_rows, summary


def write_csv(path, data, fields):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(data)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gold", required=True, type=Path, help="FarSense.csv, or a held-out subset with the same columns")
    ap.add_argument("--validate", action="store_true", help="Validate the reference CSV schema")
    ap.add_argument("--predictions", type=Path, help="CSV of model responses, in the same order as gold or with row_index")
    ap.add_argument("--pred-column", help="Predictions column (default: generated_text/prediction/response/output)")
    ap.add_argument("--output-dir", type=Path, default=Path("results"))
    args = ap.parse_args()
    if not args.validate and not args.predictions:
        ap.error("Provide --validate and/or --predictions")
    gold = read_csv(args.gold)
    if not gold or not {"task", "assistant"}.issubset(gold[0].keys()):
        ap.error("Gold CSV must have task and assistant columns")
    counts, errors = validate(gold)
    print(f"Gold records: {len(gold)}")
    for task, n in sorted(counts.items()):
        print(f"  {task}: {n}")
    print(f"Invalid gold answers: {len(errors)}")
    if errors:
        print("First errors:", errors[:5], file=sys.stderr)
        sys.exit(1)
    if args.predictions:
        predictions, column = aligned_predictions(args.predictions, args.pred_column, len(gold))
        scored, summary = evaluate(gold, predictions)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_csv(args.output_dir/'scored_predictions.csv', scored,
                  ['row_index','task','gold','prediction','parsed','exact_match','token_f1'])
        write_csv(args.output_dir/'metrics_by_task.csv', summary,
                  ['task','n','parse_rate','exact_match_accuracy','macro_precision','macro_recall','macro_f1','mean_whitespace_token_f1'])
        (args.output_dir/'metrics_by_task.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f"Evaluated {len(scored)} predictions from column '{column}' and wrote outputs to {args.output_dir}")
        for s in summary:
            print(f"  {s['task']}: n={s['n']}, exact-match={s['exact_match_accuracy']:.3f}, parsed={s['parse_rate']:.3f}")
        print("Caution: diagnostic metrics only; not verified publication scores.")


if __name__ == '__main__':
    main()
