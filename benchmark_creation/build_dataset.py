"""Step 2 of benchmark creation: scenario + task JSON -> FarSense items (one row per task).

    python benchmark_creation/build_dataset.py generated.csv farsense_new.jsonl.gz

Reads the output of generate.py (after human review) and writes items in the same format
as data/farsense_*.jsonl.gz: each task becomes an evaluation prompt (task_templates.py)
with its gold answer as a one-field JSON `target`.
"""
import argparse
import gzip
import json
import re

import pandas as pd

from task_templates import PROMPTS

# task -> key of the gold answer inside the target JSON
GOLD_KEYS = {
    "true_false_1": "true_false_1_answer",
    "true_false_2": "true_false_2_answer",
    "multiple_choice": "multiple_choice_question_answer",
    "cloze": "cloze_answer",
    "cause_effect": "cause_effect_effect",
    "counterfactual": "counterfactual_answer",
    "explanation": "explanation_answer",
}


def load_tasks(raw):
    """Parse the model's task JSON, tolerating ```json fences."""
    if not isinstance(raw, str):
        return None
    text = re.sub(r"^```(?:json)?\s*\n", "", raw.strip())
    text = re.sub(r"\n?```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def items_for(scenario: str, data: dict):
    """Yield (task, prompt, gold_key, gold_value) for one scenario."""
    mc = data.get("multiple_choice") or {}
    options = mc.get("options")
    yield "multiple_choice", PROMPTS["multiple_choice"].format(
        context_block_if_any=scenario,
        question=mc.get("question"),
        options="; ".join(options) if isinstance(options, list) else options,
    ), mc.get("answer")

    for i, tf in enumerate(data.get("true_false", [])[:2], start=1):
        yield f"true_false_{i}", PROMPTS[f"true_false_{i}"].format(
            context_block_if_any=scenario, statement=tf.get("statement")
        ), tf.get("answer")

    cloze = data.get("cloze") or {}
    yield "cloze", PROMPTS["cloze"].format(
        context_block_if_any=scenario, sentence_with_blank=cloze.get("sentence")
    ), cloze.get("answer")

    ce = data.get("cause_effect") or {}
    yield "cause_effect", PROMPTS["cause_effect"].format(
        context_block_if_any=scenario, cause_text=ce.get("cause")
    ), ce.get("effect")

    cf = data.get("counterfactual") or {}
    yield "counterfactual", PROMPTS["counterfactual"].format(
        context_block_if_any=scenario, counterfactual_question=cf.get("question")
    ), cf.get("answer")

    ex = data.get("explanation") or {}
    yield "explanation", PROMPTS["why"].format(
        context_block_if_any=scenario, why_prompt=ex.get("prompt")
    ), ex.get("answer")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="CSV from generate.py with `scenario` and `tasks_json`.")
    p.add_argument("output", help="Output .jsonl or .jsonl.gz")
    args = p.parse_args()

    df = pd.read_csv(args.input)
    rows, skipped = [], 0
    for n, rec in enumerate(df.itertuples(index=False), start=1):
        data = load_tasks(getattr(rec, "tasks_json"))
        if data is None:
            skipped += 1
            continue
        for task, prompt, answer in items_for(rec.scenario, data):
            gold_key = GOLD_KEYS[task]
            rows.append(
                {
                    "id": f"fs{n:04d}_{task}",
                    "task": task,
                    "article_title": getattr(rec, "Article_Title", None),
                    "source_paragraph": rec.Content_Paragraph,
                    "prompt": prompt,
                    "target": json.dumps({f"{gold_key}_predict": answer}, ensure_ascii=False),
                }
            )

    opener = gzip.open if args.output.endswith(".gz") else open
    with opener(args.output, "wt", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} items to {args.output} ({skipped} passages skipped: unparseable JSON)")


if __name__ == "__main__":
    main()
