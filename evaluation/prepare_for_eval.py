"""Turn generation files into the column pairs that evaluation/evaluate.py scores.

    python evaluation/prepare_for_eval.py results/raw results/ready

For every CSV written by inference/generate.py or inference/generate_api.py this:
  1. takes the model reply (`generation`; older files: `output` minus prompt and special tokens),
  2. reshapes the file to one row per scenario (seven tasks side by side),
  3. pulls the answer value out of the gold JSON (`target`) and the model's JSON reply,
  4. writes `<task>_answer` / `<task>_answer_predict` columns, plus `cloze_answer_text`
     (the cloze answer scored again with the text metrics).

This is the procedure used for the paper's results.
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd

SPECIAL_TOKENS = ["<|end_of_text|>", "<|eot_id|>", "</s>", "<|begin_of_text|>", "<s>"]

_JSON_PICKOFF = re.compile(
    r"""
    ["']?            # optional opening quote on key
    (?P<key>[^"':]+) # key text
    ["']?\s*:\s*     # colon separator
    (?P<val>
        (?:true|false|null)
      | "(?:\\.|[^"\\])*"
      | '(?:\\.|[^'\\])*'
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)


def model_reply(row) -> str:
    if isinstance(row.get("generation"), str):
        return row["generation"]
    output = row.get("output")
    if not isinstance(output, str):
        return ""
    output = output.replace(str(row.get("prompt", "")), "")
    for token in SPECIAL_TOKENS:
        output = output.replace(token, "")
    return output.strip()


def coerce_literal(v):
    if isinstance(v, str):
        txt = v.strip()
        if txt.lower() in {"true", "false"}:
            return txt.lower() == "true"
        if txt and txt[0] in {"'", '"'} and txt[-1] == txt[0]:
            return txt[1:-1]
        return txt
    return v


def extract_value(raw):
    """Value of a one-field JSON answer; regex fallback for malformed JSON."""
    if not isinstance(raw, str):
        return raw
    raw = raw.strip()
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and len(obj) == 1:
            return coerce_literal(next(iter(obj.values())))
    except Exception:
        pass
    match = _JSON_PICKOFF.search(raw)
    if match:
        return coerce_literal(match.group("val"))
    return "failed parsing"


def to_wide(df: pd.DataFrame) -> pd.DataFrame:
    """One row per scenario; each task contributes `<task>_answer` and `<task>_answer_predict`."""
    if "id" in df.columns:
        df["_scenario"] = df["id"].str.split("_", n=1).str[0]
    else:
        df["_scenario"] = df.groupby("task").cumcount()
    gold_col = "target" if "target" in df.columns else "assistant"
    df["_gold"] = df[gold_col].map(extract_value)
    df["_pred"] = df.apply(model_reply, axis=1).map(extract_value)

    gold = df.pivot(index="_scenario", columns="task", values="_gold").add_suffix("_answer")
    pred = df.pivot(index="_scenario", columns="task", values="_pred").add_suffix("_answer_predict")
    wide = pd.concat([gold, pred], axis=1).reset_index().rename(columns={"_scenario": "scenario"})
    wide["cloze_answer_text"] = wide["cloze_answer"]
    wide["cloze_answer_text_predict"] = wide["cloze_answer_predict"]
    return wide


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="A generation CSV or a folder of them.")
    p.add_argument("output", help="Output folder.")
    args = p.parse_args()

    src, dst = Path(args.input), Path(args.output)
    dst.mkdir(parents=True, exist_ok=True)
    for path in sorted(src.glob("*.csv")) if src.is_dir() else [src]:
        wide = to_wide(pd.read_csv(path, low_memory=False))
        wide.to_csv(dst / path.name, index=False)
        print(f"{path.name}: {len(wide)} scenarios -> {dst / path.name}")


if __name__ == "__main__":
    main()
