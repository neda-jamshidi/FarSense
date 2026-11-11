"""Answer the FarSense test split with an API model (GPT-4o, Gemini, DeepSeek, Qwen, Mistral).

    python inference/generate_api.py --model gpt-4o

Writes results/raw/<model>.csv in the same format as inference/generate.py, so both go
through the same evaluation. Progress is saved after every item; re-running the command
resumes where it stopped.
"""
import argparse
import time
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from llm_apis import MODELS, chat

SYSTEM_PROMPT = "You are a Farsi commonsense-reasoning engine."


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=sorted(MODELS))
    p.add_argument("--model-id", default=None, help="Override the provider model id.")
    p.add_argument("--data", default="data/farsense_test.jsonl.gz")
    p.add_argument("--output", default=None, help="Default: results/raw/<model>.csv")
    p.add_argument("--limit", type=int, default=None, help="Only run the first N items.")
    p.add_argument("--sleep", type=float, default=0.5, help="Pause between requests (seconds).")
    args = p.parse_args()

    out_path = Path(args.output or f"results/raw/{args.model}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_json(args.data, lines=True)
    if args.limit:
        df = df.head(args.limit)
    if out_path.exists():
        done = pd.read_csv(out_path).set_index("id")["generation"]
        df["generation"] = df["id"].map(done)
    else:
        df["generation"] = None

    failed = df["generation"].astype(str).str.startswith("ERROR")
    todo = df.index[df["generation"].isna() | failed]
    for i in tqdm(todo, desc=args.model):
        try:
            df.at[i, "generation"] = chat(args.model, df.at[i, "prompt"], SYSTEM_PROMPT, args.model_id)
        except Exception as exc:  # keep going; failed items are scored as wrong
            df.at[i, "generation"] = f"ERROR: {exc}"
        df.to_csv(out_path, index=False)
        time.sleep(args.sleep)

    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} generations to {out_path}")


if __name__ == "__main__":
    main()
