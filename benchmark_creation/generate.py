"""Step 1 of benchmark creation: Wikipedia passage -> everyday scenario -> six tasks (Section 3.2).

    python benchmark_creation/generate.py passages.csv generated.csv --model gemini-2.5-flash

`passages.csv` needs a `Content_Paragraph` column (other columns, e.g. `Article_Title`,
are carried over). For every passage the model is called twice:
    prompt_1(passage)  -> `scenario`
    prompt_2(scenario) -> `tasks_json`
Progress is saved after every passage; re-running the command resumes.

In the paper the 4,210 scenario/task bundles were generated with Gemini-2.5-Flash and then
reviewed and corrected by two native speakers before building the dataset (Section 3.4).
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))
from llm_apis import MODELS, chat  # noqa: E402
from prompts import SYSTEM_PROMPT, prompt_1, prompt_2  # noqa: E402


def render(template: str, text: str) -> str:
    return template.replace("{{SOURCE_TEXT}}", text)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="CSV with a Content_Paragraph column.")
    p.add_argument("output", help="Output CSV (scenario + tasks_json columns added).")
    p.add_argument("--model", default="gemini-2.5-flash", choices=sorted(MODELS))
    p.add_argument("--model-id", default=None, help="Override the provider model id.")
    p.add_argument("--sleep", type=float, default=0.5)
    args = p.parse_args()

    out_path = Path(args.output)
    df = pd.read_csv(out_path if out_path.exists() else args.input)
    for col in ("scenario", "tasks_json"):
        if col not in df.columns:
            df[col] = None

    # The paper's Gemini runs sent the prompt without a system message; the other models had one.
    system = None if MODELS[args.model][0] == "gemini" else SYSTEM_PROMPT

    for i in tqdm(df.index[df["tasks_json"].isna()], desc=args.model):
        try:
            scenario = chat(args.model, render(prompt_1, df.at[i, "Content_Paragraph"]), system, args.model_id)
            df.at[i, "scenario"] = scenario
            df.at[i, "tasks_json"] = chat(args.model, render(prompt_2, scenario), system, args.model_id)
        except Exception as exc:
            print(f"row {i}: {exc}")
        df.to_csv(out_path, index=False)
        time.sleep(args.sleep)

    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
