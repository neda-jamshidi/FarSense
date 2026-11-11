"""Generate answers for the FarSense test split with a base model or a fine-tuned LoRA adapter.

Examples:
    # zero-shot base model
    python inference/generate.py --model meta-llama/Llama-3.2-1B-Instruct

    # fine-tuned model (adapter written by training/train.py)
    python inference/generate.py --model meta-llama/Llama-3.2-1B-Instruct \
        --adapter outputs/Llama-3.2-1B-Instruct

The output CSV keeps every column of the test file and adds `generation`, the model's reply.
Decoding settings follow Appendix G.2 of the paper.
"""
import argparse
import gc
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="Base model id or path.")
    p.add_argument("--adapter", default=None, help="LoRA adapter folder. Omit to evaluate the base model.")
    p.add_argument("--data", default="data/farsense_test.jsonl.gz", help="Test file (JSONL, optionally gzipped).")
    p.add_argument("--output", default=None, help="Output CSV. Default: results/raw/<model>_{base,finetuned}.csv")
    p.add_argument("--limit", type=int, default=None, help="Only run the first N items (for a quick check).")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--repetition-penalty", type=float, default=1.1)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_model(base_name, adapter_dir):
    model = AutoModelForCausalLM.from_pretrained(
        base_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    if adapter_dir:
        model = PeftModel.from_pretrained(model, adapter_dir).merge_and_unload()
    model.eval()
    return model


@torch.no_grad()
def generate(model, tokenizer, prompt, args):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out_ids = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_tokens = out_ids[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main():
    args = parse_args()
    set_seed(args.seed)

    df = pd.read_json(args.data, lines=True)
    if args.limit:
        df = df.head(args.limit)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = load_model(args.model, args.adapter)

    generations = []
    for prompt in tqdm(df["prompt"], desc=Path(args.model).name):
        chat = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        generations.append(generate(model, tokenizer, chat, args))
    df["generation"] = generations

    if args.output:
        out_path = Path(args.output)
    else:
        suffix = "_finetuned" if args.adapter else "_base"
        out_path = Path("results/raw") / f"{Path(args.model).name}{suffix}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} generations to {out_path}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
