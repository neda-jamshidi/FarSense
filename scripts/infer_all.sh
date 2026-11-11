#!/usr/bin/env bash
# Answer the test split with every model from the paper.
#   open models: before (base) and after (finetuned) LoRA fine-tuning  -> needs a GPU
#   API models:  GPT-4o, Gemini, DeepSeek, Qwen, Mistral                -> needs keys in .env
set -euo pipefail
cd "$(dirname "$0")/.."

OPEN_MODELS=(
  meta-llama/Llama-3.2-1B-Instruct
  meta-llama/Llama-3.2-3B-Instruct
  meta-llama/Llama-3.1-8B-Instruct
  mistralai/Mistral-7B-Instruct-v0.3
  mistralai/Mistral-Nemo-Instruct-2407
  mistralai/Mistral-Small-24B-Instruct-2501
)
API_MODELS=(gpt-4o gemini-2.5-flash deepseek-chat qwen-plus mistral-large)

for model in "${OPEN_MODELS[@]}"; do
  name=$(basename "$model")
  python inference/generate.py --model "$model" --output "results/raw/${name}_base.csv"
  python inference/generate.py --model "$model" --adapter "outputs/$name" \
    --output "results/raw/${name}_finetuned.csv"
done

if [[ "${SKIP_API:-0}" != "1" ]]; then
  for model in "${API_MODELS[@]}"; do
    python inference/generate_api.py --model "$model"
  done
fi
