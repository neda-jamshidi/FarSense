#!/usr/bin/env bash
# Fine-tune the six open models from the paper with LoRA (Appendix G.1).
set -euo pipefail
cd "$(dirname "$0")/.."

MODELS=(
  meta-llama/Llama-3.2-1B-Instruct
  meta-llama/Llama-3.2-3B-Instruct
  meta-llama/Llama-3.1-8B-Instruct
  mistralai/Mistral-7B-Instruct-v0.3
  mistralai/Mistral-Nemo-Instruct-2407
  mistralai/Mistral-Small-24B-Instruct-2501
)

for model in "${MODELS[@]}"; do
  accelerate launch --config_file configs/accelerate_deepspeed_zero3.yaml \
    training/train.py configs/train_lora.json \
    --model_name_or_path "$model" \
    --output_dir "outputs/$(basename "$model")"
done
