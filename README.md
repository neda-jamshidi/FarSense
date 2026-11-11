# FarSense: A Comprehensive Commonsense Benchmark and Evaluation Framework for the Farsi Language

[![Paper](https://img.shields.io/badge/IJCNLP--AACL_2025-paper-b31b1b)](https://aclanthology.org/2025.ijcnlp-long.187/)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97_Hugging_Face-CS__Farsi-yellow)](https://huggingface.co/datasets/Kamyar-zeinalipour/CS_Farsi)
[![Code license](https://img.shields.io/badge/code-Apache_2.0-blue)](LICENSE)
[![Data license](https://img.shields.io/badge/data-CC_BY--SA_4.0-lightgrey)](data/README.md#source-and-license)

FarSense is a benchmark for context-grounded commonsense reasoning in Farsi (Persian).
Each of its 4,210 scenarios is an everyday narrative rewritten from a Farsi Wikipedia
passage, paired with six kinds of reasoning tasks. This repository contains the dataset
and all the code behind the paper: building the benchmark, fine-tuning open models,
evaluating open and API models, and computing the reported scores.

| | |
|---|---|
| Scenarios | 4,210 (train 4,000 · test 210) |
| Items | 29,470 (train 28,000 · test 1,470), seven per scenario |
| Tasks | True/False (×2), multiple choice, cloze, cause→effect, counterfactual, explanation |
| Construction | Gemini-2.5-Flash generation, two-pass native-speaker validation |
| Evaluated | 5 API models and 6 open models (1B–24B), before and after LoRA fine-tuning |

## Contents

- [Tasks](#tasks)
- [The data](#the-data)
- [Repository layout](#repository-layout)
- [Setup](#setup)
- [Reproducing the paper](#reproducing-the-paper)
- [Results](#results)
- [Building the benchmark](#building-the-benchmark)
- [Citation](#citation)
- [License](#license)

## Tasks

Every item gives the model the scenario (`Context`) and asks for exactly one JSON object.

| `task` | The model must | Answer | Scored with |
|---|---|---|---|
| `true_false_1`, `true_false_2` | judge a statement as true or false | `true` / `false` | accuracy |
| `multiple_choice` | pick one of four options | `A`–`D` | accuracy |
| `cloze` | fill the blank «_____» in a sentence | Farsi phrase | accuracy, and the four text metrics |
| `cause_effect` | state the effect of a given cause | Farsi sentence | BLEU-4, ROUGE-L, BERTScore, cosine |
| `counterfactual` | answer a "what if" question (≤ 40 words) | Farsi sentence | BLEU-4, ROUGE-L, BERTScore, cosine |
| `explanation` | answer a "why" question (≤ 40 words) | Farsi sentence | BLEU-4, ROUGE-L, BERTScore, cosine |

Scores combine as in Section 4.2 of the paper: **Classification** is the mean accuracy over
T/F-True, T/F-False, MCQ and Cloze; **Free-form** is the mean, over Cause→Effect,
Counterfactual, Explanation and Cloze-text, of the average of the four text metrics;
**Overall** is the mean of the two.

## The data

The data ships with the repository in [`data/`](data/), so no download or account is needed.

```python
import pandas as pd

train = pd.read_json("data/farsense_train.jsonl.gz", lines=True)
test = pd.read_json("data/farsense_test.jsonl.gz", lines=True)
print(test.loc[0, "prompt"])
print(test.loc[0, "target"])  # {"true_false_1_answer_predict": true}
```

The same split is on the Hugging Face Hub as
[`Kamyar-zeinalipour/CS_Farsi`](https://huggingface.co/datasets/Kamyar-zeinalipour/CS_Farsi).
Field descriptions, label distribution and notes are in the [data card](data/README.md).

## Repository layout

```
FarSense/
├── data/                       dataset (train/test) and data card
├── benchmark_creation/         how the benchmark was built (Section 3)
│   ├── prompts.py              passage → scenario and scenario → tasks prompts
│   ├── task_templates.py       templates that turn each task into a test item
│   ├── generate.py             run the generation prompts through an API model
│   └── build_dataset.py        turn generated tasks into dataset items
├── training/                   LoRA fine-tuning (Section 4.3, Appendix G)
│   ├── train.py
│   └── utils.py
├── configs/
│   ├── train_lora.json                    hyperparameters from the paper
│   └── accelerate_deepspeed_zero3.yaml    DeepSpeed ZeRO-3, 3 GPUs
├── inference/
│   ├── generate.py             answer the test set with an open model (base or fine-tuned)
│   ├── generate_api.py         answer the test set with an API model
│   └── llm_apis.py             GPT-4o, Gemini, DeepSeek, Qwen, Mistral clients
├── evaluation/
│   ├── prepare_for_eval.py     extract answers from model replies
│   ├── evaluate.py             accuracy, P/R/F1, BLEU-4, ROUGE-L, BERTScore, cosine
│   └── aggregate.py            Classification / Free-form / Overall scores
├── paper_results/              per-model metrics behind the paper's tables
├── scripts/                    train_all.sh, infer_all.sh, evaluate_all.sh
├── requirements.txt            training and inference
├── requirements-eval.txt       scoring
└── requirements-api.txt        API models
```

## Setup

The experiments ran with Python 3.11 on three NVIDIA RTX A6000 GPUs (48 GB each).

```bash
git clone https://github.com/KamyarZeinalipour/FarSense.git
cd FarSense
conda create -n farsense python=3.11 -y && conda activate farsense

pip install -r requirements.txt          # fine-tuning and open-model inference
pip install -r requirements-eval.txt     # scoring
pip install -r requirements-api.txt      # only for API models or benchmark generation
pip install flash-attn==2.6.1 --no-build-isolation   # optional, used for training
```

The Llama and Mistral models are gated on the Hugging Face Hub. Accept their terms on each
model page, then run `huggingface-cli login`. For API models, copy `.env.example` to `.env` and add
your keys.

## Reproducing the paper

All commands run from the repository root. Each step has a script in `scripts/` that runs
it for every model in the paper.

**1. Fine-tune** an open model with LoRA (≈ 1 h for Llama-3.2-1B, 13 h for Mistral-Small-24B
on 3× A6000):

```bash
accelerate launch --config_file configs/accelerate_deepspeed_zero3.yaml \
  training/train.py configs/train_lora.json \
  --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
  --output_dir outputs/Llama-3.2-1B-Instruct
```

`configs/train_lora.json` holds the paper's settings; any flag after it overrides the file.
Set `num_processes` in the accelerate config to your number of GPUs. `scripts/train_all.sh`
trains all six models.

**2. Answer the test set.** The base model, the fine-tuned model, and an API model:

```bash
python inference/generate.py --model meta-llama/Llama-3.2-1B-Instruct
python inference/generate.py --model meta-llama/Llama-3.2-1B-Instruct --adapter outputs/Llama-3.2-1B-Instruct
python inference/generate_api.py --model gpt-4o
```

Each writes a CSV to `results/raw/`. Add `--limit 20` for a quick check.
`scripts/infer_all.sh` covers every model (set `SKIP_API=1` to skip the API models).

**3. Score:**

```bash
python evaluation/prepare_for_eval.py results/raw results/ready
python evaluation/evaluate.py results/ready -o results/scores
python evaluation/aggregate.py results/scores/aggregate_metrics.csv
```

BERTScore uses `microsoft/deberta-xlarge-mnli`, so a GPU makes this step much faster.

**Check the reported numbers without a GPU.** `paper_results/` holds the per-model metrics
from the paper's runs. This prints Tables 6 and 9:

```bash
python evaluation/aggregate.py paper_results/finetuned_models.csv
python evaluation/aggregate.py paper_results/api_models.csv
```

### Training settings (Appendix G)

| Setting | Value |
|---|---|
| Method | LoRA, r = 128, α = 256, dropout 0.1 |
| LoRA modules | q, k, v, o, gate, up, down projections, `embed_tokens`, `lm_head` |
| Epochs | 3 |
| Batch | 4 sequences per GPU, 2 gradient-accumulation steps, packed to 2,048 tokens |
| Optimizer | AdamW, learning rate 1e-4, cosine decay, no warm-up, weight decay 1e-4 |
| Other | bf16, gradient clipping 1.0, DeepSpeed ZeRO-3, Flash Attention 2, seed 100 |
| Decoding | sampling, temperature 0.8, top-p 0.95, top-k 50, repetition penalty 1.1, ≤ 256 new tokens |

## Results

Open models before and after fine-tuning on FarSense (test split, Table 6):

| Model | Classification | | Free-form | | Overall | |
|---|---:|---:|---:|---:|---:|---:|
| | Base | FT | Base | FT | Base | FT |
| Llama-3.2-1B-Instruct | 0.286 | 0.582 | 0.168 | 0.319 | 0.227 | 0.450 |
| Llama-3.2-3B-Instruct | 0.593 | 0.732 | 0.305 | 0.326 | 0.449 | 0.529 |
| Llama-3.1-8B-Instruct | 0.761 | 0.813 | 0.435 | 0.471 | 0.598 | 0.642 |
| Mistral-7B-Instruct-v0.3 | 0.615 | 0.817 | 0.377 | 0.482 | 0.496 | 0.649 |
| Mistral-Nemo-Instruct-2407 | 0.789 | 0.826 | 0.419 | 0.482 | 0.604 | 0.654 |
| Mistral-Small-24B-Instruct-2501 | 0.810 | 0.845 | 0.445 | 0.499 | 0.627 | **0.672** |

API models, zero-shot (Table 9):

| Model | Classification | Free-form | Overall |
|---|---:|---:|---:|
| DeepSeek-Chat | 0.881 | 0.518 | **0.700** |
| Gemini-2.5-Flash | 0.874 | 0.518 | 0.696 |
| Mistral-Large | 0.864 | 0.497 | 0.681 |
| GPT-4o | 0.860 | 0.494 | 0.677 |
| Qwen-Plus | 0.845 | 0.481 | 0.663 |

Per-task scores for every model are in [`paper_results/`](paper_results/).

## Building the benchmark

FarSense was built in three stages (Section 3):

1. **Passages.** Lead paragraphs of 8,894 Farsi Wikipedia articles from the featured, vital
   and most-viewed article lists. Paragraphs under 100 words, sensitive pages and
   navigation or list pages are removed, leaving 4,210 passages.
2. **Generation.** Each passage is rewritten into an everyday scenario (`prompt_1`), and the six
   tasks are generated from that scenario (`prompt_2`). Gemini-2.5-Flash was chosen after
   native speakers compared outputs from five models.
3. **Validation.** Two native-speaker annotators reviewed every bundle; 60.1% needed at least
   one edit and 17.9% were regenerated.

To apply the pipeline to new passages (a CSV with a `Content_Paragraph` column):

```bash
python benchmark_creation/generate.py passages.csv generated.csv --model gemini-2.5-flash
# review and correct `scenario` / `tasks_json` in generated.csv
python benchmark_creation/build_dataset.py generated.csv my_items.jsonl.gz
```

## Citation

```bibtex
@inproceedings{zeinalipour-etal-2025-farsense,
    title = "{F}ar{S}ense: A Comprehensive Commonsense Benchmark and Evaluation Framework for the {F}arsi Language",
    author = "Zeinalipour, Kamyar  and
      Jamshidi, Neda  and
      Hejazi, Seyedehbahareh  and
      Maggini, Marco  and
      Bianchini, Monica  and
      Paoletti, Simone  and
      Gori, Marco",
    booktitle = "Proceedings of the 14th International Joint Conference on Natural Language Processing and the 4th Conference of the {A}sia-Pacific Chapter of the {A}ssociation for {C}omputational {L}inguistics",
    month = dec,
    year = "2025",
    address = "Mumbai, India",
    publisher = "The Asian Federation of Natural Language Processing and The Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.ijcnlp-long.187/",
    doi = "10.18653/v1/2025.ijcnlp-long.187",
    pages = "3529--3599",
}
```

## License

Code: [Apache License 2.0](LICENSE). The training code is adapted from the Hugging Face PEFT
examples, which use the same license.
Data: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), as it is derived from
Wikipedia text.

Questions and problems: please [open an issue](https://github.com/KamyarZeinalipour/FarSense/issues).
