import os
from enum import Enum
from pathlib import Path

import torch
from datasets import DatasetDict, load_dataset, load_from_disk
from datasets.builder import DatasetGenerationError
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

DEFAULT_CHATML_CHAT_TEMPLATE = "{% for message in messages %}\n{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% if loop.last and add_generation_prompt %}{{'<|im_start|>assistant\n' }}{% endif %}{% endfor %}"
DEFAULT_ZEPHYR_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"


class ZephyrSpecialTokens(str, Enum):
    user = "<|user|>"
    assistant = "<|assistant|>"
    system = "<|system|>"
    eos_token = "</s>"
    bos_token = "<s>"
    pad_token = "<pad>"

    @classmethod
    def list(cls):
        return [c.value for c in cls]


class ChatmlSpecialTokens(str, Enum):
    user = "<|im_start|>user"
    assistant = "<|im_start|>assistant"
    system = "<|im_start|>system"
    eos_token = "<|im_end|>"
    bos_token = "<s>"
    pad_token = "<pad>"

    @classmethod
    def list(cls):
        return [c.value for c in cls]


def to_messages(example):
    """FarSense item -> single-turn chat: the task prompt and the gold JSON answer."""
    return {
        "messages": [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["target"]},
        ]
    }


def load_farsense_split(dataset_name, split):
    """Load one split from the local `data/` folder, the Hugging Face Hub, or a `save_to_disk` folder."""
    local_file = Path(dataset_name) / f"farsense_{split}.jsonl.gz"
    if local_file.exists():
        dataset = load_dataset("json", data_files=str(local_file), split="train")
    else:
        try:
            dataset = load_dataset(dataset_name, split=split)
        except (DatasetGenerationError, FileNotFoundError, ValueError):
            dataset = load_from_disk(os.path.join(dataset_name, split))
    if "messages" not in dataset.column_names:
        dataset = dataset.map(to_messages)
    return dataset


def create_datasets(tokenizer, data_args, training_args, apply_chat_template=False):
    def preprocess(samples):
        return {
            "content": [
                tokenizer.apply_chat_template(conversation, tokenize=False)
                for conversation in samples["messages"]
            ]
        }

    raw_datasets = DatasetDict()
    for split in data_args.splits.split(","):
        dataset = load_farsense_split(data_args.dataset_name, split)
        if "train" in split:
            raw_datasets["train"] = dataset
        elif "test" in split:
            raw_datasets["test"] = dataset
        else:
            raise ValueError(f"Split type {split} not recognized as one of test or train.")

    if apply_chat_template:
        raw_datasets = raw_datasets.map(
            preprocess,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
        )

    train_data = raw_datasets["train"]
    if data_args.train_fraction < 1.0:
        train_data = train_data.shuffle(seed=training_args.seed).select(
            range(int(len(train_data) * data_args.train_fraction))
        )
        print(f"Using a fraction of the training data: {data_args.train_fraction}")
    valid_data = raw_datasets["test"]
    print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")
    print(f"A sample of train dataset: {train_data[0]}")

    return train_data, valid_data


def create_and_prepare_model(args):
    device_map = None
    bnb_config = None
    load_in_8bit = args.use_8bit_quantization

    if args.use_4bit_quantization:
        compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=args.use_4bit_quantization,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.use_nested_quant,
        )
        if compute_dtype == torch.float16:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                print("Your GPU supports bfloat16, you can accelerate training with the argument --bf16")

    if args.use_4bit_quantization or args.use_8bit_quantization:
        device_map = (
            int(os.environ.get("LOCAL_RANK", -1))
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else "auto"
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        load_in_8bit=load_in_8bit,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if args.use_flash_attn else "eager",
    )

    peft_config = None
    if args.use_peft_lora:
        peft_config = LoraConfig(
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=args.lora_target_modules.split(","),
        )

    special_tokens = None
    chat_template = None
    if args.chat_template_format == "chatml":
        special_tokens = ChatmlSpecialTokens
        chat_template = DEFAULT_CHATML_CHAT_TEMPLATE
    elif args.chat_template_format == "zephyr":
        special_tokens = ZephyrSpecialTokens
        chat_template = DEFAULT_ZEPHYR_CHAT_TEMPLATE

    if special_tokens is not None:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            pad_token=special_tokens.pad_token.value,
            bos_token=special_tokens.bos_token.value,
            eos_token=special_tokens.eos_token.value,
            additional_special_tokens=special_tokens.list(),
            trust_remote_code=True,
        )
        tokenizer.chat_template = chat_template
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
    else:
        # The instruct models used in the paper already ship a chat template.
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        if "llama" in args.model_name_or_path.lower():
            tokenizer.pad_token = "<|finetune_right_pad_id|>"
        else:
            tokenizer.pad_token = "[PAD]"
        print(f"Using pad token: {tokenizer.pad_token}")

    return model, peft_config, tokenizer
