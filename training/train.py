# coding=utf-8
# Adapted from the Hugging Face PEFT SFT example (Apache License 2.0).
"""LoRA supervised fine-tuning on FarSense.

Usage:
    accelerate launch --config_file configs/accelerate_deepspeed_zero3.yaml \
        training/train.py configs/train_lora.json \
        --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
        --output_dir outputs/Llama-3.2-1B-Instruct

The JSON file holds the paper's hyperparameters (Appendix G); any flag given after it overrides
the JSON value.
"""
import json
import sys
from dataclasses import dataclass, field
from typing import Optional

from transformers import HfArgumentParser, TrainingArguments, set_seed
from trl import SFTTrainer

from utils import create_and_prepare_model, create_datasets


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    chat_template_format: Optional[str] = field(
        default="none",
        metadata={"help": "chatml|zephyr|none. `none` keeps the model's own chat template."},
    )
    lora_alpha: Optional[int] = field(default=256)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=128)
    lora_target_modules: Optional[str] = field(
        default="q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj,embed_tokens,lm_head",
        metadata={"help": "Comma-separated list of modules to apply LoRA to."},
    )
    use_nested_quant: Optional[bool] = field(
        default=False, metadata={"help": "Activate nested quantization for 4bit base models"}
    )
    bnb_4bit_compute_dtype: Optional[str] = field(
        default="float16", metadata={"help": "Compute dtype for 4bit base models"}
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4", metadata={"help": "Quantization type fp4 or nf4"}
    )
    use_flash_attn: Optional[bool] = field(
        default=False, metadata={"help": "Enables Flash Attention 2 for training."}
    )
    use_peft_lora: Optional[bool] = field(
        default=False, metadata={"help": "Enables PEFT LoRA for training."}
    )
    use_8bit_quantization: Optional[bool] = field(
        default=False, metadata={"help": "Loads the model in 8bit."}
    )
    use_4bit_quantization: Optional[bool] = field(
        default=False, metadata={"help": "Loads the model in 4bit."}
    )
    use_reentrant: Optional[bool] = field(
        default=False, metadata={"help": "Gradient checkpointing parameter."}
    )


@dataclass
class DataTrainingArguments:
    dataset_name: Optional[str] = field(
        default="data",
        metadata={
            "help": "Folder with farsense_{train,test}.jsonl.gz, or the Hub id "
            "Kamyar-zeinalipour/CS_Farsi."
        },
    )
    packing: Optional[bool] = field(default=False, metadata={"help": "Pack samples into sequences."})
    dataset_text_field: str = field(
        default="content", metadata={"help": "Dataset field to use as input text."}
    )
    max_seq_length: Optional[int] = field(default=2048)
    append_concat_token: Optional[bool] = field(
        default=False,
        metadata={"help": "If True, appends `eos_token_id` at the end of each sample being packed."},
    )
    add_special_tokens: Optional[bool] = field(
        default=False,
        metadata={"help": "If True, the tokenizer adds special tokens to each sample being packed."},
    )
    splits: Optional[str] = field(
        default="train,test",
        metadata={"help": "Comma-separated list of the splits to use from the dataset."},
    )
    train_fraction: float = field(
        default=1.0, metadata={"help": "Portion of the train set to actually use (0.0-1.0)."}
    )


def main(model_args, data_args, training_args):
    set_seed(training_args.seed)

    model, peft_config, tokenizer = create_and_prepare_model(model_args)

    model.config.use_cache = not training_args.gradient_checkpointing
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": model_args.use_reentrant}

    train_dataset, eval_dataset = create_datasets(
        tokenizer, data_args, training_args, apply_chat_template=True
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        packing=data_args.packing,
        dataset_kwargs={
            "append_concat_token": data_args.append_concat_token,
            "add_special_tokens": data_args.add_special_tokens,
        },
        dataset_text_field=data_args.dataset_text_field,
        max_seq_length=data_args.max_seq_length,
    )
    trainer.accelerator.print(f"{trainer.model}")
    if model_args.use_peft_lora:
        trainer.model.print_trainable_parameters()
        if getattr(trainer.accelerator.state, "fsdp_plugin", None):
            from peft.utils.other import fsdp_auto_wrap_policy
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            fsdp_plugin = trainer.accelerator.state.fsdp_plugin
            kwargs = {
                "sharding_strategy": fsdp_plugin.sharding_strategy,
                "cpu_offload": fsdp_plugin.cpu_offload,
                "auto_wrap_policy": fsdp_auto_wrap_policy(trainer.model),
                "mixed_precision": fsdp_plugin.mixed_precision_policy,
                "sync_module_states": fsdp_plugin.sync_module_states,
                "use_orig_params": False,
                "limit_all_gathers": True,
                "param_init_fn": fsdp_plugin.param_init_fn,
                "device_id": trainer.accelerator.device,
            }
            trainer.model = trainer.model_wrapped = FSDP(trainer.model, **kwargs)
            trainer.args.remove_unused_columns = False

    trainer.train()

    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model()


def _json_to_cli(path):
    with open(path, encoding="utf-8") as f:
        config = json.load(f)
    return [token for key, value in config.items() for token in (f"--{key}", str(value))]


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    cli_args = sys.argv[1:]
    if cli_args and cli_args[0].endswith(".json"):
        cli_args = _json_to_cli(cli_args[0]) + cli_args[1:]
    model_args, data_args, training_args = parser.parse_args_into_dataclasses(args=cli_args)
    main(model_args, data_args, training_args)
