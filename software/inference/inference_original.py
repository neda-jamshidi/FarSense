import gc
import os
import time
import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


DEFAULT_LLAMA2_CHAT_TEMPLATE = (
    "{% for message in messages %}\n"
    "{% if message['role'] == 'system' %}\n"
    "{{ '<<SYS>>\\n' + message['content'] + '\\n<</SYS>>\\n\\n' }}\n"
    "{% elif message['role'] == 'user' %}\n"
    "{{ '[INST]' + message['content'] + '[/INST]\\n' }}\n"
    "{% elif message['role'] == 'assistant' %}\n"
    "{{ message['content'] + '[/INST]\\n' }}\n"
    "{% endif %}\n"
    "{% if loop.last and add_generation_prompt %}\n"
    "{{ '[INST]' }}\n"
    "{% endif %}\n"
    "{% endfor %}"
)

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
MODELS = [
    #"NousResearch/Llama-2-13b-chat-hf",
    #"mistralai/Mistral-Small-24B-Instruct-2501",
    #"mistralai/Mistral-Nemo-Instruct-2407",
    "meta-llama/Llama-3.2-1B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    #"NousResearch/Llama-2-7b-chat-hf",
    
    
    
]
#FRACTIONS      = [0.25, 0.5, 0.75, 1]
FRACTIONS      = [1]
INFERENCE_DIR  = "inference"
DATASET_NAME   = "Kamyar-zeinalipour/CS_Farsi"
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generation hyperparameters
MAX_NEW_TOKENS = 256
TEMPERATURE    = 0.8
TOP_K          = 50
TOP_P          = 0.95
REP_PENALTY    = 1.1

os.makedirs(INFERENCE_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# 1) Load dataset
# -----------------------------------------------------------------------------
print("Loading test split…")
raw_ds = load_dataset(DATASET_NAME, split="test")
df     = pd.DataFrame(raw_ds)

# -----------------------------------------------------------------------------
# 2) Conversation extractor
# -----------------------------------------------------------------------------
def extract_conversation(message_list, tokenizer):
    user_msg = next((m["content"] for m in message_list if m["role"] == "user"), "")
    assistant_msgs = [m["content"] for m in message_list if m["role"] == "assistant"]
    assistant_msg = "\n".join(assistant_msgs)
    prompt = tokenizer.apply_chat_template(
        message_list[:-1],
        tokenize=False,
        add_generation_prompt=True,
    ) if message_list else ""
    print("the prompt is using is :",prompt )
    return user_msg, assistant_msg, prompt

# -----------------------------------------------------------------------------
# 3) PEFT loader (no auto‑sharding, single DEVICE)
# -----------------------------------------------------------------------------
def load_peft_model(base_name, adapter_dir):
    # load base
    model_base = AutoModelForCausalLM.from_pretrained(
        base_name,
        torch_dtype=torch.bfloat16,
         device_map="auto",           # disable accelerate sharding
        low_cpu_mem_usage=True,
        offload_folder="offload",
    )
    # merge LORA / PEFT adapter
    peft_model = PeftModel.from_pretrained(model_base, adapter_dir,  device_map="auto",).merge_and_unload()
    #peft_model.to(DEVICE)
    #peft_model = dispatch_model(peft_model, device_map="auto")
    peft_model.eval()
    return peft_model

# -----------------------------------------------------------------------------
# 4) Generation helper (with attention_mask + pad_token_id)
# -----------------------------------------------------------------------------
def generate(model, tokenizer, prompt):
    model.eval()
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,    # creates attention_mask
        truncation=True,
    )
    # move to DEVICE
    #for k, v in inputs.items():
    #    inputs[k] = v.to(DEVICE)
    inputs = inputs.to('cuda')
    out_ids = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P,
        do_sample=True,
        repetition_penalty=REP_PENALTY,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.batch_decode(out_ids, skip_special_tokens=False)[0]

def unload_model(model):
    """
    Move model back to CPU, delete it, 
    force garbage collection and clear CUDA cache.
    """
    try:
        model.cpu()
    except Exception:
        pass
    # delete reference
    del model
    # collect Python garbage
    gc.collect()
    # release all unoccupied cached memory
    torch.cuda.empty_cache()


# -----------------------------------------------------------------------------
# 5) Main loop: for each model & fraction, run inference
# -----------------------------------------------------------------------------
for model_name in MODELS:
    print(f"\n=== Loading tokenizer for {model_name} ===")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # ensure pad_token_id is set
    tokenizer.pad_token_id = tokenizer.eos_token_id
    if "llama-2" in model_name.lower():
        chat_template = DEFAULT_LLAMA2_CHAT_TEMPLATE
        tokenizer.chat_template = chat_template

    # preprocess prompts once per model
    df[["user", "assistant", "prompt"]] = pd.DataFrame(
        [extract_conversation(msgs, tokenizer) for msgs in df["messages"]],
        columns=["user", "assistant", "prompt"]
    )

    for frac in FRACTIONS:
        sanitized     = model_name.replace("/", "_")
        adapter_dir   = f"outputs/{sanitized}_frac{frac}_farsi_cs_128_256"
        print(f"\n--- Inference: {model_name} @ {int(frac*100)}% data ---")
        print(f"Loading adapter from {adapter_dir}…")
        model = load_peft_model(model_name, adapter_dir)
        total = len(df)

        results = []
        for idx, row in df.iterrows():
            done = idx
            left = total - idx
            print(f"Processing example {done}/{total} ({left} left)…")
            try:
                out = generate(model, tokenizer, row["prompt"])
            except Exception as e:
                print(f"Error at row {idx}: {e}. Retrying in 60s…")
                time.sleep(60)
                out = generate(model, tokenizer, row["prompt"])
            results.append(out)

        # assemble and save
        out_df = df.copy()
        out_df["output"] = results
        out_path = os.path.join(INFERENCE_DIR, f"{sanitized}_frac{frac:.2f}.csv")
        print(f"Saving results to {out_path}")
        out_df.to_csv(out_path, index=False)

         # ---- CLEAN UP ----
        print(f"Unloading {model_name} @ {int(frac*100)}% data from GPU")
        unload_model(model)
        gc.collect()
        torch.cuda.empty_cache()
