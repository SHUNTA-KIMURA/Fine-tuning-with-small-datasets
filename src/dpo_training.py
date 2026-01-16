import json
import random
from pathlib import Path

import torch
from PRISM import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel
from trl import DPOConfig, DPOTrainer


# ==========
# Config
# ==========
# 8BはMacだと厳しいことが多いので、まずは軽いモデルで動作確認推奨
MODEL_ID = "elyza/Llama-3-ELYZA-JP-8B"
USER_JSON = "../datasets/processed_preferences/user0.json"
OUTPUT_DIR = "./output_user0_mac"

SEED = 42
TRAIN_N = 10
EVAL_N = 5

# LoRA
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "v_proj"]  # 重いならここは最小で


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)


def load_user_preferences(path: str | Path):
    """Load your JSON: [{query, chosen, reject}, ...] -> [{prompt, chosen, rejected}, ...]."""
    rows = json.load(open(path, "r", encoding="utf-8"))
    out = []
    for r in rows:
        out.append(
            {
                "prompt": r["query"],
                "chosen": r["chosen"],
                "rejected": r["reject"],
            }
        )
    return out


def train_eval_split(rows, train_n=10, eval_n=5, seed=42):
    rows = rows.copy()
    rnd = random.Random(seed)
    rnd.shuffle(rows)
    train = rows[:train_n]
    eval_ = rows[train_n:train_n + eval_n]
    if len(train) < train_n or len(eval_) < eval_n:
        raise ValueError(f"Not enough data. got train={len(train)}, eval={len(eval_)}")
    return train, eval_


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_chat_prompt(tokenizer, user_text: str, system_text: str = "あなたは日本語チャットbotです。ユーザの質問に答えてください。"):
    """ELYZA系はchat templateが使える場合があるので、あればそれを優先。"""
    messages = [{"role": "system", "content": system_text},
                {"role": "user", "content": user_text}]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return system_text + "\n" + user_text


@torch.inference_mode()
def generate(model, tokenizer, prompt_text: str, max_new_tokens=128, temperature=0.6, top_p=0.9):
    inputs = tokenizer(prompt_text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
    )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    set_seed(SEED)
    device = get_device()
    print("Device:", device)

    # ===== Data =====
    rows = load_user_preferences(USER_JSON)
    train_rows, eval_rows = train_eval_split(rows, TRAIN_N, EVAL_N, seed=SEED)
    train_ds = Dataset.from_list(train_rows)
    eval_ds = Dataset.from_list(eval_rows)

    # ===== Tokenizer =====
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ===== Model (no 4bit on Mac) =====
    # MPSは基本 fp16 が現実的
    dtype = torch.float16 if device.type == "mps" else torch.float32

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
    ).to(device)

    # ===== Apply LoRA =====
    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
    )
    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()

    # ===== Reference model for DPO =====
    # ref_model is frozen base (no LoRA). Keep it on same device.
    ref_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
    ).to(device)
    ref_model.eval()

    # ===== DPO config =====
    # Macは遅いので epoch少なめ・steps少なめ推奨
    dpo_args = DPOConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        learning_rate=1e-5,
        logging_steps=5,
        save_strategy="no",
        evaluation_strategy="no",
        beta=0.05,
        remove_unused_columns=True,
        fp16=(device.type == "mps"),
        bf16=False,
        report_to="none",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_args,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)

    # ===== Load adapter and test inference =====
    # (load on top of fresh base to avoid any trainer state)
    fresh_base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
    ).to(device)
    tuned = PeftModel.from_pretrained(fresh_base, OUTPUT_DIR).to(device)
    tuned.eval()

    prompt = "LLMのファインチューニングの手順を教えてください。"
    prompt_text = build_chat_prompt(tokenizer, prompt)
    resp = generate(tuned, tokenizer, prompt_text)
    print("Response:\n", resp)


if __name__ == "__main__":
    main()
