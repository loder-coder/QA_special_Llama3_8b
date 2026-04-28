from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorForLanguageModeling, Trainer, TrainingArguments


def format_prompt(row: dict) -> str:
    return (
        f"[카테고리]\n{row.get('category', '')}\n\n"
        f"[질문]\n{row.get('question', '')}\n\n"
        f"[답변]\n{row.get('answer', '')}"
    )


def load_prompt_dataset(path: str) -> Dataset:
    df = pd.read_json(path)
    if df.empty:
        raise ValueError(f"dataset is empty: {path}")
    df = df[["category", "question", "answer"]].fillna("")
    df["text"] = df.apply(lambda row: format_prompt(row.to_dict()), axis=1)
    return Dataset.from_pandas(df[["text"]], preserve_index=False)


def tokenize_dataset(dataset: Dataset, tokenizer: AutoTokenizer, max_length: int) -> Dataset:
    def tokenize(batch: dict) -> dict:
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    return dataset.map(tokenize, batched=True, remove_columns=["text"])


def train_one_stage(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    dataset_path: str,
    output_dir: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    max_length: int,
) -> None:
    dataset = tokenize_dataset(load_prompt_dataset(dataset_path), tokenizer, max_length=max_length)
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def train_lora(
    base_model_name: str = "meta-llama/Llama-3-8b",
    bronze_path: str = "data/bronze.json",
    gold_path: str = "data/gold.json",
    output_dir: str = "artifacts/llama_lora",
    epochs: int = 1,
    batch_size: int = 1,
    learning_rate: float = 2e-4,
    max_length: int = 2048,
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=quantization_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    train_one_stage(model, tokenizer, bronze_path, f"{output_dir}/bronze_stage", epochs, batch_size, learning_rate, max_length)
    train_one_stage(model, tokenizer, gold_path, output_dir, epochs, batch_size, learning_rate, max_length)


if __name__ == "__main__":
    train_lora()
