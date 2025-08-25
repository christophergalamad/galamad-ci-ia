import argparse
import json
import os
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("DISABLE_MLFLOW_INTEGRATION", "true")
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_labelstudio_chitumbuka_dataset(json_path: str) -> List[Dict[str, str]]:
    """
    Reads a Label Studio-style JSON export where each item contains:
      - data.english: source phrase
      - annotations[].result[].value.text: list of chitumbuka translations (strings)

    Returns a list of {"prompt": str, "answer": str} pairs.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    pairs: List[Dict[str, str]] = []
    for item in raw:
        english_text: Optional[str] = None
        try:
            english_text = item.get("data", {}).get("english")
        except Exception:
            english_text = None

        if not english_text:
            continue

        annotations = item.get("annotations", []) or []
        for ann in annotations:
            results = ann.get("result", []) or []
            for res in results:
                value = res.get("value", {}) or {}
                texts = value.get("text", []) or []
                for answer in texts:
                    answer = (answer or "").strip()
                    if not answer:
                        continue
                    prompt = (
                        "You are a helpful translator. Translate the following English phrase into Chitumbuka.\n"
                        f"English: {english_text}\n"
                        "Chitumbuka:"
                    )
                    pairs.append({"prompt": prompt, "answer": answer})

    if not pairs:
        raise ValueError("No (prompt, answer) pairs could be extracted from the dataset.")

    return pairs


def choose_torch_dtype() -> torch.dtype:
    # Favor bfloat16 where supported; otherwise use float16 on CUDA; fall back to float32
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    # On Apple Silicon (MPS), float32 is typically safest
    return torch.float32


def build_tokenized_dataset(
    tokenizer: AutoTokenizer,
    pairs: List[Dict[str, str]],
    max_seq_length: int,
) -> Dataset:
    """
    Tokenize samples so that loss is only computed on the answer portion.
    labels = [-100]*len(prompt_tokens) + answer_tokens + [eos]
    """

    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("Tokenizer must have an eos_token_id.")

    def tokenize_one(example: Dict[str, str]) -> Dict[str, List[int]]:
        prompt = example["prompt"]
        answer = example["answer"]

        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        # Leading space helps many tokenizers form proper tokens
        answer_ids = tokenizer(" " + answer, add_special_tokens=False)["input_ids"]

        input_ids = prompt_ids + answer_ids + [eos_id]
        labels = ([-100] * len(prompt_ids)) + answer_ids + [eos_id]
        attention_mask = [1] * len(input_ids)

        # If too long, keep the end (answers) by trimming from the left consistently
        if len(input_ids) > max_seq_length:
            overflow = len(input_ids) - max_seq_length
            input_ids = input_ids[overflow:]
            attention_mask = attention_mask[overflow:]
            labels = labels[overflow:]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    hf_ds = Dataset.from_list(pairs)
    tokenized = hf_ds.map(tokenize_one, remove_columns=list(hf_ds.features))
    return tokenized


@dataclass
class PadCollator:
    pad_token_id: int

    def __call__(self, batch: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(x["input_ids"]) for x in batch)

        def pad(seq: List[int], pad_id: int) -> List[int]:
            return seq + [pad_id] * (max_len - len(seq))

        input_ids = [pad(x["input_ids"], self.pad_token_id) for x in batch]
        attention_mask = [pad(x["attention_mask"], 0) for x in batch]
        labels = [pad(x["labels"], -100) for x in batch]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a LoRA adapter for Chitumbuka translation.")
    parser.add_argument("--dataset_path", type=str, default="../data/chitumbuka_fine_tuned.json")
    parser.add_argument(
        "--base_model",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="HF model id or local path. Choose a LLaMA-architecture model for best GGUF compatibility.",
    )
    parser.add_argument("--output_dir", type=str, default="../outputs/chitumbuka_lora")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_random_seed(args.seed)

    print("Loading dataset...")
    pairs = load_labelstudio_chitumbuka_dataset(args.dataset_path)
    print(f"Loaded {len(pairs)} (prompt, answer) pairs.")

    print("Loading tokenizer and base model...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        # Use EOS as PAD for causal LM training
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = choose_torch_dtype()
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch_dtype,
        device_map="auto",
    )

    # Configure LoRA for LLaMA-like attention/MLP modules
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=(
            [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
        ),
    )

    model = get_peft_model(model, lora_config)

    print("Tokenizing dataset...")
    tokenized_ds = build_tokenized_dataset(tokenizer, pairs, args.max_seq_len)
    data_collator = PadCollator(pad_token_id=tokenizer.pad_token_id)

    print("Starting training...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="no",
        save_total_limit=None,
        fp16=torch_dtype == torch.float16,
        bf16=torch_dtype == torch.bfloat16,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    print("Saving LoRA adapter...")
    adapter_dir = os.path.join(args.output_dir, "lora_adapter")
    os.makedirs(adapter_dir, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    print("\nDone. LoRA adapter saved at:")
    print(adapter_dir)
    print(
        "\nNext steps (convert to GGUF for Ollama):\n"
        "1) git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp\n"
        "2) python3 -m pip install -r requirements.txt\n"
        "3) python3 convert-lora-to-gguf.py --base-model '" + args.base_model + "' --lora-model '" + adapter_dir + "' --outfile adapter.gguf\n"
        "4) Move adapter.gguf into your project (e.g., ./adapters/adapter.gguf)\n"
        "5) Update your Modelfile to:\n\n"
        "   FROM deepseek-r1:latest\n"
        "   ADAPTER ./adapters/adapter.gguf\n"
        "   PARAMETER temperature 0.2\n"
        "   PARAMETER top_p 0.9\n\n"
        "6) Build: ollama create model_chitumbuka_fineTuned -f model_file_chitumbuka\n"
    )


if __name__ == "__main__":
    main()


