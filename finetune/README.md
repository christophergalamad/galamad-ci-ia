# Fine-tuning (LoRA) for Chitumbuka Translation

This folder contains everything needed to fine-tune a Chitumbuka translation adapter using LoRA, test it, and export to GGUF for use with llama.cpp / Ollama.

## Folder structure

```
finetune/
  README.md                # This guide
  scripts/
    train_lora_chitumbuka.py   # Train LoRA on Label Studio JSON
    convert_lora_simple.py     # Merge LoRA into base and export GGUF
    test_chitumbuka_model.py   # Quick translator test loop
  data/
    chitumbuka_fine_tuned.json # Example training JSON (Label Studio export)
    chitumbuka_english.json    # Optional additional data
  outputs/
    chitumbuka_lora/           # Training outputs (LoRA adapter saved here)
  adapters/
    chitumbuka_adapter.gguf    # Exported adapter for llama.cpp/Ollama
  tmp/
    temp_tokenizer/            # Temporary tokenizer files used during export
```

## Requirements

- Python 3.10+
- See project-level `requirements.txt` for core deps. Minimums:
  - torch>=2.3.0
  - transformers>=4.44.0
  - peft>=0.13.0
  - datasets>=2.20.0
  - accelerate>=0.34.0
  - sentencepiece>=0.1.99
  - protobuf>=4.25.3
  - gguf (for conversion script)

Install:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r ../../requirements.txt
pip install gguf
```

## Data format

The training script expects a Label Studio style JSON where each item contains:
- `data.english`: source text
- `annotations[].result[].value.text`: list of Chitumbuka translations

The script converts each to a prompt/answer pair:
- Prompt: "You are a helpful translator... English: <text>\nChitumbuka:"
- Answer: one translation string

Place your JSON under `finetune/data/` and point `--dataset_path` to it.

## Training

From the `finetune/scripts/` directory:

```bash
python train_lora_chitumbuka.py \
  --dataset_path ../data/chitumbuka_fine_tuned.json \
  --base_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --output_dir ../outputs/chitumbuka_lora \
  --num_epochs 3 \
  --learning_rate 2e-4 \
  --batch_size 4 \
  --grad_accum 2 \
  --max_seq_len 256
```

Outputs are saved to `../outputs/chitumbuka_lora/lora_adapter`.

## Quick test

```bash
python test_chitumbuka_model.py
```

You can edit the test prompts in the script. It loads the base model and applies the LoRA adapter from `../outputs/chitumbuka_lora/lora_adapter`.

## Export to GGUF

Use the provided simple converter (merges LoRA into base and writes GGUF):

```bash
python convert_lora_simple.py
```

This writes `../adapters/chitumbuka_adapter.gguf` and stores a temporary tokenizer at `../tmp/temp_tokenizer/`.

Alternatively, you can use `llama.cpp`'s official script:

```bash
# From llama.cpp repo
python3 convert-lora-to-gguf.py \
  --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --lora-model /absolute/path/to/finetune/outputs/chitumbuka_lora/lora_adapter \
  --outfile adapter.gguf
```

## Using with Ollama

Example Modelfile snippet:

```
FROM deepseek-r1:latest
ADAPTER ./adapters/chitumbuka_adapter.gguf
PARAMETER temperature 0.2
PARAMETER top_p 0.9
```

Build your model:

```bash
ollama create model_chitumbuka_fineTuned -f model_file_chitumbuka
```

## Tips

- On CUDA, the script automatically chooses fp16 or bf16 where supported; on Apple Silicon it defaults to float32.
- Ensure tokenizer has an `eos_token_id`. The trainer uses EOS as PAD when needed.
- If sequences overflow `--max_seq_len`, the script trims from the left to preserve answers.
