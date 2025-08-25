#!/usr/bin/env python3
"""
Simple LoRA to GGUF converter using available packages
"""

import os
import json
import torch
import gguf
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def convert_lora_to_gguf(base_model_path, lora_path, output_path):
    """Convert LoRA adapter to GGUF format"""
    
    print(f"Loading base model: {base_model_path}")
    print(f"Loading LoRA adapter: {lora_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float32,
        device_map="auto",
    )
    
    # Load and merge LoRA adapter
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, lora_path)
    
    # Merge LoRA weights into base model
    print("Merging LoRA weights...")
    model = model.merge_and_unload()
    
    # Create GGUF writer
    print("Creating GGUF file...")
    gguf_writer = gguf.GGUFWriter(output_path, arch="llama")
    
    # Add model parameters
    config = model.config
    gguf_writer.add_name("chitumbuka-translator")
    gguf_writer.add_context_length(config.max_position_embeddings)
    gguf_writer.add_embedding_length(config.hidden_size)
    gguf_writer.add_block_count(config.num_hidden_layers)
    gguf_writer.add_feed_forward_length(config.intermediate_size)
    gguf_writer.add_rope_dimension_count(config.hidden_size // config.num_attention_heads)
    gguf_writer.add_head_count(config.num_attention_heads)
    gguf_writer.add_head_count_kv(config.num_key_value_heads)
    gguf_writer.add_layer_norm_rms_eps(config.rms_norm_eps)
    gguf_writer.add_vocab_size(config.vocab_size)
    
    # Add tokenizer
    print("Adding tokenizer...")
    # Save tokenizer to temporary file and read it
    tokenizer.save_pretrained("../tmp/temp_tokenizer")
    with open("../tmp/temp_tokenizer/tokenizer.json", "r") as f:
        tokenizer_json = f.read()
    gguf_writer.add_tokenizer_pre(tokenizer_json)
    
    # Add model tensors
    print("Adding model tensors...")
    for name, param in model.named_parameters():
        if param.requires_grad:
            continue
        tensor = param.data
        # Convert to float32 if needed
        if tensor.dtype != torch.float32:
            tensor = tensor.to(torch.float32)
        gguf_writer.add_tensor(name, tensor)
    
    # Write the file
    print("Writing GGUF file...")
    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()
    
    print(f"GGUF file saved to: {output_path}")

if __name__ == "__main__":
    base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    lora_path = "../outputs/chitumbuka_lora/lora_adapter"
    output_path = "../adapters/chitumbuka_adapter.gguf"

    convert_lora_to_gguf(base_model, lora_path, output_path)
