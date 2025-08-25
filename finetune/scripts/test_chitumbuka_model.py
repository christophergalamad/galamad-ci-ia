#!/usr/bin/env python3
"""
Test script for the trained Chitumbuka LoRA model
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def load_model():
    """Load the base model with LoRA adapter"""
    base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    lora_path = "../outputs/chitumbuka_lora/lora_adapter"
    
    print("Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float32,
        device_map="auto",
    )
    
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, lora_path)
    
    return model, tokenizer

def translate_to_chitumbuka(model, tokenizer, english_text):
    """Translate English text to Chitumbuka"""
    prompt = f"""You are a helpful translator. Translate the following English phrase into Chitumbuka.
English: {english_text}
Chitumbuka:"""
    
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Move inputs to the same device as the model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.2,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the Chitumbuka translation part
    chitumbuka_part = response.split("Chitumbuka:")[-1].strip()
    
    return chitumbuka_part

def main():
    print("Loading Chitumbuka translation model...")
    model, tokenizer = load_model()
    
    # Test translations
    test_phrases = [
        "Hello, how are you?",
        "What is your name?",
        "I am hungry",
        "Thank you very much",
        "Good morning"
    ]
    
    print("\n=== Chitumbuka Translation Test ===\n")
    
    for phrase in test_phrases:
        print(f"English: {phrase}")
        translation = translate_to_chitumbuka(model, tokenizer, phrase)
        print(f"Chitumbuka: {translation}")
        print("-" * 50)
    
    # Interactive mode
    print("\n=== Interactive Translation Mode ===")
    print("Type 'quit' to exit")
    
    while True:
        user_input = input("\nEnter English text to translate: ").strip()
        if user_input.lower() == 'quit':
            break
        
        if user_input:
            translation = translate_to_chitumbuka(model, tokenizer, user_input)
            print(f"Translation: {translation}")

if __name__ == "__main__":
    main()
