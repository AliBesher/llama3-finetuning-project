#!/usr/bin/env python3
"""
🚀 Quick Start Script for LLAMA3 Fine-tuned Model
"""

from unsloth import FastLanguageModel
import torch

def load_model(model_name="YOUR_HF_USERNAME/llama3-finetuned"):
    """Load the fine-tuned model"""
    print("🦙 Loading LLAMA3 fine-tuned model...")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    
    FastLanguageModel.for_inference(model)
    print("✅ Model loaded successfully!")
    return model, tokenizer

def chat(model, tokenizer, message):
    """Generate response from the model"""
    messages = [{"role": "user", "content": message}]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    
    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=256,
        temperature=0.7,
        do_sample=True
    )
    
    new_tokens = outputs[0][inputs.shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return response

def main():
    """Main function for testing"""
    # Load model
    model, tokenizer = load_model()
    
    # Test cases
    test_messages = [
        "Hello! How are you?",
        "What is artificial intelligence?",
        "Write a short poem about technology",
        "Explain machine learning in simple terms"
    ]
    
    print("\n🧪 Testing the model...")
    print("=" * 50)
    
    for i, message in enumerate(test_messages, 1):
        print(f"\n{i}. 💬 User: {message}")
        response = chat(model, tokenizer, message)
        print(f"🤖 AI: {response}")
        print("-" * 30)

if __name__ == "__main__":
    main()
