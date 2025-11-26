#!/usr/bin/env python3
"""
Inference script for the fine-tuned LoRA model on sleep data.
Loads the base model + LoRA adapters and runs inference on new examples.
"""
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel
from PIL import Image
import numpy as np


def load_model_and_processor(base_model_id: str, lora_adapter_path: str):
    """Load the base model with LoRA adapters and processor."""
    print(f"Loading base model: {base_model_id}")
    
    # Load base model
    model = AutoModelForImageTextToText.from_pretrained(
        base_model_id,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    
    # Load LoRA adapters
    print(f"Loading LoRA adapters from: {lora_adapter_path}")
    model = PeftModel.from_pretrained(model, lora_adapter_path)
    model.eval()  # Set to evaluation mode
    
    # Load processor
    processor = AutoProcessor.from_pretrained(lora_adapter_path)
    
    print("Model and processor loaded successfully!")
    return model, processor


def run_inference(model, processor, messages, max_new_tokens=512, temperature=0.7):
    """
    Run inference on a single example.
    
    Args:
        model: The loaded model with LoRA adapters
        processor: The processor for tokenization and image processing
        messages: List of message dicts in chat format (can include images)
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
    
    Returns:
        Generated text response
    """
    # Extract images from messages
    images = []
    for msg in messages:
        content = msg.get("content", [])
        if not isinstance(content, list):
            content = [content]
        for element in content:
            if isinstance(element, dict) and ("image" in element or element.get("type") == "image"):
                image = element.get("image", element)
                if image is not None and hasattr(image, "convert"):
                    images.append(image.convert("RGB"))
    
    # Apply chat template
    text = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    
    # Process inputs
    inputs = processor(
        text=text,
        images=images if images else None,
        return_tensors="pt",
        padding=True
    )
    
    # Move to device
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True if temperature > 0 else False,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )
    
    # Decode output
    generated_text = processor.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the assistant's response (remove the prompt)
    prompt_text = processor.decode(inputs["input_ids"][0], skip_special_tokens=True)
    if generated_text.startswith(prompt_text):
        generated_text = generated_text[len(prompt_text):].strip()
    
    return generated_text


def create_sleep_plot_image(time_series_data):
    """
    Create a simple plot image from time series data.
    This is a placeholder - adapt to your actual data format.
    """
    import matplotlib.pyplot as plt
    import io
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time_series_data)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.set_title("Sleep Time Series")
    
    # Save to bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    
    # Convert to PIL Image
    image = Image.open(buf)
    return image


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run inference with LoRA model")
    parser.add_argument("--base-model", type=str, default="google/gemma-3-4b-pt")
    parser.add_argument("--lora-path", type=str, default="runs/gemma3-4b-pt-sleep-lora")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()
    
    # Configuration
    BASE_MODEL_ID = args.base_model
    LORA_ADAPTER_PATH = args.lora_path
    
    # Load model and processor
    model, processor = load_model_and_processor(BASE_MODEL_ID, LORA_ADAPTER_PATH)
    
    # Example 1: With image matching training prompt structure
    print("\n" + "="*80)
    print("Example 1: Sleep EEG Classification (matching training prompt)")
    print("="*80)
    
    # Create a dummy time series plot (replace with your actual data)
    dummy_data = np.random.randn(1000).cumsum()
    sleep_image = create_sleep_plot_image(dummy_data)
    
    # Match the exact prompt structure from training
    pre_prompt = """
        You are given a 30-second EEG time series segment. Your task is to classify the sleep stage based on analysis of the data.

        Instructions:
        - Analyze the data objectively without presuming a particular label.
        - Reason carefully and methodically about what the signal patterns suggest regarding sleep stage.
        - Write your reasoning as a single, coherent paragraph. Do not use bullet points, lists, or section headers.
        - Only reveal the correct class at the very end.
        - Never state that you are uncertain or unable to classify the data. You must always provide a rationale and a final answer.

        
        """
    
    post_prompt = """Possible sleep stages are:
        Wake, Non-REM stage 1, Non-REM stage 2, Non-REM stage 3, REM sleep, Movement

        - Please now write your rationale. Make sure that your last word is the answer. You MUST end your response with "Answer: """
    
    user_text = "\n\n".join([pre_prompt.strip(), post_prompt.strip()])
    
    messages_with_image = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful medical AI that analyzes sleep EEG."}],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image", "image": sleep_image},
            ]
        }
    ]
    
    response = run_inference(model, processor, messages_with_image, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
    print(f"\nResponse: {response}\n")
    
    print("\n" + "="*80)
    print("Inference completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
