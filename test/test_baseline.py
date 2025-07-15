import json
import os
import re
import sys
from typing import Type

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import pipeline

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from time_series_datasets.pamap2 import PAMAP2Dataset
from time_series_datasets.TSQADataset import TSQADataset

def format_time_series_for_text(time_series_data, time_series_text):
    """
    Convert time series data to a text representation that the language model can understand.
    """
    # Convert the time series to a readable format
    if isinstance(time_series_data[0], list):
        # Handle multiple time series (though TSQA typically has one)
        formatted_series = []
        for i, series in enumerate(time_series_data):
            series_str = ", ".join([f"{val:.4f}" for val in series[:50]])  # Limit length
            if len(series) > 50:
                series_str += "..."
            formatted_series.append(f"Series {i+1}: [{series_str}]")
        return "\n".join(formatted_series)
    else:
        # Single time series
        series_str = ", ".join([f"{val:.4f}" for val in time_series_data[:50]])
        if len(time_series_data) > 50:
            series_str += "..."
        return f"[{series_str}]"

def create_input_text(sample):
    """
    Create the full input text for the model from a TSQA sample.
    """
    pre_prompt = sample['pre_prompt']
    post_prompt = sample['post_prompt']
    time_series_text = sample['time_series_text'][0] if sample['time_series_text'] else ""
    time_series_data = sample['time_series'][0] if sample['time_series'] else []
    
    # Format the time series data
    formatted_series = format_time_series_for_text(time_series_data, time_series_text)
    
    # Combine all parts for text-only input (since we don't have images)
    input_text = f"{pre_prompt}\n\n{time_series_text}\n{formatted_series}\n\n{post_prompt}"
    
    return input_text

MODEL_IDS: list[str] = [
    "google/gemma-3n-e2b",
    #"google/gemma-3n-e2b-it",
]

DATASETS: list[Type[Dataset]] = [
    TSQADataset,
    #PAMAP2Dataset,
]

def evaluate_model_on_dataset(model_name: str, dataset_class: Type[Dataset]):
    print(f"Starting Baseline Test with model {model_name} on dataset {dataset_class.__name__}")
    print("=" * 60)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    

    # model = AutoModelForCausalLM.from_pretrained(model_name)

    # processor = AutoProcessor.from_pretrained(model_name)

    # prompt = "The capital of France is"
    # model_inputs = processor(text=prompt, return_tensors="pt").to(model.device)

    # input_len = model_inputs["input_ids"].shape[-1]

    # with torch.inference_mode():
    #     generation = model.generate(**model_inputs, max_new_tokens=10)
    #     generation = generation[0][input_len:]

    # decoded = processor.decode(generation, skip_special_tokens=True)
    # print(decoded)

    # Load model using pipeline
    print("Loading model using pipeline...")
    # Try to load the gemma-3n-e2b model
    pipe = pipeline(
        model=model_name,
        device=device,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float16,
    )
    print(f"Model loaded successfully: {model_name}")

    output = pipe("The capital of France is", max_new_tokens=20, return_full_text=False)
    print(output)
        
    
    # Load dataset
    print("Loading dataset...")
    dataset = dataset_class("test", "")
    print(f"Loaded {len(dataset)} test samples")
    
    # Initialize metrics
    total_samples = 0
    successful_inferences = 0
    
    # Results storage
    results = []
    
    print("\nRunning inference on all samples...")
    print("=" * 80)
    
    # TODO: Process samples (limit to first X for faster testing)
    max_samples = min(1, len(dataset))
    print(f"Processing first {max_samples} samples for baseline test...")
    
    # Process each sample
    for idx in tqdm(range(max_samples), desc="Processing samples"):
        try:
            sample = dataset[idx]
            
            # Create input text
            input_text = create_input_text(sample)
            target_answer = sample['answer']
            
            # Generate prediction using pipeline
            outputs = pipe(
                input_text,
                max_new_tokens=50,
                return_full_text=False,
            )
            
            # Extract generated text
            if outputs and len(outputs) > 0:
                generated_text = outputs[0]['generated_text'].strip()
                successful_inferences += 1
                
                # Store results
                result = {
                    'sample_idx': idx,
                    'input_text': input_text,
                    'target_answer': sample['answer'],
                    'generated_answer': generated_text,
                    'post_prompt': sample['post_prompt']
                }
                results.append(result)
                
                # Print progress for first few samples
                if idx < 5:
                    print(f"\nSample {idx + 1}:")
                    print(f"Question: {sample['pre_prompt'][:1000]}...")
                    print(f"Target: {target_answer}")
                    print(f"Generated: {generated_text}")
                    print("=" * 80)
            
            total_samples += 1
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue
    
    # Calculate final metrics
    if successful_inferences > 0:
        success_rate = successful_inferences / total_samples
        
        print("\n" + "=" * 80)
        print("BASELINE TEST RESULTS")
        print("=" * 80)
        print(f"Model: {model_name}")
        print(f"Total samples processed: {total_samples}")
        print(f"Successful inferences: {successful_inferences}")
        print(f"Success rate: {success_rate:.2%}")
        
        # Analyze by task type
        task_counts = {}
        for result in results:
            task = result['post_prompt']
            if task not in task_counts:
                task_counts[task] = 0
            task_counts[task] += 1
        
        print("\nSamples by task type:")
        for task, count in task_counts.items():
            print(f"  {task}: {count} samples")
        
        # Calculate simple accuracy metrics (exact match and partial match)
        exact_matches = 0
        partial_matches = 0
        
        # TODO: refactor scoring
        for result in results:
            target = result['target_answer'].lower().strip()
            generated = result['generated_answer'].lower().strip()
            
            if target == generated:
                exact_matches += 1
            elif target in generated or generated in target:
                partial_matches += 1
        
        exact_accuracy = exact_matches / successful_inferences
        partial_accuracy = (exact_matches + partial_matches) / successful_inferences
        
        print("\nAccuracy Metrics:")
        print(f"  Exact match accuracy: {exact_accuracy:.2%}")
        print(f"  Partial match accuracy: {partial_accuracy:.2%}")
        
        # Save detailed results
        normalized_model_id = re.sub(r'[^a-z0-9]', '-', model_name.lower())
        normalized_dataset_name = re.sub(r'[^a-z0-9]', '-', dataset_class.__name__.lower())
        results_file = f"baseline_test_results_{normalized_model_id}_{normalized_dataset_name}.json"
        with open(results_file, 'w') as f:
            json.dump({
                'model_name': model_name,
                'total_samples': total_samples,
                'successful_inferences': successful_inferences,
                'success_rate': success_rate,
                'exact_accuracy': exact_accuracy,
                'partial_accuracy': partial_accuracy,
                'task_counts': task_counts,
                'results': results
            }, f, indent=2)
        
        print(f"\nDetailed results saved to: {results_file}")
        
    else:
        print("No successful inferences completed!")
    
    print("\nBaseline test completed!")

if __name__ == "__main__":
    for model_id in MODEL_IDS:
        for dataset_class in DATASETS:
            evaluate_model_on_dataset(model_id, dataset_class)
