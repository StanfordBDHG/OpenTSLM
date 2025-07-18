import json
import os
import re
import sys
from typing import Type

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers.pipelines import pipeline

# Add src to path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from time_series_datasets.pamap2.PAMAP2AccQADataset import PAMAP2AccQADataset
from time_series_datasets.TSQADataset import TSQADataset

MODEL_IDS: list[str] = [
    # "google/gemma-3n-e2b",
    # "google/gemma-3n-e2b-it",
    "meta-llama/Llama-3.2-1B"
]

DATASETS: list[Type[Dataset]] = [
    TSQADataset,
    #PAMAP2AccQADataset,
]


def evaluate_model_on_dataset(model_name: str, dataset_class: Type[Dataset]):
    print(
        f"Starting Baseline Test with model {model_name} on dataset {dataset_class.__name__}"
    )
    print("=" * 60)

    # Set device
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # Load model using pipeline
    print("Loading model using pipeline...")
    pipe = pipeline(
        task="text-generation",
        model=model_name,
        device=device,
        temperature=0.1,
        max_new_tokens=100,
        # torch_dtype=torch.bfloat16 if device == "cuda" else torch.float16,
    )
    print(f"Model loaded successfully: {model_name}")

    # quick test
    output = pipe("The capital of France is", max_new_tokens=20)
    print(output)

    # Load dataset
    print("Loading dataset...")

    def format_fun(arr):
        return (
            np.array2string(
                arr,
                separator=" ",
                formatter={"all": lambda x: f'"{x:.2f}"'.replace(".", "")},
                threshold=sys.maxsize,
                max_line_width=sys.maxsize,
            )
            .removeprefix("[")
            .removesuffix("]")
        )

    dataset = dataset_class(
        "test", "", format_sample_str=True, time_series_format_function=format_fun
    )
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

            # clean up prompt for TSQADataset
            pattern = r"This is the time series, it has mean (-?\d+\.\d{4}) and std (-?\d+\.\d{4})\."
            replacement = "This is the time series:"
            sample["prompt"] = re.sub(pattern, replacement, sample["prompt"])

            # Create input text
            input_text = sample["prompt"]
            target_answer = sample["answer"]

            # Generate prediction using pipeline
            outputs = pipe(
                input_text,
                max_new_tokens=100,
                return_full_text=False,
            )

            # Extract generated text
            if outputs and len(outputs) > 0:
                generated_text = outputs[0]["generated_text"].strip()
                successful_inferences += 1

                # Store results
                result = {
                    "sample_idx": idx,
                    "input_text": input_text,
                    "target_answer": target_answer,
                    "generated_answer": generated_text,
                }
                results.append(result)

                # Print progress for first few samples
                if idx < 5:
                    print(f"\nSAMPLE {idx + 1}:")
                    print(f"PROMPT: {sample['prompt'][:1000]}...")
                    print(f"ANSWER: {target_answer}")
                    print(f"OUTPUT: {generated_text}")
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

        # Calculate simple accuracy metrics (exact match and partial match)
        exact_matches = 0
        partial_matches = 0

        # TODO: refactor scoring
        for result in results:
            target = result["target_answer"].lower().strip()
            generated = result["generated_answer"].lower().strip()

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
        normalized_model_id = re.sub(r"[^a-z0-9]", "-", model_name.lower())
        normalized_dataset_name = re.sub(
            r"[^a-z0-9]", "-", dataset_class.__name__.lower()
        )
        results_file = f"baseline_test_results_{normalized_model_id}_{normalized_dataset_name}.json"
        with open(results_file, "w") as f:
            json.dump(
                {
                    "model_name": model_name,
                    "total_samples": total_samples,
                    "successful_inferences": successful_inferences,
                    "success_rate": success_rate,
                    "exact_accuracy": exact_accuracy,
                    "partial_accuracy": partial_accuracy,
                    "results": results,
                },
                f,
                indent=2,
            )

        print(f"\nDetailed results saved to: {results_file}")

    else:
        print("No successful inferences completed!")

    print("\nBaseline test completed!")


if __name__ == "__main__":
    for model_id in MODEL_IDS:
        for dataset_class in DATASETS:
            evaluate_model_on_dataset(model_id, dataset_class)
