from get_sleep_predictions import (
    setup_device,
    load_dataset,
    extract_sleep_label,
)
from model.llm.EmbedHealthFlamingo import EmbedHealthFlamingo
import random
import torch
import numpy as np
import re
from prompt.full_prompt import FullPrompt
from prompt.text_prompt import TextPrompt
from prompt.text_time_series_prompt import TextTimeSeriesPrompt


def load_model(model_path: str, device: str, llm_id: str = "meta-llama/Llama-3.2-1B"):
    """Load the trained EmbedHealthFlamingo model."""
    print(f"Loading model from {model_path}...")

    model = EmbedHealthFlamingo(
        device=device,
        llm_id=llm_id,
    )

    model.load_from_file(model_path)
    model.eval()
    print("✅ Model loaded successfully")
    return model


def main():
    """Main function to run the evaluation."""
    print("🚀 Starting Sleep evaluation ...")
    print("=" * 60)

    # Configuration - adjust these parameters as needed
    config = {
        "model_path": "results/best_model.pt",  # Path to your trained model
        "output_path": "sleep_eval_predications.csv",  # Output CSV file
        "llm_id": "meta-llama/Llama-3.2-1B",  # LLM ID used for training
        "dataset_split": "test",  # Dataset split to use: "train", "validation", or "test"
        "max_new_tokens": 400,  # Maximum tokens to generate
        "random_seed": 42,  # Random seed for reproducibility
    }

    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    # Setup
    device = setup_device()

    # Load model
    model = load_model(config["model_path"], device, config["llm_id"])

    # Load dataset
    dataset = load_dataset(split=config["dataset_split"])

    # Run inference and collect data
    # results = run_inference_and_collect_data(
    #     model,
    #     dataset,
    #     len(dataset),  # eval whole dataset
    #     config["max_new_tokens"],
    #     config["random_seed"],
    # )
    #     "sample_index": idx,
    # "eeg_data": eeg_data,
    # "ground_truth_label": ground_truth_label,
    # "predicted_label": predicted_label,
    # "rationale": rationale,
    # "full_prediction": prediction,
    # "series_length": len(eeg_data)

    random_seed = config["random_seed"]
    max_new_tokens = config["max_new_tokens"]
    num_samples = len(dataset)

    # Set random seed for reproducibility
    random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Select random indices
    dataset_size = len(dataset)
    selected_indices = random.sample(
        range(dataset_size), min(num_samples, dataset_size)
    )

    results = []

    correct = 0
    labels = list(set(d["label"] for d in dataset))
    with torch.no_grad():
        for i, idx in enumerate(selected_indices):
            print(f"Processing sample {i + 1}/{len(selected_indices)} (index {idx})...")

            # Get the sample
            row = dataset[idx]

            # Extract raw time series data
            original_data = row.get("original_data", [])
            if len(original_data) > 0:
                eeg_data = original_data  # Original EEG data
            else:
                raise RuntimeError(f"No original data found for sample {idx}")

            # Get ground truth label and rationale
            ground_truth_label = row["label"]
            rationale = row["answer"]

            # Run inference to get prediction
            try:
                # Build the prompt for inference
                pre_prompt = TextPrompt("""You are given a 30-second EEG time series segment. Your task is to classify the sleep stage based on analysis of the data.

                Instructions:
                - Analyze the data objectively without presuming a particular label.
                - Reason carefully and methodically about what the signal patterns suggest regarding sleep stage.
                - Write your reasoning as a single, coherent paragraph. Do not use bullet points, lists, or section headers. You must always provide a rationale and a final answer.""")

                post_prompt = TextPrompt(f"""Possible sleep stages are: 
                Wake, Non-REM stage 1, Non-REM stage 2, Non-REM stage 3, REM sleep, Movement

                Please now write your answer. Make sure that your last word is the answer. The possible labels are: {labels}. You MUST end your response with "Answer:" follwed by the respective label.""")

                # Create time series prompts using the data from the dataset
                ts_prompts = []
                for ts_text, ts_data in zip(
                    row["time_series_text"], row["time_series"]
                ):
                    # fix no normalization
                    mean = np.mean(ts_data)
                    std = np.std(ts_data)
                    normalized_ts_data = (ts_data - mean) / std

                    # fix 0.000 mean and std
                    ts_text = re.sub(
                        r"mean -?\d+\.?\d+",
                        f"mean {np.mean(normalized_ts_data):.3f}",
                        ts_text,
                    )
                    ts_text = re.sub(
                        r"std -?\d+\.?\d+",
                        f"std {np.mean(normalized_ts_data):.3f}",
                        ts_text,
                    )

                    ts_prompts.append(TextTimeSeriesPrompt(ts_text, normalized_ts_data))

                # Create full prompt
                prompt = FullPrompt(pre_prompt, ts_prompts, post_prompt)

                # Run inference
                prediction = model.eval_prompt(prompt, max_new_tokens=max_new_tokens)
                predicted_label = extract_sleep_label(prediction)

                print("prompt", prompt.to_dict())
                print("predicition", prediction)

                result = {
                    "sample_index": idx,
                    "eeg_data": eeg_data,
                    "ground_truth_label": ground_truth_label,
                    "predicted_label": predicted_label,
                    "rationale": rationale,
                    "full_prediction": prediction,
                    "series_length": len(eeg_data),
                }

                results.append(result)
                print(f"  Ground truth: {ground_truth_label}")
                print(f"  Prediction: {predicted_label}")
                if predicted_label.strip() == ground_truth_label.strip():
                    correct += 1

            except Exception as e:
                print(f"  ❌ Error processing sample {idx}: {e}")
                continue

    print(f"Acc: {correct * 100 / len(dataset)}%")

    # # Save results
    # save_results_to_csv(results, config["output_path"])

    print("🎉 Sleep eval completed successfully!")


if __name__ == "__main__":
    main()
