from get_sleep_predictions import (
    setup_device,
    load_dataset,
    run_inference_and_collect_data,
    save_results_to_csv,
)
from model.llm.EmbedHealthFlamingo import EmbedHealthFlamingo


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
    results = run_inference_and_collect_data(
        model,
        dataset,
        len(dataset),  # eval whole dataset
        config["max_new_tokens"],
        config["random_seed"],
    )
    #     "sample_index": idx,
    # "eeg_data": eeg_data,
    # "ground_truth_label": ground_truth_label,
    # "predicted_label": predicted_label,
    # "rationale": rationale,
    # "full_prediction": prediction,
    # "series_length": len(eeg_data)

    correct = 0
    for r in results:
        predicted_label = r["predicted_label"]
        ground_truth_label = r["ground_truth_label"]

        print(f"'{predicted_label}' vs '{ground_truth_label}'")

        if predicted_label.strip() == ground_truth_label.strip():
            correct += 1

    print(f"Acc: {correct * 100 / len(results)}%")

    # # Save results
    # save_results_to_csv(results, config["output_path"])

    print("🎉 Sleep eval completed successfully!")


if __name__ == "__main__":
    main()
