#!/usr/bin/env python3
"""
Recompute TimeSeriesExam accuracy by mapping option letters to option texts.

The model predicts option letters (e.g., "(a)", "(b)") but the gold answers
are the actual text of the options. This script maps the letters to texts
according to the prompts and recomputes accuracy.
"""

import json
import sys
import os
import re
from datasets import load_dataset

def extract_option_letter(prediction):
    """Extract option letter from prediction string.

    Examples:
        " (b)" -> "b"
        "(a)" -> "a"
        " (c) some text" -> "c"
    """
    # Look for pattern like (a), (b), (c), etc.
    match = re.search(r'\(([a-h])\)', prediction.lower())
    if match:
        return match.group(1)
    return None


def load_test_dataset():
    """Load TimeSeriesExam1 test dataset with options."""
    print("Loading TimeSeriesExam1 dataset...")

    # Load full dataset
    ds_full = load_dataset("AutonLab/TimeSeriesExam1", split="test")

    # Apply same split as in TimeSeriesExam1QADataset
    TEST_FRAC = 0.2
    VAL_FRAC = 0.1

    # First carve out the test split
    train_val, test = ds_full.train_test_split(test_size=TEST_FRAC, seed=42).values()

    # From the remaining data, take validation
    val_frac_adj = VAL_FRAC / (1.0 - TEST_FRAC)
    train, val = train_val.train_test_split(test_size=val_frac_adj, seed=43).values()

    return test


def map_letter_to_option_text(letter, options):
    """Map option letter (a, b, c, ...) to option text.

    Args:
        letter: Option letter like "a", "b", "c"
        options: List of option texts

    Returns:
        The corresponding option text or None if not found
    """
    option_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

    if letter in option_labels:
        idx = option_labels.index(letter)
        if idx < len(options):
            return options[idx]

    return None


def recompute_accuracy(predictions_file, output_dir):
    """Recompute accuracy with proper option mapping.

    Args:
        predictions_file: Path to predictions JSONL file
        output_dir: Directory to save corrected results
    """
    # Load test dataset to get options
    test_dataset = load_test_dataset()

    # Load predictions
    print(f"Loading predictions from: {predictions_file}")
    predictions = []
    with open(predictions_file, 'r') as f:
        for line in f:
            predictions.append(json.loads(line))

    print(f"Loaded {len(predictions)} predictions")
    print(f"Test dataset has {len(test_dataset)} samples")

    if len(predictions) != len(test_dataset):
        print(f"WARNING: Number of predictions ({len(predictions)}) doesn't match test dataset size ({len(test_dataset)})")

    # Recompute accuracy
    correct = 0
    total = 0

    # Save detailed results
    os.makedirs(output_dir, exist_ok=True)
    detailed_file = os.path.join(output_dir, "detailed_predictions.jsonl")

    with open(detailed_file, 'w') as f:
        for i, (pred_entry, dataset_row) in enumerate(zip(predictions, test_dataset)):
            prediction = pred_entry['prediction']
            gold_text = pred_entry['gold'].replace('<|end_of_text|>', '').strip()

            # Extract predicted letter
            pred_letter = extract_option_letter(prediction)

            # Get options from dataset
            options = dataset_row['options']
            answer_text = dataset_row['answer']

            # Map prediction letter to option text
            if pred_letter:
                pred_option_text = map_letter_to_option_text(pred_letter, options)
            else:
                pred_option_text = None

            # Check if correct
            is_correct = False
            if pred_option_text and answer_text:
                # Compare the mapped option text with the dataset answer
                if pred_option_text.strip() == answer_text.strip():
                    is_correct = True
                    correct += 1

            total += 1

            # Save detailed entry
            detailed_entry = {
                "index": i,
                "question": dataset_row['question'],
                "options": options,
                "prediction_raw": prediction,
                "prediction_letter": pred_letter,
                "prediction_mapped_text": pred_option_text,
                "gold_text": gold_text,
                "dataset_answer": answer_text,
                "correct": is_correct
            }
            f.write(json.dumps(detailed_entry) + '\n')

    accuracy = correct / total if total > 0 else 0.0

    print("\n" + "=" * 80)
    print("CORRECTED EVALUATION RESULTS")
    print("=" * 80)
    print(f"Total samples: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("=" * 80)

    # Save corrected results
    results_file = os.path.join(output_dir, "corrected_results.json")
    results = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "original_predictions_file": predictions_file
    }

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Corrected results saved to: {results_file}")
    print(f"✅ Detailed predictions saved to: {detailed_file}")

    # Show some examples
    print("\n" + "=" * 80)
    print("SAMPLE PREDICTIONS (first 5)")
    print("=" * 80)

    with open(detailed_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= 5:
                break
            entry = json.loads(line)
            print(f"\n[Sample {i+1}] {'✓ CORRECT' if entry['correct'] else '✗ WRONG'}")
            print(f"Question: {entry['question'][:100]}...")
            print(f"Options: {entry['options']}")
            print(f"Prediction: {entry['prediction_raw']} -> letter: {entry['prediction_letter']} -> text: {entry['prediction_mapped_text']}")
            print(f"Gold: {entry['dataset_answer']}")

    return accuracy


if __name__ == "__main__":
    predictions_file = "results_interleaved/interleaved_session_20251120_220200/timeseriesexam_predictions.jsonl"
    output_dir = "results_interleaved/interleaved_session_20251120_220200/corrected_evaluation"

    if len(sys.argv) > 1:
        predictions_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]

    if not os.path.exists(predictions_file):
        print(f"Error: Predictions file not found: {predictions_file}")
        sys.exit(1)

    recompute_accuracy(predictions_file, output_dir)
