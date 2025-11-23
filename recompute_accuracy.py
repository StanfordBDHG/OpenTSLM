#!/usr/bin/env python3
"""
Recompute accuracy with lenient parsing of option letters.

This script:
1. Parses generated text by removing spaces and extracting the option letter
2. Parses gold text by removing the end of text token and extracting the option letter
"""

import json
import re
import argparse


def extract_option_from_generated(text):
    """
    Extract option letter from generated text.

    Removes spaces and looks for patterns like:
    - "(a)", "(b)", etc.
    - "a)", "b)", etc. (missing opening paren)
    - "(a", "(b", etc. (missing closing paren)
    - "a", "b", etc. (just the letter)

    Args:
        text: Generated text that may contain an option

    Returns:
        Option letter (e.g., "a", "b") or None if not found
    """
    # Remove all whitespace
    cleaned = text.strip().replace(" ", "").replace("\n", "").replace("\t", "")

    # Try to find option patterns (case insensitive)
    # Pattern 1: (a), (b), etc.
    match = re.search(r'\(([a-h])\)', cleaned, re.IGNORECASE)
    if match:
        return match.group(1).lower()

    # Pattern 2: a), b), etc. (missing opening paren)
    match = re.search(r'^([a-h])\)', cleaned, re.IGNORECASE)
    if match:
        return match.group(1).lower()

    # Pattern 3: (a, (b, etc. (missing closing paren)
    match = re.search(r'\(([a-h])', cleaned, re.IGNORECASE)
    if match:
        return match.group(1).lower()

    # Pattern 4: Just the letter at the start
    match = re.search(r'^([a-h])(?:[^a-z]|$)', cleaned, re.IGNORECASE)
    if match:
        return match.group(1).lower()

    # Pattern 5: Letter anywhere in the text (last resort)
    match = re.search(r'([a-h])', cleaned, re.IGNORECASE)
    if match:
        return match.group(1).lower()

    return None


def extract_option_from_gold(text):
    """
    Extract option letter from gold answer.

    Removes <|end_of_text|> token and extracts the option letter.

    Args:
        text: Gold answer text

    Returns:
        Option letter (e.g., "a", "b") or None if not found
    """
    # Remove end of text token
    cleaned = text.replace("<|end_of_text|>", "").strip()

    # Remove all whitespace
    cleaned = cleaned.replace(" ", "").replace("\n", "").replace("\t", "")

    # Look for option pattern
    match = re.search(r'\(([a-h])\)', cleaned, re.IGNORECASE)
    if match:
        return match.group(1).lower()

    # Fallback: just look for the letter
    match = re.search(r'([a-h])', cleaned, re.IGNORECASE)
    if match:
        return match.group(1).lower()

    return None


def compute_accuracy(predictions_file):
    """
    Compute accuracy from predictions file with lenient parsing.

    Args:
        predictions_file: Path to test_predictions.jsonl file

    Returns:
        Dictionary with accuracy metrics and examples
    """
    correct = 0
    total = 0
    errors = []

    with open(predictions_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue

            sample = json.loads(line)

            # Extract options
            pred_option = extract_option_from_generated(sample['generated'])
            gold_option = extract_option_from_gold(sample['gold'])

            total += 1

            if pred_option == gold_option:
                correct += 1
            else:
                # Store first 10 errors for debugging
                if len(errors) < 10:
                    errors.append({
                        'line': line_num,
                        'gold_raw': sample['gold'],
                        'gold_parsed': gold_option,
                        'pred_raw': sample['generated'][:100],
                        'pred_parsed': pred_option,
                    })

    accuracy = correct / total if total > 0 else 0.0

    return {
        'total': total,
        'correct': correct,
        'accuracy': accuracy,
        'errors': errors
    }


def main():
    parser = argparse.ArgumentParser(description='Recompute accuracy with lenient parsing')
    parser.add_argument('predictions_file', type=str, help='Path to test_predictions.jsonl')
    args = parser.parse_args()

    print("=" * 80)
    print("RECOMPUTING ACCURACY WITH LENIENT PARSING")
    print("=" * 80)
    print(f"File: {args.predictions_file}")
    print()

    results = compute_accuracy(args.predictions_file)

    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Total samples: {results['total']}")
    print(f"Correct: {results['correct']}")
    print(f"Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print()

    if results['errors']:
        print("=" * 80)
        print(f"FIRST {len(results['errors'])} ERRORS")
        print("=" * 80)
        for i, err in enumerate(results['errors'], 1):
            print(f"\nError {i} (line {err['line']}):")
            print(f"  Gold raw:    {err['gold_raw']}")
            print(f"  Gold parsed: {err['gold_parsed']}")
            print(f"  Pred raw:    {err['pred_raw']}")
            print(f"  Pred parsed: {err['pred_parsed']}")


if __name__ == "__main__":
    main()
