#!/usr/bin/env python3
"""Parser for converting RTF-formatted JSONL files to clean format."""

import json
import re
from pathlib import Path

def calculate_accuracy_stats(data_points):
    """Calculate accuracy statistics from data points"""
    if not data_points:
        return {}
    
    total = len(data_points)
    correct = sum(1 for point in data_points if point.get("accuracy", False))
    accuracy_percentage = (correct / total) * 100 if total > 0 else 0
    
    return {
        "total_samples": total,
        "correct_predictions": correct,
        "incorrect_predictions": total - correct,
        "accuracy_percentage": accuracy_percentage
    }

def parse_rtf_jsonl(input_file, output_file=None):
    """Parse RTF-formatted JSONL file and extract JSON objects."""
    if output_file is None:
        input_path = Path(input_file)
        output_file = str(input_path.parent / f"{input_path.stem.split('.')[0]}.clean.jsonl")
    
    print(f"Parsing {input_file}")
    print(f"Output will be saved to {output_file}")
    
    with open(input_file, 'rb') as f:
        rtf_content = f.read().decode('utf-8', errors='ignore')
    
    extracted_data = extract_structured_data(rtf_content)
    
    if extracted_data:
        print(f"Extracted {len(extracted_data)} data points")
        
        # Calculate and display accuracy statistics
        stats = calculate_accuracy_stats(extracted_data)
        print(f"\nAccuracy Statistics:")
        print(f"Total samples: {stats['total_samples']}")
        print(f"Correct predictions: {stats['correct_predictions']}")
        print(f"Incorrect predictions: {stats['incorrect_predictions']}")
        print(f"Accuracy: {stats['accuracy_percentage']:.2f}%")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in extracted_data:
                f.write(json.dumps(item, indent=2) + "\n")
        
        print(f"\nData saved to {output_file}")
        return extracted_data
    else:
        print("No data could be extracted from the file.")
        return []

def extract_structured_data(rtf_content):
    """Extract structured data from RTF content"""
    data_points = []
    
    # Find key components
    generated_pattern = r'generated":\s*"(.*?)"'
    generated_matches = re.findall(generated_pattern, rtf_content)
    
    gold_pattern = r'gold":\s*"(.*?)"'
    gold_matches = re.findall(gold_pattern, rtf_content)
    
    min_length = min(len(generated_matches), len(gold_matches))
    
    for i in range(min_length):
        model_prediction = extract_answer(generated_matches[i])
        ground_truth = extract_answer(gold_matches[i])
        
        # Calculate accuracy (exact match)
        accuracy = model_prediction == ground_truth
        
        data_point = {
            "generated": generated_matches[i],
            "model_prediction": model_prediction,
            "ground_truth": ground_truth,
            "accuracy": accuracy
        }
        data_points.append(data_point)
    
    return data_points

def extract_answer(text):
    """Extract the final answer from text"""
    if "Answer: " not in text:
        return text
    
    answer = text.split("Answer: ")[-1].strip()
    answer = re.sub(r'<\|.*?\|>$', '', answer).strip()
    return answer

if __name__ == "__main__":
    current_dir = Path(__file__).parent
    input_file = current_dir / "newest_results.jsonl"
    clean_output = current_dir / "test_predictions_pamap.clean.jsonl"
    
    parse_rtf_jsonl(input_file, clean_output)
