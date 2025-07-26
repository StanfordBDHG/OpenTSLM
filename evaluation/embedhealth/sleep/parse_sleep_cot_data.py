#!/usr/bin/env python3
"""Parser for converting sleep COT JSONL files to clean format."""

import json
import re
from pathlib import Path
from collections import Counter

def calculate_f1_score(prediction, ground_truth):
    """Calculate F1 score for classification labels"""
    # Normalize labels for comparison (lowercase, strip whitespace)
    pred_normalized = prediction.lower().strip()
    truth_normalized = ground_truth.lower().strip()
    
    # For single prediction vs single ground truth, F1 is binary
    f1 = 1.0 if pred_normalized == truth_normalized else 0.0
    
    return {
        'f1_score': f1,
        'precision': f1,  # For single-label classification, precision = recall = f1
        'recall': f1,
        'prediction_normalized': pred_normalized,
        'ground_truth_normalized': truth_normalized
    }

def calculate_f1_stats(data_points):
    """Calculate both macro-F1 and average F1 (micro-F1) statistics"""
    if not data_points:
        return {}
    
    # Calculate average F1 (micro-F1) - simple average across all predictions
    f1_scores = [point.get("f1_score", 0) for point in data_points]
    average_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
    
    # Group by ground truth class for macro-F1
    class_predictions = {}
    for point in data_points:
        gt_class = point.get("ground_truth_normalized", "")
        pred_class = point.get("prediction_normalized", "")
        
        if gt_class not in class_predictions:
            class_predictions[gt_class] = {"tp": 0, "fp": 0, "fn": 0}
        
        # True positive: prediction matches ground truth
        if pred_class == gt_class:
            class_predictions[gt_class]["tp"] += 1
        else:
            # False negative: ground truth class was not predicted
            class_predictions[gt_class]["fn"] += 1
            # False positive: predicted class that wasn't ground truth
            if pred_class in class_predictions:
                class_predictions[pred_class]["fp"] += 1
            else:
                class_predictions[pred_class] = {"tp": 0, "fp": 1, "fn": 0}
    
    # Calculate F1 per class
    class_f1_scores = {}
    total_f1 = 0
    valid_classes = 0
    
    for class_name, counts in class_predictions.items():
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        class_f1_scores[class_name] = {
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "tp": tp,
            "fp": fp,
            "fn": fn
        }
        
        total_f1 += f1
        valid_classes += 1
    
    # Calculate macro-F1 (average across all classes)
    macro_f1 = total_f1 / valid_classes if valid_classes > 0 else 0
    
    return {
        "average_f1": average_f1,
        "macro_f1": macro_f1,
        "class_f1_scores": class_f1_scores,
        "total_classes": valid_classes
    }

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

def parse_sleep_cot_jsonl(input_file, output_file=None):
    """Parse sleep COT JSONL file and extract JSON objects."""
    if output_file is None:
        input_path = Path(input_file)
        output_file = str(input_path.parent / f"{input_path.stem}.clean.jsonl")
    
    print(f"Parsing {input_file}")
    print(f"Output will be saved to {output_file}")
    
    extracted_data = extract_structured_data(input_file)
    
    if extracted_data:
        print(f"Extracted {len(extracted_data)} data points")
        
        # Calculate and display accuracy statistics
        accuracy_stats = calculate_accuracy_stats(extracted_data)
        print(f"\nAccuracy Statistics:")
        print(f"Total samples: {accuracy_stats['total_samples']}")
        print(f"Correct predictions: {accuracy_stats['correct_predictions']}")
        print(f"Incorrect predictions: {accuracy_stats['incorrect_predictions']}")
        print(f"Accuracy: {accuracy_stats['accuracy_percentage']:.2f}%")
        
        # Calculate and display F1 statistics
        f1_stats = calculate_f1_stats(extracted_data)
        print(f"\nF1 Score Statistics:")
        print(f"Average F1 Score: {f1_stats['average_f1']:.4f}")
        print(f"Macro-F1 Score: {f1_stats['macro_f1']:.4f}")
        print(f"Total Classes: {f1_stats['total_classes']}")
        
        # Display per-class F1 scores
        if f1_stats['class_f1_scores']:
            print(f"\nPer-Class F1 Scores:")
            for class_name, scores in f1_stats['class_f1_scores'].items():
                print(f"  {class_name}: F1={scores['f1']:.4f}, P={scores['precision']:.4f}, R={scores['recall']:.4f}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in extracted_data:
                f.write(json.dumps(item, indent=2) + "\n")
        
        print(f"\nData saved to {output_file}")
        return extracted_data
    else:
        print("No data could be extracted from the file.")
        return []

def extract_structured_data(input_file):
    """Extract structured data from JSONL file"""
    data_points = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                # Parse JSON line
                data = json.loads(line.strip())
                
                # Extract generated and gold fields
                generated_text = data.get('generated', '')
                gold_text = data.get('gold', '')
                
                # Extract answers from both fields
                model_prediction = extract_answer(generated_text)
                ground_truth = extract_answer(gold_text)
                
                # Calculate accuracy (exact match)
                accuracy = model_prediction == ground_truth
                
                # Calculate F1 score
                f1_result = calculate_f1_score(model_prediction, ground_truth)
                
                data_point = {
                    "generated": generated_text,
                    "model_prediction": model_prediction,
                    "ground_truth": ground_truth,
                    "accuracy": accuracy,
                    "f1_score": f1_result['f1_score'],
                    "precision": f1_result['precision'],
                    "recall": f1_result['recall'],
                    "prediction_normalized": f1_result['prediction_normalized'],
                    "ground_truth_normalized": f1_result['ground_truth_normalized'],
                    "line_number": line_num
                }
                data_points.append(data_point)
                
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Unexpected error on line {line_num}: {e}")
                continue
    
    return data_points

def extract_answer(text):
    """Extract the final answer from text"""
    if "Answer: " not in text:
        return text
    
    answer = text.split("Answer: ")[-1].strip()
    # Remove any end-of-text tokens
    answer = re.sub(r'<\|.*?\|>$', '', answer).strip()
    # Remove trailing periods and normalize
    answer = re.sub(r'\.$', '', answer).strip()
    return answer

if __name__ == "__main__":
    current_dir = Path(__file__).parent
    input_file = current_dir / "flamingo_predictions.jsonl"
    clean_output = current_dir / "flamingo_predictions.clean.jsonl"
    
    parse_sleep_cot_jsonl(input_file, clean_output)
