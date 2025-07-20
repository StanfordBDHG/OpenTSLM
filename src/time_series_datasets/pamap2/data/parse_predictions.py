#!/usr/bin/env python3
"""
Parser for converting RTF-formatted JSONL files to clean, human-readable format.
Specifically designed for PAMAP2 prediction files.
"""

import json
import os
import re
from pathlib import Path

def parse_rtf_jsonl(input_file, output_file=None):
    """
    Parse an RTF-formatted JSONL file and extract the JSON objects.
    
    Args:
        input_file (str): Path to the RTF-formatted JSONL file
        output_file (str, optional): Path to save the cleaned JSON data.
            If None, will use the input filename with '.clean.jsonl' extension.
    
    Returns:
        list: List of parsed JSON objects
    """
    # Define output file if not provided
    if output_file is None:
        input_path = Path(input_file)
        output_file = str(input_path.parent / f"{input_path.stem.split('.')[0]}.clean.jsonl")
    
    print(f"Parsing {input_file}")
    print(f"Output will be saved to {output_file}")
    
    # Read the RTF file as binary to avoid encoding issues
    with open(input_file, 'rb') as f:
        rtf_content = f.read().decode('utf-8', errors='ignore')
    
    # Create a human-readable summary file
    summary_file = str(Path(output_file).parent / f"{Path(output_file).stem}.summary.txt")
    
    # Extract content using a simpler approach
    # First, find all occurrences of the pattern that indicates a JSON object
    pattern = r'\{"pre_prompt":.*?"gold":.*?"\}\\'
    
    # We'll use a different approach - extract content line by line
    lines = rtf_content.split('\n')
    json_lines = []
    
    # Find lines that contain JSON data
    for line in lines:
        if '"pre_prompt":' in line and '"time_series_text":' in line and '"generated":' in line:
            # This looks like a line with JSON data
            # Clean up RTF formatting
            cleaned_line = line.strip()
            # Remove RTF control sequences and escape characters
            cleaned_line = re.sub(r'\\[a-z0-9]+\s?', '', cleaned_line)
            cleaned_line = cleaned_line.replace('\\', '')
            
            # Extract the JSON part
            json_start = cleaned_line.find('{')
            json_end = cleaned_line.rfind('}')
            
            if json_start != -1 and json_end != -1 and json_end > json_start:
                json_str = cleaned_line[json_start:json_end+1]
                json_lines.append(json_str)
    
    print(f"Found {len(json_lines)} potential JSON lines")
    
    # Parse the JSON objects
    parsed_objects = []
    for i, json_str in enumerate(json_lines):
        try:
            # Try to fix common issues with the JSON
            # Replace escaped newlines with actual newlines
            json_str = json_str.replace('\\n', '\n')
            # Fix escaped quotes
            json_str = json_str.replace('\\"', '"')
            # Fix double-escaped quotes
            json_str = json_str.replace('\\\\"', '\\"')
            
            obj = json.loads(json_str)
            parsed_objects.append(obj)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON #{i}: {e}")
            # Try a more aggressive cleaning approach for problematic JSONs
            try:
                # Remove all backslashes except those that are part of escape sequences
                cleaned_json = re.sub(r'\\(?!["\\bfnrt/])', '', json_str)
                obj = json.loads(cleaned_json)
                parsed_objects.append(obj)
                print(f"Successfully parsed JSON #{i} after additional cleaning")
            except json.JSONDecodeError:
                print(f"Failed to parse JSON #{i} even after additional cleaning")
    
    if not parsed_objects:
        print("No JSON objects could be parsed from the file.")
        
        # Try a completely different approach - extract just the key fields
        # This is a fallback method that extracts structured data without parsing full JSON
        print("Attempting to extract structured data without full JSON parsing...")
        
        extracted_data = extract_structured_data(rtf_content)
        
        if extracted_data:
            print(f"Extracted {len(extracted_data)} data points using fallback method")
            
            # Save the extracted data
            with open(output_file, 'w', encoding='utf-8') as f:
                for item in extracted_data:
                    f.write(json.dumps(item, indent=2) + "\n")
            
            print(f"Extracted data saved to {output_file}")
            
            # Create a summary
            create_summary(extracted_data, summary_file)
            
            return extracted_data
        
        return []
    
    print(f"Successfully parsed {len(parsed_objects)} JSON objects")
    
    # Save to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for obj in parsed_objects:
            # Format with only the fields the user is interested in
            formatted_obj = {
                "generated": obj.get("generated", ""),  # Include the raw generated text
                "model_prediction": extract_answer(obj.get("generated", "")),  # Just the final answer
                "ground_truth": extract_answer(obj.get("gold", ""))  # Just the final answer from gold
            }
            f.write(json.dumps(formatted_obj, indent=2) + "\n")
    
    print(f"Formatted data saved to {output_file}")
    
    # Create a summary
    create_summary(parsed_objects, summary_file)
    
    return parsed_objects

def extract_structured_data(rtf_content):
    """Extract structured data from RTF content without full JSON parsing"""
    data_points = []
    
    # Find all pre_prompts
    pre_prompt_pattern = r'pre_prompt":\s*"(.*?)"'
    pre_prompts = re.findall(pre_prompt_pattern, rtf_content)
    
    # Find all time_series_text arrays
    time_series_pattern = r'time_series_text":\s*\[(.*?)\]'
    time_series_matches = re.findall(time_series_pattern, rtf_content)
    
    # Find all generated responses
    generated_pattern = r'generated":\s*"(.*?)"'
    generated_matches = re.findall(generated_pattern, rtf_content)
    
    # Find all gold responses
    gold_pattern = r'gold":\s*"(.*?)"'
    gold_matches = re.findall(gold_pattern, rtf_content)
    
    # Match them up
    min_length = min(len(pre_prompts), len(time_series_matches), 
                     len(generated_matches), len(gold_matches))
    
    for i in range(min_length):
        # Extract time series stats
        time_series_text = []
        time_series_str = time_series_matches[i]
        # Extract the quoted strings from the time_series array
        ts_items = re.findall(r'"(.*?)"', time_series_str)
        time_series_text = ts_items
        
        data_point = {
            "generated": generated_matches[i],  # Include the raw generated text
            "model_prediction": extract_answer(generated_matches[i]),  # Just the final answer
            "ground_truth": extract_answer(gold_matches[i])  # Just the final answer from gold
        }
        data_points.append(data_point)
    
    return data_points

def extract_activity_labels(pre_prompt):
    """Extract the list of activity labels from the pre_prompt"""
    if "Possible activity labels are:" not in pre_prompt:
        return []
    
    labels_section = pre_prompt.split("Possible activity labels are:")[1].split("-")[0].strip()
    return [label.strip() for label in labels_section.split(",")]

def extract_stats(time_series_text):
    """Extract mean and std from time series description"""
    stats = {}
    if not time_series_text:
        return stats
    
    mean_match = re.search(r'mean ([-\d\.]+)', time_series_text)
    std_match = re.search(r'std ([-\d\.]+)', time_series_text)
    
    if mean_match:
        stats["mean"] = float(mean_match.group(1))
    if std_match:
        stats["std"] = float(std_match.group(1))
    
    return stats

def extract_answer(text):
    """Extract the final answer from the generated or gold text"""
    if "Answer: " not in text:
        return text
    
    answer = text.split("Answer: ")[-1].strip()
    # Remove any trailing markers like <|end_of_text|>
    answer = re.sub(r'<\|.*?\|>$', '', answer).strip()
    return answer

def create_summary(parsed_objects, output_file):
    """
    Create a human-readable summary of the predictions.
    
    Args:
        parsed_objects (list): List of parsed JSON objects
        output_file (str): Path to save the summary
    """
    # Create summary
    correct_count = 0
    total_count = len(parsed_objects)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"PAMAP2 Prediction Summary\n")
        f.write(f"======================\n\n")
        f.write(f"Total predictions: {total_count}\n\n")
        
        f.write("Sample Predictions:\n")
        f.write("------------------\n\n")
        
        # Show first 5 examples (or fewer if there are less than 5)
        for i, obj in enumerate(parsed_objects[:min(5, total_count)]):
            f.write(f"Example {i+1}:\n")
            
            # Handle both formats - direct time_series_stats or nested in formatted_obj
            x_axis_stats = obj.get('time_series_stats', {}).get('x_axis', {})
            y_axis_stats = obj.get('time_series_stats', {}).get('y_axis', {})
            z_axis_stats = obj.get('time_series_stats', {}).get('z_axis', {})
            
            f.write(f"  X-axis: mean={x_axis_stats.get('mean', 'N/A')}, std={x_axis_stats.get('std', 'N/A')}\n")
            f.write(f"  Y-axis: mean={y_axis_stats.get('mean', 'N/A')}, std={y_axis_stats.get('std', 'N/A')}\n")
            f.write(f"  Z-axis: mean={z_axis_stats.get('mean', 'N/A')}, std={z_axis_stats.get('std', 'N/A')}\n")
            
            model_pred = obj.get('model_prediction', 'unknown')
            ground_truth = obj.get('ground_truth', 'unknown')
            
            f.write(f"  Prediction: {model_pred}\n")
            f.write(f"  Ground Truth: {ground_truth}\n")
            f.write(f"  Correct: {model_pred == ground_truth}\n\n")
            
            if model_pred == ground_truth:
                correct_count += 1
        
        # Calculate accuracy for all examples
        if total_count > 5:
            for i in range(5, total_count):
                model_pred = parsed_objects[i].get('model_prediction', 'unknown')
                ground_truth = parsed_objects[i].get('ground_truth', 'unknown')
                if model_pred == ground_truth:
                    correct_count += 1
        
        accuracy = correct_count / total_count if total_count > 0 else 0
        f.write(f"Overall Accuracy: {accuracy:.2%} ({correct_count}/{total_count})\n")
        
        # Count predictions by activity
        activity_counts = {}
        for obj in parsed_objects:
            activity = obj.get('model_prediction', 'unknown')
            if activity not in activity_counts:
                activity_counts[activity] = 0
            activity_counts[activity] += 1
        
        f.write("\nPredictions by Activity:\n")
        for activity, count in sorted(activity_counts.items(), key=lambda x: x[1], reverse=True):
            f.write(f"  {activity}: {count} ({count/total_count:.2%})\n")
    
    print(f"Summary saved to {output_file}")


def create_human_readable_summary(input_file, output_file=None):
    """
    Create a human-readable summary of the predictions from a clean JSONL file.
    
    Args:
        input_file (str): Path to the clean JSONL file
        output_file (str, optional): Path to save the summary.
            If None, will use the input filename with '.summary.txt' extension.
    """
    if output_file is None:
        input_path = Path(input_file)
        output_file = str(input_path.parent / f"{input_path.stem}.summary.txt")
    
    # Read the clean JSONL file
    parsed_objects = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            parsed_objects.append(json.loads(line))
    
    # Create summary using the common function
    create_summary(parsed_objects, output_file)

if __name__ == "__main__":
    # Define paths
    current_dir = Path(__file__).parent
    input_file = current_dir / "test_predictions_pamap.jsonl.rtf"
    clean_output = current_dir / "test_predictions_pamap.clean.jsonl"
    summary_output = current_dir / "test_predictions_pamap.summary.txt"
    
    # Parse the RTF file
    parsed_objects = parse_rtf_jsonl(input_file, clean_output)
    
    # We don't need to call create_human_readable_summary since the summary is already created in parse_rtf_jsonl
    # The summary is saved to test_predictions_pamap.clean.summary.txt
