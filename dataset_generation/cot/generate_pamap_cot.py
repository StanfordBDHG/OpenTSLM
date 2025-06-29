#!/usr/bin/env python
import sys
import random
import json

# Add the parent directory to the path to import from time_series_datasets
sys.path.append("../../")

from time_series_datasets.pamap2.PAMAP2Dataset import PAMAP2Dataset

# Load the PAMAP2 dataset
print("Loading PAMAP2 dataset...")
dataset = PAMAP2Dataset()

# Get all unique activity labels
unique_labels = set()
for i in range(len(dataset)):
    unique_labels.add(dataset[i]["label"])
unique_labels = list(unique_labels)
print(f"Unique activity labels: {len(unique_labels)}")

# Function to create binary classification prompt
def create_binary_prompt(window, correct_label):
    # Select a random incorrect label
    other_labels = [label for label in unique_labels if label != correct_label]
    incorrect_label = random.choice(other_labels)
    
    # Randomly order the two labels
    class_options = [correct_label, incorrect_label]
    random.shuffle(class_options)
    
    # Simply use the first feature as the data description
    feature_keys = list(window.keys())
    first_feature = feature_keys[0] if feature_keys else "sensor data"
    data_description = first_feature
    
    # Create the prompt with dynamic content
    prompt = f"""Considering that this is {data_description} of a two-minute window, with classes based on whether data are captured during {class_options[0]} or {class_options[1]} activity, classify the time-series and respond only with the following options
{class_options[0]}
{class_options[1]}"""
    
    return {
        "prompt": prompt,
        "options": class_options,
        "correct_label": correct_label,
        "correct_index": class_options.index(correct_label),
        "data_description": data_description
    }

# Process windows and create prompts
print(f"\nTotal number of samples: {len(dataset)}")
all_prompts = []

# Process a subset of windows for demonstration
num_samples = min(5, len(dataset))
for i in range(num_samples):
    data_point = dataset[i]
    window = data_point["time_series"]
    label = data_point["label"]
    
    # Create binary classification prompt
    prompt_data = create_binary_prompt(window, label)
    all_prompts.append(prompt_data)
    
    # Print minimal information about this sample
    print(f"\nSample {i+1}:")
    print(f"  Label: {label}")
    print(f"  Feature used: {prompt_data['data_description']}")
    print(f"  Options: {prompt_data['options'][0]} vs {prompt_data['options'][1]}")

# Save prompts to file
output_file = "pamap2_binary_prompts.json"
with open(output_file, 'w') as f:
    json.dump(all_prompts, f, indent=2)

print(f"\nSaved {len(all_prompts)} binary classification prompts to {output_file}")

# Print a full example of the first prompt
if all_prompts:
    print("\nExample of a complete prompt:")
    print(all_prompts[0]["prompt"])
