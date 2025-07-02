#!/usr/bin/env python
import sys
import random
import json
import base64
import io
import matplotlib.pyplot as plt
from openai import OpenAI
import pandas as pd
import os

client = OpenAI()

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


def generate_classification_rationale(feature, time_series_data, label):
    # Check if time_series_data is a list of time series
    num_series = len(time_series_data)
    fig, axes = plt.subplots(num_series, 1, figsize=(10, 4 * num_series), sharex=True)
    
    # If there's only one series, axes won't be an array
    if num_series == 1:
        axes = [axes]
    
    # Plot each time series in its own subplot
    for i, series in enumerate(time_series_data):
        axes[i].plot(series, marker='o', linestyle='-', markersize=0)
        axes[i].grid(True, alpha=0.3)
        axes[i].set_title(f"{feature} - Component {i+1}")
    
    plt.tight_layout()
    
    temp_image_path = f"temp_plot.png"
    plt.savefig(temp_image_path)
    plt.close()
    
    prompt = create_classification_prompt(feature, label)

    with open(temp_image_path, "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode('utf-8')

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert in time series analysis."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}", "detail": "high"}}
                ]}
            ],
            temperature=0.5,
            max_tokens=500,
            seed=42
        )
    
    rationale = response.choices[0].message.content
    return prompt, rationale


def create_classification_prompt(feature, correct_label):
    # Select a random incorrect label for the binary classification
    other_labels = [label for label in unique_labels if label != correct_label]
    incorrect_label = random.choice(other_labels)
    
    # Randomly order the two labels
    class_options = [correct_label, incorrect_label]
    random.shuffle(class_options)
    
    prompt = f"""Considering that this is {feature} of a two-minute window, with classes based on whether data are captured during {class_options[0]} or {class_options[1]} activity, classify the time-series and respond only with the following options
{class_options[0]}
{class_options[1]}
Think step by step and answer with ONLY a rationale for the correct answer {correct_label}. You MUST end your response with \"Answer: {correct_label}\":
"""
    return prompt


def main():
    COT_FILE = f"pamap2_cot.csv"

    relevant_features = {
        "hand_acceleration": ["handAcc16_1", "handAcc16_2", "handAcc16_3"],
        "heart_rate": ["heartrate"],
    }

    num_samples = min(1, len(dataset))
    for i in range(num_samples):
        data_point = dataset[i]
        window = data_point["time_series"]
        label = data_point["label"]

        for feature_name, features in relevant_features.items():
            multi_series_data = [window[feat] for feat in features]
            
            prompt, rationale = generate_classification_rationale(feature_name, multi_series_data, label)
            
            cot_data = {
                'time_series': str([window[feat].tolist() for feat in features]),
                'label': label,
                'prompt': prompt,
                'rationale': rationale,
            }

            df = pd.DataFrame([cot_data])
            df.to_csv(COT_FILE, mode='a', header=not os.path.exists(COT_FILE), index=False)

            break
        break

if __name__ == "__main__":
    main()
