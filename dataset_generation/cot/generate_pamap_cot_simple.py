#!/usr/bin/env python
import sys
import random
import csv
import os
import time
from openai import OpenAI

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


def generate_classification_rationale(feature, correct_label, incorrect_label):
    # Randomly order the two labels
    class_options = [correct_label, incorrect_label]
    random.shuffle(class_options)
    
    prompt = f"""Considering that this is {feature} time-series data of a two-minute window, with classes based on whether data are captured during {class_options[0]} or {class_options[1]} activity, classify the time-series and respond only with the following options
{class_options[0]}
{class_options[1]}
You MUST answer in the following format:
First think step by step and write a rationale as if you would see the time-series data in front of you.
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert in physical activities and human movement patterns."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        max_tokens=500,
        seed=42
    )
    
    rationale = response.choices[0].message.content
    return rationale


def main():
    COT_FILE = "pamap2_cot_simple.csv"
    FEATURES = ["heartrate"]
    
    if not os.path.exists(COT_FILE):
        with open(COT_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['correct_activity', 'incorrect_activity', 'rationale'])
    
    # Compare each activity with every other activity
    for feature in FEATURES:
        for correct_label in unique_labels:
            for incorrect_label in unique_labels:
                if correct_label != incorrect_label:
                    time.sleep(2)
                    
                    rationale = generate_classification_rationale(feature, correct_label, incorrect_label)
                    print(f"Generated rationale for {correct_label} vs {incorrect_label}\n{rationale}")
                
                    with open(COT_FILE, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([correct_label, incorrect_label, rationale])
            break


if __name__ == "__main__":
    main()