#!/usr/bin/env python
import os
import sys
import random
import json
import base64
import io
import matplotlib.pyplot as plt
from openai import OpenAI
import pandas as pd
import importlib.util
import mne
import tempfile
import shutil
import numpy as np




# Add the parent directory to the path to import from time_series_datasets
sys.path.append("../../src")
from time_series_datasets.sleep_edf.sleepedf_loader import SleepEDFDataset, load_sleepedf_recordings

client = OpenAI()

def generate_classification_rationale(feature, time_series_data, label):
    num_series = len(time_series_data)
    fig, axes = plt.subplots(num_series, 1, figsize=(10, 4 * num_series), sharex=True)
    
    if num_series == 1:
        axes = [axes]
    
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
                {"role": "system", "content": "You are an expert in sleep EEG analysis and classification."},
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


def get_dissimilar_label(correct_label):
    # Map of sleep stages and their dissimilar stages
    dissimilar_labels_map = {
        'Wake': ['NREM3', 'REM'],
        'NREM1': ['Wake', 'NREM3'],
        'NREM2': ['Wake', 'REM'],
        'NREM3': ['Wake', 'REM'],
        'REM': ['NREM2', 'NREM3']
    }
    
    labels = dissimilar_labels_map.get(correct_label, [])
    
    if labels:
        return random.choice(labels)
    
    # Fallback if no dissimilar labels are found
    all_labels = ['Wake', 'NREM1', 'NREM2', 'NREM3', 'REM']
    other_labels = [label for label in all_labels if label != correct_label]
    return random.choice(other_labels)


def create_classification_prompt(feature, correct_label):
    # Select a dissimilar incorrect label for the binary classification
    incorrect_label = get_dissimilar_label(correct_label)
    
    # Randomly order the two labels
    class_options = [correct_label, incorrect_label]
    random.shuffle(class_options)
    
    prompt = f"""Considering that this is {feature} data from a sleep study, with classes based on whether the data represents {class_options[0]} or {class_options[1]} sleep stage, classify the time-series and respond only with the following options:
{class_options[0]}
{class_options[1]}

Sleep stage definitions:
- Wake: The awake state, characterized by high frequency, low amplitude EEG patterns.
- NREM1: Non-rapid eye movement stage 1, light sleep with theta waves.
- NREM2: Non-rapid eye movement stage 2, characterized by sleep spindles and K-complexes.
- NREM3: Non-rapid eye movement stage 3, deep sleep with slow delta waves.
- REM: Rapid eye movement sleep, characterized by rapid eye movements and low amplitude, mixed frequency EEG.

Think step by step and answer with ONLY a rationale for the correct answer {correct_label}. You MUST end your response with "Answer: {correct_label}":
"""
    return prompt


def main():
    print("Loading Sleep-EDF dataset via new loader")

    COT_FILE = "sleep_cot.csv"

    try:
        recs = load_sleepedf_recordings()
        dataset = SleepEDFDataset(
            recs,
            preload=True,
            picks=None
        )
    except Exception as e:
        print(f"Error loading Sleep-EDF dataset: {e}")
        sys.exit(1)

    label_map = {
        0: "Wake",
        1: "NREM1",
        2: "NREM2",
        3: "NREM3",
        4: "NREM3",
        5: "REM"
    }

    num_samples = min(1, len(dataset))
    for i in range(num_samples):
        data_point = dataset[i]
        window = data_point["time_series"]
        label_int = int(data_point["label"])
        label = label_map[label_int]

        # multi_series_data = [window[ch] for ch in range(window.shape[0])]
        multi_series_data = [window[0]]

        prompt, rationale = generate_classification_rationale(
            "EEG", multi_series_data, label)
        
        cot_data = {
            "time_series": str([s.tolist() for s in multi_series_data]),
            "label": label,
            "prompt": prompt,
            "rationale": rationale,
        }
        df = pd.DataFrame([cot_data])
        df.to_csv(COT_FILE, mode="a", header=not os.path.exists(COT_FILE), index=False)

        print(f"Generated sample {i+1}/{num_samples} — stage: {label}, window shape: {window.shape}")
        break

    print("✅  Finished sample generation.")

if __name__ == "__main__":
    main()
