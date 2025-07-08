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

client = OpenAI()

# Specify the path to the sleepedf_loader.py file
sleepedf_loader_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                   'time_series_datasets', 'sleep-edf', 'sleepedf_loader.py')

# Load the module
spec = importlib.util.spec_from_file_location('sleepedf_loader', sleepedf_loader_path)
sleepedf_loader = importlib.util.module_from_spec(spec)
sys.modules['sleepedf_loader'] = sleepedf_loader
spec.loader.exec_module(sleepedf_loader)

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
        # Wake is most different from deep sleep (N3) and REM
        'Wake': ['NREM3', 'REM'],
        
        # NREM1 (light sleep) is most different from wake and deep sleep
        'NREM1': ['Wake', 'NREM3'],
        
        # NREM2 (intermediate sleep) is most different from wake and REM
        'NREM2': ['Wake', 'REM'],
        
        # NREM3 (deep sleep) is most different from wake and REM
        'NREM3': ['Wake', 'REM'],
        
        # REM is most different from NREM2 and NREM3
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
    
    # Create a prompt that explains the sleep stages
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


def extract_sleep_segments(raw, annotations, window_size=30):
    """Extract segments of EEG data with corresponding sleep stage labels."""

    # Print more meaningful information about the raw object
    print(f"Raw data info:")
    print(f"- Channels: {raw.ch_names}")
    print(f"- Sampling frequency: {raw.info['sfreq']} Hz")
    print(f"- Duration: {raw.times[-1]:.2f} seconds")
    
    # Get the actual data
    data = raw.get_data()
    print(f"- Data shape: {data.shape} (channels × time points)")
    print(f"- Number of samples: {data.shape[1]}")
    print(f"- Number of channels: {data.shape[0]}")
    print(f"- Data type: {data.dtype}")
    print(f"- Data range: min={data.min():.2f}, max={data.max():.2f}, mean={data.mean():.2f}")
    print(f"- Unique values: {np.unique(data)}")
    print(f"- First 10 time points for each channel:")
    for i, ch_name in enumerate(raw.ch_names):
        if i < len(data):
            print(f"  - {ch_name}: {data[i, :10]}")
    
    # Print more meaningful information about the annotations
    print(f"\nAnnotations info:")
    if len(annotations) > 0:
        print(f"- Number of annotations: {len(annotations)}")
        print(f"- Annotation types: {set(annotations.description)}")
        print(f"- First 5 annotations: {list(zip(annotations.onset[:5], annotations.duration[:5], annotations.description[:5]))}")
    else:
        print("- No annotations found")

    data_segments = []
    labels = []
    
    # Map annotation descriptions to standardized labels
    label_mapping = {
        'Sleep stage W': 'Wake',
        'Sleep stage 1': 'NREM1',
        'Sleep stage 2': 'NREM2',
        'Sleep stage 3': 'NREM3',
        'Sleep stage 4': 'NREM3',  # Often combined with stage 3 in modern classification
        'Sleep stage R': 'REM'
    }
    
    # Get the data and sampling rate
    data, times = raw[:]
    sfreq = raw.info['sfreq']
    
    # Process each annotation
    for ann in annotations:
        if ann['description'] in label_mapping:
            # Convert to standardized label
            label = label_mapping[ann['description']]
            
            # Calculate start and end samples
            start_sample = int(ann['onset'] * sfreq)
            end_sample = start_sample + int(ann['duration'] * sfreq)
            
            # Extract windows of specified size
            for window_start in range(start_sample, end_sample, int(window_size * sfreq)):
                window_end = window_start + int(window_size * sfreq)
                if window_end <= end_sample:
                    # Extract data for all channels in this window
                    segment = data[:, window_start:window_end]
                    data_segments.append(segment)
                    labels.append(label)
    
    return data_segments, labels


def main():
    # Use the Sleep-EDF loader to get the data
    print("Loading Sleep-EDF dataset...")
    
    # Define the output file for the CoT dataset
    COT_FILE = f"sleep_cot.csv"
    
    # Try to use the loader from the imported module
    try:
        # Increase batch size to check more recordings at once
        loader = sleepedf_loader.get_sleepedf_loader(batch_size=10, shuffle=False)
        
        # Iterate through recordings until we find one with annotations
        found_annotations = False
        raw = None
        ann = None
        total_recordings = 0
        
        print("Searching for recordings with annotations...")
        for batch_idx, (raw_batch, ann_batch) in enumerate(loader):
            # Check each recording in the batch
            for rec_idx in range(len(raw_batch)):
                print(batch_idx, rec_idx)
                raw = raw_batch[rec_idx]
                ann = ann_batch[rec_idx]
                total_recordings += 1
                
                # Only print minimal info
                if len(ann) > 0:
                    print(f"\nFound recording with annotations! (#{total_recordings}, batch {batch_idx+1}, recording {rec_idx+1})")
                    print(f"Channels: {raw.ch_names}")
                    print(f"Number of annotations: {len(ann)}")
                    print(f"Annotation types: {set(ann.description)}")
                    print(f"First 5 annotations: {list(zip(ann.onset[:5], ann.duration[:5], ann.description[:5]))}")
                    found_annotations = True
                    break  # Found a recording with annotations
            
            if found_annotations:
                break  # Exit the outer loop if we found annotations
            
            # Don't limit the number of batches to check - go through all data
            print(f"Checked batch {batch_idx+1} - no annotations found yet...")
                
        if not found_annotations:
            print("No recordings with annotations found in the dataset.")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error using loader: {e}")
        sys.exit(1)
    
    # Extract segments and labels
    data_segments, labels = extract_sleep_segments(raw, ann)
    
    if not data_segments:
        print("No valid sleep segments found!")
        return
    
    print(f"Extracted {len(data_segments)} segments with labels")
    
    # Define the relevant EEG channels to use
    relevant_features = {
        "EEG": ["EEG Fpz-Cz"],  # Main EEG channel
    }
    
    # Process a subset of segments to generate CoT data
    num_samples = min(10, len(data_segments))  # Limit to 10 samples for testing
    
    for i in range(1, num_samples):
        segment = data_segments[i]
        label = labels[i]
        
        # Get channel indices for the relevant features
        for feature_name, channel_names in relevant_features.items():
            # Find the indices of the channels in raw.ch_names
            channel_indices = [raw.ch_names.index(ch) for ch in channel_names if ch in raw.ch_names]
            
            if not channel_indices:
                print(f"Warning: Channels {channel_names} not found in the data")
                continue
            
            # Extract the time series data for these channels
            multi_series_data = [segment[idx] for idx in channel_indices]
            
            # Generate the classification rationale
            # prompt, rationale = generate_classification_rationale(feature_name, multi_series_data, label)
            
            # # Save to CSV
            # cot_data = {
            #     'time_series': str([series.tolist() for series in multi_series_data]),
            #     'label': label,
            #     'prompt': prompt,
            #     'rationale': rationale,
            # }

            # df = pd.DataFrame([cot_data])
            # df.to_csv(COT_FILE, mode='a', header=not os.path.exists(COT_FILE), index=False)
            
            print(f"Generated CoT for sample {i+1}/{num_samples}, label: {label}")
            break
        break

if __name__ == "__main__":
    main()
