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
import mne
import tempfile
import shutil
import numpy as np
import zipfile
import requests
from tqdm.auto import tqdm
import warnings

# ---------------------------
# Constants
# ---------------------------
SLEEPEDF_URL = "https://physionet.org/static/published-projects/sleep-edf/sleep-edf-database-1.0.0.zip"
ZIP_NAME = "sleep-edf-database-1.0.0.zip"
DATA_DIR = "data/sleep-edf-database-1.0.0"
RAW_DATA_PATH = "data"
RECORDS_FILE = os.path.join(DATA_DIR, "RECORDS")

client = OpenAI()

# ---------------------------
# Download and extract Sleep-EDF if needed
# ---------------------------
def ensure_sleepedf_data():
    os.makedirs(RAW_DATA_PATH, exist_ok=True)
    sleepedf_dir = DATA_DIR
    zip_path = os.path.join(RAW_DATA_PATH, ZIP_NAME)
    if os.path.isdir(sleepedf_dir) and os.path.exists(RECORDS_FILE):
        return
    # Download
    print(f"Downloading Sleep-EDF from {SLEEPEDF_URL} ...")
    resp = requests.get(SLEEPEDF_URL, stream=True)
    resp.raise_for_status()
    total = int(resp.headers.get('content-length', 0))
    with open(zip_path, "wb") as f, tqdm(
        total=total, unit='B', unit_scale=True, desc="Downloading Sleep-EDF ZIP"
    ) as pbar:
        for chunk in resp.iter_content(8192):
            f.write(chunk)
            pbar.update(len(chunk))
    # Extract
    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(RAW_DATA_PATH)
    os.remove(zip_path)
    if not os.path.isdir(sleepedf_dir) or not os.path.exists(RECORDS_FILE):
        raise FileNotFoundError(f"Sleep-EDF data not found after extraction: {sleepedf_dir}")

# ---------------------------
# Load and pair recordings
# ---------------------------
def load_sleepedf_recordings():
    ensure_sleepedf_data()
    sleepedf_dir = DATA_DIR
    records_file = RECORDS_FILE
    recs = []
    files_by_basename = {}
    with open(records_file, "r") as f:
        for line in f:
            filename = line.strip()
            if not filename:
                continue
            if '.' in filename:
                basename, ext = filename.rsplit('.', 1)
                if basename not in files_by_basename:
                    files_by_basename[basename] = {'rec': None, 'hyp': None}
                if ext == 'rec':
                    files_by_basename[basename]['rec'] = os.path.join(sleepedf_dir, filename)
                elif ext == 'hyp':
                    files_by_basename[basename]['hyp'] = os.path.join(sleepedf_dir, filename)
    for _, paths in files_by_basename.items():
        if paths['rec'] and paths['hyp']:
            recs.append((paths['rec'], paths['hyp']))
    return recs

# ---------------------------
# Windowing logic
# ---------------------------
def make_windows(rec_path, hyp_path, window_size_sec=3, min_pct=0.5, picks=None):
    # 1) read raw EDF
    with tempfile.NamedTemporaryFile(suffix=".edf") as tmp:
        shutil.copyfile(rec_path, tmp.name)
        raw = mne.io.read_raw_edf(tmp.name, preload=True, verbose=False)
    # 2) read hypnogram annotations
    with tempfile.NamedTemporaryFile(suffix=".edf") as tmp:
        shutil.copyfile(hyp_path, tmp.name)
        ann = mne.io.read_raw_edf(tmp.name, preload=True, verbose=False)
    # 3) optionally pick subset of channels
    if picks is not None:
        raw = raw.copy().pick_channels(picks)
    # 4) Get data & sampling information
    raw_data = raw.get_data()    # (n_features, n_samples)
    raw_freq = raw.info["sfreq"] # 100 Hz
    ann_data = ann.get_data()    # (1, m_samples)
    ann_freq = ann.info["sfreq"] # 0.0333 Hz
    # 5) Resample annotations to raw sampling frequency
    if ann_freq != raw_freq:
        raw_len = raw_data.shape[1]
        ann_len = ann_data.shape[1]
        labels_resampled = np.interp(
            np.arange(raw_len),
            np.linspace(0, raw_len - 1, ann_len),
            ann_data[0]
        ).round().astype(int)
    else:
        labels_resampled = ann_data[0].astype(int)
    # 6) Trim to the common length
    n_samples = min(raw_data.shape[1], labels_resampled.shape[0])
    raw_data = raw_data[:, :n_samples]
    labels_resampled = labels_resampled[:n_samples]
    # 7) Downsample by 2x (100 Hz → 50 Hz)
    ds_factor = 2
    raw_data = raw_data[:, ::ds_factor]
    labels_resampled = labels_resampled[::ds_factor]
    raw_freq = raw_freq / ds_factor
    # 8) Slide through contiguous non-overlapping windows
    window_size = int(window_size_sec * raw_freq)
    n_windows = n_samples // window_size
    windows = []
    labels = []
    for w in range(n_windows):
        start = w * window_size
        end = start + window_size
        win_raw = raw_data[:, start:end]
        win_lbl = labels_resampled[start:end]
        if win_lbl.size == 0:
            continue
        mode = np.bincount(win_lbl).argmax()
        if (win_lbl == mode).sum() < min_pct * win_lbl.size:
            continue
        windows.append(win_raw)
        labels.append(int(mode))
    return windows, labels

# ---------------------------
# COT Generation logic (unchanged)
# ---------------------------
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
    all_labels = ['Wake', 'NREM1', 'NREM2', 'NREM3', 'REM']
    other_labels = [label for label in all_labels if label != correct_label]
    return random.choice(other_labels)

def create_classification_prompt(feature, correct_label):
    incorrect_label = get_dissimilar_label(correct_label)
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
    print("Loading Sleep-EDF dataset via standalone loader")
    COT_FILE = "sleep_cot.csv"
    ZIP_FILE = "sleep_cot.zip"
    warnings.filterwarnings(
        "ignore",
        message="Channels contain different highpass filters",
        category=RuntimeWarning,
        module="mne"
    )
    recs = load_sleepedf_recordings()
    label_map = {
        0: "Wake",
        1: "NREM1",
        2: "NREM2",
        3: "NREM3",
        4: "NREM3",
        5: "REM"
    }
    all_rows = []
    # For demo, just do 1 sample. For full dataset, use len(recs) and all windows.
    num_samples = min(1, len(recs))
    for i in range(num_samples):
        rec_path, hyp_path = recs[i]
        windows, labels = make_windows(rec_path, hyp_path)
        for win, label_int in zip(windows, labels):
            label = label_map.get(int(label_int), str(label_int))
            # For demo, just use first channel
            multi_series_data = [win[0]]
            prompt, rationale = generate_classification_rationale("EEG", multi_series_data, label)
            row = {
                "time_series": str([s.tolist() for s in multi_series_data]),
                "label": label,
                "prompt": prompt,
                "rationale": rationale,
            }
            all_rows.append(row)
            print(f"Generated window — stage: {label}, window shape: {win.shape}")
            break  # Remove this break for full dataset
        break  # Remove this break for full dataset
    df = pd.DataFrame(all_rows, columns=["time_series", "label", "prompt", "rationale"])
    df.to_csv(COT_FILE, index=False)
    with zipfile.ZipFile(ZIP_FILE, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(COT_FILE)
    print(f"\n✅  Finished sample generation.")
    print(f"CSV saved to: {os.path.abspath(COT_FILE)}")
    print(f"Zipped file saved to: {os.path.abspath(ZIP_FILE)}")
    print("Upload the zip file to your polybox and update the loader URL accordingly.")

if __name__ == "__main__":
    main()
