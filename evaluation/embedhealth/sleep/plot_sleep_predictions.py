#!/usr/bin/env python3
"""
Plot SleepEDF time series samples from sleep_cot_data.csv.
Each sample is plotted as a PNG with EEG data and the full_prediction as text.
"""
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

CSV_PATH = "sleep_cot_data.csv"
OUTPUT_DIR = "sleep_cot_plots"

# Publication style
plt.style.use('seaborn-v0_8')
sns.set_palette("colorblind")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_sample(row, idx):
    eeg_data = json.loads(row['eeg_data'])
    full_pred = row['full_prediction']
    gt_label = row['ground_truth_label']
    pred_label = row['predicted_label']
    sample_idx = row['sample_index']
    series_length = row['series_length']

    fig, ax = plt.subplots(figsize=(12, 8))
    t = np.arange(len(eeg_data))
    ax.plot(t, eeg_data, linewidth=2.5, color='blue', alpha=0.8)
    ax.set_xlabel('Time Step', fontsize=15)
    ax.set_ylabel('Normalized EEG Amplitude', fontsize=15)
    ax.set_title(f"Sample {sample_idx} | GT: {gt_label} | Pred: {pred_label}", fontsize=18, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=13)
    
    # Set consistent y-axis limits for all plots
    ax.set_ylim(-1.25, 1.25)
    ax.set_yticks(np.linspace(-1, 1, 5))

    # Add full_prediction as a text box below the plot
    # Pad with underscores to ensure consistent height
    pred_max = 900
    pred_text = full_pred[:pred_max]
    pred_text = full_pred
    
    # Pad with underscores to make text box consistent size
    min_chars = 1000
    if len(pred_text) < min_chars:
        padding = '_' * (min_chars - len(pred_text))
        pred_text = full_pred + padding

    plt.gcf().text(0.01, -0.02, f"Prediction:\n{pred_text}", fontsize=18, ha='left', va='top', wrap=True,
                   bbox=dict(boxstyle='round', facecolor='whitesmoke', alpha=0.9, edgecolor='gray'))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    fname = f"sample_{idx+1:03d}_gt_{gt_label.lower().replace(' ', '_').replace('-', '_')}.png"
    plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {fname}")

def main():
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} samples from {CSV_PATH}")
    for idx, row in df.iterrows():
        plot_sample(row, idx)
    print(f"All plots saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main() 