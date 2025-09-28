#!/usr/bin/env python3
"""
Visualize FaultDetectionA Dataset Samples

This script loads the FaultDetectionA dataset and creates visualizations
showing 5 sample time series for each bearing condition class.
"""

import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append("src")

from time_series_datasets.fault_prediction_a.fault_detection_a_loader import (
    load_fault_detection_a_splits,
)


def plot_fault_detection_samples():
    """Create visualization plots of sample time series for each fault class."""

    print("Loading FaultDetectionA dataset...")
    train_ds, val_ds, test_ds = load_fault_detection_a_splits()

    # Use test set for visualization
    dataset = test_ds
    print(f"Using test set with {len(dataset)} samples for visualization")

    # Collect samples by label
    samples_per_label = {0.0: [], 1.0: [], 2.0: []}
    label_names = {0.0: "Undamaged", 1.0: "Inner Damaged", 2.0: "Outer Damaged"}

    # Gather 5 samples per label
    for i, sample in enumerate(dataset):
        label = sample["label"]
        if len(samples_per_label[label]) < 5:
            samples_per_label[label].append((i, sample))

        # Stop when we have 5 samples for each label
        if all(len(samples) >= 5 for samples in samples_per_label.values()):
            break

    # Create the visualization
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    fig.suptitle(
        "FaultDetectionA Dataset: Sample Time Series by Bearing Condition",
        fontsize=16,
        fontweight="bold",
    )

    # Colors for each class
    colors = {0.0: "#2E8B57", 1.0: "#FF6B35", 2.0: "#C41E3A"}  # Green, Orange, Red

    for label_idx, (label_val, label_name) in enumerate(
        [(0.0, "Undamaged"), (1.0, "Inner Damaged"), (2.0, "Outer Damaged")]
    ):
        print(f"\nProcessing {label_name} samples...")

        for sample_idx, (dataset_idx, sample) in enumerate(
            samples_per_label[label_val]
        ):
            ax = axes[label_idx, sample_idx]

            # Get time series data
            time_series = sample["time_series"]

            # Create time axis (4 seconds at 64kHz = 5120 samples)
            time_axis = np.linspace(0, 4, len(time_series))

            # Plot the time series
            ax.plot(
                time_axis,
                time_series,
                color=colors[label_val],
                linewidth=0.5,
                alpha=0.8,
            )

            # Customize the subplot
            ax.set_title(
                f"{label_name}\nSample {sample_idx + 1} (Index {dataset_idx})",
                fontsize=10,
                fontweight="bold",
            )
            ax.set_xlabel("Time (seconds)", fontsize=8)
            ax.set_ylabel("Motor Current", fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)

            # Set consistent y-axis limits for better comparison
            ax.set_ylim(-1.0, 1.0)

            # Add some statistics as text
            mean_val = np.mean(time_series)
            std_val = np.std(time_series)
            ax.text(
                0.02,
                0.98,
                f"μ={mean_val:.3f}\nσ={std_val:.3f}",
                transform=ax.transAxes,
                fontsize=7,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the plot
    output_file = "fault_detection_samples_visualization.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"\n✅ Visualization saved as: {output_file}")

    # Close the plot to save memory
    plt.close()


def print_dataset_info():
    """Print information about the dataset."""
    print("=" * 80)
    print("FaultDetectionA Dataset Information")
    print("=" * 80)
    print("Source: Electromechanical drive system monitoring rolling bearings")
    print("Sampling: 64kHz, 4 seconds duration = 5,120 samples per recording")
    print("Classes:")
    print("  • Undamaged (9.09%)")
    print("  • Inner damaged (45.55%)")
    print("  • Outer damaged (45.55%)")
    print("\nOriginal split:")
    print("  • Train: 8,184 samples")
    print("  • Validation: 2,728 samples")
    print("  • Test: 2,728 samples")
    print("=" * 80)


if __name__ == "__main__":
    print_dataset_info()
    plot_fault_detection_samples()
    print("\n🎉 Visualization complete!")
    print("Generated file:")
    print("  • fault_detection_samples_visualization.png")
