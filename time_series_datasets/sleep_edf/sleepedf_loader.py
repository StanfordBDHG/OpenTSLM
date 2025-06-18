import os
import sys
import subprocess
import mne
import numpy as np
from datasets import Dataset
import matplotlib.pyplot as plt

# Add parent directory to sys.path to import constants
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from constants import RAW_DATA as RAW_DATA_PATH

# ---------------------------
# Constants
# ---------------------------

DATA_DIR_NAME = "sleep-edf-database-1.0.0"
SLEEPEDF_DIR = os.path.join(RAW_DATA_PATH, DATA_DIR_NAME)

# Where wget drops the .edf files:
SLEEPEDF_FILES_DIR = os.path.join(
    SLEEPEDF_DIR,
    "physionet.org", "files", "sleep-edfx", "1.0.0", "sleep-cassette"
)

DOWNLOAD_COMMAND = (
    f"wget --progress=bar:force -r -N -c -np "
    "https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/ "
    f"-P {SLEEPEDF_DIR}"
)

# Sleep stage mapping
SLEEP_STAGES = {
    2: "Sleep stage 1 (NREM1)",
    3: "Sleep stage 2 (NREM2)",
    4: "Sleep stage 3 (NREM3)",  # Combined with stage 4
    5: "Sleep stage 3 (NREM3)",  # Combined with stage 3
    7: "Sleep stage REM",
    8: "Sleep stage Wake"
}

def download_sleepedf_data_if_not_exists():
    if os.path.exists(SLEEPEDF_FILES_DIR) and os.listdir(SLEEPEDF_FILES_DIR):
        return
    os.makedirs(SLEEPEDF_DIR, exist_ok=True)
    print(f"Downloading Sleep-EDF to {SLEEPEDF_DIR} …")
    subprocess.run(DOWNLOAD_COMMAND.split(" "), check=True)

def process_recording(psg_path: str, hyp_path: str, duration: int = 30, channel: str = "EEG Fpz-Cz"):
    """Process a single recording and return its epochs and labels.
    
    Args:
        psg_path: Path to the PSG file
        hyp_path: Path to the hypnogram file
        duration: Duration of each epoch in seconds (default: 30)
        channel: EEG channel to use (default: "EEG Fpz-Cz")
    """
    # Load data
    raw = mne.io.read_raw_edf(psg_path, preload=True)
    ann = mne.read_annotations(hyp_path)
    raw.set_annotations(ann)

    # Pick channel
    raw.pick([channel])

    sfreq = raw.info['sfreq']
    n_times = raw.n_times
    epoch_samples = int(duration * sfreq)

    # Get events
    events, _ = mne.events_from_annotations(raw)
    
    # Filter out any event that would run past the end
    valid = [ev for ev in events if ev[0] + epoch_samples <= n_times]
    if not valid:
        raise RuntimeError(f"No valid epochs for {psg_path}")
    events = np.array(valid)

    # Build epochs
    tmax = duration - 1. / sfreq
    epochs = mne.Epochs(
        raw, events,
        tmin=0, tmax=tmax,
        baseline=None,
        detrend=1,
        preload=True
    )

    data = epochs.get_data()            # (n_epochs, 1, n_times)
    labels = epochs.events[:, -1]       # sleep-stage codes
    
    # Filter out movement time (1) and unknown (6) stages
    valid_mask = ~np.isin(labels, [1, 6])
    data = data[valid_mask]
    labels = labels[valid_mask]
    
    # Combine stages 3 and 4 into NREM3
    labels = np.where(np.isin(labels, [4, 5]), 4, labels)

    return data, labels

def get_sleepedf_data(duration: int = 30, channel: str = "EEG Fpz-Cz") -> Dataset:
    """Download and process Sleep-EDF data, returning a Dataset with all recordings.
    
    Args:
        duration: Duration of each epoch in seconds (default: 30)
        channel: EEG channel to use (default: "EEG Fpz-Cz")
    """
    download_sleepedf_data_if_not_exists()
    
    # Get list of PSG files
    psg_files = [f for f in os.listdir(SLEEPEDF_FILES_DIR) if f.endswith("PSG.edf")]
    
    # Process all recordings
    all_data = []
    all_labels = []
    recording_indices = []
    skipped_files = []
    
    for i, psg_file in enumerate(psg_files):
        psg_path = os.path.join(SLEEPEDF_FILES_DIR, psg_file)
        
        # Match the hypnogram by the SCXXXX prefix
        prefix = psg_file.split("-")[0][:6]  # e.g. "SC4001"
        hyp_files = [
            f for f in os.listdir(SLEEPEDF_FILES_DIR)
            if f.startswith(prefix) and f.endswith("Hypnogram.edf")
        ]
        if not hyp_files:
            print(f"Warning: No hypnogram found for {psg_file}, skipping...")
            skipped_files.append(psg_file)
            continue
            
        hyp_path = os.path.join(SLEEPEDF_FILES_DIR, hyp_files[0])
        
        try:
            # Process this recording
            data, labels = process_recording(psg_path, hyp_path, duration, channel)
            if len(labels) > 0:  # Only add if we have valid epochs
                all_data.append(data)
                all_labels.append(labels)
                recording_indices.extend([i] * len(labels))
        except Exception as e:
            print(f"Warning: Error processing {psg_file}: {str(e)}, skipping...")
            skipped_files.append(psg_file)
            continue
    
    if not all_data:
        raise RuntimeError("No valid recordings found after processing!")
        
    if skipped_files:
        print(f"\nSkipped {len(skipped_files)} files due to missing hypnograms or processing errors:")
        print("This is normal! E.g. SC4032E0 has no hypnogram.")
        for f in skipped_files:
            print(f"  - {f}")
    
    # Combine all recordings
    combined_data = np.concatenate(all_data, axis=0)  # (total_epochs, 1, n_times)
    combined_labels = np.concatenate(all_labels)  # (total_epochs,)
    
    # Reshape data to be compatible with PyArrow (flatten the channel dimension)
    # Original shape: (n_epochs, 1, n_times) -> (n_epochs, n_times)
    combined_data = combined_data.squeeze(1)
    
    # Create dataset with numpy arrays
    dataset = Dataset.from_dict({
        "data": combined_data,
        "label": combined_labels,
        "recording_idx": np.array(recording_indices)
    })
    
    # Set format to numpy for all columns
    dataset.set_format(type="numpy")
    
    # Print dataset statistics
    print("\nDataset statistics:")
    print(f"Total number of epochs: {len(dataset)}")
    print("\nSleep stage distribution:")
    unique_labels, counts = np.unique(dataset["label"], return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"{SLEEP_STAGES[label]}: {count} epochs ({count/len(dataset)*100:.1f}%)")
    
    return dataset

def plot_example_epochs(dataset: Dataset, n_examples: int = 3):
    """Plot example epochs from different sleep stages."""
    # Get unique labels and their counts
    unique_labels, counts = np.unique(dataset["label"], return_counts=True)
    
    # Create a figure with subplots for each sleep stage
    n_stages = len(unique_labels)
    fig, axes = plt.subplots(n_stages, n_examples, figsize=(15, 3*n_stages))
    if n_stages == 1:
        axes = axes[None, :]  # Make axes 2D for consistent indexing
    
    # For each sleep stage
    for i, stage in enumerate(unique_labels):
        # Get indices for this stage and convert to Python int
        stage_indices = [int(idx) for idx in np.where(dataset["label"] == stage)[0]]
        
        # Plot n_examples
        for j in range(n_examples):
            if j < len(stage_indices):
                idx = stage_indices[j]
                epoch = dataset[idx]["data"]
                ax = axes[i, j]
                
                # Plot the epoch
                ax.plot(epoch)
                ax.set_title(f"{SLEEP_STAGES[int(stage)]} (n={int(counts[i])})")
                if j == 0:  # Only show y-label for first plot in each row
                    ax.set_ylabel("Amplitude")
                if i == n_stages-1:  # Only show x-label for last row
                    ax.set_xlabel("Time (samples)")
                ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# ---------------------------
# Main (for smoke‐test)
# ---------------------------

if __name__ == "__main__":
    dataset = get_sleepedf_data()
    print(f"Dataset size: {len(dataset)}")
    print(f"Data shape: {dataset[0]['data'].shape}")
    print(f"Label: {SLEEP_STAGES[dataset[0]['label']]}")
    
    # Plot example epochs
    print("\nPlotting example epochs from each sleep stage...")
    plot_example_epochs(dataset)
    