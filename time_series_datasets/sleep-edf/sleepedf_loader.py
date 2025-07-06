import os
import zipfile
import requests
import mne
from typing import Optional
import torch
from torch.utils.data import Dataset, DataLoader
import tempfile
import shutil
import numpy as np
import matplotlib.pyplot as plt


# from constants import RAW_DATA_PATH

RAW_DATA_PATH = "./data"

# ---------------------------
# Constants
# ---------------------------

SLEEPEDF_URL = "https://physionet.org/static/published-projects/sleep-edf/sleep-edf-database-1.0.0.zip"
ZIP_NAME = "sleep-edf-database-1.0.0.zip"
DATA_DIR_NAME = "sleep-edf-database-1.0.0"

ZIP_PATH = os.path.join(RAW_DATA_PATH, ZIP_NAME)
SLEEPEDF_DIR = os.path.join(RAW_DATA_PATH, DATA_DIR_NAME)
RECORDS_FILE = os.path.join(SLEEPEDF_DIR, "RECORDS")

# ---------------------------
# Helper to ensure data
# ---------------------------


def ensure_sleepedf_data(
    raw_data_path: str = RAW_DATA_PATH,
    url: str = SLEEPEDF_URL,
    zip_name: str = ZIP_NAME
):
    """
    1) Download the Sleep-EDF ZIP if missing.
    2) Extract it to raw_data_path/DATA_DIR_NAME.
    """
    os.makedirs(raw_data_path, exist_ok=True)

    # If already extracted, nothing to do
    if os.path.isdir(SLEEPEDF_DIR):
        return

    zip_path = os.path.join(raw_data_path, zip_name)

    # 1) Download
    if not os.path.isfile(zip_path):
        print(f"Downloading Sleep-EDF from {url} …")
        resp = requests.get(url, stream=True)
        resp.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in resp.iter_content(8192):
                f.write(chunk)

    # 2) Extract
    print(f"Extracting {zip_path} …")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(raw_data_path)
        
    os.listdir()

# ---------------------------
# Core loader
# ---------------------------


def load_sleepedf_recordings(
    raw_data_path: str = RAW_DATA_PATH
):
    """
    Returns a list of (rec_path, hyp_path) for all recordings.
    """
    ensure_sleepedf_data(raw_data_path)
    recs = []
    # Read the RECORDS file and group .rec and .hyp files by their base name
    print(RECORDS_FILE)
    
    # Create a dictionary to store file paths by base name
    files_by_basename = {}
    
    with open(RECORDS_FILE, "r") as f:
        for line in f:
            filename = line.strip()
            if not filename:
                continue
                
            # Extract the base name (without extension)
            if '.' in filename:
                basename, ext = filename.rsplit('.', 1)
                
                # Initialize the dictionary entry if it doesn't exist
                if basename not in files_by_basename:
                    files_by_basename[basename] = {'rec': None, 'hyp': None}
                
                # Store the full path based on extension
                if ext == 'rec':
                    files_by_basename[basename]['rec'] = os.path.join(SLEEPEDF_DIR, filename)
                elif ext == 'hyp':
                    files_by_basename[basename]['hyp'] = os.path.join(SLEEPEDF_DIR, filename)
    
    # Create pairs of (rec_path, hyp_path) for each recording
    for _, paths in files_by_basename.items():
        if paths['rec'] and paths['hyp']:
            recs.append((paths['rec'], paths['hyp']))
    
    return recs

# ---------------------------
# PyTorch Dataset
# ---------------------------


class SleepEDFDataset(Dataset):
    """
    Yields (raw: mne.io.Raw, ann: mne.Annotations) for each PSG.
    """

    def __init__(
        self,
        recordings: list[tuple[str, str]],
        preload: bool = True,
        picks: Optional[list[str]] = None
    ):
        """
        recordings: list of (rec_path, hyp_path)
        preload:     whether to load data into memory
        picks:       list of channel names to keep (e.g. ['EEG Fpz-Cz'])
        """
        super().__init__()
        self.recordings = recordings
        self.preload = preload
        self.picks = picks

    def __len__(self):
        return len(self.recordings)

    def __getitem__(self, idx):
        rec_path, hyp_path = self.recordings[idx]
        # 1) read raw EDF
        with tempfile.NamedTemporaryFile(suffix=".edf") as tmp:
            shutil.copyfile(rec_path, tmp.name)
            raw = mne.io.read_raw_edf(
                tmp.name, preload=self.preload, verbose=False)
        # 2) read hypnogram annotations
        with tempfile.NamedTemporaryFile(suffix=".edf") as tmp:
            shutil.copyfile(hyp_path, tmp.name)
            ann = mne.io.read_raw_edf(
                tmp.name, preload=self.preload, verbose=False)
        # 3) optionally pick subset of channels
        # if self.picks is not None:
        #     raw = raw.copy().pick_channels(self.picks)
        return raw, ann


def sleepedf_collate_fn(batch):
    """
    Default collate: returns lists of raws and annotations.
    """
    raws, anns = zip(*batch)
    return list(raws), list(anns)

# ---------------------------
# DataLoader helper
# ---------------------------


def get_sleepedf_loader(
    batch_size: int = 1,
    shuffle: bool = False,
    preload: bool = True,
    picks: Optional[list[str]] = None
) -> DataLoader:
    """
    Returns a DataLoader over all Sleep-EDF recordings.

    Args:
        batch_size: number of recordings per batch (usually 1)
        shuffle:    whether to shuffle the order
        preload:    whether to preload EDF signals into memory
        picks:      list of channel names to keep (e.g. ['EEG Fpz-Cz'])
    """
    recs = load_sleepedf_recordings()
    dataset = SleepEDFDataset(recs, preload=preload, picks=picks)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=sleepedf_collate_fn
    )

# ---------------------------
# Example usage
# ---------------------------


if __name__ == "__main__":
    loader = get_sleepedf_loader(batch_size=1, shuffle=False)
    
    raw, ann = next(iter(loader))

    print(raw[0].get_data())
    print(ann[0].get_data())

    raw_data = raw[0].get_data()  
    ann_data = ann[0].get_data()

    print(len(raw_data[0]), len(ann_data[0]))
    
    raw_time = np.arange(raw_data.shape[1]) / raw[0].info['sfreq']
    ann_time = np.arange(ann_data.shape[1]) / ann[0].info['sfreq']
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    
    # Plot hypnogram
    ax1.plot(ann_time, ann_data[0], 'k-')
    ax1.set_ylabel('Sleep Stage')
    ax1.set_title('Hypnogram')
    
    # Labels for sleep stages
    stage_labels = {
        0: 'W',    # Wake
        1: 'N1',   # Non-REM stage 1
        2: 'N2',   # Non-REM stage 2
        3: 'N3',   # Non-REM stage 3
        4: 'N3',   # Non-REM stage 4 (often combined with N3)
        5: 'REM',  # REM sleep
        6: 'M',    # Movement
        9: 'Unknown'  # Unknown
    }
    
    unique_stages = sorted(np.unique(ann_data[0]))
    ax1.set_yticks(unique_stages)
    ax1.set_yticklabels([stage_labels.get(int(stage), str(stage)) for stage in unique_stages])
    
    # Plot EEG-Fpz-Cz signal (first channel)
    ax2.plot(raw_time, raw_data[0], 'b-')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Amplitude')
    
    if len(raw[0].ch_names) > 0:
        ax2.set_title(f'Raw Data - {raw[0].ch_names[0]}')
    else:
        ax2.set_title('Raw Data - Main Channel')
    
    plt.tight_layout()
    plt.show()
