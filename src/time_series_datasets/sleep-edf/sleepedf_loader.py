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
import warnings

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

warnings.filterwarnings(
    "ignore",
    message="Channels contain different highpass filters",
    category=RuntimeWarning,
    module="mne"
)

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

        self.data = self._load_data()
        self.time_series, self.labels = self._make_windows()

    def _load_data(self):
        data = []
        for rec_path, hyp_path in self.recordings:
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
            data.append((raw, ann))
        return data

    def _make_windows(self, window_size_sec: int = 3, *, min_pct: float = 0.5):
        """Segment each `(raw, ann)` tuple into fixed-length windows.

        Args:
            window_size_sec: Length of each window **in seconds** (default 3 s).
            min_pct: Minimum fraction (0-1) of samples within the window that
                must agree with the modal sleep stage for the window to be
                accepted.

        Returns:
            windows: list[np.ndarray] — each of shape (n_channels, n_steps)
            labels:  list[int]       — modal sleep-stage per window
        """
        windows: list[np.ndarray] = []
        labels: list[int] = []

        for raw, ann in self.data:
            # 1) Get data & sampling information
            raw_data = raw.get_data()    # (n_features, n_samples)
            raw_freq = raw.info["sfreq"] # 100 Hz

            ann_data = ann.get_data()    # (1, m_samples)
            ann_freq = ann.info["sfreq"] # 0.0333 Hz

            # 2) Resample annotations to raw sampling frequency
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

            # 3) Trim to the common length
            n_samples = min(raw_data.shape[1], labels_resampled.shape[0])
            raw_data = raw_data[:, :n_samples]
            labels_resampled = labels_resampled[:n_samples]

            # 4) Slide through contiguous non-overlapping windows
            window_size = int(window_size_sec * raw_freq)
            n_windows = n_samples // window_size

            for w in range(n_windows):
                start = w * window_size
                end = start + window_size
                win_raw = raw_data[:, start:end]
                win_lbl = labels_resampled[start:end]

                # Mode (most common stage) within the window
                if win_lbl.size == 0:
                    continue
                mode = np.bincount(win_lbl).argmax()
                if (win_lbl == mode).sum() < min_pct * win_lbl.size:
                    continue

                windows.append(win_raw)
                labels.append(int(mode))

        return windows, labels


    def __len__(self):
        return len(self.time_series)

    def __getitem__(self, idx):
        return {"time_series": self.time_series[idx], "label": self.labels[idx]}

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
    # loader = get_sleepedf_loader(batch_size=1, shuffle=False)
    recs = load_sleepedf_recordings()
    dataset = SleepEDFDataset(recs, preload=True, picks=None)
    data_point = next(iter(dataset))

    window = data_point['time_series']
    label = data_point['label']
    
    # Plot only EEG Fpz-Cz window (first channel)
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

    plt.figure(figsize=(15, 4))
    plt.plot(window[0], 'b-')
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')

    stage_str = stage_labels.get(int(label), str(label))
    if hasattr(window[0], 'ch_names') and len(window[0].ch_names) > 0:
        ch_name = window[0].ch_names[0]
    else:
        ch_name = 'Main Channel'
    plt.title(f'Raw EEG Window - {ch_name} | Stage: {stage_str} ({label})')

    plt.tight_layout()
    plt.show()
