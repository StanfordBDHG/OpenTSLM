import os
import subprocess
from typing import Literal, Optional
from constants import RAW_DATA
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# ---------------------------
# Constants
# ---------------------------

REPO_URL = "https://github.com/Mcompetitions/M4-methods.git"
REPO_DIR = f"{RAW_DATA}/m4"  # Local folder name after cloning

# ---------------------------
# Helper to ensure repo is available
# ---------------------------

def ensure_m4_repo(repo_dir: str = REPO_DIR):
    """Clone the M4-methods repo if it's not already present."""
    if not os.path.isdir(repo_dir):
        print(f"Cloning M4-methods into ./{repo_dir} …")
        subprocess.run(["git", "clone", REPO_URL, repo_dir], check=True)

# ---------------------------
# Core loader
# ---------------------------

def load_m4(
    frequency: Literal["Yearly", "Quarterly", "Monthly", "Weekly", "Daily", "Hourly"],
    repo_dir: str = REPO_DIR
) -> pd.DataFrame:
    """
    Load the full training data of the M4 dataset for a given frequency.

    Args:
        frequency: One of ["Yearly","Quarterly","Monthly","Weekly","Daily","Hourly"].
        repo_dir: Local path to the cloned repo.

    Returns:
        DataFrame with columns ["M4id", "x1", "x2", ..., "xN"].
    """
    ensure_m4_repo(repo_dir)
    filename = f"Dataset/Train/{frequency}-train.csv"
    path = os.path.join(repo_dir, filename)
    return pd.read_csv(path)

# ---------------------------
# PyTorch Dataset + Collation
# ---------------------------

class M4Dataset(Dataset):
    """
    PyTorch Dataset for a subset of M4 time series.
    Returns a normalized series tensor and its ID.
    """
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        id_column = self.df.columns[0]
        series_id = row[id_column]
        values = row.drop(id_column).values.astype(float)
        
        # Trim to the last non-NaN value
        if np.isnan(values).all():
            trimmed_values = np.array([])
        else:
            last_valid = np.where(~np.isnan(values))[0][-1]
            trimmed_values = values[:last_valid + 1]
        tensor = torch.tensor(trimmed_values, dtype=torch.float32)

        if tensor.numel() > 0:
            mean = tensor.mean()
            std = tensor.std()
            if std > 0:
                tensor = (tensor - mean) / std
            else:
                tensor = tensor - mean
        return tensor, series_id


def collate_fn(batch, *, patch_size: int = 1):
    """
    Pad variable-length series in the batch to the same length.
    Returns:
        - Tensor of shape (batch_size, max_len)
        - List of series IDs
    """
    series_list, ids = zip(*batch)
    max_len = max(seq.size(0) for seq in series_list)
    max_len = ((max_len + patch_size - 1) // patch_size) * patch_size

    padded = []
    for seq in series_list:
        if seq.size(0) < max_len:
            pad_len = max_len - seq.size(0)
            seq = torch.nn.functional.pad(seq, (0, pad_len), "constant", 0)
        else:
            seq = seq[:max_len]
        padded.append(seq)
    return torch.stack(padded), list(ids)

# ---------------------------
# DataLoader helper with train/val/test/all splits
# ---------------------------

def get_m4_loader(
    frequency: Literal["Yearly", "Quarterly", "Monthly", "Weekly", "Daily", "Hourly"],
    split: Literal["train", "val", "test", "all"] = "train",
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
    batch_size: int = 8,
    shuffle: Optional[bool] = None,
    repo_dir: str = REPO_DIR,
    patch_size: int = 1
) -> DataLoader:
    """
    Returns a DataLoader for train/validation/test/all splits of the M4 dataset.

    Args:
        frequency: Data frequency.
        split: One of "train", "val", "test", or "all".
        val_frac: Fraction of the original training set used for validation.
        test_frac: Fraction of the original training set used for test.
        seed: RNG seed for reproducibility.
        batch_size: Batch size.
        shuffle: Whether to shuffle (defaults to True for train/all, False otherwise).
        repo_dir: Local path to cloned repo.
        patch_size: Pad length multiple.

    Returns:
        DataLoader yielding (series_tensor, id_list).
    """
    # 1) Load full training DataFrame
    df_full = load_m4(frequency, repo_dir=repo_dir)

    if split == "all":
        df_subset = df_full
    else:
        # 2) Split into train_rest and test
        df_train_rest, df_test = train_test_split(
            df_full, test_size=test_frac, random_state=seed, shuffle=True
        )
        # 3) Split train_rest into train and val
        val_frac_adj = val_frac / (1.0 - test_frac)
        df_train, df_val = train_test_split(
            df_train_rest, test_size=val_frac_adj, random_state=seed + 1, shuffle=True
        )
        # 4) Select subset by split
        if split == "train":
            df_subset = df_train
        elif split in ("val", "validation"):
            df_subset = df_val
        elif split == "test":
            df_subset = df_test
        else:
            raise ValueError("split must be 'train', 'val', 'test', or 'all'")

    # Determine default shuffle
    if shuffle is None:
        shuffle = split in ("train", "all")

    # Wrap in DataLoader
    dataset = M4Dataset(df_subset)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: collate_fn(batch, patch_size=patch_size)
    )

# ---------------------------
# Example usage
# ---------------------------

if __name__ == "__main__":
    # Monthly all data
    all_loader = get_m4_loader("Monthly", split="all", batch_size=4)
    train_loader = get_m4_loader("Monthly", split="train", batch_size=4)
    val_loader = get_m4_loader("Monthly", split="val", batch_size=4)
    test_loader = get_m4_loader("Monthly", split="test", batch_size=4)

    print("All batch:")
    for series_batch, ids in all_loader:
        print(series_batch.shape, ids)
        break

    print("Train batch:")
    for series_batch, ids in train_loader:
        print(series_batch.shape, ids)
        break

    print("Validation batch:")
    for series_batch, ids in val_loader:
        print(series_batch.shape, ids)
        break

    print("Test batch:")
    for series_batch, ids in test_loader:
        print(series_batch.shape, ids)
        break