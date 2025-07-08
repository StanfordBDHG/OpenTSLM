import os
import zipfile
from typing import Literal, Optional
from constants import RAW_DATA_PATH
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import requests


# ---------------------------
# Constants
# ---------------------------

PAMAP2_URL = "https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip"
PAMAP2_ORG_NAME = f"{RAW_DATA_PATH}/pamap2+physical+activity+monitoring.zip"

OUTER_ZIP_NAME  = "pamap2+physical+activity+monitoring.zip"
INNER_ZIP_NAME  = "PAMAP2_Dataset.zip"

PAMAP2_DIR = f"{RAW_DATA_PATH}/PAMAP2_Dataset"
PROTOCOL_DIR = os.path.join(PAMAP2_DIR, "Protocol")
SUBJECT_IDS = list(range(101, 110))

# According to UCI, each .dat file contains 54 columns:
# 1 timestamp, 1 activity label, and 52 raw‐sensor features. https://archive.ics.uci.edu/ml/datasets/PAMAP2%2BPhysical%2BActivity%2BMonitoring
COLUMN_NAMES = ["timestamp", "activity"] + \
    [f"feature_{i}" for i in range(1, 53)]

# ---------------------------
# Helper to ensure data is present
# ---------------------------

def ensure_pamap2_data(
    raw_data_path: str = RAW_DATA_PATH,
    url: str = PAMAP2_URL,
    outer_zip_name: str = OUTER_ZIP_NAME,
    inner_zip_name: str = INNER_ZIP_NAME,
):
    """
    1) Download the outer ZIP from `url` if it's not already in `raw_data_path`
    2) Extract that ZIP to drop `PAMAP2_Dataset.zip` into `raw_data_path`
    3) Extract the inner ZIP to produce the PAMAP2_Dataset/Protocol folder
    """
    # Build all the paths
    os.makedirs(raw_data_path, exist_ok=True)
    outer_zip_path = os.path.join(raw_data_path, outer_zip_name)
    inner_zip_path = os.path.join(raw_data_path, inner_zip_name)
    protocol_dir   = os.path.join(raw_data_path, "PAMAP2_Dataset", "Protocol")

    # If we've already got the Protocol folder, nothing to do
    if os.path.isdir(protocol_dir):
        return

    # 1) Download outer ZIP
    if not os.path.isfile(outer_zip_path):
        print(f"Downloading PAMAP2 dataset from {url} …")
        resp = requests.get(url, stream=True)
        resp.raise_for_status()
        with open(outer_zip_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

    # 2) Extract outer ZIP to pull out PAMAP2_Dataset.zip
    print(f"Extracting {outer_zip_path} …")
    with zipfile.ZipFile(outer_zip_path, "r") as z:
        # some UCI zips include a top-level folder,
        # but this one directly contains PAMAP2_Dataset.zip
        z.extract(inner_zip_name, path=raw_data_path)

    # 3) Extract the inner dataset ZIP
    print(f"Extracting {inner_zip_path} …")
    with zipfile.ZipFile(inner_zip_path, "r") as z:
        z.extractall(raw_data_path)
        
        
# ---------------------------
# Core loader
# ---------------------------


def load_pamap2(
    protocol_dir: str = PROTOCOL_DIR,
    subject_ids: list = SUBJECT_IDS
) -> pd.DataFrame:
    """
    Load all 'Protocol' .dat files into one DataFrame.

    Returns columns: ['timestamp','activity','feature_1',…,'feature_52','subject_id'].
    """
    ensure_pamap2_data()
    dfs = []
    for sid in subject_ids:
        path = os.path.join(protocol_dir, f"subject{sid}.dat")
        df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMN_NAMES)
        df["subject_id"] = sid
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)
    # drop any rows with missing sensor readings
    data = data.dropna()
    # convert types
    data["timestamp"] = data["timestamp"].astype(float)
    data["activity"] = data["activity"].astype(int)
    return data

# ---------------------------
# PyTorch Dataset + Collate
# ---------------------------


class PAMAP2Dataset(Dataset):
    """
    Returns (normalized feature tensor, activity label).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[list] = None,
        label_col: str = "activity"
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols or [
            c for c in df.columns if c.startswith("feature_")]
        self.label_col = label_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        feats = row[self.feature_cols].values.astype(float)
        tensor = torch.tensor(feats, dtype=torch.float32)
        # normalize per‐sample
        tensor = (tensor - tensor.mean()) / (tensor.std() + 1e-8)
        label = int(row[self.label_col])
        return tensor, label


def collate_fn(batch):
    """
    Stack into:
      - features: FloatTensor (batch_size, n_features)
      - labels:   LongTensor  (batch_size,)
    """
    feats, labs = zip(*batch)
    return torch.stack(feats), torch.tensor(labs, dtype=torch.long)

# ---------------------------
# DataLoader helper with subject‐based splits
# ---------------------------


def get_pamap2_loader(
    split: Literal["train", "val", "test", "all"] = "train",
    batch_size: int = 32,
    shuffle: Optional[bool] = None
) -> DataLoader:
    """
    split:
      - 'train': subjects 101–107
      - 'val'  : subject 108
      - 'test' : subject 109
      - 'all'  : all subjects
    """
    df = load_pamap2()
    if split == "all":
        df_sub = df
    else:
        subject_split = {
            "train": list(range(101, 108)),
            "val": [108],
            "test": [109]
        }
        if split not in subject_split:
            raise ValueError("split must be 'train','val','test', or 'all'")
        df_sub = df[df["subject_id"].isin(subject_split[split])]

    if shuffle is None:
        shuffle = split in ("train", "all")

    dataset = PAMAP2Dataset(df_sub)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
    )

# ---------------------------
# Example usage
# ---------------------------


if __name__ == "__main__":
    for split in ["all", "train", "val", "test"]:
        loader = get_pamap2_loader(split=split, batch_size=4)
        feats, labs = next(iter(loader))
        print(
            f"{split.capitalize()} → features: {feats.shape}, labels: {labs.tolist()}")
