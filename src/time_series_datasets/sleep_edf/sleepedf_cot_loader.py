import os
import urllib.request
import zipfile
import pandas as pd
from datasets import Dataset
from typing import Tuple

# Placeholder for the polybox URL (to be filled in later)
POLYBOX_URL = "<POLYBOX_SLEEP_COT_URL>"
DATA_DIR = "data/SleepEDFCotDataset"
COT_CSV = os.path.join(DATA_DIR, "sleep_cot.csv")
ZIP_NAME = "sleep_cot.zip"

TEST_FRAC = 0.1
VAL_FRAC = 0.1


def download_and_extract_sleepedf_cot():
    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = os.path.join(DATA_DIR, ZIP_NAME)
    if os.path.exists(COT_CSV):
        return
    # Download
    print(f"Downloading Sleep-EDF COT dataset from {POLYBOX_URL} ...")
    urllib.request.urlretrieve(POLYBOX_URL, zip_path)
    # Extract
    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)
    os.remove(zip_path)
    if not os.path.exists(COT_CSV):
        raise FileNotFoundError(f"COT CSV not found after extraction: {COT_CSV}")

def load_sleepedf_cot_splits(seed: int = 42) -> Tuple[Dataset, Dataset, Dataset]:
    download_and_extract_sleepedf_cot()
    df = pd.read_csv(COT_CSV)
    # Optionally: parse time_series from string to list of floats
    def parse_series(s):
        import ast
        return [list(map(float, x)) for x in ast.literal_eval(s)]
    df['time_series'] = df['time_series'].apply(parse_series)
    # Split
    full_dataset = Dataset.from_pandas(df)
    train_val, test = full_dataset.train_test_split(test_size=TEST_FRAC, seed=seed).values()
    val_frac_adj = VAL_FRAC / (1.0 - TEST_FRAC)
    train, val = train_val.train_test_split(test_size=val_frac_adj, seed=seed+1).values()
    return train, val, test 