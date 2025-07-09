import os
from typing import Tuple
import numpy as np
from datasets import Dataset
from .sleepedf_loader import load_sleepedf_recordings, SleepEDFDataset

TEST_FRAC = 0.1
VAL_FRAC = 0.1

# Helper to convert the SleepEDFDataset to a flat list of (window, label)
def get_all_windows_and_labels(preload=True, picks=None):
    recs = load_sleepedf_recordings()
    dataset = SleepEDFDataset(recs, preload=preload, picks=picks)
    data = []
    for i in range(len(dataset)):
        item = dataset[i]
        # item['time_series']: np.ndarray (channels x length)
        # item['label']: int
        data.append({
            'time_series': item['time_series'].tolist(),
            'label': int(item['label'])
        })
    return data

def load_sleepedf_classification_splits(seed: int = 42) -> Tuple[Dataset, Dataset, Dataset]:
    data = get_all_windows_and_labels()
    full_dataset = Dataset.from_list(data)
    train_val, test = full_dataset.train_test_split(test_size=TEST_FRAC, seed=seed).values()
    val_frac_adj = VAL_FRAC / (1.0 - TEST_FRAC)
    train, val = train_val.train_test_split(test_size=val_frac_adj, seed=seed+1).values()
    return train, val, test 