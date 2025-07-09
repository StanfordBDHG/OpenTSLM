from torch.utils.data import Dataset
from typing import Tuple
from .sleepedf_classification_loader import load_sleepedf_classification_splits

class SleepEDFClassificationDataset(Dataset):
    def __init__(self, split: str = "train"):
        train, val, test = load_sleepedf_classification_splits()
        if split == "train":
            self.data = train
        elif split == "validation":
            self.data = val
        elif split == "test":
            self.data = test
        else:
            raise ValueError(f"Unknown split: {split}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item['time_series'], item['label'] 