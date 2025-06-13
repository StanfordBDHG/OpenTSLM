from typing import List, Literal, Tuple
import torch
import numpy as np
from datasets import Dataset
from prompt.text_time_series_prompt import TextTimeSeriesPrompt
from time_series_datasets.QADataset import QADataset
from time_series_datasets.sleep_edf.sleepedf_loader import get_sleepedf_data, SLEEP_STAGES

# Sleep stage mapping
SLEEP_STAGES = {
    1: "Movement time",
    2: "Sleep stage 1",
    3: "Sleep stage 2",
    4: "Sleep stage 3",
    5: "Sleep stage 4",
    6: "Sleep stage unknown",
    7: "Sleep stage REM",
    8: "Sleep stage Wake"
}

class SleepEDFQADataset(QADataset):
    def __init__(self, split: Literal["train", "test", "validation"], EOS_TOKEN: str):
        self.full_dataset = get_sleepedf_data()
        super().__init__(split, EOS_TOKEN)

    def _load_splits(self) -> Tuple[Dataset, Dataset, Dataset]:
        # Split into train/val/test (80/10/10)
        train_val, test = self.full_dataset.train_test_split(test_size=0.1, seed=42).values()
        train, val = train_val.train_test_split(test_size=0.1/0.9, seed=42).values()
        
        return train, val, test

    def _get_answer(self, row) -> str:
        return SLEEP_STAGES[row["label"]]

    def _get_pre_prompt(self, row) -> str:
        return "Given this 30-second EEG recording, what is the sleep stage? The possible stages are: Movement time, Sleep stage 1, Sleep stage 2, Sleep stage 3, Sleep stage 4, Sleep stage REM, and Sleep stage Wake."

    def _get_post_prompt(self, row) -> str:
        return "The sleep stage is:"

    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        # Get the EEG data and normalize it
        series = torch.tensor(row["data"][0], dtype=torch.float32)  # Shape: (n_times,)
        
        # Normalize the time series
        mean = series.mean()
        std = series.std()
        series = (series - mean) / (std + 1e-8)
        
        return [TextTimeSeriesPrompt("This is the EEG recording.", series.tolist())] 