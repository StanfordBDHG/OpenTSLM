import torch
from typing import List, Tuple
from datasets import Dataset
from prompt.text_time_series_prompt import TextTimeSeriesPrompt
from time_series_datasets.QADataset import QADataset
from .sleepedf_cot_loader import load_sleepedf_cot_splits

class SleepEDFCotQADataset(QADataset):
    def _load_splits(self) -> Tuple[Dataset, Dataset, Dataset]:
        return load_sleepedf_cot_splits(seed=42)

    def _get_answer(self, row) -> str:
        return row['rationale']

    def _get_pre_prompt(self, row) -> str:
        return row['prompt']

    def _get_post_prompt(self, row) -> str:
        return "Please provide a detailed rationale for your classification."

    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        # row['time_series'] is a list of lists (channels x length)
        series = row['time_series']
        # Normalize each channel
        normd = []
        for ch in series:
            t = torch.tensor(ch, dtype=torch.float32)
            mean, std = t.mean(), t.std()
            if std > 0:
                t = (t - mean) / std
            else:
                t = t - mean
            normd.append(t.tolist())
        return [TextTimeSeriesPrompt("This is the EEG time series window.", normd)] 