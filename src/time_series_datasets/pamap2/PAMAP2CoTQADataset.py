from datasets import Dataset
from typing import List, Tuple
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from prompt.text_time_series_prompt import TextTimeSeriesPrompt
from time_series_datasets.QADataset import QADataset
from time_series_datasets.pamap2.pamap2_cot_loader import load_pamap2_cot_splits
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from time_series_datasets.util import (
    extend_time_series_to_match_patch_size_and_aggregate,
)


TIME_SERIES_LABELS = [
    "The following is the accelerometer data on the x-axis",
    "The following is the accelerometer data on the y-axis",
    "The following is the accelerometer data on the z-axis",
]


class PAMAP2CoTQADataset(QADataset):
    def _load_splits(self) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Load the PAMAP2 CoT dataset splits using the pamap2_cot_loader.
        
        Returns:
            Tuple of (train, validation, test) datasets
        """
        return load_pamap2_cot_splits()

    def _get_answer(self, row) -> str:
        """
        Get the answer from the row, which is the chain-of-thought reasoning.
        
        Args:
            row: Dataset row
            
        Returns:
            Chain-of-thought reasoning as a string
        """
        return row["rationale"]

    def _get_pre_prompt(self, _row) -> str:
        """
        Get the pre-prompt text.
        
        Args:
            _row: Dataset row (unused)
            
        Returns:
            Pre-prompt text
        """
        return "You are given accelerometer data in all three dimensions. Your task is to analyze this data and provide a chain-of-thought reasoning to determine the person's activity."

    def _get_post_prompt(self, _row) -> str:
        """
        Get the post-prompt text.
        
        Args:
            _row: Dataset row (unused)
            
        Returns:
            Post-prompt text
        """
        return "Rationale:"

    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        """
        Convert the time series data into a list of TextTimeSeriesPrompt objects.
        
        Args:
            row: Dataset row containing x_axis, y_axis, and z_axis data
            
        Returns:
            List of TextTimeSeriesPrompt objects
        """
        # Extract the time series data from the row
        series = torch.tensor(
            [
                row["x_axis"],
                row["y_axis"],
                row["z_axis"],
            ],
            dtype=torch.float32,
        )
        
        # Normalize the data
        means = series.mean(dim=1, keepdim=True)
        stds = series.std(dim=1, keepdim=True)
        series = (series - means) / (stds + 1e-8)
        
        return [
            TextTimeSeriesPrompt(time_series_label, time_series)
            for time_series_label, time_series in zip(
                TIME_SERIES_LABELS, series.tolist()
            )
        ]


if __name__ == "__main__":
    dataset = PAMAP2CoTQADataset(split="train", EOS_TOKEN="")
    dataset_val = PAMAP2CoTQADataset(split="validation", EOS_TOKEN="")
    dataset_test = PAMAP2CoTQADataset(split="test", EOS_TOKEN="")

    print(f"Dataset sizes: Train: {len(dataset)}, Validation: {len(dataset_val)}, Test: {len(dataset_test)}")

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=lambda batch: extend_time_series_to_match_patch_size_and_aggregate(
            batch, patch_size=4
        ),
    )

    for batch in tqdm(dataloader, total=1):
        print("Batch keys:", batch[0].keys())
        print("Batch values:", batch[0]["answer"])
        # print("Batch time series:", batch[0]["time_series"])
        print("Batch time series text:", batch[0]["time_series_text"])
        print("Batch pre prompt:", batch[0]["pre_prompt"])
        print("Batch post prompt:", batch[0]["post_prompt"])
        break