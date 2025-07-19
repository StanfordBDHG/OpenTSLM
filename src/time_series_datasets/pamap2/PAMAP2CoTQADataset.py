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

        text = """
        You are given accelerometer data in all three dimensions. Your task is to classify the activity based on analysis of the data.

        Instructions:
        - Begin by analyzing the time series without assuming a specific label.
        - Think step-by-step about what the observed patterns suggest regarding movement intensity and behavior.
        - Write your rationale as a single, natural paragraph — do not use bullet points, numbered steps, or section headings.
        - Do **not** mention any class label until the final sentence.

        Possible activity labels are:
        lying, sitting, standing, walking, running, cycling, nordic walking, watching TV, computer work, car driving, ascending stairs, descending stairs, vacuum cleaning, ironing, folding laundry, house cleaning, playing soccer, rope jumping.
        
        - Make sure that your last word is the answer. You MUST end your response with "Answer: ".
        """

        return text

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
        Convert the time series data into a list of TextTimeSeriesPrompt objects, including mean and std in the text.
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

        # Check for invalid data
        if torch.isnan(series).any() or torch.isinf(series).any():
            print(f"❌ Invalid data detected in PAMAP2 sample")
            print(f"Row data: {row}")
            print(f"Series shape: {series.shape}")
            print(f"Series values: {series}")
            print(f"NaN positions: {torch.isnan(series).nonzero()}")
            print(f"Inf positions: {torch.isinf(series).nonzero()}")
            exit(1)

        # Normalize the data with better numerical stability
        means = series.mean(dim=1, keepdim=True)
        stds = series.std(dim=1, keepdim=True)
        
        # Handle zero or very small standard deviations
        min_std = 1e-6  # Increased from 1e-8 for better stability
        stds = torch.clamp(stds, min=min_std)
        
        series_norm = (series - means) / stds
        
        # Check for NaN/Inf after normalization
        if torch.isnan(series_norm).any() or torch.isinf(series_norm).any():
            print(f"❌ NaN/Inf detected after normalization")
            print(f"Original series: {series}")
            print(f"Means: {means}")
            print(f"Stds: {stds}")
            print(f"Normalized series: {series_norm}")
            print(f"NaN positions: {torch.isnan(series_norm).nonzero()}")
            print(f"Inf positions: {torch.isinf(series_norm).nonzero()}")
            exit(1)

        prompts = []
        for i, (time_series_label, time_series, mean, std) in enumerate(zip(TIME_SERIES_LABELS, series_norm.tolist(), means.squeeze().tolist(), stds.squeeze().tolist())):
            text_prompt = f"{time_series_label}, it has mean {mean:.4f} and std {std:.4f}:"
            prompts.append(TextTimeSeriesPrompt(text_prompt, time_series))
        return prompts


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