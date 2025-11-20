#
# This source file is part of the OpenTSLM open-source project
#
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

"""
TimeSeriesExam1QADataset.py
----------------------------
PyTorch-style QA dataset for TimeSeriesExam1 benchmark.

This module defines the TimeSeriesExam1QADataset class for the AutonLab/TimeSeriesExam1
dataset, which contains challenging time series understanding questions with multiple-choice
answers. Questions may involve one or two time series.
"""

from typing import List, Tuple
import torch
from datasets import Dataset, load_dataset

from prompt.text_time_series_prompt import TextTimeSeriesPrompt
from time_series_datasets.QADataset import QADataset


# Split ratios (since dataset only has test split, we create our own)
TEST_FRAC = 0.2
VAL_FRAC = 0.1


class TimeSeriesExam1QADataset(QADataset):
    """
    TimeSeriesExam1 Question-Answer Dataset.

    This dataset loads the AutonLab/TimeSeriesExam1 benchmark dataset containing
    challenging time series questions with multiple-choice answers. Questions test
    understanding of:
    - Distribution comparison
    - Pattern recognition
    - Seasonality and trends
    - Time series properties

    Some questions involve comparing two time series (ts1, ts2), while others
    analyze a single time series (ts).
    """

    def _load_splits(self) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Load and split the TimeSeriesExam1 data into train/validation/test sets.

        Note: The original dataset only has a 'test' split, so we create our own
        train/val/test splits from it.

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        # Load the full dataset (only has 'test' split)
        ds_full = load_dataset("AutonLab/TimeSeriesExam1", split="test")

        # First carve out the test split
        train_val, test = ds_full.train_test_split(
            test_size=TEST_FRAC, seed=42
        ).values()

        # From the remaining data, take validation
        val_frac_adj = VAL_FRAC / (1.0 - TEST_FRAC)
        train, val = train_val.train_test_split(
            test_size=val_frac_adj, seed=43
        ).values()

        return train, val, test

    def _get_answer(self, row) -> str:
        """
        Get the answer text.

        Args:
            row: Dataset row containing answer data

        Returns:
            The answer string
        """
        return row['answer']

    def _get_pre_prompt(self, row) -> str:
        """
        Format the question and options as a multiple-choice prompt.

        Args:
            row: Dataset row

        Returns:
            Formatted question with options
        """
        question = row['question']
        options = row['options']

        # Format as multiple choice question
        prompt_parts = [question]
        prompt_parts.append("\nSelect one of the following answers:")

        # Add options with (a), (b), (c), ... labels
        option_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        for idx, option in enumerate(options):
            if idx < len(option_labels):
                prompt_parts.append(f"({option_labels[idx]}) {option}")

        return "\n".join(prompt_parts)

    def _get_post_prompt(self, row) -> str:
        """
        Get the post-prompt instruction for answering.

        Args:
            row: Dataset row

        Returns:
            Post-prompt string asking for the answer
        """
        return "Answer:"

    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        """
        Create text-time series prompts from the data.

        Handles both single-series (ts) and dual-series (ts1, ts2) questions.
        Each time series is normalized to zero mean and unit standard deviation.

        Args:
            row: Dataset row containing time series data

        Returns:
            List of TextTimeSeriesPrompt objects
        """
        prompts = []

        # Check if this is a dual-series question (ts1 and ts2)
        has_ts1 = row['ts1'] is not None and len(row['ts1']) > 0
        has_ts2 = row['ts2'] is not None and len(row['ts2']) > 0

        if has_ts1 and has_ts2:
            # Dual time series question
            for idx, ts_data in enumerate([row['ts1'], row['ts2']], start=1):
                series_tensor = torch.tensor(ts_data, dtype=torch.float32)

                # Normalize
                mean = series_tensor.mean()
                std = series_tensor.std()
                if std > 1e-8:
                    normalized_series = (series_tensor - mean) / std
                else:
                    normalized_series = series_tensor - mean

                # Create prompt with statistics
                text_prompt = f"Time series {idx}, it has mean {mean:.4f} and std {std:.4f}:"
                prompts.append(TextTimeSeriesPrompt(text_prompt, normalized_series.tolist()))

        else:
            # Single time series question (use 'ts' field)
            ts_data = row['ts']
            if ts_data is not None and len(ts_data) > 0:
                series_tensor = torch.tensor(ts_data, dtype=torch.float32)

                # Normalize
                mean = series_tensor.mean()
                std = series_tensor.std()
                if std > 1e-8:
                    normalized_series = (series_tensor - mean) / std
                else:
                    normalized_series = series_tensor - mean

                # Create prompt with statistics
                text_prompt = f"This is the time series, it has mean {mean:.4f} and std {std:.4f}:"
                prompts.append(TextTimeSeriesPrompt(text_prompt, normalized_series.tolist()))

        return prompts


# ---------------------------
# Example usage
# ---------------------------

if __name__ == "__main__":
    # Create dataset instances
    print("Creating TimeSeriesExam1 dataset splits...")
    train_dataset = TimeSeriesExam1QADataset("train", "")
    val_dataset = TimeSeriesExam1QADataset("validation", "")
    test_dataset = TimeSeriesExam1QADataset("test", "")

    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Validation: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")
    print(f"  Total: {len(train_dataset) + len(val_dataset) + len(test_dataset)} samples")

    # Show example samples
    print(f"\n{'='*80}")
    print("Example samples:")
    print('='*80)

    for i in [0, 1]:
        sample = train_dataset[i]
        print(f"\nSample {i+1}:")
        print(f"Pre-prompt:\n{sample['pre_prompt'][:200]}...")
        print(f"\nNumber of time series: {len(sample['time_series'])}")
        if sample['time_series']:
            for idx, (ts, ts_text) in enumerate(zip(sample['time_series'], sample['time_series_text']), start=1):
                print(f"  Time series {idx}: {len(ts)} points")
                print(f"  Text: {ts_text}")
        print(f"\nPost-prompt: {sample['post_prompt']}")
        print(f"Answer: {sample['answer'][:100]}...")
        print('-'*80)
