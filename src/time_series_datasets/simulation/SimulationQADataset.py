import numpy as np
import torch
from typing import List, Tuple
from datasets import Dataset

from prompt.text_time_series_prompt import TextTimeSeriesPrompt
from time_series_datasets.QADataset import QADataset


class SimulationQADataset(QADataset):
    def __init__(
        self,
        split,
        EOS_TOKEN,
        length: int = 100,
        num_series: int = 1,
        format_sample_str: bool = False,
        time_series_format_function=None,
    ):
        """
        Initialize SimulationQADataset with one or more time series of variable length.

        Args:
            split: Dataset split (train/test/validation) - all return the same single item
            EOS_TOKEN: End-of-sequence token
            length: Length of the generated time series (default: 100)
            num_series: Number of time series to generate (default: 1)
            format_sample_str: Whether to format as string
            time_series_format_function: Optional time series formatting function
        """
        self.length = length
        self.num_series = num_series

        # Generate time series once in constructor
        self.generated_series_data = self._generate_time_series()

        super().__init__(
            split, EOS_TOKEN, format_sample_str, time_series_format_function
        )

    def _generate_time_series(self) -> List[Tuple[torch.Tensor, str]]:
        """
        Generate time series data once in constructor.
        Returns a list of tuples containing (normalized_series, text_description).
        """
        series_data = []

        # Generate time series for any num_series value
        for i in range(self.num_series):
            # Generate random time series
            series = torch.tensor(np.random.randn(self.length), dtype=torch.float32)

            # Check for invalid data (very unlikely with random normal data)
            if torch.isnan(series).any() or torch.isinf(series).any():
                print(f"❌ Invalid data detected in simulation series {i}")
                raise ValueError(f"Invalid data detected in series {i}")

            # Normalize the series with better numerical stability
            mean_val = series.mean().item()
            std_val = series.std().item()

            # Handle zero or very small standard deviations
            min_std = 1e-6
            std_val = max(std_val, min_std)

            normalized_series = (series - mean_val) / std_val

            # Check for NaN/Inf after normalization (very unlikely with random normal data)
            if (
                torch.isnan(normalized_series).any()
                or torch.isinf(normalized_series).any()
            ):
                print(f"❌ NaN/Inf detected after normalization in series {i}")
                raise ValueError(f"NaN/Inf detected after normalization in series {i}")

            # Create descriptive text - use consistent format for all series
            text_description = f"This is the time series, it has mean {mean_val:.4f} and std {std_val:.4f}."

            series_data.append((normalized_series, text_description))

        return series_data

    def _load_splits(self) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Creates a dataset item with the pre-generated time series data.
        The time series data was generated once in the constructor.
        """
        # Create data structure with pre-generated time series
        if self.num_series == 1:
            # Single time series (backward compatibility)
            normalized_series, text_description = self.generated_series_data[0]
            data_item = {
                "Series": normalized_series.tolist(),  # Store the actual generated series
                "Question": "What is the pattern of this time series?",
                "Answer": "This is a random pattern with noise.",
                "Task": "pattern recognition",
            }
        else:
            # Multiple time series - store actual generated data
            time_series_data = {}
            for i in range(self.num_series):
                normalized_series, text_description = self.generated_series_data[i]
                time_series_data[f"series_{i}"] = normalized_series.tolist()

            data_item = {
                **time_series_data,
                "Question": f"What are the patterns of these {self.num_series} time series?",
                "Answer": f"These are {self.num_series} different synthetic patterns including sinusoidal, trend, and noise components.",
                "Task": "multi-series pattern recognition",
            }

        # Create a dataset with single item
        single_item_dataset = Dataset.from_dict(
            {key: [value] for key, value in data_item.items()}
        )

        # Return the same dataset for all splits
        return single_item_dataset, single_item_dataset, single_item_dataset

    def _get_answer(self, row) -> str:
        """Get the answer from the data row."""
        return row["Answer"]

    def _get_pre_prompt(self, row) -> str:
        """Get the question/pre-prompt from the data row."""
        return row["Question"]

    def _get_post_prompt(self, row) -> str:
        """Get the post-prompt from the data row."""
        return "Predict the " + row["Task"] + " Answer:"

    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        """
        Use pre-generated time series data and convert to TextTimeSeriesPrompt format.
        The time series were generated once in the constructor and are reused for consistency.
        Handles both single and multiple time series.
        """
        prompts = []

        # Use the pre-generated series data
        for i, (normalized_series, text_description) in enumerate(
            self.generated_series_data
        ):
            prompts.append(
                TextTimeSeriesPrompt(text_description, normalized_series.tolist())
            )

        return prompts


if __name__ == "__main__":
    # Example usage - Single time series (backward compatibility)
    print("=== Single Time Series Dataset ===")
    dataset_single = SimulationQADataset("train", "", length=50, num_series=1)
    print(f"Dataset length: {len(dataset_single)}")
    sample_single = dataset_single[0]
    print(f"Sample keys: {sample_single.keys()}")
    print(f"Question: {sample_single['pre_prompt'][:100]}...")
    print(f"Answer: {sample_single['answer']}")
    print(f"Number of time series prompts: {len(sample_single['time_series_prompts'])}")

    print("\n=== Multiple Time Series Dataset ===")
    # Example usage - Multiple time series
    dataset_multi = SimulationQADataset("train", "", length=50, num_series=3)
    print(f"Dataset length: {len(dataset_multi)}")
    sample_multi = dataset_multi[0]
    print(f"Sample keys: {sample_multi.keys()}")
    print(f"Question: {sample_multi['pre_prompt'][:100]}...")
    print(f"Answer: {sample_multi['answer']}")
    print(f"Number of time series prompts: {len(sample_multi['time_series_prompts'])}")

    # Show time series prompt details
    for i, ts_prompt in enumerate(sample_multi["time_series_prompts"]):
        print(f"Time series {i}: {ts_prompt.text[:50]}...")

    # Test different splits (should all be the same)
    print("\n=== Testing Different Splits ===")
    train_dataset = SimulationQADataset("train", "", length=50, num_series=2)
    val_dataset = SimulationQADataset("validation", "", length=50, num_series=2)
    test_dataset = SimulationQADataset("test", "", length=50, num_series=2)

    print(f"Train length: {len(train_dataset)}")
    print(f"Val length: {len(val_dataset)}")
    print(f"Test length: {len(test_dataset)}")
