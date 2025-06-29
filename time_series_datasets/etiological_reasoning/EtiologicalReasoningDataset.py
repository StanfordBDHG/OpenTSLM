import json
from typing import Literal, List, Optional, Tuple
import random
import numpy as np

from datasets import Dataset, Value, Sequence, Features, load_dataset
from prompt.text_time_series_prompt import TextTimeSeriesPrompt
from time_series_datasets.QADataset import QADataset
from time_series_datasets.util import (
    extend_time_series_to_match_patch_size_and_aggregate,
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torch

from etiological_reasoning_loader import load_etiological_reasoning_dataset
TEST_FRAC = 0.1
VAL_FRAC = 0.1

# @Winnie: We have a common QADataset class which makes it easy
# to provide TextTimeSeries samples for our models.
# It has some functions which are automatically called by our dataset loader,
# and then they are automatically preprocessed for the model.
# You simply need to:
# 1) Implement the _load_splits to return the train, val, test splits. 
# 2) Implement _get_answer to extract the answer from a row (one row would be one sample)
# 3) Implement _get_pre_prompt to extract the question from a row.
# 4) Implement _get_post_prompt to format how the answer should be prompted. (e.g. "What does this time series represent?")
# 5) Implement _get_text_time_series_prompt_list to convert the time series data
#    into a list of TextTimeSeriesPrompt objects, which include both the text
#    description and the normalized time series data.  
#    If it is only 1 time series per sample, you can just return a list with 1 TextTimeSeriesPrompt.
# 

# You don't need to care about batching here, our loder class later does that automatically..

# I have provided you below with an example from the TSQA dataset, which you could adopt.

class EtiologicalReasoningDataset(QADataset):
    def __init__(self, split: Literal["train", "test", "validation"], EOS_TOKEN: str):
        # Generate shuffling indices for options before calling parent constructor
        self._generate_shuffling_indices()
        super().__init__(split, EOS_TOKEN)
    
    def _generate_shuffling_indices(self):
        """Generate random positions (0-3) for each sample to shuffle options."""
        # Load all datasets to get total size
        dataset = load_etiological_reasoning_dataset()
        train_data = dataset["train"]
        val_data = dataset["val"]
        test_data = dataset["test"]
        total_size = len(train_data) + len(val_data) + len(test_data)
        
        # Generate random positions for each sample
        np.random.seed(42)  # For reproducibility
        self.shuffling_indices = np.random.randint(0, 4, size=total_size)
        self.current_index = 0
    
    def _load_splits(self) -> Tuple[Dataset, Dataset, Dataset]:
        # This automatically downloads the dataset if it does not exist,
        # and returns three separate datasets for train, validation, and test.
        dataset = load_etiological_reasoning_dataset()
        return dataset["train"], dataset["val"], dataset["test"]

    def _get_answer(self, row) -> str:
        # Get the target position for this sample
        target_pos = self.shuffling_indices[self.current_index]
        self.current_index += 1
        
        # Map position to letter (0->A, 1->B, 2->C, 3->D)
        answer_letter = chr(65 + target_pos)  # 65 is ASCII for 'A'
        return answer_letter

    def _get_pre_prompt(self, row) -> str:
        options = row["options"]
        if len(options) != 4:
            return str(options)  # Fallback if not exactly 4 options
        
        # Get the target position for this sample
        target_pos = self.shuffling_indices[self.current_index - 1]  # Use same index as _get_answer
        
        # Create shuffled options: move correct answer (first option) to target position
        shuffled_options = options.copy()
        correct_answer = shuffled_options[0]
        
        # Remove correct answer from first position
        shuffled_options.pop(0)
        
        # Insert correct answer at target position
        shuffled_options.insert(target_pos, correct_answer)
        
        # Format as A, B, C, D options
        formatted_options = []
        for i, option in enumerate(shuffled_options):
            letter = chr(65 + i)  # A, B, C, D
            formatted_options.append(f"{letter}. {option}")
        
        # Combine description with options
        options_text = "\n".join(formatted_options)
        
        return f"Options:\n{options_text}"

    def _get_post_prompt(self, row) -> str:
        # Format the post prompt to ask for the answer
        return "What scenario could have produced this time series? Choose A, B, C, or D."

    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        # Get the time series data from the 'series' field
        series_data = row.get("series", [])
        
        if not series_data:
            # Return empty series if no data
            return [TextTimeSeriesPrompt("No time series data available.", [])]
        
        # Convert to tensor for normalization
        series = torch.tensor(series_data, dtype=torch.float32)
        
        # Normalize the time series
        means = series.mean(dim=0, keepdim=True)  # shape: (n_series, 1)
        stds = series.std(dim=0, keepdim=True)  # shape: (n_series, 1)
        series = (series - means) / (stds + 1e-8)  # broadcasts to (n_series, length)
        
        # Create a description for the time series
        characteristics = row.get("characteristics", "")
        description = f"Time series data: {characteristics}" if characteristics else "Time series data"
        return [TextTimeSeriesPrompt(description, series.tolist())]


if __name__ == "__main__":
    # This is just for testing.
    train = EtiologicalReasoningDataset("train", "")
    val = EtiologicalReasoningDataset("validation", "")
    test = EtiologicalReasoningDataset("test", "")


    from collections import Counter

    train_values = [
        (el[0], el[1] / len(train))
        for el in Counter(map(lambda x: x["post_prompt"], train)).items()
    ]
    print("train", train_values)
    val_values = [
        (el[0], el[1] / len(val))
        for el in Counter(map(lambda x: x["post_prompt"], val)).items()
    ]
    print("val", val_values)
    test_values = [
        (el[0], el[1] / len(test))
        for el in Counter(map(lambda x: x["post_prompt"], test)).items()
    ]
    print("test", test_values)


