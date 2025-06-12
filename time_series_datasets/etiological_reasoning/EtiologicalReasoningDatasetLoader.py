import json
from typing import Literal, List, Optional, Tuple


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

class EtiologicalReasoningDatasetLoader(QADataset):
    def _load_splits(self) -> Tuple[Dataset, Dataset, Dataset]:

        # This automatically downloads the dataset if it does not exist,
        # and returns a pandas dataframe.
        full_er_data = load_etiological_reasoning_dataset()

        # 1) Load the single built‑in "train" split (≈ 7 k rows)
        ds_full = load_dataset("ChengsenWang/TSQA", split="train")

        # 2) First carve out the test split
        train_val, test = ds_full.train_test_split(
            test_size=TEST_FRAC, seed=42
        ).values()

        # 3) From the remaining data take validation
        train, val = train_val.train_test_split(
            test_size=VAL_FRAC / (1 - TEST_FRAC), seed=42
        ).values()

        return train, val, test

    # This will be a called for each row
    # You only need to return the answer as a string.
    def _get_answer(self, row) -> str:
        return row["Answer"]

    def _get_pre_prompt(self, row) -> str:
        return row["Question"]

    def _get_post_prompt(self, row) -> str:
        # return "Answer:"
        return "Predict the " + row["Task"] + " Answer:"

    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        # TODO standardize normalization over the all datasets
        series = torch.tensor(json.loads(row["Series"]), dtype=torch.float32)

        means = series.mean(dim=0, keepdim=True)  # shape: (n_series, 1)
        stds = series.std(dim=0, keepdim=True)  # shape: (n_series, 1)
        series = (series - means) / (stds + 1e-8)  # broadcasts to (n_series, length)
        # TSQA has always only one time series
        return [TextTimeSeriesPrompt("This is the time series.", series.tolist())]


if __name__ == "__main__":
    # This is just for testing.
    train = EtiologicalReasoningDatasetLoader("train", "")
    val = EtiologicalReasoningDatasetLoader("validation", "")
    test = EtiologicalReasoningDatasetLoader("test", "")


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
