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


TEST_FRAC = 0.1
VAL_FRAC = 0.1


class TSQADataset(QADataset):
    def _load_splits(self) -> Tuple[Dataset, Dataset, Dataset]:
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

    def _get_answer(self, row) -> str:
        return row["Answer"]

    def _get_pre_prompt(self, row) -> str:
        return row["Question"]

    def _get_post_prompt(self, row) -> str:
        return "Answer:"

    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        # TODO standardize normalization over the all datasets
        series = torch.tensor(json.loads(row["Series"]), dtype=torch.float32)

        means = series.mean(dim=0, keepdim=True)  # shape: (n_series, 1)
        stds = series.std(dim=0, keepdim=True)  # shape: (n_series, 1)
        series = (series - means) / (stds + 1e-8)  # broadcasts to (n_series, length)

        # TSQA has always only one time series
        return [TextTimeSeriesPrompt("time series", series.tolist())]


if __name__ == "__main__":
    dataset = TSQADataset("train", "")

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=lambda batch: extend_time_series_to_match_patch_size_and_aggregate(
            batch, patch_size=4
        ),
    )

    for batch in tqdm(dataloader):
        print(batch)
