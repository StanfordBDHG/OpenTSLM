from collections import Counter, defaultdict
import itertools
import os
import sys
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from typing import List
from tqdm.auto import tqdm

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from model.llm.EmbedHealthFlamingo import EmbedHealthFlamingo
from src.model_config import (
    PATCH_SIZE,
)
from time_series_datasets.TSQADataset import TSQADataset
from time_series_datasets.util import (
    extend_time_series_to_match_patch_size_and_aggregate,
)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using device: {device}")

# Change based on the model you want to evaluate
model = EmbedHealthFlamingo(
    device=device,
    llm_id="google/gemma-2b",
)
model.load_from_file("../models/best_model.pt")
model.eval()

# Change based on the dataset that the model should be evaluated on
datasets = [TSQADataset]


def _merge_data_loaders(
    self,
    datasets: List[Dataset],
    shuffle: bool,
    batch_size: int,
    patch_size: int,
    distribute_data: bool = False,
) -> DataLoader:
    """Create a merged data loader from multiple datasets."""
    merged_ds = ConcatDataset(datasets)
    return DataLoader(
        merged_ds,
        shuffle=shuffle,
        batch_size=batch_size,
        collate_fn=lambda batch: extend_time_series_to_match_patch_size_and_aggregate(
            batch, patch_size=patch_size
        ),
    )


test_dataset_list = (
    [dataset("test", EOS_TOKEN=model.get_eos_token()) for dataset in datasets],
)
test_loader = _merge_data_loaders(
    test_dataset_list,
    shuffle=False,
    batch_size=1,
    patch_size=PATCH_SIZE,
    distribute_data=False,
)

# Creating a list in which the position at idx is pointing to the string name of the dataset
# e.g., mapping_index_to_dataset[0] = "TSQADataset"
mapping_index_to_dataset = itertools.chain(
    *[[dataset.__name__] * len(dataset) for dataset in test_dataset_list]
)
correct_matches_dict = defaultdict(int)
prog = tqdm(test_loader)
for idx, batch in enumerate(prog):
    output = model.generate(batch, max_new_tokens=30000)[0]

    # TODO check how we have to adapt this because exact string matching is probably too strict
    if batch[0]["answer"] == output:
        dataset_name = mapping_index_to_dataset[idx]
        correct_matches_dict[dataset_name] += 1


total_count_dict = Counter(mapping_index_to_dataset)
print("Accuracy calculations per dataset")
print("---------------------------------")
for dataset_name, correct_matches in correct_matches_dict.items():
    total_count = total_count_dict[dataset_name]
    acc = correct_matches / total_count
    print(f"{dataset_name:<30} Acc: {acc:.4f}")
