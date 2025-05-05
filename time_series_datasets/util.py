import ast
from functools import wraps
from datasets import Dataset
from typing import Literal, Optional
from model_config import PATCH_SIZE

import torch


def collate_fn(batch, *, patch_size: int = PATCH_SIZE):
    """Pad variable-length series so each sample length is a multiple of *patch_size*."""

    for idx, element in enumerate(batch):
        # print("ex", ex)
        ts = element["time_series"]
        padded_length = (ts.size(0) + patch_size - 1) // patch_size * patch_size
        if ts.size(0) < padded_length:
            pad = padded_length - ts.size(0)
            ts = torch.nn.functional.pad(ts, (0, pad), "constant", 0)
        else:
            ts = ts[:padded_length]
        batch[idx]["time_series"] = ts

    return batch


def load_qa_dataset(
    ds_full: Dataset,
    split: Literal["train", "validation", "test"],
    EOS_TOKEN,
    *,
    max_samples: Optional[int],
    val_frac: float = 0.1,
    test_frac: Optional[float] = 0.1,
    seed: int = 42,
):
    """Load the TSQA dataset with an explicit **train/validation/test** split.

    Args:
        split: which split to return.
        max_samples: optional cap on number of samples *after* splitting.
        val_frac: fraction (0–1) of the original data used for **validation**.
        test_frac: fraction (0–1) of the original data used for **test**.
        seed: RNG seed to make splits deterministic.
    Returns:
        ``datasets.Dataset`` with columns ["ts", "question", "answer"].
    """

    # Setting default values
    train = ds_full
    val = None
    test = None

    if test_frac is not None and test_frac > 0:
        # 2) First carve out the test split
        train, test = ds_full.train_test_split(test_size=test_frac, seed=seed).values()
    elif split == "test":
        test = ds_full
        train = None

    # 3) From the remaining data take validation
    if test_frac is None:
        test_frac = 0
    if val_frac > 0:
        train, val = train.train_test_split(
            test_size=val_frac / (1 - test_frac), seed=seed + 1
        ).values()

    # 4) Choose the requested split
    if split == "train":
        ds = train
    elif split in {"validation", "val"}:
        ds = val
    elif split == "test":
        ds = test
    else:
        raise ValueError("split must be 'train', 'validation', or 'test'")

    # 5) Optional size cap
    if max_samples is not None and max_samples < len(ds):
        ds = ds.select(range(max_samples))

    # 6) Pre-processing helper
    def _preprocess(ex):
        # --- normalise time‑series ---
        series = None

        series = ex["TextTimeSeriesPromptTimeSeries"]
        if isinstance(series, str):
            series = ast.literal_eval(series)

        series = torch.tensor(series, dtype=torch.float32)

        means = series.mean(dim=1, keepdim=True)  # shape: (n_series, 1)
        stds = series.std(dim=1, keepdim=True)  # shape: (n_series, 1)
        series = (series - means) / (stds + 1e-8)  # broadcasts to (n_series, length)

        # --- clean Q/A and ensure EOS token ---
        pre_prompt = ex["PrePrompt"].strip()
        post_prompt = ex["PostPrompt"].strip()
        time_series_text = ex["TextTimeSeriesPromptTexts"]
        answer = ex["Answer"].strip()

        if not answer.endswith(EOS_TOKEN):
            answer += EOS_TOKEN

        return {
            "time_series": series,
            "time_series_text": time_series_text,
            "pre_prompt": pre_prompt,
            "post_prompt": post_prompt,
            "answer": answer,
        }

    ds = ds.map(_preprocess)
    columns = ["time_series", "time_series_text", "pre_prompt", "post_prompt", "answer"]
    ds.set_format(type="torch", columns=columns)
    ds = ds.remove_columns(list(set(ds.column_names) - set(columns)))

    return ds


def torch_to_hf_generator(torch_ds):
    for val in torch_ds:
        yield {
            "PrePrompt": val["pre_prompt"],
            "TextTimeSeriesPromptTexts": val["text_time_series_prompt_texts"],
            "TextTimeSeriesPromptTimeSeries": val[
                "text_time_series_prompt_time_series"
            ],
            "PostPrompt": val["post_prompt"],
            "Answer": val["answer"],
        }


def SingletonDataset(cls):
    @wraps(cls)
    def get_instance():
        if not hasattr(cls, "_instance"):
            cls._instance = cls()
        return cls._instance

    return get_instance
