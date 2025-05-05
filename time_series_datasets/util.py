import ast
from functools import wraps
from datasets import Dataset
from typing import Literal, Optional
from model_config import PATCH_SIZE

import torch


def collate_fn(batch, *, patch_size: int = PATCH_SIZE):
    """Pad variable-length series so each sample length is a multiple of *patch_size*."""

    max_len = max(ex["ts"].size(0) for ex in batch)
    max_len = ((max_len + patch_size - 1) // patch_size) * patch_size

    ts_list, qs, ans = [], [], []
    for ex in batch:
        # print("ex", ex)
        ts = ex["ts"]
        if ts.size(0) < max_len:
            pad = max_len - ts.size(0)
            ts = torch.nn.functional.pad(ts, (0, pad), "constant", 0)
        else:
            ts = ts[:max_len]
        ts_list.append(ts)

        qs.append(ex["question"] + "\nAnswer:")
        ans.append(ex["answer"])

    # print(ts_list, qs, ans)

    return torch.stack(ts_list), qs, ans


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
        if isinstance(ex["Series"], str):
            series = torch.tensor(ast.literal_eval(ex["Series"]), dtype=torch.float32)
        else:
            series = torch.tensor(ex["Series"], dtype=torch.float32)

        series = (series - series.mean()) / (series.std() + 1e-8)

        # --- clean Q/A and ensure EOS token ---
        question = ex["Question"].strip()
        answer = ex["Answer"].strip()
        if not answer.endswith(EOS_TOKEN):
            answer += EOS_TOKEN

        return {"ts": series, "question": question, "answer": answer}

    ds = ds.map(_preprocess)
    columns = ["ts", "question", "answer"]
    ds.set_format(type="torch", columns=columns)
    ds = ds.remove_columns(list(set(ds.column_names) - set(columns)))

    return ds


def torch_to_hf_generator(torch_ds):
    for wrapped_time_series, question, value in torch_ds:
        clean = {
            "Series": list(wrapped_time_series[0]),
            "Answer": str(value),
            "Question": question,
        }
        yield clean


def SingletonDataset(cls):
    @wraps(cls)
    def get_instance():
        if not hasattr(cls, "_instance"):
            cls._instance = cls()
        return cls._instance

    return get_instance
