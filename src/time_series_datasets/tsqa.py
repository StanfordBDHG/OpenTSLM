#
# This source file is part of the OpenTSLM open-source project
#
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

from typing import Literal

from datasets import load_dataset

from src.model_config import *
from time_series_datasets.util import load_qa_dataset


def get_tsqa_dataset(
    split: Literal["train", "validation", "test"] = "train",
    *,
    EOS_TOKEN,
    max_samples: int | None = None,
):
    return load_qa_dataset(
        load_dataset("ChengsenWang/TSQA", split="train"),
        split=split,
        max_samples=max_samples,
        EOS_TOKEN=EOS_TOKEN,
    )
