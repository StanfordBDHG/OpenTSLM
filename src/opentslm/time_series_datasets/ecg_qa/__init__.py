# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
ECG-QA Dataset Module

This module provides tools for working with the ECG-QA dataset, which combines
electrocardiogram data from PTB-XL with question-answer pairs for medical AI tasks.

The ECG-QA dataset was introduced in:
"ECG-QA: A Comprehensive Question Answering Dataset Combined With Electrocardiogram"
https://github.com/Jwoo5/ecg-qa

Usage:
    from opentslm.time_series_datasets.ecg_qa.ECGQADataset import ECGQADataset

    # Create dataset instance
    dataset = ECGQADataset(split="train", EOS_TOKEN="")

    # Access samples
    sample = dataset[0]
    print(sample["question"], sample["answer"])
"""

from .ecgqa_cot_loader import does_ecg_qa_cot_exist, download_ecg_qa_cot_if_not_exists, load_ecg_qa_cot_splits
from .ecgqa_loader import (
    does_ecg_qa_exist,
    does_ptbxl_exist,
    download_ecg_qa_if_not_exists,
    download_ptbxl_if_not_exists,
    load_ecg_qa_answers,
    load_ecg_qa_ptbxl_splits,
)
from .ECGQACoTQADataset import ECGQACoTQADataset
from .ECGQADataset import ECGQADataset

__all__ = [
    "ECGQACoTQADataset",
    "ECGQADataset",
    "does_ecg_qa_cot_exist",
    "does_ecg_qa_exist",
    "does_ptbxl_exist",
    "download_ecg_qa_cot_if_not_exists",
    "download_ecg_qa_if_not_exists",
    "download_ptbxl_if_not_exists",
    "load_ecg_qa_answers",
    "load_ecg_qa_cot_splits",
    "load_ecg_qa_ptbxl_splits",
]
