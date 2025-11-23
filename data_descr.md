<!--
This source file is part of the OpenTSLM open-source project

SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)

SPDX-License-Identifier: MIT
-->

The Key Abstraction: Unified Input Format

  All datasets, regardless of their structure, are converted to a standardized format before reaching the model (QADataset.py:94-104):

  {
    "pre_prompt": str,           # Text before time series
    "time_series": List[np.ndarray],  # List of 1D time series
    "time_series_text": List[str],    # Text labels for each series
    "post_prompt": str,          # Text after time series
    "answer": str                # Expected output
  }

  How Different Datasets Map to This Format

  TSQA (TSQADataset.py:78):
  
1 time series per sample (univariate)
time_series = [single_1d_array]
Example: "This is the time series, it has mean 0.0234 and std 1.4567"

  M4 (M4QADataset.py:101):
  
1 time series per sample (univariate)
time_series = [single_1d_array]
Example: "This is the time series, it has mean 123.45 and std 67.89"

  HAR (HARCoTQADataset.py:149):
  
3 time series per sample (3-axis accelerometer: x, y, z)
time_series = [x_axis_1d, y_axis_1d, z_axis_1d]
Example: 3 separate prompts for x, y, z axes

  ECG-QA (likely similar):
  
12 time series per sample (12-lead ECG)
time_series = [lead1_1d, lead2_1d, ..., lead12_1d]

  How the Model Processes This

  The Flamingo model treats each time series as a visual token (OpenTSLMFlamingo.py:210-212):

  
Text encoding: pre_prompt + <image> + ts_text + <endofchunk> + ... + post_prompt
Time series encoding: Each 1D time series → CNN encoder → perceiver → embeddings
Cross-attention: LLM attends to time series embeddings via gated cross-attention layers

  The model doesn't care about:
  
How many time series there are (1 for TSQA/M4, 3 for HAR, 12 for ECG)
What the time series represent (stock prices, accelerometer, ECG)
How long each series is (padding handles variable lengths)

  The Genius of This Design

  Dataset-agnostic processing: Each 1D time series is independently encoded and positioned in the sequence via special tokens. The model learns:
  
Stage 1 (TSQA): "Understand single time series + answer MCQs"
Stage 2 (M4): "Generate detailed captions for single time series"
Stage 3+ (HAR/ECG): "Reason across multiple related time series"

  Variable number of series: OpenTSLMFlamingo.py:161-187 pads time series to the same length within a batch, and the text encoding loop (lines 208-214)
  handles any number of time series in the list.

  Key constraint (text_time_series_prompt.py:31): Each time series MUST be 1D. Multivariate data (like 3-axis accelerometer) is split into multiple 1D
  series, not treated as a 2D array.