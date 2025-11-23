<!--
This source file is part of the OpenTSLM open-source project

SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)

SPDX-License-Identifier: MIT
-->

# Dataset Setup Summary - Branch: ts-exam-2

## Setup Status

Successfully set up datasets on branch `ts-exam-2` (created from `main`) for OpenTSLM training.

## Datasets Ready

### 1. TSQA (Stage 1 - Multiple Choice Questions)
- **Status:** ✅ Ready
- **Location:** HuggingFace cache (`~/.cache/huggingface/datasets/ChengsenWang___tsqa`)
- **Size:** 181 MB (255 MB arrow file)
- **Samples:** 48,000
- **Source:** `ChengsenWang/TSQA` on HuggingFace
- **Tasks:** Trend, Shape, Cyclic Pattern, Outlier Detection (multiple choice)
- **Features:** Task, Size, Question, Answer, Label, Series
- **Loading:**
  ```python
  from datasets import load_dataset
  ds = load_dataset('ChengsenWang/TSQA', split='train')
  ```

### 2. M4 (Stage 2 - Time Series Captioning)
- **Status:** ✅ Ready
- **Location:** `data/M4TimeSeriesCaptionDataset/M4TimeSeriesCaptionDataset/`
- **Size:** 163 MB uncompressed (89.4 MB zip)
- **Samples:** 100,000 total
  - Train: 80,000 (80%)
  - Validation: 10,000 (10%)
  - Test: 10,000 (10%)
- **Source:** ETH Zurich Polybox
- **Frequencies:**
  - Monthly: 48,000 samples (48%)
  - Quarterly: 24,000 samples (24%)
  - Yearly: 23,000 samples (23%)
  - Daily: 4,227 samples (4%)
  - Hourly: 414 samples (0.4%)
  - Weekly: 359 samples (0.4%)
- **Files:**
  - `m4_series_{frequency}.csv` - Time series data (JSON arrays)
  - `m4_captions_{frequency}.csv` - Generated captions
- **Loading:**
  ```python
  from src.time_series_datasets.m4.m4_loader import load_all_m4_data, create_combined_dataset
  data_dict = load_all_m4_data()
  train, val, test = create_combined_dataset(data_dict)
  ```

### 3. MIMIC-IV-ECG (Stage 5 - ECG Question Answering with CoT)
- **Status:** ⏳ Downloading in background
- **Location:** `data/mimic_iv_ecg/physionet.org/files/`
- **Expected Size:** ~33.8 GB compressed, ~90.4 GB uncompressed
- **Download Log:** `mimic_download.log`
- **Process PID:** Check with `ps aux | grep wget | grep mimic`
- **Note:** Download started on `ecg-qa` branch, data is shared across branches
- **Details:** See `understand_data.md` for complete information

## Training Pipeline Stages

Based on `README.md`, the curriculum learning stages are:

1. **Stage 1 (MCQ):** TSQA dataset - Multiple choice questions ✅
2. **Stage 2 (Captioning):** M4 dataset - Time series caption generation ✅
3. **Stage 3 (CoT):** HAR dataset - Chain-of-thought on activity recognition
4. **Stage 4 (Sleep CoT):** SleepEDF dataset - Chain-of-thought on sleep stages
5. **Stage 5 (ECG CoT):** ECG-QA dataset - Chain-of-thought on ECG analysis ⏳

## Dependencies Installed

- `scikit-learn` (1.7.2) - Required for M4 dataset splitting
- `joblib` (1.5.2) - Dependency of scikit-learn
- `threadpoolctl` (3.6.0) - Dependency of scikit-learn

## Git Branch Information

- **Branch:** ts-exam-2
- **Based on:** main (commit: 0b60c37)
- **Previous branch:** ecg-qa
- **Changes:** Clean branch from main, datasets added locally (gitignored)

## Quick Verification

```bash
# Verify M4 dataset
python3 src/time_series_datasets/m4/m4_loader.py

# Verify TSQA dataset
python3 -c "from datasets import load_dataset; ds = load_dataset('ChengsenWang/TSQA', split='train'); print(f'TSQA: {len(ds)} samples')"

# Check MIMIC-IV-ECG download progress
tail -f mimic_download.log
```

## Next Steps

1. Complete MIMIC-IV-ECG download (monitor with `tail -f mimic_download.log`)
2. Download remaining datasets:
   - HAR (Human Activity Recognition)
   - SleepEDF
   - ECG-QA repository (if not already cloned)
3. Begin training Stage 1 with TSQA dataset
4. Progress through curriculum stages

## Storage Usage

```
TSQA:         181 MB (in HuggingFace cache)
M4:           163 MB (in data/)
MIMIC-IV-ECG: ~90 GB (when download completes)
Total:        ~90.3 GB
```

---
*Created: 2025-11-20*
*Branch: ts-exam-2*
