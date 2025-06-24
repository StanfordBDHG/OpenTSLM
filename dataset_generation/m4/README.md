# M4 Dataset Caption Generation & OpenAI Batch Processing

## ♻️ Summary

- **Caption Generation:**  
  Scripts in this folder load the M4 time series dataset and generate a detailed caption for each series using OpenAI's GPT-4o model.  
  The generated dataset (captions and series) is stored in intervals as `.csv` (and can be converted to `.parquet` for use with the `M4QADataset`).

- **Batch Processing:**  
  Includes scripts to generate an OpenAI Batch file, enabling batched completion requests to OpenAI (reducing API costs by half).

- **QADataset Integration:**  
  Provides `M4QADataset`, which loads the previously generated dataset as a QA-style dataset for downstream tasks.

- **Normalization:**  
  Updates normalization logic to robustly handle missing and extreme values in the time series.

---

## 📦 Batches (OpenAI)

### 1. Generate Batch Requests

Run the following script to generate a `.jsonl` file with OpenAI batch requests. This also creates `m4_series.csv` with the series IDs and time series data.

```bash
python generate_m4_captions_batch.py
```

- **Output:**  
  - `m4_caption_requests.jsonl` — OpenAI batch API requests  
  - `m4_series.csv` — Series IDs and raw time series data

### 2. Upload Batch Requests

Upload the generated requests to OpenAI and start the batch execution:

```bash
python push_batch_requests.py
```

- **Note:**  
  Results for all batches will be available within 24 hours after upload.

### 3. Download Batch Results

Download the results for the batched requests and store the results (series ID and generated caption) in `m4_captions.csv`:

```bash
python pull_batch_response.py
```

- **Output:**  
  - `m4_captions.csv` — Series IDs and generated captions

---

## 🧩 Additional Scripts

- `generate_m4_captions.py`  
  Generates captions for the M4 dataset using the OpenAI API (single-request mode, not batched). Stores results in `m4_captions_series.csv`.

---

## 🗃️ Dataset Usage

- **M4QADataset**  
  The `M4QADataset` class (in the `time_series_datasets` package) loads the generated dataset (expects a `.parquet` file, e.g., `m4_captions_series.parquet`) and provides train/validation/test splits for QA tasks.

---

## 📝 Notes

- **Normalization:**  
  All time series are normalized to handle missing and extreme values before being sent to the model.

- **File Formats:**  
  - `.jsonl` — OpenAI batch API requests  
  - `.csv` — Intermediate and final results  
  - `.parquet` — Used by `M4QADataset` for efficient loading (convert from `.csv` as needed)

- **Dependencies:**  
  - Python 3.8+  
  - `openai`, `pandas`, `torch`, `numpy`, `matplotlib`

---

## 📂 Files in this folder

- `generate_m4_captions_batch.py` — Generate OpenAI batch requests and series CSV
- `push_batch_requests.py` — Upload batch requests to OpenAI
- `pull_batch_response.py` — Download and parse batch results
- `generate_m4_captions.py` — (Optional) Generate captions in single-request mode
- `.gitignore` — Ignore intermediate files (`*.png`, `*.txt`, `*.parquet`)

--- 