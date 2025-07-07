import torch
import numpy as np
import pandas as pd

import sys
import os
print(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from model.llm.EmbedHealthFlamingo import EmbedHealthFlamingo
from prompt.text_prompt import TextPrompt
from prompt.text_time_series_prompt import TextTimeSeriesPrompt
from prompt.prompt_with_answer import PromptWithAnswer
from time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate

# 1. Load the model
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")
model = EmbedHealthFlamingo(
    device=device,
    llm_id="google/gemma-2b",  # or whatever you used for training
)

# Manual checkpoint loading (as in curriculum_learning.py)
checkpoint = torch.load("../models/best_model.pt", map_location=device)
if "llm" in checkpoint:
    model_state = checkpoint["llm"]
elif "model_state" in checkpoint:
    model_state = checkpoint["model_state"]
else:
    raise RuntimeError("No recognized model state key in checkpoint.")

# Handle DDP (DistributedDataParallel) if needed
if hasattr(model, 'module'):
    model_state = {f'module.{k}': v for k, v in model_state.items()}

# Remove 'model.' prefix if present in checkpoint keys
if all(k.startswith('model.') for k in model_state.keys()):
    model_state = {k.replace('model.', '', 1): v for k, v in model_state.items()}
print(model_state)

# Load state dict with strict=False to handle missing/unexpected keys
missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False)
if missing_keys:
    print(f"⚠️  Warning: Missing keys when loading checkpoint:")
    for key in missing_keys[:10]:
        print(f"   - {key}")
    if len(missing_keys) > 10:
        print(f"   ... and {len(missing_keys) - 10} more keys")
if unexpected_keys:
    print(f"⚠️  Warning: Unexpected keys when loading checkpoint:")
    for key in unexpected_keys[:10]:
        print(f"   - {key}")
    if len(unexpected_keys) > 10:
        print(f"   ... and {len(unexpected_keys) - 10} more keys")

model = model.to(device)
model.eval()

# 2. Load the M4 series-M42150 from CSV
csv_path = "../data/m4/m4_series_Monthly.csv"
df = pd.read_csv(csv_path)
row = df[df['id'] == 'series-M42150'].iloc[0]
series_str = row['series']
# Remove brackets and split
series = [float(x) for x in series_str.strip('[]').replace('\n', '').replace(' ', '').split(',') if x]
series = np.array(series, dtype=np.float32)
mean = series.mean()
std = series.std()
normalized_series = (series - mean) / std if std > 0 else series - mean

# 3. Build the prompt
pre_prompt = TextPrompt("You are an expert in time series analysis.")
ts_text = f"This is the time series, it has mean {mean:.4f} and std {std:.4f}:"
ts_prompt = TextTimeSeriesPrompt(ts_text, normalized_series.tolist())
post_prompt = TextPrompt("Please generate a detailed caption for this time-series, describing it as accurately as possible.")

# 4. Build the batch (list of dicts)
prompt_with_answer = PromptWithAnswer(
    pre_prompt,
    [ts_prompt],
    post_prompt,
    answer=""  # Leave blank for inference
)
batch = [prompt_with_answer.to_dict()]

# Ensure time series are padded to patch size
model.eval()
batch = extend_time_series_to_match_patch_size_and_aggregate(batch)
loss = model.compute_loss(batch)
print(f"Loss: {loss}")



# 5. Run inference
output = model.generate(batch, max_new_tokens=30000)
print("Model output:", output[0])